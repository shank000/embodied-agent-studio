"""Native RoboDojo session, imported only inside its Isaac Sim process.

Reuse the pinned EvalEnv, layout loader, IK, control interpolation, cameras and
task reward implementation. Host-driven research sessions do not start a policy
WebSocket client or write an official leaderboard result.
"""
from __future__ import annotations

import hashlib
from copy import deepcopy
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np


def array(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def seed_parts(seed):
    if type(seed) is not int or not 0 <= seed <= 4294967295:
        raise ValueError('布局种子须为非负整数。')
    # Bijection, never wrap a finite layout pool and call it a new initial state.
    return seed % 3, seed // 3


class HostPolicyBoundary:
    """EvalEnv.reset's policy reset is owned by the workbench controller.

    The native session receives explicit actions; accidental use of run_eval or
    its policy methods fails rather than silently returning dummy actions.
    """
    def __init__(self, **kwargs): pass

    def call(self, *, func_name, **kwargs):
        if func_name != 'reset':
            raise RuntimeError('交互会话由工作台提供动作，不运行官方外部策略客户端。')

    def close(self): pass


class NativeSession:
    def __init__(self, task_id, seed):
        self.app = self.env = None
        from server.robodojo import runtime_config
        config = runtime_config()
        if config is None:
            raise ValueError('未配置 RoboDojo 原生运行时；请提供包含 Isaac Sim 5.1 与官方资产的 Linux GPU 环境。')
        self.source_root = Path(config['source_root']).resolve()
        os.chdir(self.source_root)
        sys.path[:0] = [str(self.source_root), str(self.source_root / 'XPolicyLab')]
        try:
            from isaaclab.app import AppLauncher
            from env.camera_manager.capture.render_sync import add_zero_delay_kit_args
            args = SimpleNamespace(headless=True, enable_cameras=True, device='cuda:0',
                kit_args='--enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera')
            add_zero_delay_kit_args(args)
            self.app = AppLauncher(vars(args)).app
            self.reset(task_id, seed)
        except BaseException as exc:
            try: self.close()
            except Exception as cleanup: exc.add_note(f'RoboDojo 初始化清理失败：{cleanup}')
            raise

    def reset(self, task_id, seed):
        from omegaconf import OmegaConf
        from env.global_configs import ENV_CONFIG_PATH, BENCHMARK
        from utils.load_file import load_yaml
        from utils.pipeline_utils import process_config, process_randomization
        from task.RoboDojo import task_registry
        from src.eval_client import eval_env
        eval_seed, layout_index = seed_parts(seed)
        if self.env is not None:
            self.env.close()
            self.env = None
        base = Path(ENV_CONFIG_PATH)
        cfg = load_yaml(str(base / 'arx_x5.yml'))
        cfg.update(task_name=task_id, num_envs=1, device_id=0, eval_batch=False,
                   policy_name='embodiworks_interactive', additional_info='research',
                   seed=eval_seed, physx_monitor_enabled=False)
        sections = {key: load_yaml(str(base / key / (cfg['config'][key] + '.yml')))
                    for key in ('sim', 'scene', 'camera', 'robot')}
        sections['task_env'] = load_yaml(task_registry.task_config_path(
            str(self.source_root / 'task' / BENCHMARK / 'config'), task_id))
        sections['eval_cfg'] = cfg
        sections['deploy_cfg'] = {'port': 1, 'host': '127.0.0.1', 'policy_name': 'embodiworks_interactive'}
        full = OmegaConf.create(sections)
        full.sim.scene.num_envs = 1
        full.sim.seed = [0]
        full = process_randomization(full)
        full, eval_num = process_config(full, task_name=task_id)
        full.eval_cfg.eval_num = eval_num
        full.camera.default_frequency = cfg['observation']['collect_freq']
        # The official factory has no client injection argument. Bind only for
        # construction in this isolated process and restore it immediately.
        original = eval_env.WsModelClient
        try:
            eval_env.WsModelClient = HostPolicyBoundary
            self.env = eval_env.create_eval_env(full, self.app)
        finally:
            eval_env.WsModelClient = original
        # Platform video recorder owns recording; avoid an unfinalized duplicate
        # in the official evaluator's temporary video directory.
        self.env._stream_vision = lambda env_idx, frame: None
        info = self.env.seed_manager.seed_info.get(layout_index)
        if info is None:
            raise ValueError(f'{task_id} 的官方 seed={eval_seed} 没有 layout_index={layout_index}；'
                             '请选已有布局，平台不会取模复用初态。')
        path = Path(info['scene_layout'])
        self.initial_state = {'task_id': task_id, 'eval_seed': eval_seed, 'layout_index': layout_index,
                              'layout_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        self.env.reset(seed=[layout_index])
        # Official EE actions use coordinates relative to each replicated env
        # origin. This adapter exposes one world-frame environment only.
        origin = array(self.env.env_origins)[0]
        if origin.shape != (3,) or not np.allclose(origin, 0., atol=1e-7):
            raise ValueError('单场景原点不为零，不能将原生相对坐标标记为 world；请检查官方场景配置。')
        self.env.run_reward()
        if hasattr(self.env, 'get_score'): self.env.get_score()
        if getattr(self.env, 'interact', False) and hasattr(self.env, 'query_support_arm_traj'):
            self.env.query_support_arm_traj(env_idx=0)
        self.timestep = float(self.env.dt * int(self.env.obs_manager.collect_interval))
        if not np.isfinite(self.timestep) or self.timestep <= 0:
            raise ValueError('RoboDojo 官方观察/动作周期无效。')
        self.elapsed = 0.
        self._robots = [r for r in self.env.robot_manager.robot_list if r.type == 'target']
        if len(self._robots) != 2:
            raise ValueError('该官方任务当前未提供双 ARX X5 目标机械臂。')
        # Sort by native arm labels, not unspecified model construction order.
        self._robots.sort(key=lambda r: 0 if r.arm_name.startswith('left_') else 1)
        if [r.arm_name.split('_')[0] for r in self._robots] != ['left', 'right']:
            raise ValueError('无法确认官方左右臂身份。')
        self.home_action = self.hold_action()

    def robot_states(self):
        manager = self.env.robot_manager
        result = []
        for idx, robot in enumerate(self._robots):
            key = manager.robot_key[manager.robot_list.index(robot)]
            indices = robot.arm_joint_indices
            q = array(key.data.joint_pos)[0, indices]
            v = array(key.data.joint_vel)[0, indices]
            limits = array(key.data.soft_joint_pos_limits)[0, indices]
            names = [key.joint_names[i] for i in indices]
            pose = array(manager.get_real_endpose(robot, [0], is_relative=True)[0])
            actual = float(manager.get_end_effector_real_val(robot, [0])[0][0])
            lo, hi = robot.gripper_scale
            openness = (actual - lo) / (hi - lo)
            if robot.gripper_move['sign'] != 1: openness = 1 - openness
            target = manager.control_manager.prev_control[0][manager.process_name(robot.gripper_name)]['position'][0]
            target = (target - lo) / (hi - lo)
            if robot.gripper_move['sign'] != 1: target = 1 - target
            values = np.r_[q, v, limits.ravel(), pose, actual, openness, target]
            if not np.isfinite(values).all(): raise RuntimeError('RoboDojo 机器人状态出现非有限值。')
            if len(q) != 6: raise ValueError('当前双 X5 适配要求每臂 6 个关节。')
            result.append({'id': f'arm{idx+1}', 'side': ('left', 'right')[idx],
                'joint_names': names, 'qpos': q.tolist(), 'qvel': v.tolist(), 'limits': limits.tolist(),
                'pose': pose.tolist(), 'gripper_actual': actual, 'gripper_open': float(np.clip(openness, 0, 1)),
                'gripper_target': float(np.clip(target, 0, 1))})
        return result

    def hold_action(self):
        result = {}
        for r in self.robot_states():
            result[r['side'] + '_arm_joint_state'] = r['qpos']
            result[r['side'] + '_ee_joint_state'] = [r['gripper_target']]
        return result

    def advance(self, action):
        # Upstream take_action skips failed IK without surfacing an error. Check
        # all requested EE targets first, before it can move either gripper.
        manager = self.env.robot_manager
        native_action = deepcopy(action)
        for robot in self._robots:
            name = robot.arm_name.split('_')[0] + '_ee_pose'
            if name in action:
                solved = manager.solve_ik(target_pose=action[name], env_idx=0, robot=robot)
                if solved is None or solved.get('status') != 'Success':
                    raise ValueError(f'{name} 的官方 IK 无解，未执行该动作。')
                joints = array(solved['joint_value']).reshape(-1)
                if joints.size != 6 or not np.isfinite(joints).all():
                    raise ValueError(f'{name} 的官方 IK 返回无效关节结果，未执行该动作。')
                native_action.pop(name)
                native_action[robot.arm_name + '_joint_state'] = joints.tolist()
        before = int(self.env.take_action_cnt[0])
        # Feed the already verified official IK solution to the official joint
        # path. Re-running stochastic IK in take_action could fail silently.
        try:
            self.env.take_action(native_action)
        except Exception as exc:
            # The native call can advance physics before reward/capture fails.
            # Only the pre-execution IK check is a recoverable ValueError.
            raise RuntimeError(f'RoboDojo 原生动作执行失败，物理状态可能已变化：{type(exc).__name__}: {exc}') from exc
        after = int(self.env.take_action_cnt[0])
        if after > before: self.elapsed += self.timestep

    def observe(self):
        self.env.obs_manager.render_for_capture()
        return self.env.obs_manager.get_obs(env_idx_list=[0])[0]

    def evaluate(self):
        try:
            self.env.is_episode_end()
            scores = self.env.reward_manager.get_score() if hasattr(self.env, 'get_score') else None
        except Exception as exc:
            raise RuntimeError(f'RoboDojo 原生任务判定失败：{type(exc).__name__}: {exc}') from exc
        end = bool(self.env.end_flag[0])
        unstable = 0 in self.env.unstable_envs
        score = float(scores[0]) if scores is not None else None
        if score is not None and not np.isfinite(score):
            raise RuntimeError('RoboDojo 官方奖励返回非有限分数。')
        return {'terminal': end, 'success': end and bool(self.env.success[0]) and not unstable,
                'unstable': unstable, 'score': score,
                'instruction': self.env.obs_manager.instruction[0],
                'action_count': int(self.env.take_action_cnt[0]), 'action_limit': int(self.env.step_lim)}

    def close(self):
        try:
            if self.env is not None: self.env.close()
        finally:
            self.env = None
            if self.app is not None: self.app.close()
            self.app = None
