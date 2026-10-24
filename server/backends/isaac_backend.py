"""Isaac Lab 3.0 workbench: Newton kitless, or explicitly installed Isaac Sim.

The Newton route uses Isaac Lab's official MJWarp solver, USD importer, PD drives,
DifferentialIKController, and Warp renderer. It does not run the MuJoCo workbench.
The public contract is world metres/WXYZ; Isaac Lab 3.0 tensors use XYZW.
"""
from __future__ import annotations

import base64
from collections import deque
import hashlib
import importlib.util
from importlib.metadata import version
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
from PIL import Image

from ..actions import PoseAction
from ..simulators import backend_identity

MODELS = {'panda': {'name': 'Franka Panda', 'arm_dof': 7, 'hand_dof': 1,
    'folder': 'IsaacLab/Robots/FrankaEmika', 'file': 'franka_panda.usda', 'image': 'panda.png',
    'source': 'https://github.com/isaac-sim/IsaacLab/blob/v3.0.0/source/isaaclab_assets/isaaclab_assets/robots/franka.py'}}
TASKS = {
    'isaac_lift_red': {'name': '抬起红色方块', 'instruction': '抓取桌面的红色方块，将它抬到桌面上方至少 8 厘米并保持。',
                       'criterion': 'Isaac Lab 物理状态：红块中心 z > 0.08 m，连续保持 0.3 秒。'},
    'isaac_reach_up': {'name': '末端抬高 2 cm', 'instruction': '保持夹爪开度和末端姿态，将末端沿世界 Z 轴抬高 2 厘米。',
                       'criterion': 'Isaac Lab 物理状态：末端距初始位置上方 2 cm 的目标小于 6 mm，连续保持 0.3 秒。'},
}
SCENES = {'isaac_tabletop': {'name': 'Isaac Panda 桌面', 'description': '官方 Franka USD、刚体方块、原生三相机与 IK。', 'tasks': list(TASKS)}}
JOINTS = [f'panda_joint{i}' for i in range(1, 8)]+['panda_finger_joint1']


def array(value):
    if hasattr(value, 'torch'): value = value.torch
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def serial(value):
    if isinstance(value, dict): return {k: serial(v) for k, v in value.items()}
    return array(value).tolist()


def wxyz(xyzw): return np.asarray(xyzw)[[3, 0, 1, 2]].tolist()


def xyzw(wxyz_value): return np.asarray(wxyz_value)[[1, 2, 3, 0]].tolist()


def rotate(quat_wxyz, vector):
    from scipy.spatial.transform import Rotation
    return Rotation.from_quat(xyzw(quat_wxyz)).apply(vector)


def rotation_error(a, b):
    return 2*math.acos(float(np.clip(abs(np.dot(a, b)), -1., 1.)))


class IsaacEvaluator:
    REQUIRED_HOLD_S = .3

    def __init__(self, sim, initial_tcp=None):
        self.initial_tcp = np.asarray(initial_tcp if initial_tcp is not None else sim.tcp_pose()['position_m'])
        self.success = self.ever_success = self.previous_valid = False
        self.hold_s, self.last_time = 0., sim.time
        self.update(sim)

    def update(self, sim):
        finite = bool(np.isfinite(array(sim.robot.data.joint_pos)).all() and np.isfinite(array(sim.red.data.root_pose_w)).all())
        if sim.task_id == 'isaac_lift_red':
            actual = float(array(sim.red.data.root_pose_w)[0, 2]); valid = actual > .08
            item = {'key': 'object_height', 'label': '红块中心高于桌面', 'actual': actual, 'expected': '> 0.08', 'passed': valid, 'unit': 'm'}
            values = {'red_height_m': actual, 'table_top_world_z_m': 0., 'height_threshold_m': .08}
            progress = (actual-.024)/.056
        else:
            target = self.initial_tcp+[0, 0, .02]
            actual = float(np.linalg.norm(np.asarray(sim.tcp_pose()['position_m'])-target)); valid = actual < .006
            item = {'key': 'tcp_target_error', 'label': '末端距目标误差', 'actual': actual, 'expected': '< 0.006', 'passed': valid, 'unit': 'm'}
            values = {'position_error_m': actual, 'target_world_m': target.tolist(), 'position_tolerance_m': .006}
            progress = 1-actual/.02
        valid = bool(valid and finite)
        elapsed = sim.time-self.last_time
        self.hold_s = self.hold_s+elapsed if valid and self.previous_valid and elapsed >= 0 else 0.
        self.success = bool(valid and self.hold_s+1e-8 >= self.REQUIRED_HOLD_S)
        self.ever_success |= self.success
        self.previous_valid, self.last_time = valid, sim.time
        checks = [item, {'key': 'state_finite', 'label': '物理状态数值有效', 'actual': finite, 'expected': True, 'passed': finite},
                  {'key': 'hold_duration', 'label': '成功状态持续时间', 'actual': self.hold_s, 'expected': '>= 0.3', 'passed': self.success, 'unit': 's'}]
        self.last = {'task_id': sim.task_id, **TASKS[sim.task_id], 'source': sim.backend['evaluator_source'],
                     'version': 'isaaclab-task-v1', 'success': self.success, 'ever_success': self.ever_success,
                     'currently_satisfied': valid, 'hold_s': self.hold_s, 'required_hold_s': self.REQUIRED_HOLD_S,
                     'status': 'success' if self.success else 'holding' if valid else 'unsatisfied',
                     'progress': float(np.clip(progress, 0, 1)) if math.isfinite(progress) else 0.,
                     'evaluated_at_sim_s': sim.time, 'values': values, 'checks': checks,
                     'failure_reasons': [c['label'] for c in checks if not c['passed']]}

    def snapshot(self): return self.last.copy()


class IsaacSimulation:
    ENGINE_ID = 'isaac'

    def __init__(self, model_id='panda', render=True, scene_id='isaac_tabletop', seed=42, task_id=None):
        self._validate(model_id, scene_id, task_id or 'isaac_lift_red')
        if self.ENGINE_ID == 'isaacsim' and importlib.util.find_spec('isaacsim') is None:
            raise ImportError('完整 Isaac Sim 未安装。请在独立 .venv-isaacsim 按 docs/isaac.md 安装并自行接受 NVIDIA 许可；该入口不会改用 Newton。')
        self._warp_directory = None
        spec = importlib.util.find_spec('warp')
        if spec is not None and os.name == 'nt' and not str(spec.origin).isascii():
            # NVRTC on Windows rejects Unicode -I paths (builtin.h not found).
            # Load unchanged Warp Python/headers through an ASCII directory;
            # its 367 MB binary directory remains in the installed environment.
            parent = Path(tempfile.gettempdir()).resolve()
            if not str(parent).isascii(): raise ValueError('Windows NVRTC 需要英文编译路径，请将 TEMP 设为可写英文目录。')
            self._warp_directory = tempfile.TemporaryDirectory(prefix='embodiworks-isaac-warp-')
            target = Path(self._warp_directory.name).resolve()
            if target.parent != parent or not target.name.startswith('embodiworks-isaac-warp-'): raise RuntimeError('Warp 临时目录超出预期位置。')
            source = Path(spec.origin).parent
            shutil.copytree(source, target/'warp', ignore=shutil.ignore_patterns('bin', '__pycache__', 'tests', 'examples'))
            try: os.symlink(source/'bin', target/'warp/bin', target_is_directory=True)
            except OSError as exc:
                self._warp_directory.cleanup(); self._warp_directory = None
                raise RuntimeError('Windows NVRTC 中文路径适配需要创建目录符号链接。请启用 Windows 开发者模式，或把独立 Isaac 环境安装到英文路径。') from exc
            sys.path.insert(0, str(target))
        global torch, sim_utils, SimulationContext, subtract_frame_transforms
        import torch
        import isaaclab.sim as sim_utils
        from isaaclab.sim import SimulationContext
        from isaaclab.utils.math import subtract_frame_transforms
        self.model_id, self.scene_id, self.seed, self.task_id = model_id, scene_id, seed, task_id or 'isaac_lift_red'
        self.backend = {**backend_identity(self.ENGINE_ID, version('isaaclab' if self.ENGINE_ID == 'isaac' else 'isaacsim')),
                        'runtime_packages': {p: version(p) for p in ('isaaclab', 'torch', 'warp-lang', 'numpy', 'pillow')}}
        if self.ENGINE_ID == 'isaac': self.backend['runtime_packages']['newton'] = version('newton')
        else: self.backend['runtime_packages']['isaacsim'] = version('isaacsim')
        self.backend['physics_settings'] = ({'solver': 'mjwarp', 'integrator': 'implicitfast', 'substeps': 2,
                                             'robot_gravity_compensation': 1., 'timestep_s': .01, 'cuda_graph': True} if self.ENGINE_ID == 'isaac'
                                            else {'solver': 'isaacsim_physx', 'timestep_s': .01})
        self.render_enabled, self.render_suspended = render, False
        self.generation, self.speed, self.active_robot = 1, 1., 'arm1'
        self.events, self.event_id = deque(maxlen=30), 0
        self.flags = {'contacts': False, 'joints': False, 'collision': False}
        self.sim, self.launcher = None, None
        self._load()
        self.event('已加载 Isaac Lab 官方 Franka、原生动力学、DifferentialIK 与相机。')

    @staticmethod
    def _validate(model, scene, task):
        if model != 'panda' or scene != 'isaac_tabletop' or task not in TASKS:
            raise ValueError('Isaac 当前支持 Panda、isaac_tabletop 场景及 isaac_lift_red / isaac_reach_up。')

    def _load(self):
        from isaaclab.app import launch_simulation
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
        from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
        from isaaclab.scene import InteractiveSceneCfg
        from isaaclab.sensors.camera import CameraCfg
        from isaaclab.utils import clone, configclass, instantiate, replace
        from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG
        if self.ENGINE_ID == 'isaac':
            from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
            from isaaclab_newton.renderers import NewtonWarpRendererCfg
            # Isaac Lab's bare MJWarp config defaults to Euler, unlike
            # Newton's own solver default. Use the maintained reach task's
            # implicitfast/two-substep preset for the high-PD Panda drives.
            physics = NewtonCfg(solver_cfg=MJWarpSolverCfg(integrator='implicitfast'),
                                num_substeps=2, use_cuda_graph=True)
            renderer = NewtonWarpRendererCfg(use_cuda_graph=True, enable_textures=False)
        else:
            from isaaclab_physx.physics import PhysxCfg
            from isaaclab_physx.renderers import IsaacRtxRendererCfg
            physics, renderer = PhysxCfg(), IsaacRtxRendererCfg()
        robot_cfg = replace(FRANKA_PANDA_HIGH_PD_CFG, prim_path='{ENV_REGEX_NS}/Robot')
        robot_cfg.spawn.variants['Physics'] = 'mujoco' if self.ENGINE_ID == 'isaac' else 'physx'
        if self.ENGINE_ID == 'isaac':
            from isaaclab_newton.sim.schemas import MujocoRigidBodyCfg
            # The maintained Franka reach IK preset authors both the PhysX
            # disable-gravity flag and MJWarp's native gravcomp attribute.
            robot_cfg.spawn.rigid_props = [clone(robot_cfg.spawn.rigid_props), MujocoRigidBodyCfg(gravcomp=1.)]

        @configclass
        class SceneCfg(InteractiveSceneCfg):
            robot = robot_cfg
            ground = AssetBaseCfg(prim_path='/World/Ground', spawn=sim_utils.GroundPlaneCfg(),
                                  init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, -.7)))
            table = AssetBaseCfg(prim_path='{ENV_REGEX_NS}/Table',
                spawn=sim_utils.CuboidCfg(size=(1.5, 1.2, .07), collision_props=sim_utils.UsdPhysicsCollisionCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.65, .72, .68))),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(.3, 0, -.035)))
            red = RigidObjectCfg(prim_path='{ENV_REGEX_NS}/Red',
                spawn=sim_utils.CuboidCfg(size=(.048,)*3, rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
                    mass_props=sim_utils.MassCfg(mass=.08), collision_props=sim_utils.UsdPhysicsCollisionCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.8, .18, .13))),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(.4, -.2, .08)))
            light = AssetBaseCfg(prim_path='/World/Light', spawn=sim_utils.DomeLightCfg(intensity=2500.))

        scene_cfg = SceneCfg(num_envs=1, env_spacing=2.)
        if self.render_enabled:
            for name in ('main', 'front', 'side', 'wrist'):
                setattr(scene_cfg, name, CameraCfg(prim_path='{ENV_REGEX_NS}/Camera_'+name,
                    height=640 if name == 'main' else 426, width=960 if name == 'main' else 640,
                    update_period=0, data_types=['rgb'], renderer_cfg=clone(renderer),
                    spawn=sim_utils.PinholeCameraCfg(focal_length=24., horizontal_aperture=23.4, clipping_range=(.01, 10.))))
        sim_cfg = sim_utils.SimulationCfg(dt=.01, device='cuda:0', physics=physics, create_stage_in_memory=True)
        self.launcher = launch_simulation(SimpleNamespace(sim=sim_cfg, scene=scene_cfg),
            {'headless': True, 'device': 'cuda:0', 'enable_cameras': self.render_enabled, 'visualizer': []})
        self.launcher.__enter__()
        self.sim = SimulationContext(sim_cfg)
        self.scene = instantiate(scene_cfg)
        self.sim.reset()
        self.robot, self.red = self.scene['robot'], self.scene['red']
        self.joint_ids, _ = self.robot.find_joints(JOINTS[:7], preserve_order=True)
        self.finger_ids, _ = self.robot.find_joints(['panda_finger_joint1', 'panda_finger_joint2'], preserve_order=True)
        self.hand_index = self.robot.find_bodies('panda_hand')[0][0]
        self.jacobian_index = self.hand_index-1 if self.robot.is_fixed_base else self.hand_index
        self.timestep = self.sim.get_physics_dt()
        self.ik = DifferentialIKController(DifferentialIKControllerCfg(command_type='pose', use_relative_mode=False, ik_method='dls'), 1, self.sim.device)
        limits = array(self.robot.data.joint_pos_limits)[0]
        self.actuators = [{'index': i, 'name': name, 'robot_id': 'arm1', 'is_gripper': i == 7,
                          'min': 0. if i == 7 else float(limits[self.joint_ids[i], 0]),
                          'max': 255. if i == 7 else float(limits[self.joint_ids[i], 1]),
                          'qpos_index': self.finger_ids[0] if i == 7 else self.joint_ids[i], 'unit': 'command' if i == 7 else 'rad'} for i, name in enumerate(JOINTS)]
        self.cameras = {name: self.scene[name] for name in ('main', 'front', 'side', 'wrist')} if self.render_enabled else {}
        # SimulationContext.reset initializes views and solvers; it does not
        # apply an articulation config's default joint state. Match the official
        # tutorial's explicit writes before capturing our reusable reset state.
        self.robot.write_joint_position_to_sim_index(position=self.robot.data.default_joint_pos.torch.clone())
        self.robot.write_joint_velocity_to_sim_index(velocity=self.robot.data.default_joint_vel.torch.clone())
        self.scene.update(0.)
        self.initial_state = self.scene.get_state()
        self.reset()
        # First physics and camera calls compile kernels. Keep this work in the
        # native startup window, before the parent applies its shorter command
        # timeout, then restore the user's clean initial state and clock.
        self.step()
        if self.cameras:
            self.render()
            self.observe()
        self.reset()
        # The final reset must leave usable observations at t=0, not merely
        # after an unrecorded settle/physics step.
        if self.cameras: self.observe()

    def event(self, message, kind='info'):
        self.event_id += 1
        self.events.append({'id': self.event_id, 'time': self.time, 'message': message, 'kind': kind})

    def _q(self): return array(self.robot.data.joint_pos)[0]

    def reset(self):
        self.scene.reset_to(self.initial_state)
        self.scene.reset(); self.scene.update(0.)
        rng = np.random.default_rng(self.seed)
        pose = self.red.data.root_pose_w.torch.clone()
        pose[0, :3] = torch.tensor([.4, -.2, .08], device=self.sim.device)
        if self.seed != 42: pose[0, :2] += torch.tensor(rng.uniform(-.025, .025, 2), device=self.sim.device, dtype=pose.dtype)
        self.red.write_root_pose_to_sim_index(root_pose=pose)
        q = self._q(); self.ctrl = np.r_[q[self.joint_ids], np.clip(q[self.finger_ids[0]]/.04, 0, 1)*255]
        self.home = self.ctrl.copy(); self._apply_ctrl()
        self.time, self.physics_steps = 0., 0
        self.motion, self.pose_goal, self.playback = None, None, None
        self.paused, self.run_deadline, self.stop_on_success = True, None, False
        self.task = {'status': 'idle', 'title': '等待指令', 'steps': [], 'step': -1, 'progress': 0.}
        self.evaluator = IsaacEvaluator(self)
        self.set_view('perspective')
        if self.cameras:
            self._camera_view('front', [.9, -.9, .7], [.3, 0, .2])
            self._camera_view('side', [.7, .85, .7], [.3, 0, .2])

    def tcp_pose(self):
        hand = array(self.robot.data.body_pose_w)[0, self.hand_index]
        quat = wxyz(hand[3:7]); pos = hand[:3]+rotate(quat, [0, 0, .1034])
        return {'frame': 'world', 'position_m': pos.tolist(), 'quaternion_wxyz': quat}

    def _actuators(self):
        q = self._q()
        forces = array(self.robot.actuators.applied_effort)[0]
        return [{**a, 'position': float(q[a['qpos_index']]), 'actual': float(q[a['qpos_index']]),
                 'target': float(self.ctrl[i]), 'force': float(forces[a['qpos_index']])} for i, a in enumerate(self.actuators)]

    def robot_states(self):
        q, v = self._q(), array(self.robot.data.joint_vel)[0]
        return [{'id': 'arm1', 'name': 'Franka Panda', 'model': 'panda', 'active': True, 'arm_dof': 7, 'hand_dof': 1,
                 'tcp': self.tcp_pose(), 'gripper': {'position': float(q[self.finger_ids[0]]), 'unit': 'm_per_finger', 'target_open_fraction': float(self.ctrl[7]/255)},
                 'joints': [{'name': a['name'], 'position_rad': float(q[self.joint_ids[i]]), 'velocity_rad_s': float(v[self.joint_ids[i]]), 'limits_rad': [a['min'], a['max']]} for i, a in enumerate(self.actuators[:7])],
                 'actuators': self._actuators()}]

    def state(self):
        pose, robots = self.tcp_pose(), self.robot_states()
        return {'engine': 'Isaac Lab / Newton' if self.ENGINE_ID == 'isaac' else 'Isaac Sim / PhysX',
                'version': self.backend['engine_version'], 'backend': self.backend.copy(), 'model': 'panda', 'name': 'Franka Panda',
                'active_robot': 'arm1', 'robots': robots, 'generation': self.generation, 'paused': self.paused,
                'speed': self.speed, 'time': self.time, 'timestep': self.timestep, 'steps': self.physics_steps,
                'contacts': None, 'contact_scope': 'not_reported', 'gravity': True, 'view': self.view, 'flags': self.flags.copy(),
                'nq': len(self._q()), 'nv': len(self._q()), 'nu': 8, 'meshes': 0, 'arm_dof': 7, 'hand_dof': 1,
                'actuators': robots[0]['actuators'], 'model_note': '官方 Franka USD、高 PD 预设、DifferentialIK；Isaac Lab 3.0 XYZW 转换为工作台 WXYZ。Newton 路径使用官方 MJWarp 求解器与 Warp 渲染；不等同完整 Isaac Sim RTX。',
                'scene_id': self.scene_id, 'scene_name': SCENES[self.scene_id]['name'], 'seed': self.seed,
                'evaluation': self.evaluator.snapshot(), 'playback': self.playback, 'tcp': pose['position_m'], 'tcp_pose': pose,
                'task': self.task.copy(), 'events': list(self.events), 'has_wrist_camera': bool(self.cameras),
                'source': 'https://github.com/isaac-sim/IsaacLab', 'compute_backend': 'newton_mjwarp' if self.ENGINE_ID == 'isaac' else 'isaacsim_physx',
                'render_backend': 'newton_warp' if self.ENGINE_ID == 'isaac' else 'isaacsim_rtx'}

    def cancel_motion(self):
        if self.motion is not None or self.task['status'] == 'settling': self.task['status'] = 'cancelled'
        self.motion, self.pose_goal = None, None

    def start_motion(self, poses, title, pose_goal=None, policy=False):
        if self.motion is not None or self.task['status'] == 'settling': raise ValueError('当前动作尚未完成，请等待或停止。')
        self.motion = {'poses': poses, 'index': 0, 'elapsed': 0., 'from': self.ctrl.copy(), 'policy': policy}
        self.pose_goal = pose_goal
        self.task = {'status': 'running', 'title': title, 'steps': [p['name'] for p in poses], 'step': 0, 'progress': 0.}
        self.paused = False

    def _apply_ctrl(self):
        q = self.robot.data.joint_pos.torch.clone()
        q[0, self.joint_ids] = torch.as_tensor(self.ctrl[:7], device=self.sim.device, dtype=q.dtype)
        q[0, self.finger_ids] = float(self.ctrl[7]/255*.04)
        self.robot.actuators.target_command.set_position_index(value=q)
        self.scene.write_data_to_sim()

    def _ik_step(self):
        action = self.pose_goal
        world_hand = np.asarray(action.position_m)-rotate(action.quaternion_wxyz, [0, 0, .1034])
        goal = torch.tensor([[*world_hand, *xyzw(action.quaternion_wxyz)]], device=self.sim.device, dtype=torch.float32)
        root = self.robot.data.root_pose_w.torch
        target_p, target_q = subtract_frame_transforms(root[:, :3], root[:, 3:7], goal[:, :3], goal[:, 3:7])
        self.ik.set_command(torch.cat([target_p, target_q], dim=-1))
        hand = self.robot.data.body_pose_w.torch[:, self.hand_index]
        current_p, current_q = subtract_frame_transforms(root[:, :3], root[:, 3:7], hand[:, :3], hand[:, 3:7])
        ids = [i+self.robot.num_base_dofs for i in self.joint_ids]
        jacobian = self.robot.data.body_link_jacobian_w.torch[:, self.jacobian_index, :, ids]
        # The robot root has identity rotation in this fixed tabletop scene.
        targets = self.ik.compute(current_p, current_q, jacobian, self.robot.data.joint_pos.torch[:, self.joint_ids])
        values = array(targets)[0]
        lower = np.array([a['min'] for a in self.actuators[:7]])
        upper = np.array([a['max'] for a in self.actuators[:7]])
        self.ctrl[:7] = np.clip(values, np.maximum(lower, self.ctrl[:7]-2*self.timestep), np.minimum(upper, self.ctrl[:7]+2*self.timestep))

    def step(self):
        if self.motion is not None:
            motion = self.motion; pose = motion['poses'][motion['index']]
            motion['elapsed'] += self.timestep; fraction = min(motion['elapsed']/pose['duration'], 1.)
            if self.pose_goal is None: self.ctrl = motion['from']+(pose['ctrl']-motion['from'])*(fraction*fraction*(3-2*fraction))
            else: self.ctrl[7] = motion['from'][7]+(pose['ctrl'][7]-motion['from'][7])*fraction
            self.task['progress'] = (motion['index']+fraction)/len(motion['poses'])
            if fraction >= 1-1e-8:
                motion['index'] += 1
                if motion['index'] == len(motion['poses']):
                    self.motion = None; self.task['status'] = 'complete' if motion['policy'] else 'settling'; self.settle_deadline = self.time+3.
                else:
                    motion['elapsed'], motion['from'] = 0., self.ctrl.copy(); self.task['step'] = motion['index']
        if self.pose_goal is not None and self.task['status'] in ('running', 'settling'): self._ik_step()
        self._apply_ctrl(); self.sim.step(render=False); self.scene.update(self.timestep)
        self.physics_steps += 1; self.time = self.physics_steps*self.timestep
        if not np.isfinite(self._q()).all(): raise RuntimeError('Isaac 物理状态出现非有限数值，请重置并检查控制目标。')
        self.evaluator.update(self)
        if self.task['status'] == 'settling':
            reached = np.max(np.abs(self._q()[self.joint_ids]-self.ctrl[:7])) < .025
            if self.pose_goal:
                current = self.tcp_pose()
                ep = float(np.linalg.norm(np.asarray(current['position_m'])-self.pose_goal.position_m))
                er = rotation_error(current['quaternion_wxyz'], self.pose_goal.quaternion_wxyz)
                self.task.update(position_error_m=ep, orientation_error_rad=er); reached = ep < .006 and er < .05
            if reached and np.max(np.abs(array(self.robot.data.joint_vel)[0, self.joint_ids])) < .05: self.task['status'] = 'complete'
            elif self.time > self.settle_deadline: self.task.update(status='failed', message='Isaac 原生 IK/PD 未收敛，请缩小目标位姿并检查碰撞和关节限制。')
        if (self.run_deadline is not None and self.time >= self.run_deadline-1e-8) or (self.stop_on_success and self.evaluator.success):
            self.cancel_motion(); self.paused = True
            self.task['status'] = 'verified' if self.stop_on_success and self.evaluator.success else 'limit'

    def move_tcp(self, payload):
        if payload['generation'] != self.generation: raise ValueError('场景已变化，本轮动作失效。')
        action = PoseAction.model_validate(payload['pose'])
        if action.status != 'move': raise ValueError('只有 move 动作可以进入 IK。')
        if action.robot_id not in (None, 'arm1') or action.hand_joint_positions_rad is not None: raise ValueError('Isaac Panda 支持 arm1 与双指开合。')
        current = self.tcp_pose()
        if np.linalg.norm(np.asarray(action.position_m)-current['position_m']) > .12+1e-8: raise ValueError('单轮末端平移超过 0.12 m。')
        if action.position_m[2] < .008: raise ValueError('目标末端位于桌面或桌面以下。')
        if rotation_error(action.quaternion_wxyz, current['quaternion_wxyz']) > math.radians(35): raise ValueError('单轮姿态变化超过 35°。')
        ctrl = self.ctrl.copy(); ctrl[7] = action.gripper_open*255
        self.start_motion([{'name': 'Isaac DifferentialIK → 原生 PD → 末端反馈', 'ctrl': ctrl, 'duration': action.duration_s}], 'Agent 末端动作', action)
        return {'robot_id': 'arm1', 'target': action.model_dump(), 'controller': 'isaaclab.DifferentialIKController'}

    def policy_chunk(self, payload):
        if payload['generation'] != self.generation: raise ValueError('场景已变化，策略动作块失效。')
        if payload.get('robot_id') not in (None, 'arm1'): raise ValueError('Isaac 当前仅支持 arm1。')
        targets, dt = np.asarray(payload['actions'], dtype=float), float(payload['dt_s'])
        if targets.ndim != 2 or targets.shape[1] != 8 or not 1 <= len(targets) <= 50 or not np.isfinite(targets).all(): raise ValueError('动作块须为 1–50 帧、每帧 8 个有限执行器目标。')
        if payload['joint_names'] != JOINTS: raise ValueError('策略关节顺序与机器人不一致。')
        if not .02 <= dt <= .5: raise ValueError('动作间隔须为 0.02–0.5 秒。')
        lower, upper = np.array([a['min'] for a in self.actuators]), np.array([a['max'] for a in self.actuators])
        current, clipped, poses = self.ctrl.copy(), 0, []
        for target in targets:
            rate = np.array([2.]*7+[255.])*dt
            bounded = np.clip(target, np.maximum(lower, current-rate), np.minimum(upper, current+rate))
            clipped += int(np.sum(np.abs(bounded-target) > 1e-8)); current = bounded
            poses.append({'name': '策略动作块', 'ctrl': bounded, 'duration': dt})
        self.start_motion(poses, '策略模型动作块', policy=True)
        return {'robot_id': 'arm1', 'frames': len(poses), 'limited_targets': clipped, 'executed_targets': [p['ctrl'].tolist() for p in poses]}

    def checkpoint(self):
        value = {'backend': self.backend.copy(), 'model': self.model_id, 'scene_id': self.scene_id, 'task_id': self.task_id,
                 'seed': self.seed, 'state': serial(self.scene.get_state()), 'ctrl': self.ctrl.tolist(), 'home': self.home.tolist(),
                 'time': self.time, 'steps': self.physics_steps, 'initial_tcp': self.evaluator.initial_tcp.tolist(), 'active_robot': 'arm1'}
        value['state_id'] = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]
        return value

    def restore(self, checkpoint):
        identity = checkpoint.get('backend', {})
        if any(identity.get(k) != self.backend[k] for k in ('engine_id', 'engine_version', 'state_schema', 'runtime_packages', 'physics_settings')): raise ValueError('检查点的引擎、SDK 版本、物理配置或状态格式与当前 Isaac 不一致。')
        self._validate(checkpoint['model'], checkpoint['scene_id'], checkpoint['task_id'])
        ctrl, home, initial = np.asarray(checkpoint['ctrl']), np.asarray(checkpoint['home']), np.asarray(checkpoint['initial_tcp'])
        if ctrl.shape != (8,) or home.shape != (8,) or initial.shape != (3,) or not all(np.isfinite(v).all() for v in (ctrl, home, initial)): raise ValueError('检查点控制目标或末端参考无效。')
        if (type(checkpoint['steps']) is not int or checkpoint['steps'] < 0 or type(checkpoint['seed']) is not int
            or not 0 <= checkpoint['seed'] <= 4294967295 or not isinstance(checkpoint['time'], (int, float))
            or not math.isfinite(checkpoint['time']) or abs(checkpoint['time']-checkpoint['steps']*self.timestep) > 1e-6): raise ValueError('检查点种子或时间无效。')
        def tensors(value, template):
            if isinstance(template, dict):
                if not isinstance(value, dict) or value.keys() != template.keys(): raise ValueError('检查点状态字段不匹配。')
                return {k: tensors(value[k], v) for k, v in template.items()}
            data = np.asarray(value)
            if tuple(data.shape) != tuple(template.shape) or not np.isfinite(data).all(): raise ValueError('检查点状态维度或数值无效。')
            return torch.as_tensor(data, dtype=template.dtype, device=template.device)
        restored = tensors(checkpoint['state'], self.scene.get_state())
        self.scene.reset_to(restored); self.scene.reset(); self.scene.update(0.)
        self.ctrl, self.home, self.time, self.physics_steps = ctrl, home, checkpoint['time'], checkpoint['steps']
        self.seed, self.task_id = checkpoint['seed'], checkpoint['task_id']
        self._apply_ctrl(); self.cancel_motion(); self.paused = True; self.generation += 1
        self.run_deadline, self.stop_on_success, self.playback = None, False, None
        self.task = {'status': 'idle', 'title': '检查点状态已恢复', 'steps': [], 'step': -1, 'progress': 0.}
        self.evaluator = IsaacEvaluator(self, initial)
        return {'state_id': checkpoint['state_id']}

    def trajectory_frame(self):
        return {'time': self.time, 'backend': self.backend.copy(), 'active_robot': 'arm1', 'checkpoint': self.checkpoint(),
                'qpos': self._q().tolist(), 'qvel': array(self.robot.data.joint_vel)[0].tolist(), 'ctrl': self.ctrl.tolist(),
                'actuator_force': array(self.robot.actuators.applied_effort)[0].tolist(), 'contacts': None,
                'contact_scope': 'not_reported', 'evaluation': self.evaluator.snapshot()}

    def replay_frame(self, frame):
        if frame.get('backend') != frame.get('checkpoint', {}).get('backend'): raise ValueError('轨迹帧后端身份与检查点不一致。')
        self.restore(frame['checkpoint'])
        self.playback = {'time': self.time, 'frame_index': frame.get('index'), 'recorded_evaluation': frame.get('evaluation')}
        self.task['title'] = '记录状态回放 · 未重新推理或执行动作'
        return self.playback

    def verify_task(self, generation, on_step=None):
        if generation != self.generation: raise ValueError('场景已切换，完成声明未被验收。')
        if not self.paused or self.motion is not None: raise ValueError('请在动作结束并暂停后验收任务。')
        self.evaluator.update(self); started = self.time; deadline = started+.3
        if self.run_deadline is not None: deadline = min(deadline, self.run_deadline)
        try:
            while not self.evaluator.success and self.time+self.timestep <= deadline+1e-8:
                self.step()
                if on_step is not None: on_step()
        finally: self.paused = True
        return {**self.evaluator.snapshot(), 'verification_sim_time_s': self.time-started}

    @staticmethod
    def _jpeg(rgb, quality=86):
        output = io.BytesIO(); Image.fromarray(rgb).save(output, 'JPEG', quality=quality)
        return output.getvalue()

    def _camera_view(self, name, eye, target):
        self.cameras[name].set_world_poses_from_view(torch.tensor([eye], device=self.sim.device, dtype=torch.float32),
                                                  torch.tensor([target], device=self.sim.device, dtype=torch.float32))

    def _wrist(self):
        tcp = self.tcp_pose(); pos = np.asarray(tcp['position_m'])+rotate(tcp['quaternion_wxyz'], [.045, 0, -.04])
        # Preserve the physical mount's full orientation, including roll, even
        # when the optical axis aligns with world Z (ROS forward +Z, up -Y).
        self.cameras['wrist'].set_world_poses(
            positions=torch.tensor([pos.tolist()], device=self.sim.device, dtype=torch.float32),
            orientations=torch.tensor([xyzw(tcp['quaternion_wxyz'])], device=self.sim.device, dtype=torch.float32),
            convention='ros')

    def _camera_rgb(self, name):
        camera = self.cameras[name]
        self.sim.render(); camera.update(0., force_recompute=True)
        return array(camera.data.output['rgb'])[0, ..., :3].astype(np.uint8)

    def observe(self):
        if not self.cameras: raise ValueError('视觉观察需要 Isaac 原生渲染器。')
        self._wrist(); cameras = []
        for name in ('front', 'side', 'wrist'):
            cam = self.cameras[name]; rgb = self._camera_rgb(name)
            calibration = {'position_world_m': array(cam.data.pos_w)[0].tolist(),
                           'quaternion_world_xyzw': array(cam.data.quat_w_world)[0].tolist(),
                           'intrinsic_cv': array(cam.data.intrinsic_matrices)[0].tolist()}
            for field, value in calibration.items():
                if not np.isfinite(value).all():
                    raise ValueError(f'Isaac 相机 {name}.{field} 含 NaN/Infinity，SDK 尚未返回有效标定；请重新加载环境。')
            cameras.append({'name': name, 'image_base64': base64.b64encode(self._jpeg(rgb, 88)).decode(),
                'width': 640, 'height': 426, 'time': self.time, **calibration})
        robots = self.robot_states()
        return {'backend': {'engine_id': self.ENGINE_ID, 'engine_version': self.backend['engine_version'], 'control_frame': 'world_z_up_m_wxyz'},
                'generation': self.generation, 'time': self.time, 'robot': 'panda', 'active_robot': 'arm1', 'robots': robots,
                'arm_dof': 7, 'tcp': self.tcp_pose(), 'joints': robots[0]['joints'], 'gripper': robots[0]['gripper'],
                'limits': {'translation_per_round_m': .12, 'rotation_per_round_deg': 35, 'position_tolerance_m': .006, 'orientation_tolerance_rad': .05},
                'actuators': robots[0]['actuators'], 'hand_joints': [], 'cameras': cameras}

    def set_view(self, view):
        if view not in ('perspective', 'top', 'front', 'wrist'): raise ValueError('未知视角。')
        self.view, self.lookat, self.distance = view, np.array([.24, 0, .3]), 2.15
        self.azimuth = {'perspective': -48, 'top': -90, 'front': -90, 'wrist': 0}[view]
        self.elevation = {'perspective': 26, 'top': 89, 'front': 8, 'wrist': 0}[view]

    def render_rgb(self):
        if not self.cameras: raise ValueError('Isaac 渲染器未开启。')
        if self.view == 'wrist': self._wrist(); name = 'wrist'
        else:
            name = 'main'; az, el = math.radians(self.azimuth), math.radians(self.elevation)
            offset = self.distance*np.array([math.cos(el)*math.cos(az), math.cos(el)*math.sin(az), math.sin(el)])
            self._camera_view(name, (self.lookat+offset).tolist(), self.lookat.tolist())
        rgb = self._camera_rgb(name)
        return np.asarray(Image.fromarray(rgb).resize((960, 640))) if rgb.shape[:2] != (640, 960) else rgb

    def render(self): return self._jpeg(self.render_rgb())

    def instruction(self, text):
        clean = re.sub(r'[\s，。！!]', '', text).removeprefix('请')
        if clean in ('机械臂归位', '归位', '回到初始姿态'): self.command({'action': 'home'})
        elif clean in ('张开夹爪', '打开夹爪', '闭合夹爪', '关闭夹爪'):
            self.cancel_motion(); self.ctrl[7] = 255. if clean.startswith(('张开', '打开')) else 0.
        elif clean in ('暂停', '暂停仿真'): self.paused = True
        elif clean in ('继续', '继续仿真', '开始仿真'): self.paused = False
        else: raise ValueError('快捷命令支持归位、开闭夹爪、暂停/继续。视觉任务请使用 Agent 接管。')
        return '指令已交给 Isaac Lab 原生控制接口。'

    def command(self, payload):
        action = payload['action']
        if action in ('load', 'scene'):
            model, scene, task = payload.get('model', self.model_id), payload.get('scene_id', self.scene_id), payload.get('task_id') or self.task_id
            self._validate(model, scene, task)
            self.seed, self.task_id = payload.get('seed', self.seed), task; self.reset(); self.generation += 1
        elif action == 'reset': self.reset(); self.generation += 1
        elif action == 'pause': self.paused = bool(payload['value'])
        elif action == 'render_suspend': self.render_suspended = bool(payload['value']); self.paused = True
        elif action == 'step':
            if not self.paused: raise ValueError('请先暂停，再执行单步。')
            self.step()
        elif action == 'control':
            index, value = payload['index'], payload['value']
            if not 0 <= index < 8 or not math.isfinite(value): raise ValueError('执行器编号或控制数值无效。')
            if not self.actuators[index]['min'] <= value <= self.actuators[index]['max']: raise ValueError('控制目标超出关节范围。')
            self.cancel_motion(); self.ctrl[index] = value
        elif action == 'home': self.start_motion([{'name': '回到初始关节姿态', 'ctrl': self.home.copy(), 'duration': 2.5}], '机械臂归位')
        elif action == 'stop': self.cancel_motion(); self.paused = True; self.run_deadline = None; self.stop_on_success = False
        elif action == 'run_limits': self.evaluator = IsaacEvaluator(self); self.run_deadline = self.time+payload['duration_s']; self.stop_on_success = payload['evaluator_stop']
        elif action == 'select_robot':
            if payload['robot_id'] != 'arm1': raise ValueError('Isaac 当前仅支持 arm1。')
            return {'active_robot': 'arm1', 'generation': self.generation}
        elif action == 'instruction': return self.instruction(payload['text'])
        elif action == 'observe': return self.observe()
        elif action == 'move_tcp': return self.move_tcp(payload)
        elif action == 'policy_chunk': return self.policy_chunk(payload)
        elif action == 'checkpoint': return self.checkpoint()
        elif action == 'restore': return self.restore(payload['checkpoint'])
        elif action == 'replay_frame': return self.replay_frame(payload['frame'])
        elif action == 'verify_task': return self.verify_task(payload['generation'])
        elif action == 'settle':
            self.cancel_motion()
            for _ in range(round(1/self.timestep)): self.step()
            self.paused = True; self.evaluator = IsaacEvaluator(self)
        elif action == 'view': self.set_view(payload['view'])
        elif action == 'camera':
            self.view = 'custom'; self.azimuth += payload.get('dx', 0)
            self.elevation = float(np.clip(self.elevation-payload.get('dy', 0), 3, 89))
            self.distance = float(np.clip(self.distance*math.exp(payload.get('zoom', 0)), .4, 5))
            az, el = math.radians(self.azimuth), math.radians(self.elevation)
            right = np.array([-math.sin(az), math.cos(az), 0.]); up = np.array([-math.sin(el)*math.cos(az), -math.sin(el)*math.sin(az), math.cos(el)])
            self.lookat += self.distance*(-payload.get('pan_dx', 0)*right+payload.get('pan_dy', 0)*up)
        elif action == 'speed':
            if payload['value'] not in (.25, .5, 1., 2.): raise ValueError('不支持的仿真速度。')
            self.speed = payload['value']
        else: raise ValueError(f'Isaac 未实现此操作：{action}')
        return '操作已完成。'

    def close(self):
        try:
            if self.sim is not None: SimulationContext.clear_instance(); self.sim = None
        finally:
            try:
                if self.launcher is not None: self.launcher.__exit__(None, None, None); self.launcher = None
            finally:
                if self._warp_directory is not None:
                    folder = self._warp_directory.name
                    if folder in sys.path: sys.path.remove(folder)
                    self._warp_directory.cleanup(); self._warp_directory = None


class IsaacSimSimulation(IsaacSimulation):
    """Same workbench contract with real Isaac Sim PhysX and RTX, never Newton."""
    ENGINE_ID = 'isaacsim'
