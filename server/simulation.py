"""MuJoCo state and renderer, owned exclusively by one simulation thread."""
from __future__ import annotations

import io
import base64
import hashlib
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image
from .control import PoseAction, tcp_pose, rotation_error, solve_ik
from .scenes import SCENES, TASKS, add_scene, TaskEvaluator
from .robot_assets import rm65_gaia_xml

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / 'vendor' / 'mujoco_menagerie'
MODELS = {
    'panda': {'name': 'Franka Panda', 'folder': 'franka_emika_panda', 'file': 'panda.xml',
              'arm_dof': 7, 'scale': 1.0, 'tip_body': 'hand', 'tip_offset': [0, 0, 0.1034],
              'home': [0, -0.4, 0, -2.1, 0, 1.7, 0.7854], 'gripper_open': 255., 'gripper_closed': 0.,
              'image': 'panda.png'},
    'so101': {'name': 'SO-101', 'folder': 'robotstudio_so101', 'file': 'so101.xml',
              'arm_dof': 5, 'scale': 0.42, 'tip_site': 'gripperframe',
              'home': [0, -0.8, 0.8, 0.5, 0], 'gripper_open': 1.25, 'gripper_closed': 0.,
              'image': 'so101.png',
              'model_note': 'SO-101 有 5 个机械臂自由度；部分位置与姿态组合不可达，IK 会返回原因。默认归位姿态已抬高以避开场景设施。'},
    'rm65_gaia': {'name': 'RM65 + GaiaHand', 'folder': 'realman', 'file': 'rm_65.urdf + right_gaiahand.urdf',
                  'arm_dof': 6, 'hand_dof': 15, 'scale': 1., 'tip_site': 'studio_tcp',
                  'home': [3., 0, 1.2, 0, 1., 0], 'gripper_open': 1., 'gripper_closed': 0.,
                  'image': None, 'source': 'https://github.com/Stella-robot/GaiaHand',
                  'model_note': '官方公开 GaiaHand 15 关节版本；非 Gaia Hand 20。安装变换与伺服参数为仿真设置。'},
}


def make_model(model_id: str, scene_id='tabletop', seed=42):
    """Use upstream robot dynamics unchanged; add a workbench and free bodies."""
    config = MODELS[model_id]
    if model_id == 'rm65_gaia':
        root, assets, _ = rm65_gaia_xml()
    else:
        path = VENDOR / config['folder'] / config['file']
        if not path.exists():
            raise FileNotFoundError(f'缺少模型文件 {path}；请先运行模型下载步骤。')
        root = ET.parse(path).getroot()
        assets = {mesh.get('file'): (path.parent / 'assets' / mesh.get('file')).read_bytes()
                  for mesh in root.findall('asset/mesh')}
    # MuJoCo's Windows file loader cannot open this workspace's Unicode path.
    # Pass the upstream mesh bytes through its supported virtual filesystem.
    root.find('compiler').set('meshdir', '')
    for keyframes in root.findall('keyframe'):
        root.remove(keyframes)  # nq changes when the scene's free bodies are added.
    option = root.find('option')
    if option is None:
        option = ET.SubElement(root, 'option')
    option.set('timestep', '0.002')
    option.set('integrator', 'implicitfast')
    option.set('gravity', '0 0 -9.81')
    visual = root.find('visual')
    if visual is None:
        visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', offwidth='960', offheight='640')
    ET.SubElement(visual, 'headlight', ambient='.20 .20 .20', diffuse='.35 .35 .35', specular='.05 .05 .05')
    ET.SubElement(visual, 'rgba', haze='.93 .95 .92 1')
    asset = root.find('asset')
    ET.SubElement(asset, 'texture', type='skybox', builtin='gradient', rgb1='.87 .91 .86', rgb2='.94 .96 .92', width='512', height='3072')
    ET.SubElement(asset, 'texture', name='studio_grid', type='2d', builtin='checker',
                  rgb1='.86 .89 .84', rgb2='.92 .94 .89', width='512', height='512')
    ET.SubElement(asset, 'material', name='studio_floor', texture='studio_grid', texrepeat='10 10', texuniform='true', reflectance='.05')
    world = root.find('worldbody')
    scale = config['scale']
    def vector(values):
        return ' '.join(str(v * scale) for v in values)
    # Fixed calibrated views are independent of the user's interactive camera.
    for name, eye in [('front', [.8, -1.05, .7]), ('side', [.8, .85, .65])]:
        zaxis = np.array(eye) - np.array([.30, 0, .18])
        zaxis /= np.linalg.norm(zaxis)
        xaxis = np.cross([0, 0, 1], zaxis); xaxis /= np.linalg.norm(xaxis)
        yaxis = np.cross(zaxis, xaxis)
        ET.SubElement(world, 'camera', name=f'agent_{name}', pos=vector(eye), fovy='52',
                      xyaxes=' '.join(str(v) for v in np.r_[xaxis, yaxis]))
    for upstream_light in world.findall('light'):
        upstream_light.set('diffuse', '.20 .20 .20')
        upstream_light.set('specular', '.05 .05 .05')
    ET.SubElement(world, 'light', pos=vector([-.4, -1.2, 2.5]), dir='.2 .4 -1', directional='true', diffuse='.40 .40 .40', specular='.05 .05 .05', castshadow='true')
    ET.SubElement(world, 'light', pos=vector([1, 1, 1.8]), dir='-.4 -.2 -1', directional='true', diffuse='.15 .15 .15', specular='0 0 0', castshadow='false')
    ET.SubElement(world, 'geom', name='studio_floor', type='plane', size='5 5 .05', pos=vector([0, 0, -.69]), material='studio_floor')
    ET.SubElement(world, 'geom', name='workbench', type='box', size=vector([.8, .65, .035]), pos=vector([.25, 0, -.037]), rgba='.80 .85 .79 1', friction='1 .005 .0001')
    for x in [-.4, .9]:
        for y in [-.5, .5]:
            ET.SubElement(world, 'geom', type='box', size=vector([.025, .025, .3]), pos=vector([x, y, -.375]), rgba='.33 .40 .35 1')
    add_scene(world, scale, scene_id, seed)
    if model_id == 'panda':
        hand = root.find(".//body[@name='hand']")
        ET.SubElement(hand, 'site', name='studio_tcp', pos='0 0 .1034', size='.007', rgba='.2 .7 .4 1', group='4')
        ET.SubElement(hand, 'camera', name='wrist_cam', pos='.045 0 .035', xyaxes='1 0 0 0 -1 0', fovy='75')
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'), assets)


class Simulation:
    def __init__(self, model_id='panda', render=True, scene_id='tabletop', seed=42, task_id=None):
        self.renderer = None
        self.render_enabled = render
        self.render_suspended = False
        self.generation = 0
        self.view = 'perspective'
        self.speed = 1.
        self.flags = {'contacts': False, 'joints': False, 'collision': False}
        self.events = []
        self.event_seq = 0
        self.scene_id, self.seed, self.task_id = scene_id, seed, task_id or SCENES[scene_id]['tasks'][0]
        self.load(model_id)

    def event(self, message, kind='info'):
        self.event_seq += 1
        self.events.append({'id': self.event_seq, 'message': message, 'kind': kind})
        self.events = self.events[-25:]

    def load(self, model_id, scene_id=None, seed=None, task_id=None):
        if model_id not in MODELS:
            raise ValueError('不支持的机械臂型号。')
        scene_id = scene_id or self.scene_id
        seed = self.seed if seed is None else seed
        if scene_id not in SCENES: raise ValueError('未知场景。')
        task_id = task_id or (self.task_id if self.task_id in SCENES[scene_id]['tasks'] else SCENES[scene_id]['tasks'][0])
        if task_id not in SCENES[scene_id]['tasks']: raise ValueError('该任务不适用于当前场景。')
        model = make_model(model_id, scene_id, seed)
        data = mujoco.MjData(model)
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        self.model, self.data = model, data
        self.model_id, self.config = model_id, MODELS[model_id]
        self.scene_id, self.seed, self.task_id = scene_id, seed, task_id
        self.generation += 1
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.cam)
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[3:] = 0
        self.home_ctrl = np.array(self.config['home'] + ([0.]*15 if model_id == 'rm65_gaia' else [self.config['gripper_open']]), dtype=float)
        self.site_id = model.site(self.config.get('tip_site', 'studio_tcp')).id
        self.actuators = []
        for index in range(model.nu):
            is_gripper = index >= self.config['arm_dof']
            joint_id = int(model.actuator_trnid[index, 0]) if not (is_gripper and model_id == 'panda') else model.joint('finger_joint1').id
            control_range = model.actuator_ctrlrange[index].copy()
            if model.actuator_trntype[index] == mujoco.mjtTrn.mjTRN_JOINT:
                control_range[0] = max(control_range[0], model.jnt_range[joint_id, 0])
                control_range[1] = min(control_range[1], model.jnt_range[joint_id, 1])
            self.actuators.append({'index': index, 'name': mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id),
                                   'is_gripper': is_gripper, 'min': float(control_range[0]), 'max': float(control_range[1]),
                                   'qpos_index': int(model.jnt_qposadr[joint_id]),
                                   'unit': 'command' if is_gripper and model_id == 'panda' else 'rad'})
        self.reset(log=False)
        self.set_view('perspective')
        self.set_flags(self.flags)
        if self.render_enabled:
            self.renderer = mujoco.Renderer(model, height=640, width=960)
        self.event(f"已加载 {self.config['name']}：{self.config['arm_dof']} 个机械臂关节 + {model.nu-self.config['arm_dof']} 个手部执行器，MuJoCo {mujoco.__version__}。")

    def reset(self, log=True):
        mujoco.mj_resetData(self.model, self.data)
        for item in self.actuators:
            index = item['index']
            if item['unit'] == 'command':
                self.data.joint('finger_joint1').qpos[0] = .04
                self.data.joint('finger_joint2').qpos[0] = .04
            else:
                self.data.qpos[item['qpos_index']] = self.home_ctrl[index]
        self.data.ctrl[:] = self.home_ctrl
        self.model.opt.gravity[:] = [0, 0, -9.81]
        self.paused = False
        self.motion = None
        self.pose_goal = None
        self.task = {'status': 'idle', 'title': '等待指令', 'steps': [], 'step': -1, 'progress': 0}
        self.physics_steps = 0
        self.run_deadline = None
        self.stop_on_success = False
        mujoco.mj_forward(self.model, self.data)
        self.evaluator = TaskEvaluator(self, self.task_id)
        if log:
            self.event('场景已重置，仿真时间、机械臂和自由物体已恢复。')

    def set_view(self, view):
        if view not in ('perspective', 'top', 'front', 'wrist'):
            raise ValueError('未知视角。')
        if view == 'wrist':
            if self.model.ncam == 0:
                raise ValueError('此模型未配置腕部相机。')
            self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.cam.fixedcamid = self.model.camera('wrist_cam').id
        else:
            self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            self.cam.lookat[:] = np.array([.24, 0, .30]) * self.config['scale']
            self.cam.distance = 2.15 * self.config['scale']
            self.cam.azimuth = {'perspective': 135, 'top': 90, 'front': 90}[view]
            self.cam.elevation = {'perspective': -26, 'top': -89, 'front': -8}[view]
        self.view = view

    def set_flags(self, flags):
        self.flags.update(flags)
        self.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = self.flags['contacts']
        self.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = self.flags['contacts']
        self.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = self.flags['joints']
        self.opt.geomgroup[2] = not self.flags['collision']
        self.opt.geomgroup[3] = self.flags['collision']
        self.opt.geomgroup[4] = self.flags['collision']

    def set_control(self, index, value, log=True):
        if not 0 <= index < self.model.nu:
            raise ValueError('执行器编号不存在。')
        item = self.actuators[index]
        if not math.isfinite(value) or not item['min'] <= value <= item['max']:
            raise ValueError(f"{item['name']} 的控制范围是 {item['min']:.3f} 至 {item['max']:.3f}。")
        self.cancel_motion()
        self.data.ctrl[index] = value
        if log:
            amount = f'{value:.1f}' if item['unit'] == 'command' else f'{math.degrees(value):.1f}°'
            self.event(f"已设置 {item['name']} 目标：{amount}" + ('；仿真暂停中，继续后生效。' if self.paused else '。'))

    def cancel_motion(self):
        if self.motion is not None or self.task['status'] == 'settling':
            self.motion = None
            self.task['status'] = 'cancelled'
            self.event('当前动作计划已被手动控制中断。')
        self.pose_goal = None

    def start_motion(self, poses, title):
        if self.motion is not None:
            raise ValueError('当前动作计划尚未完成，请先停止或重置。')
        self.motion = {'poses': poses, 'index': 0, 'elapsed': 0., 'from': self.data.ctrl.copy()}
        self.pose_goal = None
        self.task = {'status': 'running', 'title': title, 'steps': [p['name'] for p in poses], 'step': 0, 'progress': 0}
        self.paused = False
        self.event(f'开始执行：{title}。')

    def home(self):
        self.start_motion([{'name': '回到初始关节姿态', 'ctrl': self.home_ctrl.copy(), 'duration': 2.5}], '机械臂归位')

    def demo(self):
        poses = []
        for name, joint1, gripper in [('回到初始姿态', 0., self.config['gripper_open']), ('向左转动基座', .45, self.config['gripper_open']),
                                      ('闭合夹爪', .45, self.config['gripper_closed']), ('向右转动基座', -.45, self.config['gripper_closed']),
                                      ('张开夹爪', -.45, self.config['gripper_open']), ('返回初始姿态', 0., self.config['gripper_open'])]:
            target = self.home_ctrl.copy(); target[0] = joint1
            self.set_hand_target(target, (gripper-self.config['gripper_closed'])/(self.config['gripper_open']-self.config['gripper_closed']))
            poses.append({'name': name, 'ctrl': target, 'duration': 2.0})
        self.start_motion(poses, '关节与夹爪演示')

    def instruction(self, text):
        clean = re.sub(r'[\s，。！!]', '', text).removeprefix('请')
        if clean in ('机械臂归位', '归位', '回到初始姿态'):
            self.home()
        elif clean in ('运行关节演示', '关节演示', '演示运动'):
            self.demo()
        elif clean in ('张开夹爪', '打开夹爪', '闭合夹爪', '关闭夹爪'):
            self.cancel_motion()
            self.set_hand_target(self.data.ctrl, 1. if clean.startswith(('张开', '打开')) else 0.)
        elif clean in ('暂停', '暂停仿真'):
            self.paused = True
        elif clean in ('继续', '继续仿真', '开始仿真'):
            self.paused = False
        else:
            match = re.fullmatch(r'(?:把)?关节([1-7])(?:转到|设为)([-+]?\d+(?:\.\d+)?)(?:度|°)', clean)
            if not match:
                raise ValueError('快捷命令支持：机械臂归位、张开夹爪、闭合夹爪、运行关节演示，以及“关节1转到30度”。视觉任务请使用 Agent 接管入口。')
            index = int(match.group(1)) - 1
            if index >= self.config['arm_dof']:
                raise ValueError(f"{self.config['name']} 只有 {self.config['arm_dof']} 个机械臂关节。")
            self.set_control(index, math.radians(float(match.group(2))))
        return '指令已交给 MuJoCo 控制接口。' + ('当前暂停，控制目标将在继续后执行。' if self.paused else '')

    def step(self):
        if self.motion is not None:
            motion = self.motion; pose = motion['poses'][motion['index']]
            motion['elapsed'] += self.model.opt.timestep
            fraction = min(motion['elapsed'] / pose['duration'], 1.)
            smooth = fraction * fraction * (3 - 2 * fraction)
            self.data.ctrl[:] = motion['from'] + (pose['ctrl'] - motion['from']) * smooth
            self.task['progress'] = (motion['index'] + fraction) / len(motion['poses'])
            if fraction >= 1:
                motion['index'] += 1
                if motion['index'] == len(motion['poses']):
                    self.motion = None
                    # Completing the control trajectory is not proof of reaching the target.
                    self.task['status'] = 'complete' if motion.get('policy_chunk') else 'settling'
                    self.settle_deadline = self.data.time + 3.
                else:
                    self.task['step'] = motion['index']; motion['elapsed'] = 0.; motion['from'] = self.data.ctrl.copy()
        mujoco.mj_step(self.model, self.data)
        self.physics_steps += 1
        self.evaluator.update(self)
        if (self.run_deadline is not None and self.data.time >= self.run_deadline - 1e-8) or (self.stop_on_success and self.evaluator.success):
            self.cancel_motion()
            self.paused = True
            self.task['status'] = 'verified' if self.stop_on_success and self.evaluator.success else 'limit'
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise RuntimeError('MuJoCo 出现非有限状态，仿真已停止。请重置并检查控制目标。')
        if self.task['status'] == 'settling':
            errors = [abs(self.data.qpos[a['qpos_index']] - self.data.ctrl[a['index']]) for a in self.actuators if not a['is_gripper']]
            reached = max(errors) < .025
            if self.pose_goal:
                pose = tcp_pose(self.model, self.data, self.site_id)
                position_error = np.linalg.norm(np.array(pose['position_m']) - self.pose_goal.position_m)
                angle_error = np.linalg.norm(rotation_error(self.pose_goal.quaternion_wxyz, self.data.site_xmat[self.site_id]))
                self.task['position_error_m'] = float(position_error)
                self.task['orientation_error_rad'] = float(angle_error)
                reached = position_error < .006 * self.config['scale'] and angle_error < .05
            if reached and np.max(np.abs(self.data.qvel[:self.config['arm_dof']])) < .05:
                self.task['status'] = 'complete'; self.event('动作完成：实际末端已达到目标容差。' if self.pose_goal else '动作完成：关节已收敛到目标姿态。', 'success')
            elif self.data.time > self.settle_deadline:
                self.task['status'] = 'failed'
                self.task['joint_errors_rad'] = {a['name']: float(self.data.qpos[a['qpos_index']] - self.data.ctrl[a['index']])
                                                for a in self.actuators if not a['is_gripper']}
                contacts = set()
                for contact in self.data.contact:
                    bodies = [int(self.model.geom_bodyid[g]) for g in contact.geom]
                    robot = [b in self.evaluator.robot_bodies for b in bodies]
                    if robot[0] == robot[1]: continue
                    r = 0 if robot[0] else 1; other = 1-r
                    fixture = self.model.geom(int(contact.geom[other])).name or self.model.body(bodies[other]).name or '场景物体'
                    contacts.add((self.model.body(bodies[r]).name, fixture))
                self.task['contacts'] = [{'robot_body': robot, 'object': fixture} for robot, fixture in sorted(contacts)]
                deviations = sorted(self.task['joint_errors_rad'].items(), key=lambda item: abs(item[1]), reverse=True)
                message = '动作未到位：' + '、'.join(f'{name} 偏差 {abs(math.degrees(error)):.1f}°' for name, error in deviations[:3]) + '。'
                if self.pose_goal:
                    message += f"末端位置误差 {position_error*1000:.2f} mm，姿态误差 {math.degrees(angle_error):.2f}°。"
                if contacts:
                    message += '当前接触：' + '、'.join(f'{robot} ↔ {fixture}' for robot, fixture in sorted(contacts)[:3]) + '。请先移开接触部位，再调整目标。'
                else:
                    message += '请缩小动作或检查关节目标与执行器限制。'
                self.task['message'] = message
                self.event(message, 'error')

    def move_tcp(self, payload):
        if payload['generation'] != self.generation:
            raise ValueError('机械臂已经切换，本轮动作已失效。')
        action = PoseAction.model_validate(payload['pose'])
        if action.status != 'move': raise ValueError('只有 move 动作可以进入 IK。')
        joints = solve_ik(self, action)
        ctrl = self.data.ctrl.copy()
        ctrl[:self.config['arm_dof']] = joints
        self.set_hand_target(ctrl, action.gripper_open)
        if action.hand_joint_positions_rad is not None:
            if self.model_id != 'rm65_gaia' or len(action.hand_joint_positions_rad) != 15:
                raise ValueError('独立手指控制需要 RM65 + GaiaHand 的 15 个手部目标角度。')
            for item, value in zip(self.actuators[6:], action.hand_joint_positions_rad):
                if not item['min'] <= value <= item['max']: raise ValueError(f"手指关节 {item['name']} 超出范围。")
            ctrl[6:] = action.hand_joint_positions_rad
        self.start_motion([{'name': 'IK → 执行器 → 实际末端反馈', 'ctrl': ctrl, 'duration': action.duration_s}], 'Agent 末端动作')
        self.pose_goal = action
        return {'joint_targets_rad': joints.tolist(), 'target': action.model_dump()}

    def set_hand_target(self, ctrl, open_fraction):
        if self.model_id == 'rm65_gaia':
            for item in self.actuators[6:]:
                suffix = item['name'].rsplit('_', 1)[-1]
                closed = (.45 if 'thumb' in item['name'] else 0.) if suffix == '1' else min(1.1, item['max'])
                ctrl[item['index']] = (1-open_fraction)*closed
        else:
            ctrl[-1] = self.config['gripper_closed'] + open_fraction * (self.config['gripper_open']-self.config['gripper_closed'])

    def policy_chunk(self, payload):
        if payload['generation'] != self.generation: raise ValueError('场景已变化，策略动作块失效。')
        targets = np.asarray(payload['actions'], dtype=float)
        dt = float(payload['dt_s'])
        if targets.ndim != 2 or targets.shape[1] != self.model.nu or not 1 <= len(targets) <= 50 or not np.isfinite(targets).all():
            raise ValueError(f'动作块必须为 1–50 帧、每帧 {self.model.nu} 个有限执行器目标。')
        if payload['joint_names'] != [a['name'] for a in self.actuators]: raise ValueError('策略关节顺序与当前机器人不一致。')
        if not .02 <= dt <= .5: raise ValueError('策略动作间隔应在 0.02–0.5 秒。')
        lower = np.array([a['min'] for a in self.actuators]); upper = np.array([a['max'] for a in self.actuators])
        clipped = 0; current = self.data.ctrl.copy(); poses = []
        rate = np.array([255. if a['unit'] == 'command' else 2. for a in self.actuators])
        for target in targets:
            bounded = np.clip(target, np.maximum(lower, current-rate*dt), np.minimum(upper, current+rate*dt))
            clipped += int(np.sum(np.abs(bounded-target) > 1e-8))
            poses.append({'name': '策略动作块', 'ctrl': bounded, 'duration': dt}); current = bounded
        self.start_motion(poses, '策略模型动作块'); self.motion['policy_chunk'] = True
        return {'frames': len(poses), 'limited_targets': clipped, 'executed_targets': [p['ctrl'].tolist() for p in poses]}

    def checkpoint(self):
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        values = np.empty(mujoco.mj_stateSize(self.model, spec))
        mujoco.mj_getState(self.model, self.data, values, spec)
        identity = f'{self.model_id}:{self.scene_id}:{self.seed}:{self.task_id}'.encode() + values.tobytes()
        return {'model': self.model_id, 'scene_id': self.scene_id, 'seed': self.seed, 'task_id': self.task_id,
                'state': values.tolist(), 'state_id': hashlib.sha256(identity).hexdigest()[:16]}

    def trajectory_frame(self):
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        values = np.empty(mujoco.mj_stateSize(self.model, spec))
        mujoco.mj_getState(self.model, self.data, values, spec)
        return {'time': float(self.data.time), 'integration_state': values.tolist(),
                'qpos': self.data.qpos.tolist(), 'qvel': self.data.qvel.tolist(),
                'ctrl': self.data.ctrl.tolist(), 'actuator_force': self.data.actuator_force.tolist(),
                'contacts': int(self.data.ncon), 'evaluation': self.evaluator.snapshot()}

    def replay_frame(self, frame):
        values = np.asarray(frame['integration_state'], dtype=float)
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        if values.shape != (mujoco.mj_stateSize(self.model, spec),) or not np.all(np.isfinite(values)):
            raise ValueError('记录状态的维度或数值与当前模型不符。')
        self.cancel_motion()
        self.paused = True
        mujoco.mj_setState(self.model, self.data, values, spec)
        mujoco.mj_forward(self.model, self.data)
        self.playback = {'time': float(self.data.time), 'frame_index': frame.get('index'),
                         'recorded_evaluation': frame.get('evaluation')}
        self.task = {'status': 'idle', 'title': '记录状态回放 · 未重新推理或执行动作',
                     'steps': [], 'step': -1, 'progress': 0}
        return self.playback

    def restore(self, checkpoint):
        if (self.model_id, self.scene_id, self.seed, self.task_id) != (checkpoint['model'], checkpoint['scene_id'], checkpoint['seed'], checkpoint['task_id']):
            self.load(checkpoint['model'], checkpoint['scene_id'], checkpoint['seed'], checkpoint['task_id'])
        self.cancel_motion()
        mujoco.mj_setState(self.model, self.data, np.asarray(checkpoint['state']), mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(self.model, self.data)
        self.generation += 1; self.paused = True; self.speed = 1.
        self.run_deadline = None; self.stop_on_success = False
        self.task = {'status': 'idle', 'title': '对比实验初始状态', 'steps': [], 'step': -1, 'progress': 0}
        self.physics_steps = 0; self.evaluator = TaskEvaluator(self, self.task_id)
        return {'state_id': checkpoint['state_id']}

    def observe(self):
        """Three RGB images and proprioception, all from the same physics state."""
        if not self.renderer: raise ValueError('三路视觉观察需要原生渲染器。')
        mujoco.mj_forward(self.model, self.data)
        pose = tcp_pose(self.model, self.data, self.site_id)
        observations = []
        options = mujoco.MjvOption(); options.geomgroup[3:] = 0
        for name, camera in [('front', 'agent_front'), ('side', 'agent_side'), ('wrist', 'wrist_cam')]:
            camera_id = self.model.camera(camera).id
            self.renderer.update_scene(self.data, camera=camera_id, scene_option=options)
            pixels = self.renderer.render()
            out = io.BytesIO(); Image.fromarray(pixels).resize((640, 426)).save(out, 'JPEG', quality=88)
            observations.append({'name': name, 'image_base64': base64.b64encode(out.getvalue()).decode(),
                                 'width': 640, 'height': 426, 'time': float(self.data.time),
                                 'position_world_m': self.data.cam_xpos[camera_id].tolist(),
                                 'rotation_camera_to_world': self.data.cam_xmat[camera_id].reshape(3,3).tolist(),
                                 'fovy_deg': float(self.model.cam_fovy[camera_id])})
        arm = self.actuators[:self.config['arm_dof']]
        return {'generation': self.generation, 'time': float(self.data.time), 'robot': self.model_id,
                'arm_dof': self.config['arm_dof'], 'tcp': pose,
                'joints': [{'name': a['name'], 'position_rad': float(self.data.qpos[a['qpos_index']]),
                            'velocity_rad_s': float(self.data.joint(a['name']).qvel[0]),
                            'limits_rad': [a['min'], a['max']]} for a in arm],
                'gripper': {'mode': 'individual_hand_joints', 'joint_count': 15} if self.model_id == 'rm65_gaia' else {'position': float(self.data.qpos[self.actuators[-1]['qpos_index']]),
                            'unit': 'm_per_finger' if self.model_id == 'panda' else 'rad',
                            'target_open_fraction': float((self.data.ctrl[-1]-self.config['gripper_closed']) /
                                                         (self.config['gripper_open']-self.config['gripper_closed']))},
                'limits': {'translation_per_round_m': .12 * self.config['scale'], 'rotation_per_round_deg': 35,
                           'position_tolerance_m': .006 * self.config['scale'], 'orientation_tolerance_rad': .05},
                'actuators': [{'name': a['name'], 'unit': a['unit'], 'position': float(self.data.qpos[a['qpos_index']]),
                               'target': float(self.data.ctrl[a['index']]), 'min': a['min'], 'max': a['max']} for a in self.actuators],
                'hand_joints': [{'name': a['name'], 'position_rad': float(self.data.qpos[a['qpos_index']]), 'limits_rad': [a['min'], a['max']]} for a in self.actuators[self.config['arm_dof']:]] if self.model_id == 'rm65_gaia' else [],
                'cameras': observations}

    def drop(self):
        body_id = self.model.body('block_red').id
        joint_id = self.model.body_jntadr[body_id]
        qadr, vadr = self.model.jnt_qposadr[joint_id], self.model.jnt_dofadr[joint_id]
        self.data.qpos[qadr:qadr+3] = np.array([.40, -.22, .35]) * self.config['scale']
        self.data.qpos[qadr+3:qadr+7] = [1, 0, 0, 0]
        self.data.qvel[vadr:vadr+6] = 0
        mujoco.mj_forward(self.model, self.data)
        self.paused = False
        self.event('已从桌面上方释放红色方块，由 MuJoCo 计算自由落体和桌面接触。')

    def command(self, payload):
        action = payload['action']
        if action != 'replay_frame' and action not in ('view', 'camera', 'flags', 'checkpoint', 'observe'):
            self.playback = None
        if action == 'load': self.load(payload['model'], payload.get('scene_id'), payload.get('seed'), payload.get('task_id'))
        elif action == 'scene': self.load(self.model_id, payload['scene_id'], payload.get('seed'), payload.get('task_id'))
        elif action == 'reset': self.reset()
        elif action == 'pause': self.paused = bool(payload['value'])
        elif action == 'render_suspend':
            self.render_suspended = bool(payload['value'])
            self.paused = True
        elif action == 'step':
            if not self.paused: raise ValueError('请先暂停仿真，再执行单步。')
            self.step()
        elif action == 'control': self.set_control(payload['index'], payload['value'])
        elif action == 'home': self.home()
        elif action == 'demo': self.demo()
        elif action == 'stop':
            self.cancel_motion(); self.paused = True
            self.run_deadline = None; self.stop_on_success = False
        elif action == 'run_limits':
            self.run_deadline = self.data.time + payload['duration_s']
            self.stop_on_success = payload['evaluator_stop']
        elif action == 'instruction': return self.instruction(payload['text'])
        elif action == 'observe': return self.observe()
        elif action == 'move_tcp': return self.move_tcp(payload)
        elif action == 'policy_chunk': return self.policy_chunk(payload)
        elif action == 'checkpoint': return self.checkpoint()
        elif action == 'restore': return self.restore(payload['checkpoint'])
        elif action == 'replay_frame': return self.replay_frame(payload['frame'])
        elif action == 'settle':
            self.cancel_motion()
            for _ in range(500): self.step()
            self.paused = True; self.evaluator = TaskEvaluator(self, self.task_id)
        elif action == 'view': self.set_view(payload['view'])
        elif action == 'camera':
            self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            self.cam.azimuth += payload.get('dx', 0)
            self.cam.elevation = float(np.clip(self.cam.elevation + payload.get('dy', 0), -89, -3))
            self.cam.distance = float(np.clip(self.cam.distance * math.exp(payload.get('zoom', 0)), .5*self.config['scale'], 5*self.config['scale']))
            # Screen-up is (-sin(elevation) * horizontal_forward, cos(elevation)).
            # The opposite horizontal sign introduces a dolly component (pure
            # dolly at -45 degrees), which makes vertical dragging look like zoom.
            # Only this interactive camera moves; calibrated cameras do not.
            if payload.get('pan_dx', 0) or payload.get('pan_dy', 0):
                elevation = math.radians(self.cam.elevation)
                mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_MOVE_V,
                    payload.get('pan_dx', 0), payload.get('pan_dy', 0)*math.cos(elevation), self.cam)
                mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_MOVE_H,
                    0., -payload.get('pan_dy', 0)*math.sin(elevation), self.cam)
            self.view = 'custom'
        elif action == 'speed':
            if payload['value'] not in (.25, .5, 1., 2.): raise ValueError('不支持的仿真速度。')
            self.speed = payload['value']
        elif action == 'flags': self.set_flags(payload['flags'])
        elif action == 'gravity':
            self.model.opt.gravity[2] = -9.81 if payload['value'] else 0
            self.event('已开启重力。' if payload['value'] else '已关闭重力，控制参数保持不变。')
        elif action == 'drop': self.drop()
        else: raise ValueError('未知仿真操作。')
        return '操作已完成。'

    def state(self):
        pose = tcp_pose(self.model, self.data, self.site_id)
        if self.pose_goal and self.task['status'] in ('complete', 'failed', 'settling'):
            self.task['position_error_m'] = float(np.linalg.norm(np.array(pose['position_m']) - self.pose_goal.position_m))
            self.task['orientation_error_rad'] = float(np.linalg.norm(rotation_error(self.pose_goal.quaternion_wxyz, self.data.site_xmat[self.site_id])))
        arm = []
        for item in self.actuators:
            arm.append({**item, 'target': float(self.data.ctrl[item['index']]), 'actual': float(self.data.qpos[item['qpos_index']]),
                        'force': float(self.data.actuator_force[item['index']])})
        return {'engine': 'MuJoCo', 'version': mujoco.__version__, 'model': self.model_id, 'name': self.config['name'],
                'generation': self.generation, 'paused': self.paused, 'speed': self.speed, 'time': float(self.data.time),
                'timestep': self.model.opt.timestep, 'steps': self.physics_steps, 'contacts': int(self.data.ncon),
                'gravity': bool(self.model.opt.gravity[2]), 'view': self.view, 'flags': self.flags.copy(),
                'nq': self.model.nq, 'nv': self.model.nv, 'nu': self.model.nu, 'meshes': self.model.nmesh,
                'arm_dof': self.config['arm_dof'], 'mass': float(self.model.body_mass.sum()), 'actuators': arm,
                'hand_dof': self.model.nu-self.config['arm_dof'], 'model_note': self.config.get('model_note', ''),
                'scene_id': self.scene_id, 'scene_name': SCENES[self.scene_id]['name'], 'seed': self.seed,
                'evaluation': self.evaluator.snapshot(), 'playback': getattr(self, 'playback', None),
                'tcp': pose['position_m'], 'tcp_pose': pose,
                'task': self.task.copy(), 'events': list(self.events),
                'blocks': {name: self.data.body(f'block_{name}').xpos.tolist() for name in ['red', 'blue', 'yellow']},
                'has_wrist_camera': self.model.ncam > 0,
                'source': self.config.get('source', f"https://github.com/google-deepmind/mujoco_menagerie/tree/822c2d8f877dd166c5b7d3c9f7e3c3b6589473b7/{self.config['folder']}")}

    def render_rgb(self):
        self.renderer.update_scene(self.data, camera=self.cam, scene_option=self.opt)
        return self.renderer.render()

    def render(self):
        pixels = self.render_rgb()
        output = io.BytesIO()
        Image.fromarray(pixels).save(output, format='JPEG', quality=86)
        return output.getvalue()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
