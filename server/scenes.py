"""Seeded manipulation scenes and evaluator-only ground truth; never policy observations."""
import xml.etree.ElementTree as ET
import numpy as np

SCENES = {
    'tabletop': {'name': '桌面抓取', 'description': '彩色方块与目标托盘，测试观察、接近和抓取。', 'tasks': ['lift_red', 'place_red', 'reach_up']},
    'sorting': {'name': '多物体分拣', 'description': '方块、圆柱、球体与双收纳盒。', 'tasks': ['place_red', 'lift_red']},
    'stacking': {'name': '方块堆叠', 'description': '将红色方块稳定放在蓝色方块上。', 'tasks': ['stack_red', 'lift_red']},
    'drawer': {'name': '抽屉操作', 'description': '带滑轨和把手的真实可动抽屉。', 'tasks': ['open_drawer', 'lift_red']},
    'obstacles': {'name': '障碍与杂物', 'description': '立柱和散落物体，测试受限空间中的操作。', 'tasks': ['place_red', 'lift_red']},
    'assembly': {'name': '圆柱插装', 'description': '自由圆柱销与四壁插座，测试对齐与插入。', 'tasks': ['insert_peg', 'lift_red']},
}
TASKS = {
    'lift_red': {'name': '抬起红色方块', 'instruction': '把红色方块抓起来，保持在桌面上方。', 'criterion': '红色方块中心高于桌面 8 cm（随工作台缩放），持续 0.3 秒。'},
    'place_red': {'name': '红块放入绿盒', 'instruction': '把红色方块放入绿色收纳盒内，然后松开并移开手。', 'criterion': '红块完全位于绿盒内部、高度正确、速度低于阈值，且与手无接触，持续 0.3 秒。'},
    'stack_red': {'name': '红块叠在蓝块上', 'instruction': '把红色方块叠放在蓝色方块上，松开并移开手。', 'criterion': '两块水平对齐、红块比蓝块高约一个块高、静止且与手无接触，持续 0.3 秒。'},
    'open_drawer': {'name': '拉开抽屉', 'instruction': '抓住前方抽屉的把手，把抽屉向外拉开。', 'criterion': '抽屉滑轨位移超过 12 cm（随工作台缩放），持续 0.3 秒。'},
    'insert_peg': {'name': '圆柱销插入插座', 'instruction': '把橙色圆柱销竖直插入右侧的深色方形插座，松开并移开手。', 'criterion': '销轴竖直对中，底端进入插座，静止且与手无接触，持续 0.3 秒。'},
    'reach_up': {'name': '末端抬高 2 cm', 'instruction': '保持夹爪开度和当前末端姿态，将末端沿世界坐标 Z 轴抬高 2 厘米。', 'criterion': '末端距初始位置正上方 2 cm 的目标小于 6 mm（小机械臂按比例），持续 0.3 秒。'},
}


def add_scene(world, scale, scene_id, seed):
    if scene_id not in SCENES: raise ValueError('未知场景。')
    rng = np.random.default_rng(seed)
    occupied = []
    clearance = .012  # Unscaled metres; also scales with the SO-101 workbench.
    def vec(values): return ' '.join(str(x * scale) for x in values)
    def half_xy(kind, size):
        return np.asarray(size[:2] if kind == 'box' else [size[0], size[0]])
    def geom(parent, name, kind, size, pos, color, **attrs):
        result = ET.SubElement(parent, 'geom', name=name, type=kind, size=vec(size), pos=vec(pos),
                               rgba=color, friction='1 .005 .0001', **attrs)
        parent_xy = np.fromstring(parent.get('pos', '0 0 0'), sep=' ')[:2] / scale
        occupied.append((parent_xy + np.asarray(pos[:2]), half_xy(kind, size)))
        return result
    def free(name, kind, size, pos, color, jitter=False):
        # Reserve the vertical drop footprint, including fixtures and earlier objects.
        # Sampling cannot rely on the solver to push an intersecting object out later.
        half = half_xy(kind, size)
        for attempt in range(64):
            candidate = np.asarray(pos, dtype=float).copy()
            if jitter or attempt:
                candidate[:2] += rng.uniform(-.025, .025, 2)
            if all(np.any(np.abs(candidate[:2] - center) >= half + other_half + clearance)
                   for center, other_half in occupied):
                pos = candidate
                break
        else:
            raise ValueError(f'场景 {scene_id}、种子 {seed} 无法为 {name} 生成无重叠布局；请调整物体基准位置或随机范围。')
        body = ET.SubElement(world, 'body', name=name, pos=vec(pos))
        ET.SubElement(body, 'freejoint', name=name + '_joint')
        geom(body, name + '_geom', kind, size, [0, 0, 0], color, mass=str(.08 * scale), condim='4')
        return body
    def tray(name, center, color, half=(.15, .16)):
        x, y = center; hx, hy = half
        geom(world, name + '_base', 'box', [hx, hy, .008], [x, y, .007], color)
        for i, (dx, dy, sx, sy) in enumerate([(-hx, 0, .008, hy), (hx, 0, .008, hy), (0, -hy, hx, .008), (0, hy, hx, .008)]):
            geom(world, name + f'_wall{i}', 'box', [sx, sy, .025], [x+dx, y+dy, .032], color)
    # Build fixtures before sampling loose objects so their footprints are reserved.
    if scene_id != 'drawer': tray('tray', [.5, .25], '.40 .61 .48 1')
    if scene_id in ('sorting', 'obstacles'):
        tray('blue_bin', [.17, .26], '.32 .49 .65 1', (.105, .12))
    if scene_id == 'obstacles':
        for i, (x, y) in enumerate([(.42, -.02), (.67, .08), (.27, .07)]):
            geom(world, f'obstacle_{i}', 'cylinder', [.026, .085], [x, y, .085], '.44 .48 .48 1')
    if scene_id == 'drawer':
        # Open-front cabinet; the drawer is a separate dynamic slide joint.
        for name, size, pos in [('back', [.16, .01, .10], [.51, .22, .11]),
                                ('left', [.01, .16, .10], [.35, .06, .11]),
                                ('right', [.01, .16, .10], [.67, .06, .11]),
                                ('top', [.17, .16, .01], [.51, .06, .21])]:
            geom(world, 'cabinet_' + name, 'box', size, pos, '.61 .67 .62 1')
        drawer = ET.SubElement(world, 'body', name='drawer', pos=vec([.51, .06, .055]))
        ET.SubElement(drawer, 'joint', name='drawer_slide', type='slide', axis='0 -1 0',
                      range=vec([0, .20]), damping='3', frictionloss='.15')
        geom(drawer, 'drawer_bottom', 'box', [.135, .135, .008], [0, 0, 0], '.68 .49 .32 1', mass='.25')
        geom(drawer, 'drawer_front', 'box', [.14, .012, .035], [0, -.14, .025], '.70 .51 .34 1', mass='.12')
        geom(drawer, 'drawer_handle', 'box', [.065, .010, .012], [0, -.175, .025], '.24 .30 .28 1', mass='.04')
    if scene_id == 'assembly':
        x, y = .55, .05
        for i, (dx, dy, sx, sy) in enumerate([(-.034, 0, .020, .054), (.034, 0, .020, .054), (0, -.034, .014, .020), (0, .034, .014, .020)]):
            geom(world, f'socket_{i}', 'box', [sx, sy, .035], [x+dx, y+dy, .035], '.28 .35 .36 1')
    positions = {'red': [.40, -.22, .09], 'blue': [.55, -.12, .09], 'yellow': [.32, -.37, .09]}
    if scene_id == 'stacking': positions['blue'] = [.46, .02, .06]
    if scene_id == 'drawer': positions['blue'] = [.55, -.24, .09]
    colors = {'red': '.78 .26 .21 1', 'blue': '.20 .43 .67 1', 'yellow': '.85 .65 .17 1'}
    for name, pos in positions.items():
        # Seed 42 uses the safe nominal layout; other seeds retain +/-25 mm jitter.
        free('block_' + name, 'box', [.024]*3, pos, colors[name], jitter=seed != 42)
    if scene_id in ('sorting', 'obstacles'):
        free('cylinder', 'cylinder', [.020, .035], [.62, -.29, .08], '.42 .32 .63 1')
        free('ball', 'sphere', [.025], [.30, -.12, .07], '.79 .45 .20 1')
    if scene_id == 'assembly':
        free('peg', 'cylinder', [.009, .045], [.43, -.32, .08], '.87 .48 .15 1')


class TaskEvaluator:
    def __init__(self, sim, task_id):
        self.task_id, self.scale = task_id, sim.config['scale']
        self.initial_tcp = sim.data.site_xpos[sim.site_id].copy()
        self.hold_s = 0.
        self.success = False
        self.steps = 0
        self.robot_bodies = set()
        for actuator in sim.actuators:
            body = int(sim.model.jnt_bodyid[sim.model.joint(actuator['name']).id])
            self.robot_bodies.add(body)
        # Include fixed descendants (fingertips) of moving robot bodies.
        for body in range(1, sim.model.nbody):
            if int(sim.model.body_parentid[body]) in self.robot_bodies: self.robot_bodies.add(body)
        self.last = {'success': False, 'progress': 0., 'values': {}}

    def touches_robot(self, sim, body_name):
        body_id = sim.model.body(body_name).id
        for contact in sim.data.contact:
            b1, b2 = (int(sim.model.geom_bodyid[g]) for g in contact.geom)
            if (b1 == body_id and b2 in self.robot_bodies) or (b2 == body_id and b1 in self.robot_bodies): return True
        return False

    def update(self, sim):
        self.steps += 1
        s = self.scale; data = sim.data
        red = data.body('block_red').xpos / s
        red_joint = sim.model.joint('block_red_joint').id
        speed = np.linalg.norm(data.qvel[sim.model.jnt_dofadr[red_joint]:sim.model.jnt_dofadr[red_joint]+3]) / s
        if self.task_id == 'lift_red':
            valid = red[2] > .08
            progress = (red[2] - .024) / (.08 - .024)
            values = {'red_height_m': float(red[2]*s)}
        elif self.task_id == 'place_red':
            error = np.abs(red[:2] - [.5, .25])
            valid = bool(np.all(error < [.115, .125]) and .030 < red[2] < .052 and speed < .03 and not self.touches_robot(sim, 'block_red'))
            progress = max(0., 1 - np.linalg.norm(error)/.5)
            values = {'target_distance_m': float(np.linalg.norm(error)*s), 'object_speed_m_s': float(speed*s)}
        elif self.task_id == 'stack_red':
            blue = data.body('block_blue').xpos / s
            xy = np.linalg.norm(red[:2] - blue[:2]); dz = red[2]-blue[2]
            valid = xy < .015 and abs(dz-.048) < .008 and speed < .03 and not self.touches_robot(sim, 'block_red')
            progress = max(0., 1 - np.linalg.norm(red-blue-[0, 0, .048])/.3)
            values = {'alignment_error_m': float(xy*s), 'height_difference_m': float(dz*s)}
        elif self.task_id == 'open_drawer':
            distance = float(data.joint('drawer_slide').qpos[0]); valid = distance > .12*s
            progress = distance / (.12*s); values = {'drawer_open_m': distance}
        elif self.task_id == 'insert_peg':
            peg = data.body('peg'); pos = peg.xpos / s; vertical = abs(peg.xmat.reshape(3,3)[2,2])
            error = np.linalg.norm(pos[:2] - [.55, .05])
            joint = sim.model.joint('peg_joint').id; qadr = sim.model.jnt_dofadr[joint]
            speed = np.linalg.norm(data.qvel[qadr:qadr+3]) / s
            valid = error < .004 and vertical > .985 and .040 < pos[2] < .063 and speed < .03 and not self.touches_robot(sim, 'peg')
            progress = max(0., 1 - error/.4)
            values = {'alignment_error_m': float(error*s), 'axis_vertical_cosine': float(vertical), 'peg_height_m': float(pos[2]*s)}
        else:
            target = self.initial_tcp + [0, 0, .02]
            error = np.linalg.norm(data.site_xpos[sim.site_id] - target)
            valid = error < .006*s
            progress = 1-error/.02; values = {'position_error_m': float(error)}
        self.hold_s = self.hold_s + sim.model.opt.timestep if valid else 0.
        self.success = self.success or self.hold_s >= .3
        self.last = {'success': bool(self.success), 'currently_satisfied': bool(valid),
                     'progress': float(np.clip(progress, 0, 1)), 'hold_s': round(self.hold_s, 3), 'values': values}

    def snapshot(self):
        return {'task_id': self.task_id, **TASKS[self.task_id], **self.last}
