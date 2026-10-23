"""Seeded manipulation scenes and evaluator-only ground truth; never policy observations."""
import xml.etree.ElementTree as ET
import numpy as np
import mujoco

SCENES = {
    'tabletop': {'name': '桌面抓取', 'description': '彩色方块与目标托盘，测试观察、接近和抓取。', 'tasks': ['lift_red', 'place_red', 'reach_up']},
    'sorting': {'name': '多物体分拣', 'description': '方块、圆柱、球体与双收纳盒。', 'tasks': ['place_red', 'lift_red']},
    'stacking': {'name': '方块堆叠', 'description': '将红色方块稳定放在蓝色方块上。', 'tasks': ['stack_red', 'lift_red']},
    'drawer': {'name': '抽屉操作', 'description': '带滑轨和把手的真实可动抽屉。', 'tasks': ['open_drawer', 'lift_red']},
    'obstacles': {'name': '障碍与杂物', 'description': '立柱和散落物体，测试受限空间中的操作。', 'tasks': ['place_red', 'lift_red']},
    'assembly': {'name': '圆柱插装', 'description': '自由圆柱销与四壁插座，测试对齐与插入。', 'tasks': ['insert_peg', 'lift_red']},
    'cooperative': {'name': '多臂共享分拣', 'description': '共享物体与两个收纳盒，搭配双/三 Panda，研究轮间协作、避让与任务分工。', 'tasks': ['sort_red_blue', 'place_red', 'lift_red']},
}
TASKS = {
    'lift_red': {'name': '抬起红色方块', 'instruction': '把红色方块抓起来，保持在桌面上方。', 'criterion': '红色方块中心高于桌面 8 cm（随工作台缩放），持续 0.3 秒。'},
    'place_red': {'name': '红块放入绿盒', 'instruction': '把红色方块放入绿色收纳盒内，然后松开并移开手。', 'criterion': '红块完全位于绿盒内部、高度正确、速度低于阈值，且与手无接触，持续 0.3 秒。'},
    'stack_red': {'name': '红块叠在蓝块上', 'instruction': '把红色方块叠放在蓝色方块上，松开并移开手。', 'criterion': '两块水平对齐、红块比蓝块高约一个块高、静止且与手无接触，持续 0.3 秒。'},
    'open_drawer': {'name': '拉开抽屉', 'instruction': '抓住前方抽屉的把手，把抽屉向外拉开。', 'criterion': '抽屉滑轨位移超过 12 cm（随工作台缩放），持续 0.3 秒。'},
    'insert_peg': {'name': '圆柱销插入插座', 'instruction': '把橙色圆柱销竖直插入右侧的深色方形插座，松开并移开手。', 'criterion': '销轴竖直对中，底端进入插座，静止且与手无接触，持续 0.3 秒。'},
    'reach_up': {'name': '末端抬高 2 cm', 'instruction': '保持夹爪开度和当前末端姿态，将末端沿世界坐标 Z 轴抬高 2 厘米。', 'criterion': '目标位移固定为世界坐标 Z 轴正向 2 cm；末端距目标小于 6 mm（容差随工作台缩放），持续 0.3 秒。'},
    'sort_red_blue': {'name': '红蓝方块分别入盒', 'instruction': '把红色方块放入绿色收纳盒，把蓝色方块放入蓝色收纳盒；可按 robot_id 分配机械臂，完成后所有手都松开并移开。', 'criterion': '红蓝方块分别完整位于对应盒内、高度正确、静止且与任何机械臂无接触，持续 0.3 秒；评价物理结果，不要求使用指定数量的机械臂。'},
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
    if scene_id in ('sorting', 'obstacles', 'cooperative'):
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
    """Read privileged physics state; model declarations never set task success.

    Success describes the present state after a continuous hold, not an earlier
    transient achievement. ``ever_success`` records that separate historical fact.
    Paused observations cannot contribute simulated hold time.
    """
    VERSION = 'task-evaluation-v2'
    REQUIRED_HOLD_S = .3

    def __init__(self, sim, task_id):
        if task_id not in TASKS:
            raise ValueError(f'未知评价任务：{task_id}')
        self.task_id, self.scale = task_id, sim.config['scale']
        self.initial_tcp = sim.data.site_xpos[sim.site_id].copy()
        self.initial_tcps = {key: sim.data.site_xpos[robot['site_id']].copy() for key, robot in sim.robots.items()}
        self.hold_s = 0.
        self.success = self.ever_success = False
        self.steps = 0
        self.last_time = float(sim.data.time)
        self.previous_valid = False
        self.robot_bodies = set()
        for actuator in sim.all_actuators:
            body = int(sim.model.jnt_bodyid[sim.model.joint(actuator['name']).id])
            while body:
                self.robot_bodies.add(body)
                body = int(sim.model.body_parentid[body])
        # Include fixed descendants (fingertips) of moving robot bodies.
        for body in range(1, sim.model.nbody):
            if int(sim.model.body_parentid[body]) in self.robot_bodies: self.robot_bodies.add(body)
        self.update(sim)

    def touches_robot(self, sim, body_name):
        body_id = sim.model.body(body_name).id
        for contact in sim.data.contact:
            b1, b2 = (int(sim.model.geom_bodyid[g]) for g in contact.geom)
            if (b1 == body_id and b2 in self.robot_bodies) or (b2 == body_id and b1 in self.robot_bodies): return True
        return False

    def update(self, sim):
        self.steps += 1
        s = self.scale; data = sim.data; model = sim.model
        # mj_step integrates qpos after computing xpos. Refresh positions without
        # mj_forward's changes to dynamics/warm-start state.
        mujoco.mj_kinematics(model, data)
        checks = []
        def plain(value):
            if isinstance(value, (list, tuple, np.ndarray)):
                return [plain(item) for item in value]
            if isinstance(value, (float, np.floating)):
                return float(value) if np.isfinite(value) else None
            if isinstance(value, (bool, np.bool_)): return bool(value)
            if isinstance(value, np.integer): return int(value)
            if isinstance(value, dict): return {key: plain(item) for key, item in value.items()}
            return value
        def check(key, label, actual, expected, passed, unit=None):
            checks.append({'key': key, 'label': label, 'actual': plain(actual),
                           'expected': expected, 'passed': bool(passed), 'unit': unit})
        finite = bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all() and np.isfinite(data.time))
        check('state_finite', '物理状态数值有效', finite, True, finite)
        def speed_of(name):
            adr = model.jnt_dofadr[model.joint(name + '_joint').id]
            return float(np.linalg.norm(data.qvel[adr:adr+3]))
        def placed(name, tray):
            body_name = 'block_' + name
            pos = data.body(body_name).xpos
            geom = data.geom(body_name + '_geom')
            # The rotated box's extent is measured from actual MuJoCo geometry.
            # A center-only test can accept corners protruding through bin walls.
            half = np.abs(geom.xmat.reshape(3, 3)) @ model.geom(body_name + '_geom').size
            lower = np.array([data.geom(tray+'_wall0').xpos[0] + model.geom(tray+'_wall0').size[0],
                              data.geom(tray+'_wall2').xpos[1] + model.geom(tray+'_wall2').size[1]])
            upper = np.array([data.geom(tray+'_wall1').xpos[0] - model.geom(tray+'_wall1').size[0],
                              data.geom(tray+'_wall3').xpos[1] - model.geom(tray+'_wall3').size[1]])
            bounds = [pos[:2] - half[:2], pos[:2] + half[:2]]
            contained = bool(np.all(bounds[0] > lower) and np.all(bounds[1] < upper))
            prefix = name + '_'
            check(prefix+'inside_bin_xy', f'{name} 方块完整位于盒内', bounds,
                  plain([lower, upper]), contained, 'm')
            check(prefix+'height', f'{name} 方块高度', pos[2], f'{.030*s:g} < z < {.052*s:g}',
                  .030*s < pos[2] < .052*s, 'm')
            speed = speed_of(body_name)
            check(prefix+'speed', f'{name} 方块线速度', speed, f'< {.03*s:g}', speed < .03*s, 'm/s')
            touched = self.touches_robot(sim, body_name)
            check(prefix+'released', f'{name} 方块已脱离机械臂接触', not touched, True, not touched)
            center = data.geom(tray+'_base').xpos[:2]
            return all(item['passed'] for item in checks[-4:]), float(np.linalg.norm(pos[:2]-center)), speed

        red = data.body('block_red').xpos
        speed = speed_of('block_red')
        if self.task_id == 'lift_red':
            table = model.geom('workbench')
            table_top = float(table.pos[2] + table.size[2])
            height = float(red[2] - table_top)
            check('red_height', '红块中心高于桌面', height, f'> {.08*s:g}', height > .08*s, 'm')
            progress = (height/s - .024) / (.08 - .024)
            values = {'red_height_m': float(red[2]), 'height_above_table_m': height,
                      'table_top_world_z_m': table_top, 'height_threshold_m': .08*s}
        elif self.task_id == 'place_red':
            _, distance, speed = placed('red', 'tray')
            progress = max(0., 1 - distance / (.5*s))
            values = {'target_distance_m': distance, 'object_speed_m_s': speed}
        elif self.task_id == 'sort_red_blue':
            satisfied = {}; distances = {}
            for name, tray in [('red', 'tray'), ('blue', 'blue_bin')]:
                satisfied[name], distances[name], _ = placed(name, tray)
            progress = sum(satisfied.values()) / 2
            values = {'objects_placed': satisfied, 'target_distances_m': distances}
        elif self.task_id == 'stack_red':
            blue = data.body('block_blue').xpos
            xy = float(np.linalg.norm(red[:2] - blue[:2])); dz = float(red[2]-blue[2])
            check('stack_alignment', '红蓝块水平对齐误差', xy, f'< {.015*s:g}', xy < .015*s, 'm')
            check('stack_height', '红蓝块中心高度差', dz, f'|dz - {.048*s:g}| < {.008*s:g}', abs(dz-.048*s) < .008*s, 'm')
            check('red_speed', '红块线速度', speed, f'< {.03*s:g}', speed < .03*s, 'm/s')
            touched = self.touches_robot(sim, 'block_red')
            check('red_released', '红块已脱离机械臂接触', not touched, True, not touched)
            progress = max(0., 1 - np.linalg.norm(red-blue-[0, 0, .048*s])/(.3*s))
            values = {'alignment_error_m': xy, 'height_difference_m': dz, 'object_speed_m_s': speed}
        elif self.task_id == 'open_drawer':
            joint = model.joint('drawer_slide')
            position = float(data.joint('drawer_slide').qpos[0])
            closed, maximum = map(float, joint.range)
            distance = position - closed
            threshold = .12*s
            # MuJoCo joint limits are soft; small solver excursions at the stop
            # are valid, while a displaced/off-rail state is not.
            limit_tolerance = .002*s
            check('drawer_displacement', '抽屉相对闭合位置的位移', distance, f'> {threshold:g}', distance > threshold, 'm')
            check('drawer_joint_limits', '抽屉滑轨处于物理行程内', position,
                  f'{closed:g} - {limit_tolerance:g} <= qpos <= {maximum:g} + {limit_tolerance:g}',
                  closed-limit_tolerance <= position <= maximum+limit_tolerance, 'm')
            progress = distance / threshold
            values = {'drawer_open_m': distance, 'drawer_threshold_m': threshold,
                      'drawer_joint_qpos_m': position, 'drawer_closed_qpos_m': closed,
                      'drawer_joint_limits_m': [closed, maximum], 'limit_tolerance_m': limit_tolerance,
                      'drawer_velocity_m_s': float(data.joint('drawer_slide').qvel[0])}
        elif self.task_id == 'insert_peg':
            peg = data.body('peg'); pos = peg.xpos
            vertical = float(abs(peg.xmat.reshape(3,3)[2,2]))
            error = float(np.linalg.norm(pos[:2] - np.array([.55, .05])*s))
            speed = speed_of('peg')
            check('peg_alignment', '销轴水平对中误差', error, f'< {.004*s:g}', error < .004*s, 'm')
            check('peg_vertical', '销轴与竖直方向夹角余弦绝对值', vertical, '> 0.985', vertical > .985)
            check('peg_height', '销轴中心高度', pos[2], f'{.040*s:g} < z < {.063*s:g}', .040*s < pos[2] < .063*s, 'm')
            check('peg_speed', '销轴线速度', speed, f'< {.03*s:g}', speed < .03*s, 'm/s')
            touched = self.touches_robot(sim, 'peg')
            check('peg_released', '销轴已脱离机械臂接触', not touched, True, not touched)
            progress = max(0., 1 - error/(.4*s))
            values = {'alignment_error_m': error, 'axis_vertical_cosine': vertical,
                      'peg_height_m': float(pos[2]), 'object_speed_m_s': speed}
        elif self.task_id == 'reach_up':
            target = self.initial_tcps[sim.active_robot] + [0, 0, .02]
            error = float(np.linalg.norm(data.site_xpos[sim.site_id] - target))
            check('tcp_target_error', '末端距抬高目标的误差', error, f'< {.006*s:g}', error < .006*s, 'm')
            progress = 1-error/.02
            values = {'position_error_m': error, 'target_world_m': target.tolist(), 'position_tolerance_m': .006*s}
        valid = all(item['passed'] for item in checks)
        now = float(data.time)
        elapsed = now - self.last_time
        if valid and self.previous_valid and np.isfinite(elapsed) and elapsed >= 0:
            self.hold_s += elapsed
        else:
            self.hold_s = 0.
        self.previous_valid = valid
        self.last_time = now
        self.success = bool(valid and self.hold_s + 1e-9 >= self.REQUIRED_HOLD_S)
        self.ever_success = self.ever_success or self.success
        check('hold_duration', '物理条件连续满足的仿真时间', self.hold_s,
              f'>= {self.REQUIRED_HOLD_S:g}', self.success, 's')
        self.last = {'version': self.VERSION, 'source': 'mujoco_privileged_state',
                     'success': self.success, 'ever_success': self.ever_success,
                     'currently_satisfied': valid,
                     'status': 'success' if self.success else 'holding' if valid else 'unsatisfied',
                     'progress': float(np.clip(progress, 0, 1)) if np.isfinite(progress) else 0.,
                     'hold_s': round(self.hold_s, 6), 'required_hold_s': self.REQUIRED_HOLD_S,
                     'evaluated_at_sim_s': plain(now), 'values': plain(values), 'checks': checks,
                     'failure_reasons': [item['label'] for item in checks if not item['passed']]}

    def snapshot(self):
        return {'task_id': self.task_id, **TASKS[self.task_id], **self.last}
