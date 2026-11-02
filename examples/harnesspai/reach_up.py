"""平台提供的接口示例，不是上游发布的任务解法。

只使用相机/机器人状态观察与末端动作；环境真值由平台独立验收。
目标固定为初始位置上方 2 cm，重复反馈修正时不累加目标。
"""


def run(ctx):
    observation = ctx.observe()
    target = dict(observation['tcp'])
    target['position_m'] = list(target['position_m'])
    target['position_m'][2] += 0.02
    target.update(status='move', gripper_open=1.0, duration_s=1.0,
                  explanation='沿世界坐标 Z 轴抬高 2 cm，保持初始姿态。')
    for attempt in range(3):
        feedback = ctx.move_tcp(target)
        ctx.note(f'第 {attempt + 1} 次末端反馈：{feedback["motion"]["status"]}')
        # 程序只能检查机器人执行反馈；任务成功由环境验收。
        current = ctx.observe()['tcp']['position_m']
        if sum((a - b)**2 for a, b in zip(current, target['position_m']))**0.5 < 0.004:
            break
