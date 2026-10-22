"""Absolute TCP action contract and joint-limited local inverse kinematics."""
from typing import Literal

import mujoco
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


class PoseAction(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False, strict=True)
    status: Literal['move', 'done', 'blocked']
    frame: Literal['world']
    position_m: list[float] = Field(min_length=3, max_length=3)
    quaternion_wxyz: list[float] = Field(min_length=4, max_length=4)
    gripper_open: float = Field(ge=0, le=1)
    duration_s: float = Field(ge=.5, le=4)
    explanation: str = Field(min_length=1, max_length=1000)
    hand_joint_positions_rad: list[float] | None = None
    robot_id: str | None = Field(default=None, min_length=1, max_length=40)

    @classmethod
    def action_schema(cls):
        schema = cls.model_json_schema()
        schema['required'] = list(schema['properties'])
        for prop in schema['properties'].values(): prop.pop('default', None)
        return schema

    @model_validator(mode='after')
    def unit_quaternion(self):
        norm = np.linalg.norm(self.quaternion_wxyz)
        if abs(norm - 1) > .01:
            raise ValueError('quaternion_wxyz 必须为单位四元数，顺序为 w / x / y / z。')
        self.quaternion_wxyz = (np.array(self.quaternion_wxyz) / norm).tolist()
        return self


def tcp_pose(model, data, site_id):
    # mj_step integrates qpos after computing derived positions; refresh kinematics.
    mujoco.mj_kinematics(model, data)
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, data.site_xmat[site_id])
    return {'frame': 'world', 'position_m': data.site_xpos[site_id].tolist(),
            'quaternion_wxyz': quat.tolist()}


def rotation_error(target_quat, current_matrix):
    target = np.empty(9)
    mujoco.mju_quat2Mat(target, np.asarray(target_quat, dtype=float))
    delta = target.reshape(3, 3) @ current_matrix.reshape(3, 3).T
    quat, error = np.empty(4), np.empty(3)
    mujoco.mju_mat2Quat(quat, delta.ravel())
    if quat[0] < 0: quat *= -1
    mujoco.mju_quat2Vel(error, quat, 1.)
    return error


def solve_ik(sim, action):
    """Use MuJoCo's site Jacobian on scratch data; never move live robot qpos."""
    model = sim.model
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = sim.data.qpos
    arm = sim.actuators[:sim.config['arm_dof']]
    qids = np.array([a['qpos_index'] for a in arm])
    vids = np.array([model.jnt_dofadr[model.joint(a['name']).id] for a in arm])
    lower, upper = np.array([a['min'] for a in arm]), np.array([a['max'] for a in arm])
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    position = np.array(action.position_m)
    current = np.array(tcp_pose(model, sim.data, sim.site_id)['position_m'])
    limit = .12 * sim.config['scale']
    if np.linalg.norm(position-current) > limit + 1e-8:
        raise ValueError(f'单轮末端平移超过 {limit:.3f} m，请输出更近的绝对目标。')
    if position[2] < .008 * sim.config['scale']:
        raise ValueError('目标末端位于桌面或桌面以下，请提高目标 Z。')
    if np.linalg.norm(rotation_error(action.quaternion_wxyz, sim.data.site_xmat[sim.site_id])) > np.deg2rad(35):
        raise ValueError('单轮姿态变化超过 35°，请输出更近的绝对姿态。')
    weight = .15 * sim.config['scale']
    for _ in range(180):
        mujoco.mj_fwdPosition(model, scratch)
        ep = position - scratch.site_xpos[sim.site_id]
        er = rotation_error(action.quaternion_wxyz, scratch.site_xmat[sim.site_id])
        if np.linalg.norm(ep) < .0008 * sim.config['scale'] and np.linalg.norm(er) < .008:
            if np.max(np.abs(scratch.qpos[qids] - sim.data.qpos[qids])) > .65:
                raise ValueError('IK 所需关节变化超过单轮 0.65 rad，请缩小目标动作。')
            return scratch.qpos[qids].copy()
        mujoco.mj_jacSite(model, scratch, jacp, jacr, sim.site_id)
        jac = np.vstack((jacp[:, vids], weight * jacr[:, vids]))
        error = np.r_[ep, weight * er]
        delta = jac.T @ np.linalg.solve(jac @ jac.T + np.eye(6) * 1e-6, error)
        delta *= min(1., .15 / max(np.max(np.abs(delta)), 1e-12))
        scratch.qpos[qids] = np.clip(scratch.qpos[qids] + delta, lower, upper)
    raise ValueError(f'IK 未找到满足关节限制的目标：位置误差 {np.linalg.norm(ep)*1000:.2f} mm，'
                     f'姿态误差 {np.rad2deg(np.linalg.norm(er)):.2f}°。'
                     f'{sim.config["name"]} 为 {sim.config["arm_dof"]} 自由度，请调整目标姿态。')
