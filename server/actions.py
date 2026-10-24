"""Simulator-independent absolute world TCP action schema."""
from typing import Literal

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
