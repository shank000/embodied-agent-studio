"""Configured inference primitives; credentials stay in the owning server process."""
import httpx
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator
from typing import Literal

from .policy import PolicyConfig, PolicyOutput, predict_policy, http_result
from .runlog import model_output
from .telemetry import track_inference


class HarnessTool(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: str = Field(pattern=r'^[a-z][a-z0-9_]{0,39}$')
    kind: Literal['perception', 'decision'] = 'perception'
    model: str = Field(min_length=1, max_length=300)
    description: str = Field(min_length=1, max_length=2000)
    endpoint: str = Field(min_length=1)
    api_key: SecretStr = SecretStr('')
    timeout_s: int = Field(default=60, ge=1, le=600)

    @field_validator('endpoint')
    @classmethod
    def valid_endpoint(cls, value):
        return PolicyConfig.valid_endpoint(value)


def primitive_contract(policy=None, tools=()):
    # Bind to capability interfaces, not model names: compatible backends may be
    # replaced without rewriting code, but every new experiment is revalidated.
    return {'version': 'harnesspai-primitives-v1',
            'policy': 'embodied-policy-v1' if policy else None,
            'tools': sorted([{'name': t.name, 'kind': t.kind} for t in tools], key=lambda t:t['name'])}


@track_inference('harnesspai_tool')
async def call_tool(config, body):
    headers = {'Authorization': 'Bearer '+config.api_key.get_secret_value()} if config.api_key.get_secret_value() else {}
    from .telemetry import routing_request_context
    headers.update({f'X-EmbodiWorks-{name.replace("_", "-")}': value
                    for name, value in routing_request_context().items() if isinstance(value, str)})
    async with httpx.AsyncClient(timeout=config.timeout_s, trust_env=False) as client:
        response = await client.post(config.endpoint, json=body, headers=headers)
    try: returned = response.json()
    except ValueError: returned = {'content': response.text}
    model_output({'provider':config.kind, 'model':config.model, 'response':returned})
    raw = http_result(response)
    if not isinstance(raw, dict) or raw.get('protocol') != 'embodied-tool-v1' or 'result' not in raw:
        raise ValueError('工具服务须返回 embodied-tool-v1 协议及 result 字段。')
    return raw['result']


async def call_primitive(request, policy, tools):
    observation = request['observation']
    if request['kind'] == 'policy':
        if policy is None: raise ValueError('未配置动作策略模型。')
        body = {'protocol':'embodied-policy-v1', 'model':policy.model, 'device':policy.device,
                'task':request['instruction'], 'observation':observation,
                'seed':request['seed'], 'reset':request['reset'], 'horizon':policy.horizon, 'dt_s':policy.dt_s}
        return PolicyOutput.model_validate(await predict_policy(policy, body)).model_dump()
    config = next((t for t in tools if t.name == request.get('name')), None)
    if config is None: raise ValueError('未配置该工具：'+str(request.get('name')))
    return await call_tool(config, {'protocol':'embodied-tool-v1', 'model':config.model,
        'task':request['instruction'], 'observation':observation, 'input':request.get('input',{})})
