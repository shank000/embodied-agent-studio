"""Local SmolVLA inference sidecar. Run in .venv-policy, independent of MuJoCo."""
import base64
import gc
import io
import os
import threading
import time

# Xet's parallel transfer buffers exhausted available host memory during setup.
# Use Hugging Face's supported ordinary HTTP download path for the same weights.
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '120')

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

app = FastAPI(title='Embodied Studio · SmolVLA')
lock = threading.Lock()
loaded = {}
status = {'status': 'idle', 'message': '尚未加载 SmolVLA。'}


class LoadRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    model: str = 'lerobot/smolvla_base'
    device: str = 'cpu'


class InferenceRequest(LoadRequest):
    protocol: str = 'embodied-policy-v1'
    task: str
    observation: dict
    seed: int = 42
    reset: bool = False
    horizon: int = Field(default=10, ge=1, le=50)
    dt_s: float = Field(default=.05, ge=.02, le=.5)


def load_policy(model_path, device):
    global loaded
    if loaded.get('model') == model_path and loaded.get('device') == device: return
    status.update(status='loading', message=f'正在加载 {model_path}；首次使用需要下载官方权重。')
    import torch
    torch.set_num_threads(4)
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from huggingface_hub import snapshot_download
    if device not in ('cuda', 'cpu'): raise ValueError('推理设备仅支持 cuda 或 cpu。')
    if device == 'cuda' and not torch.cuda.is_available(): raise ValueError('CUDA 不可用，请在策略配置中选择 CPU。')
    checkpoint_path = model_path if os.path.isdir(model_path) else snapshot_download(
        model_path, allow_patterns=['*.json','*.safetensors'], max_workers=1)
    # The policy checkpoint already contains the VLM weights.
    config = SmolVLAConfig.from_pretrained(checkpoint_path)
    config.device = device; config.load_vlm_weights = False
    if config.input_features['observation.state'].shape != (6,) and list(config.input_features['observation.state'].shape) != [6]:
        raise ValueError('本地 SmolVLA 适配器要求 SO-101 的 6 维状态；其他本体请使用自定义策略接口。')
    loaded = {}
    if device == 'cuda': torch.cuda.empty_cache()
    # Empty initialization + assign loads one copy of the official weights.
    # A normal initialization plus checkpoint exceeded this Windows host's
    # system commit limit (os error 1455). No parameters remain randomly initialized.
    from accelerate import init_empty_weights
    from safetensors.torch import load_file
    with init_empty_weights():
        policy = SmolVLAPolicy(config)
    weights = load_file(os.path.join(checkpoint_path, 'model.safetensors'), device=device, backend='pread')
    aliases = {}
    for name, parameter in policy.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(parameter), []).append(name)
    for group in aliases.values():
        source = next((name for name in group if name in weights), None)
        if source:
            for name in group:
                if name not in weights: weights[name] = weights[source]
    policy.load_state_dict(weights, strict=True, assign=True)
    del weights
    policy.to(device).eval()
    gc.collect()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=checkpoint_path,
        preprocessor_overrides={'device_processor': {'device': device}})
    loaded = {'model': model_path, 'device': device, 'policy': policy, 'pre': pre, 'post': post}
    status.update(status='ready', message='SmolVLA 已加载。基础权重未在当前仿真任务上微调。', model=model_path, device=device)


@app.get('/health')
def health():
    return {'service': 'embodied-smolvla', **status}


@app.post('/load')
def load(request: LoadRequest):
    with lock:
        try: load_policy(request.model, request.device)
        except Exception as exc:
            status.update(status='error', message=f'{type(exc).__name__}: {exc}')
            raise HTTPException(400, status['message']) from exc
    return health()


@app.post('/predict')
def predict(request: InferenceRequest):
    if request.protocol != 'embodied-policy-v1': raise HTTPException(400, '不支持的策略协议。')
    obs = request.observation
    if obs.get('robot') != 'so101': raise HTTPException(400, '默认 SmolVLA 适配 SO-101；请切换机械臂，或接入匹配当前机器人的自定义策略。')
    with lock:
        try:
            load_policy(request.model, request.device)
            import numpy as np
            import torch
            from PIL import Image
            policy = loaded['policy']
            if request.reset: policy.reset()
            torch.manual_seed(request.seed)
            controls = obs['actuators']
            if len(controls) != 6: raise ValueError('SO-101 应有 6 个执行器。')
            # LeRobot SO motor units: arm -100..100, gripper 0..100.
            # This simulation calibration maps each MJCF joint's physical limits
            # linearly; the mapping is returned and logged with every inference.
            lower = np.array([a['min'] for a in controls]); upper = np.array([a['max'] for a in controls])
            position = np.array([a['position'] for a in controls])
            normalized = (position-lower)/(upper-lower)*200-100
            normalized[-1] = (position[-1]-lower[-1])/(upper[-1]-lower[-1])*100
            batch = {'observation.state': torch.tensor(normalized, dtype=torch.float32), 'task': request.task}
            cameras = {c['name']: c for c in obs['cameras']}
            keys = list(policy.config.image_features)
            if len(keys) != 3: raise ValueError('本地适配器要求三路相机；请使用带有三路观察配置的 SmolVLA 权重。')
            for key, name in zip(keys, ('front', 'side', 'wrist')):
                pixels = np.array(Image.open(io.BytesIO(base64.b64decode(cameras[name]['image_base64']))).convert('RGB'))
                batch[key] = torch.from_numpy(pixels.copy()).permute(2, 0, 1).float()/255
            started = time.perf_counter()
            precision = torch.autocast(request.device, dtype=torch.bfloat16)
            with torch.inference_mode(), precision:
                processed = loaded['pre'](batch)
                raw = policy.predict_action_chunk(processed)
                actions = loaded['post'](raw)[0, :request.horizon].float().cpu().numpy()
            if actions.shape[1] != 6 or not np.isfinite(actions).all(): raise ValueError('SmolVLA 输出形状或数值无效。')
            targets = (actions+100)/200*(upper-lower)+lower
            targets[:, -1] = actions[:, -1]/100*(upper[-1]-lower[-1])+lower[-1]
            return {'protocol': 'embodied-policy-v1', 'action_type': 'actuator_targets',
                    'joint_names': [a['name'] for a in controls], 'actions': targets.tolist(), 'dt_s': request.dt_s,
                    'model': request.model, 'inference_s': time.perf_counter()-started,
                    'summary': 'SmolVLA 直接预测关节动作块；基础权重尚未在本仿真场景微调。',
                    'model_output': actions.tolist(), 'mapping': {'state_units': 'SO normalized [-100,100]; gripper [0,100]',
                    'joint_limits_rad': np.stack([lower, upper], axis=1).tolist(), 'camera_keys': dict(zip(keys, ('front','side','wrist')))}}
        except Exception as exc:
            status.update(status='error', message=f'{type(exc).__name__}: {exc}')
            raise HTTPException(400, status['message']) from exc
