"""OmniGibson runtime configuration and lightweight checks; never imports its SDK."""
from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_FILE = ROOT / '.venv-omnigibson' / 'runtime.json'
RELEASE = 'v3.9.4'
REVISION = 'fe66cc74538f7eba3c57255da0ee88e8ce113c87'
SCHEMA = 'embodiworks-omnigibson-runtime-v1'
FIELDS = {'schema_version', 'kind', 'python', 'source_root', 'data_path', 'probe'}


def validate_config(config):
    if not isinstance(config, dict) or set(config) - FIELDS:
        raise ValueError('OmniGibson runtime.json 含未知字段；请保留原文件并使用配置工具检查。')
    if config.get('kind') != 'native' or config.get('schema_version') != SCHEMA:
        raise ValueError('OmniGibson 需要独立本机 Python 配置，schema_version 或 kind 不匹配。')
    for key in ('python', 'source_root', 'data_path'):
        value = config.get(key)
        if not isinstance(value, str) or not value or any(c in value for c in '\r\n\x00') or not Path(value).is_absolute():
            raise ValueError(f'OmniGibson {key} 必须为本机绝对路径。')
    executable = Path(config['python']).name.lower()
    if executable not in ('python', 'python3', 'python3.11', 'python.exe'):
        raise ValueError('OmniGibson python 必须指向独立 Python 3.11 解释器。')
    if 'probe' in config and not isinstance(config['probe'], dict):
        raise ValueError('OmniGibson probe 须由配置工具生成。')
    if 'probe' in config:
        probe = config['probe']
        if (not isinstance(probe.get('signature'), dict) or type(probe.get('packages_ready')) is not bool
                or not isinstance(probe.get('revision'), str) or not isinstance(probe.get('gpu'), dict)):
            raise ValueError('OmniGibson probe 已损坏；请重新使用配置工具检查。')
    return dict(config)


def runtime_config():
    if not RUNTIME_FILE.is_file():
        return None
    return validate_config(json.loads(RUNTIME_FILE.read_text(encoding='utf-8-sig')))


def total_memory_gib():
    try:
        if sys.platform == 'win32':
            class MemoryStatus(ctypes.Structure):
                _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong),
                            *[(name, ctypes.c_ulonglong) for name in
                              ('total', 'available', 'total_page', 'available_page', 'total_virtual', 'available_virtual', 'extended')]]
            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return None
            return status.total / 1024**3
        return os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 1024**3
    except (AttributeError, OSError, ValueError):
        return None


def _signature(config):
    executable = Path(config['python'])
    stat = executable.stat()
    return {'python': str(executable.resolve()), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
            'source_root': str(Path(config['source_root']).resolve()), 'data_path': str(Path(config['data_path']).resolve())}


def readiness(config=None):
    """Read configuration and filesystem only. A successful check is not native acceptance."""
    checks = []
    def add(key, label, passed, detail):
        checks.append({'id': key, 'label': label, 'status': 'passed' if passed else 'missing', 'detail': detail})
    try:
        config = validate_config(config) if config is not None else runtime_config()
    except (ValueError, OSError) as exc:
        add('configuration', '运行时配置', False, str(exc))
        config = None
    if not any(x['id'] == 'configuration' for x in checks):
        add('configuration', '独立运行时配置', config is not None,
            str(RUNTIME_FILE) if config else '使用 python -m tools.omnigibson_setup 配置官方源码、数据根目录与独立 Python。')
    add('os', 'Windows / Linux', sys.platform in ('win32', 'linux'), sys.platform)
    memory = total_memory_gib()
    # Installed 32 GB systems report slightly less than 32 GiB after reserved memory.
    add('ram', '至少 32 GB 系统内存', memory is not None and memory >= 30,
        f'检测到 {memory:.1f} GiB；按 Isaac Sim 5.1 官方最低配置预检。' if memory is not None else '无法读取内存容量。')
    if config:
        python = Path(config['python'])
        source, data = Path(config['source_root']), Path(config['data_path'])
        add('python', '独立 Python 解释器', python.is_file(), str(python))
        add('source', '固定版本官方源码', (source/'OmniGibson/omnigibson').is_dir() and (source/'bddl3').is_dir(),
            f'{source}；要求 {RELEASE} / {REVISION[:12]}。')
        for folder, expected in (('behavior-1k-assets', '3.9.0'), ('omnigibson-robot-assets', '3.8.2')):
            target = data/folder
            try:
                version = (target/'VERSION').read_text(encoding='utf-8').strip()
            except OSError:
                version = None
            add(folder, folder, version == expected,
                f'{target}；资产版本 {version or "未找到"}，要求 {expected}。')
        key = data/'omnigibson.key'
        add('asset_access', '官方资产访问文件', key.is_file() and key.stat().st_size > 0,
            '仅检查访问文件是否存在，不读取或显示密钥；缺失时请通过上游许可流程安装。')
        from .backends.omnigibson_catalog import catalog
        _, scenes, _ = catalog(data)
        instances_ready = any(ids for scene in scenes.values() for ids in scene.get('instances', {}).values())
        add('instances', '可选择的官方预采样任务实例', instances_ready,
            '已发现预采样模板；各实例加载和物理结果仍待验收。' if instances_ready else
            '没有发现可运行的官方预采样模板，请安装匹配的任务实例。')
        probe = config.get('probe', {})
        try:
            signature_matches = probe.get('signature') == _signature(config)
        except OSError:
            signature_matches = False
        package_ok = signature_matches and probe.get('packages_ready') is True and probe.get('revision') == REVISION
        add('dependencies', '解释器、SDK 与源码版本检查', package_ok,
            '已检查 Python 3.11、OmniGibson、BDDL、Isaac Sim 5.1；启动前再次检查。' if package_ok else
            '运行 python -m tools.omnigibson_setup --check；包版本检查不会启动仿真。')
        gpu = probe.get('gpu', {}) if signature_matches else {}
        memory_mib = gpu.get('memory_mib')
        gpu_ok = type(memory_mib) in (int, float) and memory_mib >= 15360
        add('gpu', 'NVIDIA GPU，至少 16 GB 显存', gpu_ok,
            f'{gpu.get("name", "未检查")}；{memory_mib or "未知"} MiB。RT 核心与实际渲染仍须原生验收。')
    else:
        add('assets', '官方环境、机器人资产及许可', False, '通过上游安装脚本准备数据；平台不下载或代替用户接受许可。')
        add('gpu', 'NVIDIA GPU，至少 16 GB 显存', False, '配置工具将检查 GPU 容量；原生渲染另行验收。')
    ready = all(item['status'] == 'passed' for item in checks)
    return {'ready': ready, 'checks': checks, 'release': RELEASE, 'revision': REVISION,
            'native_validation': 'not_recorded',
            'message': '静态准备检查通过；首次原生加载、动作和任务评价仍待验收。' if ready else
                       'OmniGibson 尚未就绪。请完成下方准备项；不会改用其他引擎。'}


_PROBE = '''import importlib.metadata as m, importlib.util as u, json, sys
def version(name):
    try: return m.version(name)
    except m.PackageNotFoundError: return None
def origin(name):
    spec=u.find_spec(name)
    return spec.origin if spec else None
print(json.dumps({'python':list(sys.version_info[:2]), 'omnigibson':version('omnigibson'),
    'isaacsim':version('isaacsim'), 'bddl':origin('bddl'), 'sdk_origin':origin('omnigibson'),
    'pillow':version('Pillow'), 'torch':version('torch')}))
'''


def probe_runtime(config):
    """Execute only version/spec queries and GPU inventory, never import or start the SDK."""
    config = validate_config(config)
    kwargs = {'capture_output': True, 'text': True, 'encoding': 'utf-8', 'errors': 'replace', 'timeout': 25}
    if sys.platform == 'win32':
        kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW
    result = subprocess.run([config['python'], '-I', '-c', _PROBE], **kwargs)
    if result.returncode:
        raise ValueError('OmniGibson 解释器检查失败：' + (result.stderr or result.stdout).strip()[-2000:])
    try:
        packages = json.loads(result.stdout)
    except ValueError as exc:
        raise ValueError('独立解释器未返回有效的依赖检查结果。') from exc
    revision = subprocess.run(['git', '-C', config['source_root'], 'rev-parse', 'HEAD'], **kwargs)
    if revision.returncode:
        raise ValueError('无法读取官方源码 Git 版本：' + revision.stderr.strip()[-1000:])
    revision = revision.stdout.strip()
    expected_source = Path(config['source_root'])/'OmniGibson/omnigibson'
    origin = packages.get('sdk_origin')
    source_matches = bool(origin) and Path(origin).resolve().is_relative_to(expected_source.resolve())
    bddl_origin = packages.get('bddl')
    bddl_matches = bool(bddl_origin) and Path(bddl_origin).resolve().is_relative_to((Path(config['source_root'])/'bddl3').resolve())
    # The v3.9.4 tag still declares distribution version 3.9.3; pin the source commit as well.
    ready = (packages.get('python') == [3, 11] and packages.get('omnigibson') == '3.9.3'
             and packages.get('isaacsim') in ('5.1.0', '5.1.0.0') and bddl_matches
             and str(packages.get('torch', '')).split('+')[0] == '2.7.0'
             and bool(packages.get('pillow')) and source_matches and revision == REVISION)
    if not ready:
        raise ValueError(f'运行时版本不匹配：{packages}；源码 {revision}。要求 {RELEASE} ({REVISION[:12]})、Python 3.11、Isaac Sim 5.1，且 SDK 为该源码的 editable 安装。')
    gpu = {}
    try:
        query = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'], **kwargs)
        candidates = []
        if query.returncode == 0:
            for row in query.stdout.splitlines():
                name, memory = row.rsplit(',', 1)
                candidates.append({'name': name.strip(), 'memory_mib': int(memory.strip())})
            # OmniGibson GPU selection follows the same explicit environment variable.
            index = int(os.environ.get('OMNIGIBSON_GPU_ID', '0'))
            if index >= 0 and index < len(candidates):
                gpu = candidates[index]
    except (OSError, ValueError, subprocess.TimeoutExpired):
        gpu = {}
    return {'signature': _signature(config), 'packages_ready': ready, 'packages': packages,
            'revision': revision, 'gpu': gpu}


def configure(config, *, replace=False):
    """Write checked local paths. Existing unrelated configuration is never overwritten."""
    config = validate_config(config)
    if RUNTIME_FILE.exists():
        runtime_config()  # Reject unknown/legacy formats even when replace was requested.
        if not replace:
            raise ValueError(f'配置已存在：{RUNTIME_FILE}。确认路径后使用 --replace 更新；原文件会保留为 runtime.previous.json。')
        if RUNTIME_FILE.with_name('runtime.previous.json').exists():
            raise ValueError('已有 runtime.previous.json 配置备份；请先自行保留该备份，平台不会覆盖。')
    config['probe'] = probe_runtime(config)
    RUNTIME_FILE.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='runtime-', suffix='.json', dir=RUNTIME_FILE.parent)
    pending = Path(temporary)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            json.dump(config, stream, ensure_ascii=False, indent=2)
        if RUNTIME_FILE.exists():
            backup = RUNTIME_FILE.with_name('runtime.previous.json')
            if backup.exists():
                raise ValueError(f'已有配置备份 {backup}；请先自行保留该备份，平台不会覆盖。')
            backup.write_bytes(RUNTIME_FILE.read_bytes())
        os.replace(pending, RUNTIME_FILE)
    finally:
        pending.unlink(missing_ok=True)
    return readiness(config)
