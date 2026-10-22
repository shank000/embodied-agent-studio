"""Editable names and reversible result deletion, separate from experiment evidence."""
from datetime import datetime, timezone
import json
from threading import RLock

from .runlog import LIVE_RUNS, live_snapshot, run_folder

MANAGEMENT_LOCK = RLock()


class RunInUseError(ValueError):
    pass


def read_run_record(run_id):
    folder = run_folder(run_id)
    summary = folder / 'summary.json'
    if summary.is_file():
        return json.loads(summary.read_text(encoding='utf-8'))
    live = LIVE_RUNS.get(run_id)
    if live is not None and not live._finished:
        return live_snapshot(live)
    with (folder / 'events.jsonl').open(encoding='utf-8') as stream:
        event = json.loads(stream.readline())
    return {**event['data'], 'started_at': event['data'].get('started_at') or event.get('timestamp'),
            'status': 'interrupted', 'message': '服务中断或未正常结束，已有事件仍可下载。'}


def primary_models(record, *, _seen=()):
    """Use declared primary models; a hybrid's decision helpers are not its planner."""
    if record.get('primary_models'):
        return list(dict.fromkeys(record['primary_models']))
    recipe = record.get('recipe') or {}
    config = (recipe.get('policy') if recipe.get('mode') == 'policy' else recipe.get('agent')) or {}
    model = record.get('model') or config.get('model') or (record.get('settings') or {}).get('model')
    if model:
        return [str(model)]
    models = []
    for row in record.get('rows') or []:
        child_id = row.get('run_id')
        if not child_id or child_id in _seen: continue
        try: child = read_run_record(child_id)
        except (FileNotFoundError, ValueError): continue
        models.extend(primary_models(child, _seen=(*_seen, child_id)))
    return list(dict.fromkeys(models))


def _management(folder):
    path = folder / 'record.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}


def record_labels(record):
    run_id = record['run_id']
    folder = run_folder(run_id)
    management = _management(folder)
    started = record.get('started_at')
    try:
        instant = datetime.fromisoformat(started) if started else datetime.strptime(run_id[:15], '%Y%m%dT%H%M%S').replace(tzinfo=timezone.utc)
        stamp = instant.astimezone().strftime('%Y-%m-%d %H:%M:%S')
    except (ValueError, TypeError): stamp = run_id[:15]
    models = primary_models(record, _seen=(run_id,))
    label = ' + '.join(models) or {'comparison': '多方法对比', 'research_experiment': '多方法研究实验'}.get(record.get('kind'), '模型未记录')
    default_name = f'{label} · {stamp}'
    custom_name = management.get('name') or None
    return {'display_name': custom_name or default_name, 'default_name': default_name,
            'custom_name': custom_name, 'deleted_at': management.get('deleted_at')}


def _update(run_id, changes):
    folder = run_folder(run_id)
    data = {**_management(folder), **changes}
    pending = folder / 'record.pending.json'
    pending.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    pending.replace(folder / 'record.json')
    return {'run_id': run_id, **record_labels(read_run_record(run_id))}


def rename_run(run_id, name):
    if not isinstance(name, str): raise ValueError('实验名称必须是文字。')
    name = name.strip()
    if len(name) > 160: raise ValueError('实验名称最多 160 个字符。')
    if any(ord(char) < 32 or ord(char) == 127 for char in name): raise ValueError('实验名称不能包含换行或控制字符。')
    with MANAGEMENT_LOCK:
        return _update(run_id, {'name': name or None})


def delete_run(run_id):
    with MANAGEMENT_LOCK:
        run_folder(run_id)
        for live in list(LIVE_RUNS.values()):
            if live._finished: continue
            if live.id == run_id: raise RunInUseError('实验正在运行，请先停止实验，再移入回收站。')
            if any(row.get('run_id') == run_id for row in live_snapshot(live).get('rows', [])):
                raise RunInUseError('正在运行的批量实验仍在使用此记录，请等待批量实验结束。')
        return _update(run_id, {'deleted_at': datetime.now(timezone.utc).isoformat()})


def batch_delete_runs(run_ids):
    """Move each eligible record to trash and report individual failures."""
    if not isinstance(run_ids, list) or not 1 <= len(run_ids) <= 1000:
        raise ValueError('每次批量删除请选择 1 至 1000 条实验记录。')
    if any(not isinstance(run_id, str) for run_id in run_ids):
        raise ValueError('实验记录编号必须是文字。')
    deleted, failed = [], []
    for run_id in dict.fromkeys(run_ids):
        try:
            delete_run(run_id)
        except (ValueError, OSError) as exc:
            failed.append({'run_id': run_id, 'detail': str(exc)})
        else:
            deleted.append(run_id)
    return {'deleted': deleted, 'failed': failed}


def restore_run(run_id):
    with MANAGEMENT_LOCK:
        return _update(run_id, {'deleted_at': None})
