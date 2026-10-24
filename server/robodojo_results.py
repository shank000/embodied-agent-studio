"""Normalize one local RoboDojo _result.json without SDK or filesystem access.

The format and score units follow src/eval_client/eval_env.py at the revision
recorded in robodojo_catalog. A single file is one task/configuration/seed,
never an endorsed leaderboard result or a complete 42-task benchmark.
"""
from copy import deepcopy
import json
import math
import re

from .robodojo_catalog import BENCHMARK_ID, REVISION, VERSION, UPSTREAM, EMBODIMENTS, task_metadata


def _integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f'{name} 必须是非负整数。')
    return value


def _finite_number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _layout_ids(payload, name):
    if name not in payload:
        return None
    values = payload[name]
    if not isinstance(values, list):
        raise ValueError(f'{name} 必须是布局编号数组。')
    for value in values:
        _integer(value, name)
    if len(set(values)) != len(values):
        raise ValueError(f'{name} 含重复布局编号。')
    return values.copy()


def parse_result(payload: dict, task_id: str, seed: int, policy: str, embodiment: str) -> dict:
    """Validate terminal detail rows and derive single-file metrics.

    Detail scores are fractions (0..1); summary.score follows upstream's
    percentage (0..100). summary.success_rate is a fraction. Reported summary
    discrepancies are retained, never substituted for computed detail metrics.

    Exclusion fields are optional audit extensions: native _result.json does
    not record all abandoned/unstable attempts. assigned_episodes must be
    explicitly supplied to derive an all-assigned success rate; exclusion
    counters are never added together to guess that denominator.
    """
    if not isinstance(payload, dict):
        raise ValueError('RoboDojo 结果必须是 JSON 对象。')
    try:
        json.dumps(payload, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError('RoboDojo 结果包含非 JSON 数据或非有限数值。') from exc
    task = task_metadata(task_id)
    _integer(seed, 'seed')
    if seed > 4294967295:
        raise ValueError('seed 必须处于 0–4294967295。')
    if not isinstance(policy, str) or not policy.strip():
        raise ValueError('必须提供策略名称。')
    if not isinstance(embodiment, str) or not embodiment.strip():
        raise ValueError('必须提供机器人配置名称。')
    details = payload.get('details')
    if not isinstance(details, dict) or not details:
        raise ValueError('结果必须包含非空终局 details；顶层 success 或初始状态不能证明任务完成。')
    episodes, record_ids, layout_ids = [], set(), set()
    warnings = []
    for key, row in details.items():
        if type(key) is int and key >= 0:
            record_id = key
        elif isinstance(key, str) and re.fullmatch(r'[0-9]+', key):
            record_id = int(key)
        else:
            raise ValueError('details 记录编号必须是非负整数或其十进制字符串。')
        if record_id in record_ids:
            raise ValueError('details 包含重复的规范化记录编号。')
        record_ids.add(record_id)
        if not isinstance(row, dict):
            raise ValueError(f'details[{key}] 必须是终局记录对象。')
        for terminal in ('terminal', 'terminated', 'end_flag'):
            if terminal in row and row[terminal] is not True:
                raise ValueError(f'details[{key}] 显式声明尚未终止，不能导入为终局结果。')
        layout_id = _integer(row.get('layout_id'), f'details[{key}].layout_id')
        if layout_id in layout_ids:
            raise ValueError('details 含重复 layout_id，不能将同一布局重复计入分母。')
        layout_ids.add(layout_id)
        success, score = row.get('success'), row.get('score')
        if type(success) is not bool:
            raise ValueError(f'details[{key}].success 必须是布尔值。')
        if not _finite_number(score) or not 0 <= score <= 1:
            raise ValueError(f'details[{key}].score 必须是 0–1 的有限数值。')
        score = float(score)
        if success and not math.isclose(score, 1., abs_tol=1e-9):
            warnings.append({'code': 'success_score_disagreement', 'record_id': str(record_id),
                             'message': '该记录原生 success=true，但 score 不是 1；两项原样分开保留。'})
        episodes.append({'record_id': str(record_id), 'layout_id': layout_id,
                         'native_success': success, 'score': score})
    episodes.sort(key=lambda row: int(row['record_id']))
    completed = len(episodes)
    successes = sum(row['native_success'] for row in episodes)
    summary = {'eval_time': completed, 'successes': successes, 'failures': completed-successes,
               'success_rate': successes/completed,
               'score': math.fsum(row['score'] for row in episodes)/completed*100}
    reported = {key: deepcopy(payload[key]) for key in ('eval_time', 'success_rate', 'score') if key in payload}
    differences = []
    for key, value in reported.items():
        valid = type(value) is int and value >= 0 if key == 'eval_time' else _finite_number(value)
        equal = value == summary[key] if key == 'eval_time' else valid and math.isclose(value, summary[key], rel_tol=1e-9, abs_tol=1e-6)
        if not valid or not equal:
            differences.append({'field': key, 'reported': value, 'computed': summary[key]})
    if differences:
        warnings.append({'code': 'summary_disagreement', 'message': '源文件汇总与终局 details 不一致；展示按 details 重算值，原汇总保留。'})
    if set(reported) != {'eval_time', 'success_rate', 'score'}:
        warnings.append({'code': 'summary_incomplete', 'message': '源文件缺少部分官方汇总字段，结果仅按现有终局 details 计算。'})
    unstable = _integer(payload['unstable_nums'], 'unstable_nums') if 'unstable_nums' in payload else None
    abandoned = _layout_ids(payload, 'abandoned_layout_ids')
    excluded = _layout_ids(payload, 'excluded_layout_ids')
    assigned = _integer(payload['assigned_episodes'], 'assigned_episodes') if 'assigned_episodes' in payload else None
    if assigned is not None and assigned < completed:
        raise ValueError('assigned_episodes 小于已完成终局记录数量。')
    if (set(abandoned or []) | set(excluded or [])) & layout_ids:
        warnings.append({'code': 'exclusion_overlap', 'message': '源文件同时将某些布局列为已完成和已排除；未据此推算额外分母。'})
    denominator = {'native_valid_completed': completed, 'unstable_nums': unstable,
                   'abandoned_layout_ids': abandoned, 'excluded_layout_ids': excluded,
                   'assigned_episodes': assigned,
                   'all_assigned_success_rate': successes/assigned if assigned is not None else None}
    if assigned is None:
        warnings.append({'code': 'assigned_denominator_unknown', 'message': '单份 _result.json 未提供完整分配分母；不将缺失排除信息视为零。'})
    expected = task['native_episodes']
    scope = 'task_seed_complete' if completed == expected else 'overfull' if completed > expected else 'smoke' if completed == 1 else 'partial'
    if scope == 'overfull':
        warnings.append({'code': 'overfull_result', 'message': '记录数超过该任务的原生数量；全部保留，未静默截取前 N 项来宣称完整。'})
    known_embodiment = embodiment.strip() in {value['id'] for value in EMBODIMENTS}
    official_seed = seed in (0, 1, 2)
    native_videos = [{'record_id': row['record_id'], 'layout_id': row['layout_id'], 'camera': camera,
                      'suggested_name': f'episode_{int(row["record_id"]):07d}_cam_{camera}_{"success" if row["native_success"] else "fail"}.mp4'}
                     for row in episodes for camera in ('head', 'left_wrist', 'right_wrist')]
    return {'benchmark_id': BENCHMARK_ID, 'revision': REVISION, 'version': VERSION,
            'task_id': task_id, 'task': task, 'seed': seed, 'policy': policy.strip(), 'embodiment': embodiment.strip(),
            'episodes': episodes, 'summary': summary, 'reported_summary': reported,
            'summary_differences': differences, 'denominator': denominator,
            'coverage': {'scope': scope, 'completed_episodes': completed, 'expected_episodes': expected,
                         'task_seed_complete': completed == expected, 'official_overall_complete': False},
            'provenance': {'source': 'local_import', 'format': 'robodojo-_result.json',
                           'upstream_url': UPSTREAM, 'upstream_revision': REVISION,
                           'revision_role': 'parser_and_catalog_reference', 'source_revision_verified': False,
                           'official_verified': False, 'official_overall_complete': False,
                           'known_embodiment': known_embodiment, 'seed_in_official_protocol': official_seed},
            'warnings': warnings, 'native_videos': native_videos, 'raw': deepcopy(payload)}
