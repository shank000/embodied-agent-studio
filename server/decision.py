"""Typed candidate selection. Decision services never submit actuator commands."""
from __future__ import annotations

import asyncio
import time
from typing import Literal
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .policy import http_result
from .runlog import model_output
from .telemetry import track_inference


class DecisionConfig(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    kind: Literal['typesafe', 'custom'] = 'custom'
    url: str = ''
    model: str = 'jev-latest'
    vision: bool = False
    question_name: str = Field(default='next_action', pattern=r'^[A-Za-z_][A-Za-z0-9_]{0,63}$')
    timeout_s: float = Field(default=120, ge=1, le=600)
    weight: float = Field(default=1., gt=0, le=100)

    @field_validator('url')
    @classmethod
    def valid_url(cls, value):
        if not value: return value
        parsed = urlsplit(value)
        if (parsed.scheme not in ('http', 'https') or not parsed.hostname or
                parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise ValueError('决策接口须为不含账号、参数或密钥的 HTTP(S) URL。')
        if parsed.scheme == 'http' and parsed.hostname not in ('localhost', '127.0.0.1', '::1'):
            raise ValueError('远程决策接口请使用 HTTPS。')
        return value.rstrip('/')

    def endpoint(self):
        return self.url or ('https://api.typesafe.ai/v1/systemone' if self.kind == 'typesafe' else '')

    def validate_credentials(self, key):
        endpoint = self.endpoint()
        if not endpoint: raise ValueError('请配置自定义决策接口地址。')
        if not key and urlsplit(endpoint).hostname not in ('localhost', '127.0.0.1', '::1'):
            raise ValueError('远程决策服务需要在运行时配置 API Key；密钥不写入方法版本。')


class DecisionResult(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    protocol: Literal['embodied-decision-v1'] = 'embodied-decision-v1'
    episode_id: str
    observation_id: int
    status: Literal['selected', 'blocked', 'replan'] = 'selected'
    candidate_id: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    probabilities: dict[str, float] | None = None
    model: str | None = None
    explanation: str = Field(default='', max_length=2000)
    usage: dict | None = None
    aggregation: dict | None = None
    reasoning_content: str | None = None
    thinking: str | None = None
    summary: str | None = None

    @model_validator(mode='after')
    def selected_id(self):
        if self.status == 'selected' and not self.candidate_id:
            raise ValueError('决策结果 selected 必须引用 candidate_id。')
        if self.status != 'selected' and self.candidate_id is not None:
            raise ValueError('blocked/replan 决策不能携带待执行动作。')
        if self.probabilities and any(not 0 <= p <= 1 for p in self.probabilities.values()):
            raise ValueError('决策概率必须在 0 至 1 之间。')
        return self


@track_inference('decision')
async def choose_candidate(config, context, candidates, *, api_key='', cameras=()):
    """Bind a typed response to this pending observation and a finite candidate set."""
    config = DecisionConfig.model_validate(config)
    config.validate_credentials(api_key)
    ids = [candidate['id'] for candidate in candidates]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('决策需要非空、ID 唯一的动作候选；未调用服务。')
    headers = {'Authorization': 'Bearer ' + api_key} if api_key else {}
    if config.kind == 'custom':
        body = {'protocol': 'embodied-decision-v1', **context,
                'model': config.model, 'candidates': candidates}
        if config.vision: body['cameras'] = list(cameras)
    else:
        state = {**context, 'candidates': candidates, 'current_camera_images_available': config.vision}
        body = {'model': config.model, 'state': state, 'questions': {
            config.question_name: {'type': 'choice',
                'instructions': 'Choose one listed candidate for the current subgoal. Return its ID. '
                                'All commands remain subject to local IK and execution validation.',
                'criteria': {candidate['id']: candidate['description'] for candidate in candidates}}}}
        if config.vision:
            if len(cameras) != 3: raise ValueError('启用 Jev 图像输入时需要当前三路相机。')
            headers['TypeSafe-Danger-Zone-Images'] = 'true'
            body['assets'] = {}
            state['cameras'] = {}
            for camera in cameras:
                reference = f"obs_{context['observation_id']}_{camera['name']}"
                body['assets'][reference] = {'type': 'image', 'format': 'jpg', 'data_base64': camera['image_base64']}
                # References are values in state, not merely dictionary keys.
                state['cameras'][camera['name']] = reference
    async with httpx.AsyncClient(timeout=config.timeout_s, trust_env=False) as client:
        response = await client.post(config.endpoint(), json=body, headers=headers)
    try: returned = response.json()
    except ValueError: returned = {'content': response.text}
    model_output({'provider': config.kind, 'model': config.model, 'response': returned})
    raw = http_result(response)
    if config.kind == 'custom':
        if raw.get('protocol') != 'embodied-decision-v1':
            raise ValueError('自定义决策服务未返回 embodied-decision-v1 协议。')
        result = DecisionResult.model_validate(raw)
    else:
        answer = raw.get('answers', {}).get(config.question_name, {})
        if answer.get('type') != 'choice': raise ValueError('Jev 未返回配置的 Choice 问题结果。')
        result = DecisionResult(episode_id=context['episode_id'], observation_id=context['observation_id'],
            candidate_id=answer.get('choice'), confidence=answer.get('confidence'),
            probabilities=answer.get('probabilities'), model=raw.get('model'), usage=raw.get('usage'))
    if result.episode_id != context['episode_id'] or result.observation_id != context['observation_id']:
        raise ValueError('决策结果属于其他 episode 或观察，已拒绝执行。')
    if result.status == 'selected' and result.candidate_id not in ids:
        raise ValueError('决策引用了未提供的动作候选，已拒绝执行。')
    if result.probabilities and any(key not in ids for key in result.probabilities):
        raise ValueError('决策概率引用了未知候选。')
    return result


async def choose_decisions(configs, strategy, context, candidates, *, api_keys=None, cameras=(),
                           min_confidence=0., on_result=None, chooser=choose_candidate):
    """Single choice, serial veto, or declared-weight voting over one observation."""
    configs = [DecisionConfig.model_validate(config) for config in configs]
    if strategy not in ('single', 'cascade', 'weighted_vote'):
        raise ValueError('未知决策组合策略。')
    if not configs or (strategy == 'single' and len(configs) != 1) or (strategy != 'single' and len(configs) < 2):
        raise ValueError('single 需要一个决策器；cascade/weighted_vote 需要至少两个决策器。')
    if not 0 <= min_confidence <= 1: raise ValueError('决策置信度阈值须在 0 至 1 之间。')
    if any(candidate['id'] == 'request_replan' for candidate in candidates):
        raise ValueError('request_replan 为决策组合保留的候选 ID。')
    api_keys = api_keys or {}
    replan_option = {'id': 'request_replan', 'description': '拒绝当前动作并请求主 Agent 重新规划。', 'action_type': 'replan'}
    async def ask(index, offered, previous=None):
        config = configs[index]
        started = time.monotonic()
        record = {'decision_id': str(index), 'kind': config.kind, 'url': config.endpoint(),
                  'declared_model': config.model, 'weight': config.weight,
                  'episode_id': context['episode_id'], 'observation_id': context['observation_id']}
        try:
            result = await chooser(config, {**context, 'decision_id': str(index),
                'current_camera_images_available': config.vision, 'previous_selection': previous},
                offered, api_key=api_keys.get(str(index), ''), cameras=cameras)
            if (result.episode_id != context['episode_id'] or result.observation_id != context['observation_id'] or
                    (result.status == 'selected' and result.candidate_id not in {candidate['id'] for candidate in offered})):
                raise ValueError('决策组合收到过期观察或未提供的候选。')
            record.update(status='complete', result=result.model_dump(), usage=result.usage)
            return result
        except BaseException as exc:
            record.update(status='cancelled' if isinstance(exc, asyncio.CancelledError) else 'error',
                          error=str(exc), usage=getattr(exc, 'usage', None))
            raise
        finally:
            record['duration_s'] = time.monotonic()-started
            if on_result is not None: on_result(record)
    def accepted(result):
        return (result.status == 'selected' and result.candidate_id != 'request_replan' and
                (min_confidence == 0 or result.confidence is not None and result.confidence >= min_confidence))
    def replan(message, aggregation):
        return DecisionResult(episode_id=context['episode_id'], observation_id=context['observation_id'],
            status='replan', explanation=message, aggregation=aggregation)
    if strategy in ('single', 'cascade'):
        selected = None
        for index in range(len(configs)):
            offered = candidates if selected is None else [next(candidate for candidate in candidates if candidate['id'] == selected.candidate_id)]
            result = await ask(index, offered + [replan_option], selected.model_dump() if selected else None)
            if strategy == 'single' and result.status == 'blocked': return result
            if not accepted(result):
                return replan(result.explanation or '决策器拒绝、请求重规划或未达到置信度要求。',
                              {'strategy': strategy, 'rejected_by': str(index)})
            selected = result
        return selected.model_copy(update={'aggregation': {'strategy': strategy, 'approved_by': list(map(str, range(len(configs))))}})
    tasks = [asyncio.create_task(ask(index, candidates+[replan_option])) for index in range(len(configs))]
    try:
        results = await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            if not task.done(): task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    totals = {}
    total_weight = sum(config.weight for config in configs)
    for config, result in zip(configs, results):
        candidate_id = result.candidate_id if accepted(result) else 'request_replan'
        totals[candidate_id] = totals.get(candidate_id, 0.) + config.weight
    winner = max(totals, key=totals.get)
    aggregation = {'strategy': strategy, 'declared_weight_votes': totals,
                   'total_declared_weight': total_weight, 'winner': winner,
                   'vote_share': totals[winner]/total_weight, 'vote_share_semantics': 'declared_weight_fraction_not_calibrated_probability'}
    if winner == 'request_replan' or totals[winner] <= total_weight/2:
        return replan('加权选票未形成可执行动作的过半共识，请重新规划。', aggregation)
    return DecisionResult(episode_id=context['episode_id'], observation_id=context['observation_id'],
                          candidate_id=winner, aggregation=aggregation)
