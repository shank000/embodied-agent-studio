const esc=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const states={pending:'等待执行',active:'进行中',holding:'等待稳定',achieved:'已达成',complete:'已达成',revoked:'条件已失效',blocked:'等待前置步骤',failed:'任务已失败'};

export function taskGraphMarkup(graph){
  if(!graph||!Array.isArray(graph.stages))return '';
  return `<section class="task-graph"><div class="task-graph-heading"><strong>复合任务进度</strong><small>${esc(graph.version)}</small></div><ol>${graph.stages.map(stage=>`<li class="task-stage ${stage.satisfied?'satisfied':''}"><span class="task-stage-state">${esc(states[stage.status]||stage.status||'等待')}</span><strong>${esc(stage.label||stage.id)}</strong>${stage.after?.length?`<small>前置：${stage.after.map(id=>esc(graph.stages.find(s=>s.id===id)?.label||id)).join('、')}</small>`:''}</li>`).join('')}</ol>${graph.terminal_failure?`<p class="config-error">任务已进入失败终态：${esc(graph.failure_reason||'违反任务规则')}</p>`:''}<p class="evaluation-note">阶段由物理状态独立判定；暂时达成后失效仍会记录，最终以完整任务验收为准。</p>${graph.events?.length?`<details><summary>阶段事件 · ${graph.events.length} 条</summary><ul>${graph.events.slice(-12).map(e=>`<li>${Number.isFinite(e.time)?e.time.toFixed(2)+' s · ':''}${esc(graph.stages.find(s=>s.id===e.stage_id)?.label||e.stage_id||'任务')} · ${esc({achieved:'达成',revoked:'失效',violation:'违规'}[e.type]||e.type)}${e.reason?` · ${esc(e.reason)}`:''}</li>`).join('')}</ul></details>`:''}</section>`;
}

export function routingMarkup(call){
  const route=call.routing||{};
  if(!call.trace_id&&!route.availability)return '';
  return `<div class="routing-evidence"><p><strong>路由与调用关联</strong> · 请求 ${esc(call.requested_model||'未记录')} → 实际 ${esc(route.resolved_model||call.model||'未返回')}</p>${route.availability==='returned'?`<p>能力 ${esc(route.selected_capability||'未返回')} · 路由版本 ${esc(route.router_policy_version||'未返回')} · 选择概率 ${Number.isFinite(route.selection_probability)?route.selection_probability:'未返回'}</p>${route.reason_code?`<p>选择原因：${esc(route.reason_code)}</p>`:''}`:'<p>服务未返回路由选择明细；不能据最终模型名推断内部决策。</p>'}<details><summary>关联编号与计费范围</summary><p>决策 ${esc(call.decision_id||'未记录')}<br>调用 ${esc(call.call_id)}<br>父调用 ${esc(call.parent_call_id||'无')}<br>计费 ${esc(call.cost?.billing_scope||call.billing?.scope||'self')} · ${esc(call.cost?.evidence_type||'未定价')}</p></details></div>`;
}
