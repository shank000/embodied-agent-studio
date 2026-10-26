const views=new WeakMap();
const esc=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const value=text=>{if(typeof text!=='string')return JSON.stringify(text,null,2);try{return JSON.stringify(JSON.parse(text),null,2);}catch{return text;}};
const secs=n=>Number.isFinite(n)?`${n.toFixed(2)} s`:'未记录';
const number=n=>Number.isFinite(n)?n.toLocaleString('zh-CN'):'未返回';
const phases={agent_inference:'Agent 推理',policy_inference:'策略推理',repair_diagnosis:'故障诊断',observation:'观察',inference:'推理与恢复',planning:'规划',agent:'Agent',policy:'策略',decision_aggregation:'组合决策',execution:'执行与核验',feedback:'执行后观察'};
const statuses={running:'进行中',complete:'本轮结束',verified:'环境确认成功',blocked:'无法继续',limit:'达到预算',replan:'重新决策',error:'失败',cancelled:'已停止',interrupted:'服务中断',unmeasured:'历史记录未测量轮次时间'};
const labels={task_evaluation:'最终任务评价',observation:'本轮输入与观察',observation_after:'执行后观察',action:'模型动作',policy_action:'策略动作块',selection:'最终动作选择',execution:'执行反馈',transition:'状态转移',completion_verification:'完成核验',recoverable_error:'触发恢复',recovery_retry:'恢复后重试',recovery_exhausted:'恢复预算耗尽',decision:'决策结果',plan:'子目标',agent_action:'Agent 输出',candidates:'候选动作',decision_result:'决策器反馈',replan_trigger:'重新规划原因',error:'错误',phase:'阶段耗时'};

export function roundMarkup(row,runId){
  const timing=Object.entries(row.phase_s||{}).map(([name,n])=>`${phases[name]||name} ${secs(n)}`).join(' · ');
  const calls=(row.calls||[]).map(call=>`<section class="round-call"><h4>${esc(call.model||'模型未记录')} · ${esc(phases[call.phase]||call.phase||'调用')} · ${secs(call.inference_s??call.current_wait_s)}</h4><p>${esc(call.provider)} · ${esc(call.status==='complete'?'调用完成':statuses[call.status]||call.status)} · 推理强度 ${esc(['provider_default','not_requested'].includes(call.reasoning_effort)?'服务默认':call.reasoning_effort||'服务默认')} · 尝试 ${number((call.attempt??0)+1)}</p><p>输入 ${number(call.input_tokens)} / 输出 ${number(call.output_tokens)} / 推理 ${number(call.reasoning_tokens)} tokens</p><h5>公开输出</h5><pre>${esc(value(call.public_text??'等待服务返回公开输出。'))}</pre><h5>服务返回的推理文字或摘要</h5><pre>${esc(call.reasoning?.text||'服务尚未返回可展示的推理文字或摘要。')}</pre>${call.error?`<p class="config-error">${esc(call.error)}</p>`:''}</section>`).join('');
  const events=(row.events||[]).map(event=>{
    const observation=event.data?.observation;
    const cameras=(observation?.cameras||[]).filter(camera=>camera.image_ref).map(camera=>`<figure><img loading="lazy" alt="${esc(camera.name)}" src="/api/runs/${encodeURIComponent(runId)}/artifact/${camera.image_ref.split('/').map(encodeURIComponent).join('/')}"><figcaption>${esc(camera.name)}</figcaption></figure>`).join('');
    return `<details class="round-event"><summary>${esc(labels[event.event]||event.event)}</summary>${cameras?`<div class="round-cameras">${cameras}</div>`:''}<pre>${esc(value(event.data))}</pre></details>`;
  }).join('');
  return `<summary><strong>第 ${number(row.round)} 轮 · ${esc(statuses[row.status]||row.status)}</strong><span>整轮 ${secs(row.elapsed_s)}</span></summary><p class="round-phase-times">${esc(timing||'历史记录缺少完整轮次计时；已有调用仍可查看。')}</p>${row.started_at?`<p>开始 ${esc(new Date(row.started_at).toLocaleString('zh-CN',{hour12:false}))}${row.ended_at?` · 结束 ${esc(new Date(row.ended_at).toLocaleTimeString('zh-CN',{hour12:false}))}`:''}</p>`:''}${calls||'<p>此轮尚无模型调用记录。</p>'}${row.error?`<p class="config-error">${esc(row.error)}</p>`:''}${row.feedback_error?`<p class="config-error">执行后观察未保存：${esc(row.feedback_error)}</p>`:''}<details class="round-event"><summary>原生任务评价与奖励证据</summary><pre>${esc(value({before:row.before,after:row.after,reward:row.reward,reward_version:row.reward_version}))}</pre></details>${events}`;
}

export function renderRoundTimeline(root,controller){
  if(!root)return;
  let view=views.get(root);
  if(!view||view.id!==controller?.run_id){
    view={id:controller?.run_id,offset:0,last:0,busy:false,status:null};views.set(root,view);root.replaceChildren();
    if(!view.id){root.textContent='开始任务后，这里按轮次记录完整输入、公开输出、执行反馈与耗时。';return;}
    root.innerHTML='<div class="round-list-heading"><strong>逐轮记录</strong><span class="round-list-count"></span></div><p class="round-list-note">整轮时间含观察、模型调用、恢复等待、执行与核验；各调用耗时含网络和排队。</p><div class="round-list"></div><div class="round-pagination"><button type="button" class="secondary small">较新轮次</button><button type="button" class="secondary small">较早轮次</button></div><p class="round-list-error" role="status"></p>';
    const buttons=root.querySelectorAll('button');
    buttons[0].onclick=()=>{view.offset=Math.max(0,view.offset-20);view.last=0;renderRoundTimeline(root,view.controller);};
    buttons[1].onclick=()=>{view.offset+=20;view.last=0;renderRoundTimeline(root,view.controller);};
  }
  view.controller=controller;
  if(!view.id||view.busy||root.closest('[hidden]'))return;
  if(Date.now()-view.last<1500&&view.status===controller.status)return;
  if(view.last&&view.status===controller.status&&!controller.active)return;
  view.busy=true;view.last=Date.now();view.status=controller.status;
  fetch(`/api/runs/${encodeURIComponent(view.id)}/rounds?limit=20&offset=${view.offset}`).then(async response=>{const data=await response.json();if(!response.ok)throw new Error(data.detail||'轮次记录读取失败');return data;}).then(data=>{
    if(views.get(root)!==view)return;
    const list=root.querySelector('.round-list');const opened=new Set([...list.querySelectorAll('details[open][data-round]')].map(el=>el.dataset.round));
    const wasEmpty=!list.children.length;const nodes=[];
    for(const [index,row] of data.rounds.entries()){
      const old=list.querySelector(`[data-round="${row.round}"]`);const key=JSON.stringify(row);
      if(old?._contentKey===key){nodes.push(old);continue;}
      const node=document.createElement('details');node.className='round-record round-complete';node.dataset.round=String(row.round);node.open=opened.has(String(row.round))||(wasEmpty&&index===0);node.innerHTML=roundMarkup(row,view.id);node._contentKey=key;
      if(old){const previousText=[...old.querySelectorAll('pre')];node.querySelectorAll('pre').forEach((el,i)=>{const prev=previousText[i];if(prev&&el.textContent.startsWith(prev.textContent)){el.scrollTop=prev.scrollTop;el.scrollLeft=prev.scrollLeft;}});const previous=[...old.querySelectorAll('.round-event')].map(el=>el.open);node.querySelectorAll('.round-event').forEach((el,i)=>el.open=previous[i]||false);}
      nodes.push(node);
    }
    list.replaceChildren(...nodes);root.querySelector('.round-list-count').textContent=`共 ${data.total} 轮 · ${view.offset+1}–${Math.min(view.offset+20,data.total)}`;
    const buttons=root.querySelectorAll('.round-pagination button');buttons[0].disabled=view.offset===0;buttons[1].disabled=view.offset+20>=data.total;
    root.querySelector('.round-list-error').textContent=data.issues?.join('；')||(!data.total?'等待第一轮观察。':'');
  }).catch(error=>{if(views.get(root)===view){root.querySelector('.round-list-error').textContent=error.message;view.last=0;view.status=null;}}).finally(()=>view.busy=false);
}
