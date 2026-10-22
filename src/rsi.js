import './rsi.css';
import {renderRunVideo,runVideoMarkup,renderRunTimer} from './research.js';

const $=selector=>document.querySelector(selector);
const escape=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const modes={agent:'视觉 Agent',policy:'策略模型',hybrid:'组合方法'};
const statuses={idle:'等待任务',queued:'等待执行',preparing:'准备实验',loading:'加载模型',running:'运行中',observing:'采集观察',planning:'主 Agent 规划',thinking:'模型推理',deciding:'选择候选',executing:'执行动作',repairing:'故障暂停 · 正在诊断',repair_paused:'故障暂停 · 等待处理',retrying:'重试当前轮',verified:'评价器确认成功',model_done:'模型报告完成',blocked:'无法继续',error:'运行错误',failed:'执行未到位',stopped:'已停止',limit:'达到预算',round_complete:'本步结束',complete:'已完成',completed:'已完成',unfinished:'未结束'};
let context,defaults,catalog,methods=[],protocols=[],experiments=[],registry={},episode=null,replayRun='',lastRegistry='',lastExperiment='',lastRun='',lastFinished='',currentState={},busy=false;
const methodId=method=>method.method_id||method.id;
const protocolId=protocol=>protocol.protocol_id||protocol.id;
const experimentId=experiment=>experiment.experiment_id||experiment.run_id;
const methodName=id=>methods.find(method=>methodId(method)===id)?.recipe?.name||id||'—';
const pretty=value=>JSON.stringify(value??{},null,2);
const seconds=value=>value==null?'—':`${Number(value).toFixed(2)} s`;

async function request(path,payload){
  const response=await fetch(`/api/${path}`,payload===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  let data;try{data=await response.json();}catch{throw new Error(`服务未返回 JSON（HTTP ${response.status}），请检查本地服务。`);}
  if(!response.ok){const detail=data.detail;throw new Error(typeof detail==='string'?detail:Array.isArray(detail)?detail.map(item=>item.msg||JSON.stringify(item)).join('；'):JSON.stringify(detail??data));}
  return data;
}
function report(error){$('#rsi-error').textContent=error.message;$('#rsi-error').hidden=false;context?.toast?.(error.message);}
async function action(work){if(busy)return;busy=true;$('#rsi-error').hidden=true;renderLocks();try{await work();}catch(error){report(error);}finally{busy=false;renderLocks();}}
function selectOptions(select,items,emptyLabel){
  const previous=select.value;
  select.replaceChildren();
  if(emptyLabel!==undefined){const option=document.createElement('option');option.value='';option.textContent=emptyLabel;select.append(option);}
  for(const item of items){const option=document.createElement('option');option.value=item.id;option.textContent=item.name;select.append(option);}
  if([...select.options].some(option=>option.value===previous))select.value=previous;
}
function parseObject(selector,label){
  let value;try{value=JSON.parse($(selector).value||'{}');}catch(error){throw new Error(`${label} JSON 格式错误：${error.message}`);}
  if(!value||Array.isArray(value)||typeof value!=='object')throw new Error(`${label}必须是 JSON 对象。`);
  rejectSecrets(value,label);return value;
}
function rejectSecrets(value,label){for(const [key,entry] of Object.entries(value)){if(/^(api_key|apikey|token|password|authorization|decision_api_keys)$/i.test(key))throw new Error(`${label}不能包含密钥字段 ${key}；请在专用密钥输入框填写。`);if(entry&&typeof entry==='object')rejectSecrets(entry,label);}}
function parseDecisions(){
  let list;try{list=JSON.parse($('#rsi-decisions-json').value);}catch{throw new Error('多决策配置 JSON 格式错误，请填写 2–8 个决策器对象组成的数组。');}
  if(!Array.isArray(list)||list.length<2||list.length>8||list.some(item=>!item||Array.isArray(item)||typeof item!=='object'))throw new Error('多决策组合必须包含 2–8 个决策器对象。');
  rejectSecrets(list,'多决策配置');return list.map(item=>{const weight=item.weight??1;if(typeof weight!=='number'||!Number.isFinite(weight)||weight<=0)throw new Error('每个决策器的 weight 必须是大于 0 的数字。');return {...item,weight};});
}
function parseKeyMap(selector){
  const raw=$(selector).value.trim();if(!raw)return {};
  let map;try{map=JSON.parse(raw);}catch{throw new Error('决策器密钥 JSON 格式错误；请填写索引到字符串的对象，例如 {"0":"Key A","1":"Key B"}。');}
  if(!map||Array.isArray(map)||typeof map!=='object'||Object.entries(map).some(([index,key])=>!/^[0-7]$/.test(index)||typeof key!=='string'||!key.trim()))throw new Error('决策器密钥须为索引 0–7 到非空字符串的对象。');
  return map;
}
function seedList(selector){const raw=$(selector).value.trim();if(!/^\d+(\s*[,，]\s*\d+)*$/.test(raw))throw new Error('种子须为用逗号分隔的非负整数。');const values=raw.split(/[,，]/).map(Number);if(values.some(value=>!Number.isSafeInteger(value)||value>4294967295))throw new Error('种子必须在 0–4294967295 之间。');if(new Set(values).size!==values.length)throw new Error('同一集合不能重复填写种子。');return values;}
function fillConfigs(source){
  $('#rsi-agent-json').value=pretty(source.agent||defaults?.agent);
  $('#rsi-policy-json').value=pretty(source.policy||defaults?.policy?.[$('#rsi-policy-kind').value]);
  $('#rsi-skills-json').value=pretty(source.skills);
  $('#rsi-controller-json').value=pretty(source.controller);
  const list=source.decisions?.length?source.decisions:source.decision?[source.decision]:[],decision=list[0]||defaults?.decision||{};
  $('#rsi-decision-strategy').value=source.decision_strategy||'single';
  $('#rsi-decisions-json').value=pretty(list.length>1?list:[{...decision,weight:decision.weight??1},{kind:'custom',url:'',model:'',vision:false,question_name:'next_action',timeout_s:30,weight:1}]);
  $('#rsi-decision-kind').value=decision.kind||'typesafe';$('#rsi-decision-url').value=decision.url||'';$('#rsi-decision-model').value=decision.model||'';
  $('#rsi-decision-vision').checked=!!decision.vision;$('#rsi-question-name').value=decision.question_name||'next_action';$('#rsi-decision-timeout').value=decision.timeout_s||30;
  $('#rsi-decision-key').value='';$('#rsi-decision-keys').value='';$('#rsi-hybrid-policy').checked=!!source.policy;
  updateMode();
}
function updateAgentFields(agent){
  $('#rsi-agent-source').value=agent.provider||'codex';$('#rsi-agent-model').value=agent.model||'';$('#rsi-agent-url').value=agent.base_url||'';
  $('#rsi-agent-url-field').hidden=agent.provider==='codex';
  $('#rsi-agent-provider-help').textContent=agent.provider==='custom'?'自定义 Agent 服务使用 embodied-agent-v1：填写完整请求地址与实际模型／Agent 标识，响应须返回符合动作或子目标 Schema 的 output。任意 Agent 实现均须遵循相同观察与执行权限。':'可填写此服务实际可用的任意模型 ID。保存会冻结模型和调用配置；当前工作台配置只是默认起点。';
}
function updateAgentJson(providerChanged=false){
  try{const agent=parseObject('#rsi-agent-json','Agent 配置');agent.provider=$('#rsi-agent-source').value;agent.model=$('#rsi-agent-model').value.trim();agent.base_url=providerChanged?({codex:'https://api.openai.com/v1',openai:'https://api.openai.com/v1',compatible:'',anthropic:'https://api.anthropic.com',custom:''}[agent.provider]):$('#rsi-agent-url').value.trim();$('#rsi-agent-json').value=pretty(agent);updateMode();}catch(error){report(error);}
}
function updateDecisionStrategy(){
  const strategy=$('#rsi-decision-strategy').value;$('#rsi-single-decision-config').hidden=strategy!=='single';$('#rsi-multi-decision-config').hidden=strategy==='single';
  $('#rsi-decision-strategy-help').textContent={single:'一个决策器选择候选动作；只有该服务可使用单决策器密钥。',cascade:'第一个决策器提出首选，后续决策器依次复核；未达成一致则请求主 Agent 重规划。',weighted_vote:'各决策器分别选择候选，按 weight 计算票份额；须获得严格多数，没有多数则重规划。票份额不等同于校准概率。'}[strategy];
}
function updateMode(){
  const mode=$('#rsi-mode').value,policy=mode==='policy'||(mode==='hybrid'&&$('#rsi-hybrid-policy').checked);
  $('#rsi-decision-fields').hidden=mode!=='hybrid';$('#rsi-policy-kind-field').hidden=!policy;
  $('#rsi-agent-json-field').hidden=mode==='policy';$('#rsi-agent-config-fields').hidden=mode==='policy';$('#rsi-policy-json-field').hidden=!policy;
  let agent={},p={};try{agent=JSON.parse($('#rsi-agent-json').value||'{}');p=JSON.parse($('#rsi-policy-json').value||'{}');}catch{}
  updateAgentFields(agent);updateDecisionStrategy();
  $('#rsi-config-summary').textContent=[mode!=='policy'?`主 Agent：${agent.model||'当前模型'} · ${agent.provider||'当前接口'}`:'',policy?`策略：${p.model||'当前策略'} · ${p.kind||$('#rsi-policy-kind').value}`:'',mode==='hybrid'?`决策组合：${{single:'单决策器',cascade:'级联复核',weighted_vote:'加权多数票'}[$('#rsi-decision-strategy').value]}，未配置服务时会明确报错。`:''].filter(Boolean).join(' / ');
}
function fillTasks(){
  if(!catalog)return;const scene=catalog.scenes.find(item=>item.id===$('#rsi-protocol-scene').value);if(!scene)return;
  selectOptions($('#rsi-protocol-task'),scene.tasks.map(id=>({id,name:catalog.tasks[id].name})));updateCriterion();
}
function updateCriterion(){if(catalog)$('#rsi-criterion').textContent=catalog.tasks[$('#rsi-protocol-task').value]?.criterion||'';}
function renderMethods(){
  const options=methods.map(method=>({id:methodId(method),name:`${method.recipe.name} · G${method.recipe.evolution_generation??0} · ${methodId(method).slice(-8)}`}));
  for(const selector of ['#rsi-run-method','#rsi-promotion-candidate','#rsi-promotion-parent'])selectOptions($(selector),options);
  selectOptions($('#rsi-parent'),options,'新建根代');
  $('#rsi-version-count').textContent=`${methods.length} 个版本`;
  $('#rsi-method-list').replaceChildren();
  if(!methods.length)$('#rsi-method-list').innerHTML='<p class="empty-state">保存根代方法后，从父代创建下一个版本。</p>';
  for(const method of [...methods].reverse()){
    const recipe=method.recipe,id=methodId(method),entry=document.createElement('article');entry.className='rsi-version';
    entry.innerHTML=`<div><strong>${escape(recipe.name)}</strong><span class="result-pill">G${recipe.evolution_generation??0} · ${escape(modes[recipe.mode]||recipe.mode)}</span></div><code>${escape(id)}</code><p>${escape(recipe.change_summary||'未填写变更说明')}</p><small>父代：${escape(recipe.parent_id?methodName(recipe.parent_id):'根代')} · ${escape(method.created_at||'')}</small><div class="rsi-version-actions"><button type="button" class="text-button" data-inherit="${escape(id)}">从此版本创建子代 ↗</button><button type="button" class="text-button" data-inspect="${escape(id)}">查看配方</button></div>`;
    entry.querySelector('[data-inherit]').addEventListener('click',()=>inheritMethod(id));
    entry.querySelector('[data-inspect]').addEventListener('click',()=>{let detail=entry.querySelector('pre');if(detail){detail.remove();return;}detail=document.createElement('pre');detail.textContent=pretty(method);entry.append(detail);});$('#rsi-method-list').append(entry);
  }
  const selected=new Set([...document.querySelectorAll('#rsi-compare-methods input:checked')].map(input=>input.value));
  $('#rsi-compare-methods').replaceChildren();
  for(const method of methods){const label=document.createElement('label');label.innerHTML=`<input type="checkbox" value="${escape(methodId(method))}"><span><strong>${escape(method.recipe.name)}</strong><small>G${method.recipe.evolution_generation??0} · ${escape(modes[method.recipe.mode])}</small></span>`;label.querySelector('input').checked=selected.has(methodId(method));$('#rsi-compare-methods').append(label);}
  if(!methods.length)$('#rsi-compare-methods').innerHTML='<p class="rsi-help">需要先保存方法版本。</p>';
  updatePromotionParent();renderRegistry();renderLocks();
}
function inheritMethod(id){const recipe=methods.find(method=>methodId(method)===id)?.recipe;if(!recipe)return;$('#rsi-parent').value=id;$('#rsi-mode').value=recipe.mode;$('#rsi-name').value=`${recipe.name} · 子代`;$('#rsi-change').value='';if(recipe.policy?.kind)$('#rsi-policy-kind').value=recipe.policy.kind;fillConfigs(recipe);$('#rsi-method-message').textContent=`已载入父代 ${id} 的完整配方；修改并保存会生成新版本。`;$('#rsi-name').focus();}
function renderProtocols(){
  selectOptions($('#rsi-protocol-select'),protocols.map(protocol=>({id:protocolId(protocol),name:`${protocol.spec.name} · ${protocolId(protocol).slice(-8)}`})));describeProtocol();
}
function describeProtocol(){
  const protocol=protocols.find(item=>protocolId(item)===$('#rsi-protocol-select').value);if(!protocol){$('#rsi-protocol-description').textContent='先保存一份协议。';return;}
  const spec=protocol.spec;$('#rsi-protocol-description').textContent=`${spec.robot} / ${spec.scene_id} / ${spec.task_id}；开发 [${spec.development_seeds.join(', ')}]，验证 [${spec.validation_seeds.join(', ')}]，测试 [${spec.test_seeds.join(', ')}]。每次仿真 ${spec.max_sim_time_s}s / 墙钟 ${spec.max_wall_time_s}s / ${spec.max_steps} 步；晋升要求成功率差 > ${spec.min_success_delta}、错误增加 ≤ ${spec.max_error_increase}，至少 ${spec.min_pairs??3} 对。`;
}
function renderExperiments(){
  const validation=experiments.filter(experiment=>(experiment.split||experiment.request?.split)==='validation'&&!experiment.active&&['complete','completed'].includes(experiment.status));
  selectOptions($('#rsi-promotion-experiment'),validation.map(experiment=>({id:experimentId(experiment),name:`验证 · ${experimentId(experiment)} `})),validation.length?undefined:'尚无已完成的验证实验');
  selectOptions($('#rsi-experiment-history'),experiments.map(experiment=>({id:experimentId(experiment),name:`${{development:'开发集',validation:'验证集',test:'封存测试集'}[experiment.split||experiment.request?.split]||'实验'} · ${statuses[experiment.status]||experiment.status} · ${experimentId(experiment)}`})),'查看当前实验');
  if($('#rsi-experiment-history').value)renderExperiment(experiments.find(experiment=>experimentId(experiment)===$('#rsi-experiment-history').value)||{});
}
async function refresh(){
  const results=await Promise.all([request('research/defaults'),request('research/methods'),request('research/protocols'),request('research/experiments'),request('scenes')]);
  [defaults,methods,protocols,experiments,catalog]=results;
  if(!$('#rsi-agent-json').value)fillConfigs({});
  selectOptions($('#rsi-protocol-scene'),catalog.scenes);fillTasks();renderMethods();renderProtocols();renderExperiments();
}
function updatePromotionParent(){const recipe=methods.find(method=>methodId(method)===$('#rsi-promotion-candidate').value)?.recipe;if(recipe?.parent_id)$('#rsi-promotion-parent').value=recipe.parent_id;}
function renderRegistry(){
  const id=registry.active_method_id;$('#rsi-champion').textContent=id?methodName(id):'尚未晋升方法';
  const promotion=[...(registry.history||[])].reverse().find(item=>item.action==='promote'&&item.method_id===id),delta=promotion?.paired_success_delta,ci=promotion?.paired_bootstrap_ci95;
  $('#rsi-champion-note').textContent=id?`当前方法 ${id}${delta==null?'':` · 验证成功率差 ${Number(delta).toFixed(3)}${ci?`，配对 bootstrap 95% 区间 [${ci.map(value=>Number(value).toFixed(3)).join(', ')}]`:''}（探索性验收）`}`:'通过验证集门槛后，候选才能成为当前方法。';
}
function renderLocks(){
  if(!context)return;
  const state=currentState,locked=!!(state.agent?.active||state.policy?.active||state.comparison?.active||state.research?.active||state.experiment?.active);
  $('#rsi-run-start').disabled=busy||locked||!methods.length;$('#rsi-run-once').disabled=busy||locked||!methods.length;$('#rsi-run-stop').disabled=busy||!state.research?.active;
  $('#rsi-experiment-start').disabled=busy||locked||methods.length<2||!protocols.length;$('#rsi-experiment-stop').disabled=busy||!state.experiment?.active;
  $('#rsi-save').disabled=busy||locked;$('#rsi-save-protocol').disabled=busy||locked;$('#rsi-save-runtime').disabled=busy||locked||!methods.length;$('#rsi-refresh').disabled=busy;$('#rsi-promote').disabled=busy||locked||!$('#rsi-promotion-experiment').value;$('#rsi-rollback').disabled=busy||locked||!registry.active_method_id;
  $('#rsi-restore-frame').disabled=busy||locked||!episode?.available;for(const selector of ['#rsi-run-method','#rsi-run-task','#rsi-run-sim','#rsi-run-wall','#rsi-run-steps','#rsi-run-record-video'])$(selector).disabled=busy||!!state.research?.active;
  $('#rsi-experiment-record-video').disabled=busy||locked;
}
async function startMethod(oneStep){const payload={method_id:$('#rsi-run-method').value,task:$('#rsi-run-task').value.trim(),max_sim_time_s:Number($('#rsi-run-sim').value),max_wall_time_s:Number($('#rsi-run-wall').value),max_steps:Number($('#rsi-run-steps').value),one_step:oneStep,record_video:$('#rsi-run-record-video').checked};if(!payload.method_id||!payload.task)throw new Error('请选择方法版本并填写任务。');const result=await request('research/start',payload);$('#rsi-run-message').textContent=result.message||'任务已提交。';}

export async function initRsi(callbacks){
  context=callbacks;
  $('#rsi-refresh').addEventListener('click',()=>action(refresh));
  $('#rsi-mode').addEventListener('change',updateMode);$('#rsi-hybrid-policy').addEventListener('change',updateMode);
  $('#rsi-agent-source').addEventListener('change',()=>updateAgentJson(true));$('#rsi-agent-model').addEventListener('change',()=>updateAgentJson());$('#rsi-agent-url').addEventListener('change',()=>updateAgentJson());
  $('#rsi-decision-strategy').addEventListener('change',()=>{$('#rsi-decision-key').value='';$('#rsi-decision-keys').value='';updateMode();});
  $('#rsi-parent').addEventListener('change',()=>{if($('#rsi-parent').value)inheritMethod($('#rsi-parent').value);});
  $('#rsi-agent-json').addEventListener('input',updateMode);$('#rsi-policy-json').addEventListener('input',updateMode);
  $('#rsi-policy-kind').addEventListener('change',()=>{if(defaults?.policy){$('#rsi-policy-json').value=pretty(defaults.policy[$('#rsi-policy-kind').value]);updateMode();}});
  $('#rsi-use-current').addEventListener('click',()=>action(async()=>{defaults=await request('research/defaults');const current={agent:defaults.agent,policy:defaults.policy[$('#rsi-policy-kind').value]};$('#rsi-agent-json').value=pretty(current.agent);$('#rsi-policy-json').value=pretty(current.policy);updateMode();$('#rsi-method-message').textContent='已载入当前工作台模型配置，保存时冻结为此版本。';}));
  $('#rsi-method-form').addEventListener('submit',event=>{event.preventDefault();action(async()=>{
    const mode=$('#rsi-mode').value,needsPolicy=mode==='policy'||(mode==='hybrid'&&$('#rsi-hybrid-policy').checked),strategy=mode==='hybrid'?$('#rsi-decision-strategy').value:'single';
    const agent=mode!=='policy'?{...parseObject('#rsi-agent-json','Agent 配置'),provider:$('#rsi-agent-source').value,model:$('#rsi-agent-model').value.trim(),base_url:$('#rsi-agent-url').value.trim()}:null;
    const recipe={schema_version:'embodied-method-v1',name:$('#rsi-name').value.trim(),mode,parent_id:$('#rsi-parent').value||null,change_summary:$('#rsi-change').value.trim(),agent,policy:needsPolicy?{...parseObject('#rsi-policy-json','策略配置'),kind:$('#rsi-policy-kind').value}:null,decision_strategy:strategy,decision:mode==='hybrid'&&strategy==='single'?{kind:$('#rsi-decision-kind').value,url:$('#rsi-decision-url').value.trim(),model:$('#rsi-decision-model').value.trim(),vision:$('#rsi-decision-vision').checked,question_name:$('#rsi-question-name').value.trim(),timeout_s:Number($('#rsi-decision-timeout').value),weight:1}:null,decisions:mode==='hybrid'&&strategy!=='single'?parseDecisions():[],skills:parseObject('#rsi-skills-json','技能配置'),controller:parseObject('#rsi-controller-json','控制配置')};
    const credentials={};if(mode==='hybrid'){if(strategy==='single'&&$('#rsi-decision-key').value)credentials.decision_api_key=$('#rsi-decision-key').value;if(strategy!=='single'){const keys=parseKeyMap('#rsi-decision-keys');if(Object.keys(keys).some(index=>Number(index)>=recipe.decisions.length))throw new Error('密钥索引超出本方法的决策器列表。');if(Object.keys(keys).length)credentials.decision_api_keys=keys;}}
    const saved=await request('research/methods',{recipe,credentials});$('#rsi-decision-key').value='';$('#rsi-decision-keys').value='';
    methods=await request('research/methods');renderMethods();$('#rsi-run-method').value=methodId(saved);$('#rsi-method-message').textContent=`已保存 ${methodId(saved)}。后续修改请创建子代版本。`;
  });});
  $('#rsi-run-form').addEventListener('submit',event=>{event.preventDefault();action(()=>startMethod(false));});$('#rsi-run-once').addEventListener('click',()=>{if($('#rsi-run-form').reportValidity())action(()=>startMethod(true));});
  $('#rsi-run-stop').addEventListener('click',()=>action(async()=>{await request('research/stop',{});}));
  $('#rsi-run-method').addEventListener('change',()=>{for(const type of ['agent','policy','decision'])$(`#rsi-runtime-${type}-key`).value='';$('#rsi-runtime-decision-keys').value='';$('#rsi-runtime-message').textContent='';});
  $('#rsi-save-runtime').addEventListener('click',()=>action(async()=>{
    const id=$('#rsi-run-method').value;if(!id)throw new Error('请先选择一个方法版本。');const credentials={},recipe=methods.find(method=>methodId(method)===id)?.recipe;
    for(const type of ['agent','policy','decision']){const value=$(`#rsi-runtime-${type}-key`).value;if(value)credentials[`${type}_api_key`]=value;}
    const keys=parseKeyMap('#rsi-runtime-decision-keys');if(Object.keys(keys).length)credentials.decision_api_keys=keys;
    const count=recipe?.decisions?.length||(recipe?.decision?1:0);if(count>1&&credentials.decision_api_key)throw new Error('多决策方法需要按索引分别填写密钥，不能使用单决策器密钥。');if(Object.keys(keys).some(index=>Number(index)>=count))throw new Error('密钥索引超出所选方法的决策器列表。');
    if(!Object.keys(credentials).length)throw new Error('请填写需要更新的运行时密钥。');await request(`research/methods/${encodeURIComponent(id)}/credentials`,credentials);for(const type of ['agent','policy','decision'])$(`#rsi-runtime-${type}-key`).value='';$('#rsi-runtime-decision-keys').value='';$('#rsi-runtime-message').textContent='凭据已应用于所选版本，各决策器密钥独立，仅在本次服务会话生效。';
  }));
  $('#rsi-protocol-scene').addEventListener('change',fillTasks);$('#rsi-protocol-task').addEventListener('change',updateCriterion);$('#rsi-protocol-select').addEventListener('change',describeProtocol);
  $('#rsi-protocol-form').addEventListener('submit',event=>{event.preventDefault();action(async()=>{
    const payload={name:$('#rsi-protocol-name').value.trim(),robot:$('#rsi-protocol-robot').value,scene_id:$('#rsi-protocol-scene').value,task_id:$('#rsi-protocol-task').value,development_seeds:seedList('#rsi-seeds-development'),validation_seeds:seedList('#rsi-seeds-validation'),test_seeds:seedList('#rsi-seeds-test'),max_sim_time_s:Number($('#rsi-protocol-sim').value),max_wall_time_s:Number($('#rsi-protocol-wall').value),max_steps:Number($('#rsi-protocol-steps').value),schedule_seed:Number($('#rsi-schedule-seed').value),min_success_delta:Number($('#rsi-min-delta').value),max_error_increase:Number($('#rsi-error-increase').value),min_pairs:Number($('#rsi-min-pairs').value)};
    const all=[...payload.development_seeds,...payload.validation_seeds,...payload.test_seeds];if(new Set(all).size!==all.length)throw new Error('开发、验证与测试的种子集合必须互不重叠。');
    const saved=await request('research/protocols',payload);protocols=await request('research/protocols');renderProtocols();$('#rsi-protocol-select').value=protocolId(saved);describeProtocol();$('#rsi-experiment-message').textContent=`协议 ${protocolId(saved)} 已冻结。`;
  });});
  $('#rsi-experiment-form').addEventListener('submit',event=>{event.preventDefault();action(async()=>{const method_ids=[...document.querySelectorAll('#rsi-compare-methods input:checked')].map(input=>input.value);if(method_ids.length<2)throw new Error('请选择至少两个方法版本进行配对比较。');const result=await request('research/experiments',{protocol_id:$('#rsi-protocol-select').value,method_ids,split:$('#rsi-split').value,record_video:$('#rsi-experiment-record-video').checked});$('#rsi-experiment-history').value='';$('#rsi-experiment-message').textContent=result.message||'配对实验已提交。';});});
  $('#rsi-experiment-history').addEventListener('change',()=>renderExperiment($('#rsi-experiment-history').value?experiments.find(experiment=>experimentId(experiment)===$('#rsi-experiment-history').value)||{}:currentState.experiment||{}));
  $('#rsi-experiment-stop').addEventListener('click',()=>action(async()=>{await request('research/experiments/stop',{});}));
  $('#rsi-promotion-candidate').addEventListener('change',updatePromotionParent);
  $('#rsi-promotion-form').addEventListener('submit',event=>{event.preventDefault();action(async()=>{const result=await request('research/promote',{experiment_id:$('#rsi-promotion-experiment').value,candidate_id:$('#rsi-promotion-candidate').value,parent_id:$('#rsi-promotion-parent').value});registry=result.registry||result;renderRegistry();$('#rsi-promotion-message').textContent=result.message||`已晋升 ${methodName($('#rsi-promotion-candidate').value)}；完整判定已存入实验档案。`;});});
  $('#rsi-rollback').addEventListener('click',()=>action(async()=>{const result=await request('research/rollback',{});registry=result.registry||result;renderRegistry();context.toast('已回滚上次晋升，版本与实验记录保留。');}));
  $('#rsi-replay-form').addEventListener('submit',event=>{event.preventDefault();action(()=>loadEpisode($('#rsi-replay-run').value.trim()));});$('#rsi-observation-index').addEventListener('input',renderObservation);
  $('#rsi-restore-frame').addEventListener('click',()=>action(async()=>{const result=await request('research/replay',{run_id:replayRun,frame_index:Number($('#rsi-state-index').value)});$('#rsi-replay-message').textContent=result.message||'已载入记录的物理状态；请在仿真工作台查看原生画面。';document.querySelector('[data-page="workspace"]').click();}));
  $('#rsi-experiment-rows').addEventListener('click',event=>{const button=event.target.closest('[data-replay-run]');if(button){$('#rsi-replay-run').value=button.dataset.replayRun;action(()=>loadEpisode(button.dataset.replayRun));$('#rsi-replay-form').scrollIntoView({behavior:'smooth',block:'center'});}});
  try{await refresh();}catch(error){report(error);}
}

export function renderRsi(state){
  if(!context)return;currentState=state;registry=state.registry||registry;
  const registryKey=JSON.stringify(registry);if(registryKey!==lastRegistry){lastRegistry=registryKey;renderRegistry();}
  const run=state.research||{};$('#rsi-run-status').textContent=statuses[run.status]||run.status||'等待任务';$('#rsi-run-message').textContent=run.message||'使用当前场景。运行记录包含模型公开输出、同步观察与实际物理状态。';$('#rsi-run-bundle').hidden=!run.run_id;if(run.run_id)$('#rsi-run-bundle').href=`/api/runs/${encodeURIComponent(run.run_id)}/bundle`;
  renderRunVideo($('#rsi-run-video'),run.run_id,run.video);
  renderRunTimer($('#rsi-run-time'),run);renderRunTimer($('#rsi-experiment-time'),state.experiment);
  const runKey=JSON.stringify([run.run_id,run.status,run.step,run.max_steps,run.subgoal,run.selected_candidate,run.timings,run.history,run.metrics]);if(runKey!==lastRun){lastRun=runKey;renderRunFeedback(run);}
  const experiment=state.experiment||{},key=JSON.stringify([experiment.run_id,experiment.status,experiment.message,experiment.rows,experiment.aggregates]);if(key!==lastExperiment){lastExperiment=key;if(!$('#rsi-experiment-history').value)renderExperiment(experiment);}
  if(experiment.run_id&&!experiment.active&&['complete','completed','stopped','error'].includes(experiment.status)&&experiment.run_id!==lastFinished){lastFinished=experiment.run_id;request('research/experiments').then(value=>{experiments=value;renderExperiments();renderLocks();}).catch(report);}
  renderLocks();
}
function renderRunFeedback(run){
  $('#rsi-run-feedback').hidden=!run.run_id;
  const timings=run.timings||{},metrics=run.metrics||{},labels={planning:'主 Agent 规划',agent_inference:'Agent 推理',policy_inference:'策略推理',decision:'决策服务',execution:'动作执行',policy_loading:'策略加载'};
  $('#rsi-run-metrics').innerHTML=`<div><small>决策步骤</small><strong>${run.step||0} / ${run.max_steps||'—'}</strong></div>`+Object.entries(timings).map(([phase,record])=>`<div><small>${escape(labels[phase]||phase)} · ${record.calls??0} 次</small><strong>${seconds(record.total_s)}</strong></div>`).join('')+(run.metrics?`<div><small>物理评价</small><strong>${metrics.success?'成功':'未达成'}</strong></div><div><small>墙钟 / 仿真</small><strong>${seconds(metrics.wall_time_s)} / ${seconds(metrics.sim_time_s)}</strong></div>`:'');
  $('#rsi-run-subgoal').textContent=[run.subgoal?`当前子目标：${run.subgoal}`:'',run.selected_candidate?`所选候选：${run.selected_candidate}`:''].filter(Boolean).join(' / ');
  $('#rsi-run-history').innerHTML=(run.history||[]).slice(-4).reverse().map(record=>{const result=record.result||{};return `<div class="round-record"><strong>步骤 ${record.step} · ${escape(record.candidate_id)} · ${escape(statuses[result.status]||result.status)}</strong><p>${escape(record.action?.description||record.action?.pose?.explanation||'')}${result.message?`<br>${escape(result.message)}`:''}${result.position_error_m==null?'':`<br>末端位置误差 ${(Number(result.position_error_m)*1000).toFixed(2)} mm · 姿态误差 ${Number(result.orientation_error_rad||0).toFixed(4)} rad`}</p></div>`;}).join('')||'<p class="rsi-help">尚无动作执行反馈。</p>';
}
function renderExperiment(experiment){
  $('#rsi-experiment-message').textContent=experiment.message||'尚未开始版本评测。';const rows=experiment.rows||[];
  $('#rsi-experiment-rows').innerHTML=rows.length?rows.map(row=>{const metrics=row.metrics||{};return `<tr><td><strong>${escape(row.method_name||methodName(row.method_id))}</strong><small>种子 ${row.seed}</small></td><td><code>${escape(row.initial_state_id||'待分配')}</code><small title="${escape(row.message)}">${escape(statuses[row.status]||row.status)}</small></td><td>${row.metrics?(metrics.success?'<span class="result-pill success">成功</span>':'<span class="result-pill">未达成</span>'):'—'}</td><td>${seconds(metrics.sim_time_s)}<small>墙钟 ${seconds(metrics.wall_time_s)}</small></td><td>${metrics.inference_count??'—'}</td><td>${row.run_id?`<button class="text-button" type="button" data-replay-run="${escape(row.run_id)}">观察与视频回放</button><a class="text-button" href="/api/runs/${encodeURIComponent(row.run_id)}/bundle">完整 ZIP ↗</a>${runVideoMarkup(row.run_id,row.video,true)}`:'—'}</td></tr>`;}).join(''):'<tr><td colspan="6" class="empty-state">实际运行结果会保存在这里。</td></tr>';
  $('#rsi-experiment-summary').innerHTML=(experiment.aggregates||[]).map(summary=>`<div><strong>${escape(summary.name||methodName(summary.method_id))}</strong><span>${summary.successes??0} / ${summary.assigned??0}</span><small>成功 / 全部分配 · 完成 ${summary.completed??0}<br>错误 ${summary.errors??0} · 停止 ${summary.stopped??0} · 墙钟 ${seconds(summary.wall_time_s)}</small></div>`).join('');
}
async function loadEpisode(runId){
  if(!runId)throw new Error('请填写运行编号。');const loaded=await request(`runs/${encodeURIComponent(runId)}/episode`);if(!loaded.available&&!loaded.video)throw new Error(loaded.message||'此运行没有完整轨迹记录；历史日志不会自动生成观察图像。');
  episode=loaded;replayRun=runId;$('#rsi-replay-run').value=runId;$('#rsi-replay-content').hidden=false;$('#rsi-observation-index').max=Math.max(0,(episode.observations||[]).length-1);$('#rsi-observation-index').value='0';
  const frameCount=Array.isArray(episode.trajectory_frames)?episode.trajectory_frames.length:Number(episode.trajectory_frames)||0;$('#rsi-state-index').max=Math.max(0,frameCount-1);$('#rsi-state-index').value='0';$('#rsi-replay-bundle').href=`/api/runs/${encodeURIComponent(runId)}/bundle`;
  $('#rsi-replay-state-controls').hidden=!episode.available;renderRunVideo($('#rsi-replay-video'),runId,episode.video);
  $('#rsi-replay-message').textContent=`${runId} · ${episode.observations?.length||0} 个观察步骤 / ${frameCount} 个物理状态帧。${episode.available?'载入状态会暂停当前仿真。':episode.message||'本次没有可载入的完整物理状态。'}`;
  renderObservation();
}
function renderObservation(){
  if(!episode)return;const observation=episode.observations?.[Number($('#rsi-observation-index').value)];$('#rsi-replay-cameras').replaceChildren();if(!observation){$('#rsi-observation-label').textContent='没有图像观察记录';return;}
  $('#rsi-observation-label').textContent=`步骤 ${observation.step??$('#rsi-observation-index').value} · t=${Number(observation.time??observation.sim_time_s??0).toFixed(3)} s`;
  for(const camera of observation.cameras||[]){if(!camera.image_ref)continue;const figure=document.createElement('figure'),img=document.createElement('img'),caption=document.createElement('figcaption');img.src=`/api/runs/${encodeURIComponent(replayRun)}/artifact/${camera.image_ref.split('/').map(encodeURIComponent).join('/')}`;img.alt=`记录相机 ${camera.name||camera.id||''}`;caption.textContent=camera.name||camera.id||'相机';figure.append(img,caption);$('#rsi-replay-cameras').append(figure);}
}
