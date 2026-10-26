import './repair.css';
const $=selector=>document.querySelector(selector);
let context,config=null,current={},lastRendered='',resumePending=false;
const recoveryStatuses={diagnosing:'故障暂停 · 正在诊断',waiting:'故障暂停 · 等待处理',retrying:'正在重试当前轮',recovered:'当前轮已恢复',stopped:'恢复已停止',exhausted:'恢复预算已耗尽',failed:'恢复失败',skipped:'未启动故障修复'};
async function request(path,payload){
  const response=await fetch(`/api/repair/${path}`,payload===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  const data=await response.json();if(!response.ok)throw new Error(typeof data.detail==='string'?data.detail:JSON.stringify(data.detail||'修复请求失败'));return data;
}
function payload(){const codex=$('#repair-provider').value==='codex';return {provider:$('#repair-provider').value,model:$('#repair-model').value.trim(),base_url:codex?'https://api.openai.com/v1':$('#repair-base').value.trim(),api_key:codex?'':$('#repair-key').value,reasoning:$('#repair-effort').value.trim()||null,reasoning_native:true,timeout_s:Number($('#repair-timeout').value),max_output_tokens:codex?(config?.max_output_tokens??16384):Number($('#repair-tokens').value),extra_instructions:$('#repair-instructions').value,auto_analyze:$('#repair-auto').checked,auto_recover:$('#repair-auto-recover').checked,max_recovery_attempts:Number($('#repair-recovery-budget').value)};}
function providerFields(){const codex=$('#repair-provider').value==='codex';$('#repair-api-fields').hidden=codex;$('#repair-budget-field').hidden=codex;$('#repair-api-fields').querySelectorAll('input').forEach(input=>input.disabled=codex);$('#repair-tokens').disabled=codex;}
function automaticFields(){
  const enabled=$('#repair-auto').checked;
  $('#repair-auto-recover').disabled=!enabled;
  $('#repair-auto-note').textContent=enabled?($('#repair-auto-recover').checked?'调用失败后自动诊断，并在恢复预算内重试同一轮。':'调用失败后自动诊断，保留当前轮并等待你确认重试。'):'自动分析已关闭：调用失败时不会启动修复 Agent，自动重试选项也不会生效。';
}
function renderRecovery(){
  const automatic=current.automatic||config;
  $('#repair-automatic-status').textContent=!automatic?'正在读取自动修复配置…':!automatic.auto_analyze?'自动分析：关闭 · 故障不会自动修复':automatic.auto_recover?`自动诊断并重试 · 每次实验最多恢复 ${automatic.max_recovery_attempts} 次`:'自动诊断：开启 · 重试需手动确认';
  const recovery=current.recovery,visible=!!recovery?.failure_id;
  $('#repair-recovery').hidden=!visible;$('#repair-resume').disabled=resumePending||!recovery?.can_resume;$('#repair-resume').textContent=resumePending?'正在提交恢复请求…':'重试当前轮并继续（消耗模型额度）';
  $('#repair-card-status').textContent=visible?(recoveryStatuses[recovery.status]||recovery.status):current.active?'正在处理':current.restart_required?'待重启':'独立编码模型';
  if(!visible){$('#repair-card-message').textContent='读取调用报错与接入代码，生成可验证的修复建议。';return;}
  const owner={agent:'视觉 Agent',policy:'策略模型',research:'RSI 方法',composition:'RSI 方法'}[recovery.owner]||recovery.owner||'主任务';
  const phase={agent_inference:'Agent 推理',policy_inference:'策略推理',planning:'主 Agent 规划',decision:'决策服务',execution:'动作执行',policy_loading:'策略加载'}[recovery.phase]||recovery.phase||'阶段未记录';
  const source=`${owner} · 实验 ${recovery.source_run_id||'—'} · 第 ${recovery.round??'—'} 轮 · ${phase}`,budget=`本次实验恢复次数 ${recovery.attempts_used??0} / ${recovery.max_attempts??'—'}`;
  $('#repair-card-message').textContent=`${source}。${budget}。${recovery.message||''}`;
  $('#repair-recovery-source').textContent=source;$('#repair-recovery-status').textContent=recoveryStatuses[recovery.status]||recovery.status||'';$('#repair-recovery-budget-used').textContent=budget;$('#repair-recovery-message').textContent=recovery.message||'主任务保留在当前轮，等待诊断或恢复。';
  $('#repair-recovery-code-note').hidden=recovery.status!=='waiting'||!current.proposal?.edits?.length;
}
function render(data){
  const previousJob=current.job_id,previousFailure=current.recovery?.failure_id;current=data||{};if(previousJob!==current.job_id)$('#repair-allow-execution').checked=false;if(previousFailure!==current.recovery?.failure_id)$('#repair-recovery-error').textContent='';renderRecovery();
  $('#repair-message').textContent=current.message||'选择报错实验，分析并生成候选修复。';
  const elapsed=typeof current.elapsed_s==='number'&&Number.isFinite(current.elapsed_s)?current.elapsed_s:null,timeout=typeof current.timeout_s==='number'&&Number.isFinite(current.timeout_s)?current.timeout_s:null;
  $('#repair-timing').hidden=elapsed==null;$('#repair-timing').textContent=elapsed==null?'':`${current.active?'已用时间':'总耗时'} ${elapsed.toFixed(1)} 秒${current.active&&current.status==='analyzing'&&timeout!=null?` · 模型请求超时设定 ${timeout} 秒`:''}`;
  $('#repair-stop').disabled=!current.active;$('#repair-start').disabled=!!current.active||['diagnosing','waiting','retrying'].includes(current.recovery?.status);$('#repair-save').disabled=!!current.active;
  $('#repair-validate').disabled=current.active||current.status!=='proposed';
  $('#repair-apply').disabled=current.active||current.status!=='validated';$('#repair-rollback').disabled=current.active||!['applied','applying','apply_failed','restoring','restore_failed'].includes(current.status);
  const proposal=current.proposal;$('#repair-proposal').hidden=!proposal;
  $('#repair-diagnosis').textContent=proposal?.diagnosis||'';
  for(const [id,values] of [['repair-evidence',proposal?.evidence],['repair-advice',proposal?.recommendations]]){const list=$(`#${id}`);list.replaceChildren();for(const text of values||[]){const li=document.createElement('li');li.textContent=text;list.append(li);}}
  $('#repair-diff').textContent=current.diff||'本次未提出代码修改。';$('#repair-validation').textContent=current.validation?.output||'尚未执行候选验证。';
  $('#repair-validation-scope').textContent=current.validation?.scope||'候选回归须先在原始实现上复现断言失败，再通过固定接口契约与相同回归。源码副本不是操作系统沙箱；测试作为当前用户的本机 Python 代码执行，具有该账户的文件与网络权限，请先查看全部候选差异和测试代码。';
  $('#repair-run-link').hidden=!current.run_id;$('#repair-run-link').textContent=current.run_id?`${current.run_id===(current.recovery?.source_run_id||current.source_run_id)?'诊断已记录到原实验':'本次诊断日志'} ${current.run_id}`:'';
}
async function sources(){const rows=await request('sources');const select=$('#repair-source'),selected=current.recovery?.source_run_id||select.value;select.replaceChildren(new Option('不选实验，仅分析下方描述',''));for(const row of rows)select.append(new Option(`${row.display_name||row.model||row.run_id} · ${{repairing:'故障诊断中',repair_paused:'故障暂停',retrying:'重试当前轮'}[row.status]||row.status}`,row.run_id));if(rows.some(row=>row.run_id===selected))select.value=selected;else if(rows.length)select.value=rows[0].run_id;}
async function show(){try{config=await request('config');$('#repair-provider').value=config.provider;$('#repair-model').value=config.model;$('#repair-base').value=config.base_url;$('#repair-key').value='';$('#repair-key').placeholder=config.has_api_key?'已配置，留空保留当前修复模型密钥':'仅保存在当前本机服务内存';$('#repair-effort').value=config.reasoning||'';$('#repair-timeout').value=config.timeout_s;$('#repair-tokens').value=config.max_output_tokens;$('#repair-auto').checked=config.auto_analyze;$('#repair-auto-recover').checked=config.auto_recover??true;$('#repair-recovery-budget').value=config.max_recovery_attempts??2;$('#repair-instructions').value=config.extra_instructions;$('#repair-skills').textContent=`始终加载的内置技能：${config.skills.join(' / ')}。`;providerFields();automaticFields();render(await request('state'));await sources();$('#repair-dialog').showModal();}catch(error){context.toast(error.message);}}
async function act(path,body={}){try{render(await request(path,body));}catch(error){$('#repair-message').textContent=error.message;context.toast(error.message);}}
async function resume(){
  const failureId=current.recovery?.failure_id;if(resumePending||!failureId||!current.recovery.can_resume)return;
  resumePending=true;$('#repair-recovery-error').textContent='';renderRecovery();
  try{render(await request('resume',{failure_id:failureId}));}
  catch(error){if(current.recovery?.failure_id===failureId)$('#repair-recovery-error').textContent=error.message;context.toast(error.message);}
  finally{resumePending=false;renderRecovery();}
}
export function initRepair(callbacks){
  context=callbacks;
  $('#log-filter').append(new Option('故障修复 Agent','repair'));
  const card=document.createElement('section');card.className='repair-card';card.innerHTML='<div><strong>模型故障修复 Agent</strong><span id="repair-card-status">独立编码模型</span></div><p id="repair-automatic-status" role="status">正在读取自动修复配置…</p><p id="repair-card-message">读取调用报错与接入代码，生成可验证的修复建议。</p><button id="open-repair" class="secondary" type="button">配置、诊断与恢复 ↗</button>';
  $('.agent.panel').append(card);
  const dialog=document.createElement('dialog');dialog.id='repair-dialog';dialog.innerHTML=`
  <button id="close-repair" class="dialog-close" aria-label="关闭修复 Agent">×</button><div class="eyebrow">MODEL REPAIR / CODING AGENT</div><h2>模型故障修复 Agent</h2>
  <p class="config-help">默认使用 Astra 高推理强度，独立于机械臂主策略。可替换为擅长代码的模型或自定义 Agent。诊断会将所选实验的脱敏错误片段与模型接入源码发送给配置的模型，不发送密钥或相机图像。</p>
  <form id="repair-config-form" class="config-form"><label class="record-video-option"><input id="repair-auto" type="checkbox">模型调用报错后暂停当前轮并自动分析（由后端触发，会使用诊断模型额度）</label><label class="record-video-option"><input id="repair-auto-recover" type="checkbox" checked>允许诊断后自动重试当前轮（会继续消耗主模型额度）</label><label>整次实验最多恢复次数<input id="repair-recovery-budget" type="number" min="1" step="1" value="2" required></label><p id="repair-auto-note" class="config-help" role="status"></p><p class="config-help">自动恢复仅在开启自动分析时生效，恢复预算覆盖整次实验。配置或瞬态错误可在处理后重试同一轮；需要修改源码时保留暂停状态，查看建议后再处理。候选代码仍须审阅、验证、手动应用并重启。</p><details><summary>模型配置与内置修复技能</summary>
  <div class="config-columns"><label>修复模型接入<select id="repair-provider"><option value="codex">Codex · 本机登录</option><option value="openai">OpenAI · Responses</option><option value="compatible">OpenAI 兼容 / DeepSeek</option><option value="anthropic">Claude · Messages</option><option value="custom">自定义 Agent</option></select></label><label>修复模型名称<input id="repair-model" value="gpt-6-astra" required></label></div>
  <div id="repair-api-fields"><label>修复模型服务地址<input id="repair-base" type="url" value="https://api.openai.com/v1" required></label><label>修复模型 API Key<input id="repair-key" type="password" autocomplete="new-password"></label></div>
  <div class="config-columns"><label>原生推理强度<input id="repair-effort" list="repair-efforts" placeholder="留空使用服务默认"><datalist id="repair-efforts"></datalist></label><label>超时 / 秒<input id="repair-timeout" type="number" min="15" value="300" required></label><label id="repair-budget-field">输出预算 / tokens<input id="repair-tokens" type="number" min="512" value="16384" required></label></div><p class="config-help">编码诊断可能耗时较长，可按需延长超时；诊断超时会保留错误和主任务的暂停状态。</p>
  <button id="repair-read-capabilities" class="secondary" type="button">读取修复模型原生推理等级</button><p id="repair-capabilities" class="config-help"></p>
  <p id="repair-skills" class="config-help"></p><p class="config-help">内置技能固定包含根因定位、接口与动作契约、最小修复和回归验证。模型提交限定范围的修复建议；不会自行执行 shell、修改物理安全约束或切换主模型。工作台按恢复配置决定是否重试当前轮。</p>
  <label>补充修复要求<textarea id="repair-instructions" rows="2" maxlength="3000"></textarea></label></details><button id="repair-save" class="primary-button" type="submit">保存修复 Agent 配置</button></form>
  <section id="repair-recovery" class="repair-recovery" hidden><h3>主任务暂停与恢复</h3><p id="repair-recovery-source"></p><p><strong id="repair-recovery-status"></strong> · <span id="repair-recovery-budget-used"></span></p><p id="repair-recovery-message" role="status"></p><p id="repair-recovery-code-note" hidden>此建议包含源码修改，请先停止当前实验，再审阅、验证、应用并重启后端。代码尚未在当前进程生效，不能直接恢复本轮。</p><button id="repair-resume" type="button" class="primary-button" disabled>重试当前轮并继续（消耗模型额度）</button><p class="config-help">恢复会沿用当前实验编号与轮次。需要结束实验时，请使用主任务的停止按钮。</p><p id="repair-recovery-error" role="alert"></p></section>
  <div class="repair-section"><label>报错实验<select id="repair-source"></select></label><button id="repair-refresh-sources" type="button" class="secondary">刷新报错列表</button><label>补充问题描述<textarea id="repair-description" rows="3" maxlength="4000" placeholder="也可以不选实验，直接描述接入问题。"></textarea></label><div class="repair-actions"><button id="repair-start" type="button" class="primary-button">分析并生成候选修复</button><button id="repair-stop" type="button" class="secondary">停止分析 / 验证</button></div><p id="repair-message" role="status" class="config-help"></p><p id="repair-timing" class="repair-timing" role="timer" aria-live="off" hidden></p><p id="repair-run-link" class="config-help" hidden></p></div>
  <section id="repair-proposal" hidden><h3>根因诊断</h3><p id="repair-diagnosis"></p><h4>证据</h4><ul id="repair-evidence"></ul><h4>处理建议</h4><ul id="repair-advice"></ul><details><summary>查看候选代码差异</summary><pre id="repair-diff"></pre></details></section>
  <div class="repair-section"><p id="repair-validation-scope" class="config-help"></p><label class="record-video-option"><input id="repair-allow-execution" type="checkbox">我已查看候选差异与测试源码，允许在本机 Python 执行候选代码</label><div class="repair-actions"><button id="repair-validate" class="secondary" type="button">执行候选 Python 代码并验证</button><button id="repair-apply" class="primary-button" type="button">应用已验证的修复</button><button id="repair-rollback" class="secondary" type="button">恢复应用前代码</button></div><details><summary>查看候选验证输出</summary><pre id="repair-validation"></pre></details><p class="config-help">源码修改须先停止当前实验，再验证和应用。应用前检查源码是否被其他操作修改，并保留备份；应用或恢复代码后需重启后端，当前进程不会自动载入。诊断调用与重试的费用会记录到对应实验结果。</p></div>`;
  document.body.append(dialog);
  request('config').then(data=>{config=data;renderRecovery();}).catch(error=>{$('#repair-automatic-status').textContent='无法读取自动修复配置：'+error.message;});
  for(const id of ['repair-auto','repair-auto-recover'])$('#'+id).addEventListener('change',automaticFields);
  $('#open-repair').addEventListener('click',show);$('#close-repair').addEventListener('click',()=>dialog.close());
  $('#repair-provider').addEventListener('change',()=>{const presets={openai:'https://api.openai.com/v1',compatible:'https://api.deepseek.com',anthropic:'https://api.anthropic.com',custom:'http://127.0.0.1:9000/agent'};if(presets[$('#repair-provider').value])$('#repair-base').value=presets[$('#repair-provider').value];$('#repair-key').value='';providerFields();});
  $('#repair-config-form').addEventListener('submit',async event=>{event.preventDefault();try{config=await request('config',payload());$('#repair-key').value='';current.automatic=config;automaticFields();renderRecovery();context.toast('修复 Agent 配置已保存。');}catch(error){$('#repair-message').textContent=error.message;}});
  $('#repair-read-capabilities').addEventListener('click',async()=>{try{const data=await request('model-capabilities',payload());$('#repair-efforts').replaceChildren(...(data.levels||[]).map(value=>new Option(value,value)));$('#repair-capabilities').textContent=`${data.message} ${(data.levels||[]).join(' / ')}；可留空使用服务默认。`;}catch(error){$('#repair-capabilities').textContent=error.message;}});
  $('#repair-refresh-sources').addEventListener('click',()=>sources().catch(error=>context.toast(error.message)));
  $('#repair-resume').addEventListener('click',resume);
  $('#repair-start').addEventListener('click',()=>act('start',{source_run_id:$('#repair-source').value||null,description:$('#repair-description').value}));
  $('#repair-validate').addEventListener('click',()=>{if(!$('#repair-allow-execution').checked){context.toast('请先查看候选差异与测试源码，并勾选允许在本机执行。');return;}act('validate',{allow_local_code_execution:true});});
  for(const path of ['stop','apply','rollback'])$(`#repair-${path}`).addEventListener('click',()=>act(path));
}
export function renderRepair(state){
  if(!$('#repair-dialog'))return;
  const signature=JSON.stringify(state.repair||{});if(signature!==lastRendered){lastRendered=signature;render(state.repair);}
}
