import './repair.css';
const $=selector=>document.querySelector(selector);
let context,config=null,current={},lastRendered='',started=false,autoPending=false;
const analyzedErrors=new Set();
async function request(path,payload){
  const response=await fetch(`/api/repair/${path}`,payload===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  const data=await response.json();if(!response.ok)throw new Error(typeof data.detail==='string'?data.detail:JSON.stringify(data.detail||'修复请求失败'));return data;
}
function payload(){return {provider:$('#repair-provider').value,model:$('#repair-model').value.trim(),base_url:$('#repair-base').value.trim(),api_key:$('#repair-key').value,reasoning:$('#repair-effort').value.trim()||null,reasoning_native:true,timeout_s:Number($('#repair-timeout').value),max_output_tokens:Number($('#repair-tokens').value),extra_instructions:$('#repair-instructions').value,auto_analyze:$('#repair-auto').checked};}
function providerFields(){const codex=$('#repair-provider').value==='codex';$('#repair-api-fields').hidden=codex;$('#repair-budget-field').hidden=codex;}
function render(data){
  const previousJob=current.job_id;current=data||{};if(previousJob!==current.job_id)$('#repair-allow-execution').checked=false;$('#repair-card-status').textContent=current.active?'正在处理':current.restart_required?'待重启':'独立编码模型';
  $('#repair-message').textContent=current.message||'选择报错实验，分析并生成候选修复。';
  $('#repair-stop').disabled=!current.active;$('#repair-start').disabled=!!current.active;$('#repair-save').disabled=!!current.active;
  $('#repair-validate').disabled=current.active||current.status!=='proposed';
  $('#repair-apply').disabled=current.active||current.status!=='validated';$('#repair-rollback').disabled=current.active||!['applied','applying','apply_failed','restoring','restore_failed'].includes(current.status);
  const proposal=current.proposal;$('#repair-proposal').hidden=!proposal;
  $('#repair-diagnosis').textContent=proposal?.diagnosis||'';
  for(const [id,values] of [['repair-evidence',proposal?.evidence],['repair-advice',proposal?.recommendations]]){const list=$(`#${id}`);list.replaceChildren();for(const text of values||[]){const li=document.createElement('li');li.textContent=text;list.append(li);}}
  $('#repair-diff').textContent=current.diff||'本次未提出代码修改。';$('#repair-validation').textContent=current.validation?.output||'尚未执行候选验证。';
  $('#repair-validation-scope').textContent=current.validation?.scope||'候选回归须先在原始实现上复现断言失败，再通过固定接口契约与相同回归。源码副本不是操作系统沙箱；测试作为当前用户的本机 Python 代码执行，具有该账户的文件与网络权限，请先查看全部候选差异和测试代码。';
  $('#repair-run-link').hidden=!current.run_id;$('#repair-run-link').textContent=current.run_id?`本次诊断日志 ${current.run_id}`:'';
}
async function sources(){const rows=await request('sources');const select=$('#repair-source');select.replaceChildren(new Option('不选实验，仅分析下方描述',''));for(const row of rows)select.append(new Option(`${row.display_name||row.model||row.run_id} · ${row.status}`,row.run_id));if(rows.length)select.value=rows[0].run_id;}
async function show(){try{config=await request('config');$('#repair-provider').value=config.provider;$('#repair-model').value=config.model;$('#repair-base').value=config.base_url;$('#repair-key').value='';$('#repair-key').placeholder=config.has_api_key?'已配置，留空保留当前修复模型密钥':'仅保存在当前本机服务内存';$('#repair-effort').value=config.reasoning||'';$('#repair-timeout').value=config.timeout_s;$('#repair-tokens').value=config.max_output_tokens;$('#repair-auto').checked=config.auto_analyze;$('#repair-instructions').value=config.extra_instructions;$('#repair-skills').textContent=`始终加载的内置技能：${config.skills.join(' / ')}。`;providerFields();await sources();render(await request('state'));$('#repair-dialog').showModal();}catch(error){context.toast(error.message);}}
async function act(path,body={}){try{render(await request(path,body));}catch(error){$('#repair-message').textContent=error.message;context.toast(error.message);}}
export function initRepair(callbacks){
  context=callbacks;
  $('#log-filter').append(new Option('故障修复 Agent','repair'));
  const card=document.createElement('section');card.className='repair-card';card.innerHTML='<div><strong>模型故障修复 Agent</strong><span id="repair-card-status">独立编码模型</span></div><p>读取调用报错与接入代码，生成可验证的修复建议。</p><button id="open-repair" class="secondary" type="button">配置与诊断报错 ↗</button>';
  $('.agent.panel').append(card);
  const dialog=document.createElement('dialog');dialog.id='repair-dialog';dialog.innerHTML=`
  <button id="close-repair" class="dialog-close" aria-label="关闭修复 Agent">×</button><div class="eyebrow">MODEL REPAIR / CODING AGENT</div><h2>模型故障修复 Agent</h2>
  <p class="config-help">默认使用 Astra 高推理强度，独立于机械臂主策略。可替换为擅长代码的模型或自定义 Agent。诊断会将所选实验的脱敏错误片段与模型接入源码发送给配置的模型，不发送密钥或相机图像。</p>
  <details><summary>模型与内置修复技能</summary><form id="repair-config-form" class="config-form">
  <div class="config-columns"><label>修复模型接入<select id="repair-provider"><option value="codex">Codex · 本机登录</option><option value="openai">OpenAI · Responses</option><option value="compatible">OpenAI 兼容 / DeepSeek</option><option value="anthropic">Claude · Messages</option><option value="custom">自定义 Agent</option></select></label><label>修复模型名称<input id="repair-model" value="gpt-6-astra" required></label></div>
  <div id="repair-api-fields"><label>修复模型服务地址<input id="repair-base" type="url" value="https://api.openai.com/v1" required></label><label>修复模型 API Key<input id="repair-key" type="password" autocomplete="new-password"></label></div>
  <div class="config-columns"><label>原生推理强度<input id="repair-effort" list="repair-efforts" placeholder="留空使用服务默认"><datalist id="repair-efforts"></datalist></label><label>超时 / 秒<input id="repair-timeout" type="number" min="15" max="300" value="300" required></label><label id="repair-budget-field">输出预算 / tokens<input id="repair-tokens" type="number" min="512" value="16384" required></label></div>
  <button id="repair-read-capabilities" class="secondary" type="button">读取修复模型原生推理等级</button><p id="repair-capabilities" class="config-help"></p>
  <p id="repair-skills" class="config-help"></p><p class="config-help">内置技能固定包含根因定位、接口与动作契约、最小修复和回归验证。模型仅提交限定文件的文本修改；不会自行执行 shell、修改物理安全约束、重试机械臂实验或切换主模型。</p>
  <label>补充修复要求<textarea id="repair-instructions" rows="2" maxlength="3000"></textarea></label><label class="record-video-option"><input id="repair-auto" type="checkbox">新模型调用报错后自动分析（会使用所配置模型的额度；应用代码仍需手动操作）</label><button id="repair-save" class="primary-button" type="submit">保存修复 Agent 配置</button></form></details>
  <div class="repair-section"><label>报错实验<select id="repair-source"></select></label><button id="repair-refresh-sources" type="button" class="secondary">刷新报错列表</button><label>补充问题描述<textarea id="repair-description" rows="3" maxlength="4000" placeholder="也可以不选实验，直接描述接入问题。"></textarea></label><div class="repair-actions"><button id="repair-start" type="button" class="primary-button">分析并生成候选修复</button><button id="repair-stop" type="button" class="secondary">停止分析 / 验证</button></div><p id="repair-message" role="status" class="config-help"></p><p id="repair-run-link" class="config-help" hidden></p></div>
  <section id="repair-proposal" hidden><h3>根因诊断</h3><p id="repair-diagnosis"></p><h4>证据</h4><ul id="repair-evidence"></ul><h4>处理建议</h4><ul id="repair-advice"></ul><details><summary>查看候选代码差异</summary><pre id="repair-diff"></pre></details></section>
  <div class="repair-section"><p id="repair-validation-scope" class="config-help"></p><label class="record-video-option"><input id="repair-allow-execution" type="checkbox">我已查看候选差异与测试源码，允许在本机 Python 执行候选代码</label><div class="repair-actions"><button id="repair-validate" class="secondary" type="button">执行候选 Python 代码并验证</button><button id="repair-apply" class="primary-button" type="button">应用已验证的修复</button><button id="repair-rollback" class="secondary" type="button">恢复应用前代码</button></div><details><summary>查看候选验证输出</summary><pre id="repair-validation"></pre></details><p class="config-help">应用前检查源代码是否被其他操作修改，并保留备份。应用或恢复后需重启后端；不会中断当前实验或自动重新发送付费模型请求。诊断调用的费用会单独记录到实验结果。</p></div>`;
  document.body.append(dialog);
  $('#open-repair').addEventListener('click',show);$('#close-repair').addEventListener('click',()=>dialog.close());
  $('#repair-provider').addEventListener('change',()=>{const presets={openai:'https://api.openai.com/v1',compatible:'https://api.deepseek.com',anthropic:'https://api.anthropic.com',custom:'http://127.0.0.1:9000/agent'};if(presets[$('#repair-provider').value])$('#repair-base').value=presets[$('#repair-provider').value];$('#repair-key').value='';providerFields();});
  $('#repair-config-form').addEventListener('submit',async event=>{event.preventDefault();try{config=await request('config',payload());$('#repair-key').value='';context.toast('修复 Agent 配置已保存。');}catch(error){$('#repair-message').textContent=error.message;}});
  $('#repair-read-capabilities').addEventListener('click',async()=>{try{const data=await request('model-capabilities',payload());$('#repair-efforts').replaceChildren(...(data.levels||[]).map(value=>new Option(value,value)));$('#repair-capabilities').textContent=`${data.message} ${(data.levels||[]).join(' / ')}；可留空使用服务默认。`;}catch(error){$('#repair-capabilities').textContent=error.message;}});
  $('#repair-refresh-sources').addEventListener('click',()=>sources().catch(error=>context.toast(error.message)));
  $('#repair-start').addEventListener('click',()=>act('start',{source_run_id:$('#repair-source').value||null,description:$('#repair-description').value}));
  $('#repair-validate').addEventListener('click',()=>{if(!$('#repair-allow-execution').checked){context.toast('请先查看候选差异与测试源码，并勾选允许在本机执行。');return;}act('validate',{allow_local_code_execution:true});});
  for(const path of ['stop','apply','rollback'])$(`#repair-${path}`).addEventListener('click',()=>act(path));
  request('config').then(data=>{config=data;started=true;}).catch(()=>{});
}
export function renderRepair(state){
  if(!$('#repair-dialog'))return;
  const signature=JSON.stringify(state.repair||{});if(signature!==lastRendered){lastRendered=signature;render(state.repair);}
  if(!started){request('config').then(data=>{config=data;started=true;}).catch(()=>{});return;}
  if(config?.auto_analyze&&!state.repair?.active&&!autoPending&&!state.repair?.restart_required){for(const key of ['agent','policy','research']){const source=state[key];if(source?.status==='error'&&!source.active&&source.run_id&&!analyzedErrors.has(source.run_id)){analyzedErrors.add(source.run_id);autoPending=true;request('start',{source_run_id:source.run_id}).catch(error=>context.toast(error.message)).finally(()=>{autoPending=false;});break;}}}
}
