import './style.css';
import './research.css';
import {initResearch,renderResearch,renderRunVideo,renderRunTimer} from './research.js';
import {initRsi,renderRsi} from './rsi.js';
import {initLayout} from './layout.js';
import {initRepair,renderRepair} from './repair.js';
import {initMultiarm,renderMultiarm} from './multiarm.js';

const icons={
  cube:'<path d="m12 3 8 4.5v9L12 21l-8-4.5v-9L12 3Z"/><path d="m4 7.5 8 4.5 8-4.5M12 12v9M8 5.2l8 4.6"/>',
  help:'<circle cx="12" cy="12" r="9"/><path d="M9.5 9a2.5 2.5 0 0 1 5 .4c0 1.6-2.5 1.9-2.5 3.6m0 3h.01"/>',
  reset:'<path d="M4 9a8 8 0 1 1 0 6M4 4v5h5"/>', focus:'<path d="M9 4H4v5m11-5h5v5M4 15v5h5m11-5v5h-5"/><circle cx="12" cy="12" r="3"/>',
  spark:'<path d="m12 3 2.2 6.8L21 12l-6.8 2.2L12 21l-2.2-6.8L3 12l6.8-2.2L12 3Zm7 0v4m-2-2h4"/>',
  info:'<circle cx="12" cy="12" r="9"/><path d="M12 11v6m0-10h.01"/>', mouse:'<rect x="6" y="3" width="12" height="18" rx="6"/><path d="M12 7v4"/>',
  pause:'<path d="M8 5v14M16 5v14"/>', play:'<path d="m8 4 12 8-12 8V4Z"/>', arrow:'<path d="M12 19V5m-6 6 6-6 6 6"/>',
  route:'<circle cx="5" cy="5" r="2"/><circle cx="19" cy="19" r="2"/><path d="M9 5h6a4 4 0 0 1 0 8H9a3 3 0 0 0 0 6h6"/>'
};
const icon=name=>`<svg viewBox="0 0 24 24" aria-hidden="true">${icons[name]||''}</svg>`;
document.querySelectorAll('[data-icon]').forEach(el=>el.innerHTML=icon(el.dataset.icon));
document.querySelectorAll('[data-icon-before]').forEach(el=>el.insertAdjacentHTML('afterbegin',icon(el.dataset.iconBefore)));
const $=s=>document.querySelector(s);
let state=null, generation=-1, activeRobot='', connected=false, busy=false, toastTimer, lastTaskKey='', lastEvent=0, frameStarted=false;
let agentConfig=null, lastObservation=-1, lastHistory='';
const frame=$('#sim-frame');
function toast(text){$('#toast').textContent=text;$('#toast').classList.add('visible');clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('#toast').classList.remove('visible'),3500);}
async function api(payload){
  const response=await fetch('/api/command',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  let data;
  try{data=await response.json();}catch{throw new Error('MuJoCo 服务没有返回有效响应，请检查本地服务。');}
  if(!response.ok)throw new Error(typeof data.detail==='string'?data.detail:'控制参数无效，请检查输入。');
  return data;
}
async function agentApi(path,payload){
  const response=await fetch(`/api/agent/${path}`,payload===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  const data=await response.json();
  if(!response.ok){const detail=data.detail;throw new Error(typeof detail==='string'?detail:Array.isArray(detail)?detail.map(e=>e.msg).join('；'):'Agent 请求失败');}
  return data;
}
function enableControls(){
  const enabled=connected&&!busy;
  document.querySelectorAll('.sim-action, [data-model], #reset, #view-reset, #pause, #gripper-open, #gripper-close, #stop, #send, #command, #speed, #apply-scene, #scene-select, #task-select, #scene-seed, .viewer-options input, #joint-controls input, [data-view]').forEach(el=>el.disabled=!enabled);
  $('#step').disabled=!enabled||!state?.paused;
  const active=['running','settling'].includes(state?.task.status);
  document.querySelectorAll('[data-action="home"],[data-action="demo"]').forEach(el=>el.disabled=!enabled||active);
  const takeover=!!(state?.agent?.active||state?.policy?.active||state?.comparison?.active||state?.research?.active||state?.experiment?.active);
  $('#agent-record-video').disabled=!enabled||takeover;
  $('#agent-stop').disabled=!connected||!state?.agent?.active;
  $('#agent-one-round').disabled=!enabled||takeover;
  $('#capture-observation').disabled=!enabled||takeover;
  const agentConfigEditable=state?.agent?.status==='repair_paused'&&!state?.policy?.active&&!state?.comparison?.active&&!state?.research?.active&&!state?.experiment?.active;
  $('#agent-settings').disabled=(takeover&&!agentConfigEditable)||busy;
  if(takeover)document.querySelectorAll('[data-model],#joint-controls input,.sim-action,#gripper-open,#gripper-close,#command,#send,#speed,#gravity,#step,#apply-scene,#scene-select,#task-select,#scene-seed').forEach(el=>el.disabled=true);
}
async function operate(payload,{message=false}={}){
  if(!connected||busy)return;
  try{
    busy=true;enableControls();
    if(payload.action==='load'){$('#scene-error').hidden=false;$('#scene-error').textContent='正在编译模型与加载原生渲染资源…';}
    const result=await api(payload);await poll();if(message)toast(result.message);
  }catch(error){toast(error.message);}
  finally{busy=false;enableControls();if(connected)$('#scene-error').hidden=true;}
}
const toDisplay=(a,value)=>a.unit==='command'?value/255*100:value*180/Math.PI;
const fromDisplay=(a,value)=>a.unit==='command'?value/100*255:value*Math.PI/180;
const targetText=(a,value)=>`${toDisplay(a,value).toFixed(1)}${a.unit==='command'?'%':'°'}`;
const actualText=a=>a.unit==='command'?`${(a.actual*2*1000).toFixed(1)} mm`:`${(a.actual*180/Math.PI).toFixed(1)}°`;

function buildJoints(){
  const root=$('#joint-controls');root.replaceChildren();
  state.actuators.forEach((a,localIndex)=>{
    const row=document.createElement('div');row.className='joint-control';
    const header=document.createElement('div');header.className='joint-header';
    const label=document.createElement('label');label.htmlFor=`joint-${a.index}`;
    const finger=Object.entries({thumb:'拇指',index:'食指',middle:'中指',ring:'无名指',little:'小指'}).find(([key])=>a.name.includes(key));
    const jointLabel=finger?`${finger[1]} · ${a.name.split('_').at(-1)}`:a.is_gripper?'夹爪':`关节 ${localIndex+1}`;
    label.innerHTML=`<b>${a.is_gripper?'G':String(a.index+1).padStart(2,'0')}</b><span>${jointLabel}<small>${a.name}</small></span>`;
    const output=document.createElement('output');output.id=`target-${a.index}`;output.textContent=targetText(a,a.target);header.append(label,output);
    const input=document.createElement('input');input.type='range';input.id=`joint-${a.index}`;input.min=String(Math.ceil(toDisplay(a,a.min)*10)/10);input.max=String(Math.floor(toDisplay(a,a.max)*10)/10);input.step='.1';input.value=String(toDisplay(a,a.target));input.setAttribute('aria-label',a.is_gripper?'夹爪目标':`关节 ${a.index+1} 目标角度`);
    input.setAttribute('aria-label',`${jointLabel} 目标角度`);
    input.addEventListener('input',()=>{output.textContent=targetText(a,fromDisplay(a,Number(input.value)));input.dataset.editing='true';});
    input.addEventListener('change',async()=>{input.dataset.editing='false';await operate({action:'control',index:a.index,value:fromDisplay(a,Number(input.value))});});
    const meta=document.createElement('div');meta.className='joint-meta';meta.innerHTML=`<span>${toDisplay(a,a.min).toFixed(0)}${a.unit==='command'?'%':'°'}</span><span id="actual-${a.index}">实际 ${actualText(a)}</span><span>${toDisplay(a,a.max).toFixed(0)}${a.unit==='command'?'%':'°'}</span>`;
    row.append(header,input,meta);root.append(row);
  });
  $('#engine-version').textContent=`MuJoCo ${state.version}`;$('#watermark-version').textContent=state.version;
  $('#actuator-count').textContent=`${state.nu} 个`;$('#mesh-count').textContent=`${state.meshes} 个`;$('#timestep').textContent=`${(state.timestep*1000).toFixed(1)} ms`;
    $('#model-source').href=state.source;$('#model-source').textContent='官方模型来源 ↗';$('#scene-name').textContent=state.name;
  document.querySelectorAll('[data-model]').forEach(el=>{const selected=el.dataset.model===state.model;el.classList.toggle('selected',selected);el.setAttribute('aria-pressed',String(selected));});
  $('[data-view="wrist"]').hidden=!state.has_wrist_camera;$('#gripper-label').textContent=state.model.includes('panda')?'夹爪开度':'夹爪角度';lastTaskKey='';
}
function renderTask(){
  const task=state.task,key=JSON.stringify(task.steps)+task.title,plan=$('#plan');
  if(!task.steps.length){plan.hidden=true;$('#empty-plan').hidden=false;lastTaskKey='';return;}
  plan.hidden=false;$('#empty-plan').hidden=true;
  if(key!==lastTaskKey){
    lastTaskKey=key;plan.replaceChildren();
    const title=document.createElement('div');title.className='plan-title';title.textContent=task.title;plan.append(title);
    const track=document.createElement('div');track.className='progress-track';track.innerHTML='<div class="progress-fill" id="progress-fill"></div>';plan.append(track);
    task.steps.forEach((step,i)=>{const row=document.createElement('div');row.className='plan-step';const number=document.createElement('span');number.className='step-number';number.textContent=i+1;const label=document.createElement('div');label.className='step-content';label.textContent=step;row.append(number,label);plan.append(row);});
    const result=document.createElement('div');result.id='task-result';result.className='task-result';plan.append(result);
  }
  document.querySelectorAll('.plan-step').forEach((el,i)=>{el.classList.toggle('active',i===task.step&&task.status==='running');el.classList.toggle('done',i<task.step||task.status==='complete');el.querySelector('.step-number').textContent=i<task.step||task.status==='complete'?'✓':i+1;});
  $('#progress-fill').style.width=`${task.progress*100}%`;
  const status={running:'控制序列执行中',settling:'等待关节收敛至目标',complete:'✓ 已完成，实际关节已达到目标容差。',failed:'目标未达到，请检查关节限制或碰撞。',cancelled:'动作计划已被手动控制中断。'};
  $('#task-result').textContent=task.status==='failed'&&task.message?task.message:status[task.status]||task.status;
}
function renderState(){
  if(state.generation!==generation||state.active_robot!==activeRobot){generation=state.generation;activeRobot=state.active_robot;buildJoints();}
  $('#connection').innerHTML='<i></i>MuJoCo 已连接';$('#fps').textContent=`${state.fps.toFixed(0)} FPS`;$('#sim-time').textContent=`t = ${state.time.toFixed(3)} s`;
  const active=['running','settling'].includes(state.task.status);
  $('#scene-status').textContent=state.paused?'仿真已暂停':active?(state.task.steps[state.task.step]||'执行中'):'仿真运行中';
  $('#status-dot').classList.toggle('running',!state.paused);$('#robot-status').innerHTML=`<i></i>${state.paused?'已暂停':active?'执行中':'运行中'}`;
  $('#physics-rate').textContent=`${(1/state.timestep).toFixed(0)} Hz · ${state.steps} 步`;
  $('#position').textContent=state.tcp.map(v=>v.toFixed(3)).join(' / ');$('#contact-count').textContent=state.contacts;
  $('#gripper-status').textContent=actualText(state.actuators.at(-1));$('#gripper-detail').textContent=state.model.includes('panda')?'两指间距（实际）':'夹爪关节（实际）';
  $('#frame-info').textContent=`渲染与物理帧 ${state.render_ms.toFixed(0)} ms`;
  $('#pause').innerHTML=icon(state.paused?'play':'pause')+(state.paused?'继续':'暂停');
  $('#speed').value=String(state.speed);$('#gravity').checked=state.gravity;
  for(const key of ['contacts','joints','collision'])$(`#show-${key}`).checked=state.flags[key];
  const viewNames={perspective:'透视视图',top:'俯视视图',front:'正视视图',wrist:'腕部相机',custom:'自由视图'};$('#view-tag').textContent=viewNames[state.view];
  document.querySelectorAll('[data-view]').forEach(el=>{const selected=el.dataset.view===state.view;el.classList.toggle('selected',selected);el.setAttribute('aria-pressed',String(selected));});
  state.actuators.forEach(a=>{const input=$(`#joint-${a.index}`);if(input.dataset.editing!=='true'){input.value=String(toDisplay(a,a.target));$(`#target-${a.index}`).textContent=targetText(a,a.target);}$(`#actual-${a.index}`).textContent=`实际 ${actualText(a)}`;});
  if(state.events.length&&state.events.at(-1).id!==lastEvent){lastEvent=state.events.at(-1).id;const log=$('#event-log');log.replaceChildren();state.events.slice(-6).reverse().forEach(e=>{const row=document.createElement('p');row.className=e.kind;row.textContent=e.message;log.append(row);});}
  renderTask();renderAgent();enableControls();renderResearch(state);renderRsi(state);renderRepair(state);renderMultiarm(state,{disabled:!connected||busy});
}
function renderAgent(){
  const agent=state.agent;if(!agent)return;
  const names={idle:'等待任务',observing:'采集观察',thinking:'模型决策中',executing:'正在执行',repairing:'故障暂停 · 正在诊断',repair_paused:'故障暂停 · 等待处理',retrying:'重试当前轮',verified:'评价器确认成功',model_done:'模型判断已完成',blocked:'模型无法继续',error:'调用或执行失败',stopped:'接管已停止',round_complete:'本轮已结束',limit:'已达到运行预算'};
  $('#agent-status').textContent=names[agent.status]||agent.status;
  $('#agent-message').textContent=agent.message;
  const roundLimit=Object.hasOwn(agent,'max_rounds')?agent.max_rounds:agentConfig?.max_rounds;
  $('#agent-round').textContent=roundLimit==null?`${agent.round} 轮 · 不限轮数`:`${agent.round} / ${roundLimit} 轮`;
  renderRunTimer($('#agent-run-time'),agent);
  renderRunVideo($('#agent-video'),agent.run_id,agent.video);
  document.querySelectorAll('[data-stage]').forEach(el=>el.classList.toggle('active',el.dataset.stage===agent.status));
  const action=agent.action;$('#agent-action').hidden=!action;
  if(action){$('#action-position').textContent=action.position_m.map(x=>x.toFixed(4)).join(' / ');$('#action-quaternion').textContent=action.quaternion_wxyz.map(x=>x.toFixed(3)).join(' / ');$('#action-gripper').textContent=`张开 ${(action.gripper_open*100).toFixed(0)}% · ${action.duration_s.toFixed(1)} s`;$('#action-explanation').textContent=action.explanation;}
  const historyKey=JSON.stringify(agent.history);
  if(historyKey!==lastHistory){lastHistory=historyKey;const root=$('#agent-history');root.replaceChildren();
    (agent.history||[]).slice(-5).reverse().forEach(record=>{const row=document.createElement('div');row.className='round-record';const title=document.createElement('strong');title.textContent=`第 ${record.round} 轮 · ${record.action.status==='move'?'末端动作':record.action.status==='done'?'模型完成判断':'无法继续'}`;const text=document.createElement('p');
      const result=record.result;
      text.textContent=typeof result==='string'?result:result?.message|| (result?`${result.status==='complete'?'实际末端到位':'动作未达到容差'}${result.position_error_m==null?'':` · 位置误差 ${(result.position_error_m*1000).toFixed(2)} mm`}`:'等待执行反馈');row.append(title,text);root.append(row);});
  }
  const sourceName=state.policy?.active||($('#policy-tab').getAttribute('aria-selected')==='true'&&state.policy?.observation_time!=null)?'policy':'agent';
  const source=state[sourceName],observationKey=`${sourceName}:${source?.observation_id}:${source?.observation_robot}`;
  if(observationKey!==lastObservation||source?.observation_robot!==state.model){
    lastObservation=observationKey;const valid=source?.observation_time!=null&&source.observation_robot===state.model;
    $('#observation-time').textContent=valid?`t = ${source.observation_time.toFixed(3)} s · 第 ${source.observation_id} 次采集`:'尚未采集当前机械臂';
    for(const name of ['front','side','wrist']){const img=$(`#camera-${name}`);img.hidden=!valid;img.nextElementSibling.hidden=valid;if(valid)img.src=`/api/${sourceName}/camera/${name}?observation=${source.observation_id}`;else img.removeAttribute('src');}
  }
}
function disconnected(message){
  connected=false;$('#connection').textContent='MuJoCo 未连接';$('#scene-error').hidden=false;$('#scene-error').replaceChildren();
  const title=document.createElement('strong');title.textContent='MuJoCo 服务未连接';const detail=document.createElement('p');detail.textContent=message;
  const help=document.createElement('p');help.textContent='请启动项目中的 MuJoCo 服务，再点击下方按钮重新连接。';const retry=document.createElement('button');retry.textContent='重新连接';retry.className='secondary';retry.addEventListener('click',()=>{frameStarted=false;poll();});$('#scene-error').append(title,detail,help,retry);
  frame.removeAttribute('src');frameStarted=false;$('#robot-status').textContent='离线';$('#scene-status').textContent='引擎离线';$('#status-dot').classList.remove('running');enableControls();
}
let polling=false;
async function poll(){
  if(polling)return;polling=true;
  try{
    const response=await fetch('/api/state',{signal:AbortSignal.timeout(4000)});if(!response.ok)throw new Error(`本地服务返回 ${response.status}`);
    const next=await response.json();if(!next.ready)throw new Error(next.error||'引擎尚未准备好');
    state=next;connected=true;if(!frameStarted){frame.src=`/api/stream?session=${Date.now()}`;frameStarted=true;}
    $('#scene-error').hidden=true;renderState();
  }catch(error){disconnected(error.message);}
  finally{polling=false;}
}
document.querySelectorAll('[data-model]').forEach(el=>el.addEventListener('click',()=>{if(state?.model!==el.dataset.model)operate({action:'load',model:el.dataset.model});}));
document.querySelectorAll('.sim-action').forEach(el=>el.addEventListener('click',()=>operate({action:el.dataset.action})));
$('#reset').addEventListener('click',()=>operate({action:'reset'}));$('#pause').addEventListener('click',()=>operate({action:'pause',value:!state.paused}));$('#step').addEventListener('click',()=>operate({action:'step'}));$('#stop').addEventListener('click',()=>operate({action:'stop'}));
$('#speed').addEventListener('change',e=>operate({action:'speed',value:Number(e.target.value)}));$('#gravity').addEventListener('change',e=>operate({action:'gravity',value:e.target.checked}));
for(const name of ['contacts','joints','collision'])$(`#show-${name}`).addEventListener('change',e=>operate({action:'flags',flags:{[name]:e.target.checked}}));
document.querySelectorAll('[data-view]').forEach(el=>el.addEventListener('click',()=>operate({action:'view',view:el.dataset.view})));
$('#view-reset').addEventListener('click',()=>operate({action:'view',view:'perspective'}));
$('#gripper-open').addEventListener('click',()=>operate({action:'instruction',text:'张开夹爪'}));$('#gripper-close').addEventListener('click',()=>operate({action:'instruction',text:'闭合夹爪'}));
function selectTab(name){for(const item of ['joints','agent','policy']){$(`#${item}-tab`).setAttribute('aria-selected',String(item===name));$(`#${item}-tab`).classList.toggle('selected',item===name);$(`#${item}-panel`).hidden=item!==name;}}
$('#joints-tab').addEventListener('click',()=>selectTab('joints'));$('#agent-tab').addEventListener('click',()=>selectTab('agent'));
async function submit(event,oneRound=false){
  event?.preventDefault();if(!connected||busy)return;
  const text=$('#command').value.trim();if(!text){toast('请输入一个机械臂任务。');return;}
  busy=true;enableControls();
  try{await agentApi('start',{task:text,one_round:oneRound,record_video:$('#agent-record-video').checked});await poll();}
  catch(error){toast(error.message);}
  finally{busy=false;enableControls();}
}
$('#command-form').addEventListener('submit',submit);$('#command').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey&&!e.isComposing){e.preventDefault();submit();}});
$('#help').addEventListener('click',()=>$('#help-dialog').showModal());$('#close-help').addEventListener('click',()=>$('#help-dialog').close());
$('#agent-one-round').addEventListener('click',()=>submit(null,true));
$('#example-task').addEventListener('click',()=>{$('#command').value='保持夹爪开度和当前末端姿态，将末端沿世界坐标 Z 轴抬高 2 厘米。';$('#command').focus();});
$('#agent-stop').addEventListener('click',async()=>{try{await agentApi('stop',{});await poll();}catch(error){toast(error.message);}});
$('#capture-observation').addEventListener('click',async()=>{busy=true;enableControls();try{await agentApi('observe',{});await poll();}catch(error){toast(error.message);}finally{busy=false;enableControls();}});
const providerNames={codex:'Codex · 本机登录',openai:'OpenAI · Responses API',compatible:'兼容服务 · Chat Completions',anthropic:'Claude · Messages API',custom:'自定义 Agent 服务'};
function updateConfigSummary(){if(!agentConfig)return;$('#agent-model-name').textContent=agentConfig.model;$('#agent-provider-name').textContent=providerNames[agentConfig.provider];$('#agent-run-budget').textContent=agentConfig.max_rounds==null?'不限制轮数':`最多 ${agentConfig.max_rounds} 轮`;}
let capabilityGeneration=0,capabilityTimer=null,capabilityLoading=false;
function selectedReasoning(){return $('#config-reasoning').value==='__manual'?$('#config-custom-reasoning').value.trim()||null:$('#config-reasoning').value||null;}
function reasoningFields(){
  const manual=$('#config-reasoning').value==='__manual';$('#custom-reasoning-field').hidden=!manual;
  const disabled=$('#config-reasoning').disabled;$('#config-custom-reasoning').disabled=disabled||!manual;$('#config-custom-reasoning').required=manual&&!disabled;
}
function renderModelCapabilities(capability,previous){
  const select=$('#config-reasoning');select.replaceChildren();
  const add=(value,label)=>{const option=document.createElement('option');option.value=value;option.textContent=label;select.append(option);};
  add('',capability.default_level?`服务默认（${capability.default_level}）`:'服务默认');
  for(const level of capability.levels||[])add(level,level);
  if(capability.levels==null)add('__manual','手动填写 · 能力尚未确认');
  const known=(capability.levels||[]).includes(previous);
  if(previous&&known)select.value=previous;
  else if(previous&&capability.levels==null){select.value='__manual';$('#config-custom-reasoning').value=previous;}
  else select.value='';
  const source={codex_models_cache:'本机 Codex 模型目录',service_metadata:'服务模型元数据',deepseek_models_api:'DeepSeek 模型元数据',anthropic_models_api:'Claude 模型元数据',official_docs_2026_10_22:'官方文档配置',unavailable:'尚未确认',unknown:'尚未确认'}[String(capability.source).replaceAll('-','_')]||capability.source||'尚未确认';
  $('#model-capability-help').textContent=[capability.message,`来源：${source}`,previous&&!known&&capability.levels!=null?`原配置 ${previous} 不在当前模型的能力清单中，请重新选择；当前为服务默认。`:'',capability.source_url?`文档：${capability.source_url}`:'','这些等级是各模型的原生参数，不代表跨模型统一的推理量。'].filter(Boolean).join(' ');
  providerFields();
}
async function refreshModelCapabilities(previous=selectedReasoning()){
  clearTimeout(capabilityTimer);const generation=++capabilityGeneration,model=$('#config-model').value.trim();
  if(!model){capabilityLoading=false;$('#refresh-model-capabilities').disabled=false;renderModelCapabilities({levels:null,source:'unknown',message:'请先填写模型 ID。'},null);return;}
  capabilityLoading=true;$('#refresh-model-capabilities').disabled=true;providerFields();$('#model-capability-help').textContent='正在读取模型能力（不发起推理调用）…';
  try{const capability=await agentApi('model-capabilities',{provider:$('#config-provider').value,model,base_url:$('#config-base-url').value.trim(),api_key:$('#config-api-key').value});if(generation===capabilityGeneration)renderModelCapabilities(capability,previous);}
  catch(error){if(generation===capabilityGeneration)renderModelCapabilities({levels:null,source:'unknown',message:`读取失败：${error.message}。可使用服务默认，或按服务文档手动填写。`},previous);}
  finally{if(generation===capabilityGeneration){capabilityLoading=false;$('#refresh-model-capabilities').disabled=false;providerFields();}}
}
function scheduleModelCapabilities(){clearTimeout(capabilityTimer);capabilityGeneration++;capabilityLoading=true;$('#refresh-model-capabilities').disabled=true;providerFields();capabilityTimer=setTimeout(()=>refreshModelCapabilities(),450);}
function providerFields(){
  const provider=$('#config-provider').value;
  let deepseek=false;
  try{deepseek=provider==='compatible'&&new URL($('#config-base-url').value).hostname==='api.deepseek.com';}catch{}
  $('#api-config-fields').hidden=provider==='codex';
  $('#api-budget-fields').hidden=['codex','custom'].includes(provider);
  $('#api-config-fields').querySelectorAll('input').forEach(input=>input.disabled=provider==='codex');
  $('#config-output-tokens').disabled=['codex','custom'].includes(provider);
  $('#output-budget-help').hidden=['codex','custom'].includes(provider);
  $('#deepseek-thinking-field').hidden=!deepseek;
  $('#reasoning-field').hidden=false;
  $('#config-reasoning').disabled=capabilityLoading||(deepseek&&$('#config-deepseek-thinking').value==='disabled');
  reasoningFields();
  $('#provider-help').textContent=provider==='codex'?'使用本机 Codex CLI 的现有登录，无需 API Key。每轮调用计入当前账号的 Codex 用量。':provider==='custom'?'接入你自己的 Agent 或编排服务，填写完整 POST 地址与服务模型 ID。使用 embodied-agent-v1 协议接收三路图像和输出契约；本机服务可不填 Key，远程服务使用 HTTPS 和专用 Key。':'请填写该服务的准确模型 ID 和 API 地址。模型需支持多图输入。API Key 仅存于本机服务内存，重启后需重新填写。';
  if(deepseek)$('#provider-help').textContent='deepseek-flash 支持图像输入。思考模式和推理强度会传给 DeepSeek 官方接口。API Key 仅存于本机服务内存，重启后需重新填写。';
  else if(provider==='compatible')$('#provider-help').textContent='适用于支持多图输入和 JSON 输出的 OpenAI 兼容模型服务。模型 ID 可自定义；密钥仅保存在本机服务内存。';
}
$('#config-base-url').addEventListener('input',()=>{providerFields();scheduleModelCapabilities();});
$('#config-model').addEventListener('input',scheduleModelCapabilities);
$('#config-api-key').addEventListener('change',scheduleModelCapabilities);
$('#config-reasoning').addEventListener('change',reasoningFields);
$('#refresh-model-capabilities').addEventListener('click',()=>refreshModelCapabilities());
$('#config-deepseek-thinking').addEventListener('change',providerFields);
$('#config-provider').addEventListener('change',()=>{const provider=$('#config-provider').value;$('#config-base-url').value=provider==='custom'?'http://127.0.0.1:8020/agent':provider==='anthropic'?'https://api.anthropic.com':provider==='compatible'?'https://api.deepseek.com':'https://api.openai.com/v1';$('#config-api-key').value='';if(['anthropic','compatible','custom'].includes(provider)){$('#config-model').value='';$('#config-model').placeholder=provider==='custom'?'填写自定义 Agent 服务模型 ID':'填写支持多图输入的模型 ID';}else $('#config-model').value='gpt-6-astra';providerFields();refreshModelCapabilities(null);});
$('#agent-settings').addEventListener('click',async()=>{try{agentConfig=await agentApi('config');const c=agentConfig;$('#config-provider').value=c.provider;$('#config-model').value=c.model;$('#config-base-url').value=c.base_url;$('#config-deepseek-thinking').value=c.deepseek_thinking??'enabled';$('#config-output-tokens').value=c.max_output_tokens??16384;$('#config-rounds').value=c.max_rounds??'';$('#config-timeout').value=c.timeout_s;$('#config-api-key').value='';$('#config-api-key').placeholder=c.has_api_key?'当前地址已有密钥，留空保留':'仅保留在本机服务内存中';$('#config-error').hidden=true;providerFields();$('#agent-config-dialog').showModal();refreshModelCapabilities(c.reasoning);}catch(error){toast(error.message);}});
$('#close-agent-config').addEventListener('click',()=>$('#agent-config-dialog').close());
$('#agent-config-form').addEventListener('submit',async event=>{event.preventDefault();try{if(capabilityLoading)throw new Error('模型能力正在读取，请稍候再应用。');agentConfig=await agentApi('config',{provider:$('#config-provider').value,model:$('#config-model').value.trim(),base_url:$('#config-base-url').value.trim(),api_key:$('#config-api-key').value,reasoning:selectedReasoning(),reasoning_native:true,deepseek_thinking:$('#config-deepseek-thinking').value,max_output_tokens:$('#config-output-tokens').disabled?(agentConfig?.max_output_tokens??16384):Number($('#config-output-tokens').value),max_rounds:Number($('#config-rounds').value)||null,timeout_s:Number($('#config-timeout').value)});$('#config-api-key').value='';updateConfigSummary();$('#agent-config-dialog').close();toast('模型配置已应用。');}catch(error){$('#config-error').textContent=error.message;$('#config-error').hidden=false;}});
agentApi('config').then(config=>{agentConfig=config;updateConfigSummary();}).catch(()=>{});
initResearch({operate,toast,selectTab});
initRsi({operate,toast,selectTab});
initRepair({toast});
initMultiarm({operate});
initLayout({toast});
// Camera gestures modify MuJoCo's camera. There is no client-side 3D simulation.
let pointer=null,cameraPending={dx:0,dy:0,zoom:0,pan_dx:0,pan_dy:0},cameraSending=false;
frame.title='左键拖动旋转；右键、中键或 Shift + 左键拖动平移；滚轮缩放。点击预设视角可复位。';
async function flushCamera(){
  if(cameraSending||!connected||busy||!Object.values(cameraPending).some(Boolean))return;
  cameraSending=true;const payload=cameraPending;cameraPending={dx:0,dy:0,zoom:0,pan_dx:0,pan_dy:0};
  try{await api({action:'camera',dx:Math.max(-180,Math.min(180,payload.dx)),dy:Math.max(-180,Math.min(180,payload.dy)),zoom:Math.max(-1,Math.min(1,payload.zoom)),pan_dx:Math.max(-2,Math.min(2,payload.pan_dx)),pan_dy:Math.max(-2,Math.min(2,payload.pan_dy))});}
  catch(error){toast(error.message);}
  finally{cameraSending=false;if(Object.values(cameraPending).some(Boolean))flushCamera();}
}
frame.addEventListener('contextmenu',e=>e.preventDefault());
frame.addEventListener('pointerdown',e=>{if(![0,1,2].includes(e.button))return;e.preventDefault();pointer={id:e.pointerId,x:e.clientX,y:e.clientY,pan:e.button!==0||e.shiftKey};frame.setPointerCapture(e.pointerId);});
frame.addEventListener('pointermove',e=>{
  if(!pointer||pointer.id!==e.pointerId)return;
  const dx=e.clientX-pointer.x,dy=e.clientY-pointer.y;
  if(pointer.pan){
    const bounds=frame.getBoundingClientRect();
    const imageHeight=Math.max(1,Math.min(bounds.height,bounds.width*(frame.naturalHeight||640)/(frame.naturalWidth||960)));
    cameraPending.pan_dx+=dx/imageHeight;cameraPending.pan_dy+=dy/imageHeight;
  }else{cameraPending.dx-=dx*.35;cameraPending.dy-=dy*.25;}
  pointer.x=e.clientX;pointer.y=e.clientY;flushCamera();
});
for(const type of ['pointerup','pointercancel','lostpointercapture'])frame.addEventListener(type,e=>{if(pointer?.id===e.pointerId)pointer=null;});
frame.addEventListener('wheel',e=>{e.preventDefault();cameraPending.zoom+=e.deltaY*.001;flushCamera();},{passive:false});
frame.addEventListener('error',()=>{frameStarted=false;if(connected){$('#scene-error').hidden=false;$('#scene-error').textContent='原生画面连接中断，正在重新连接。';}});
poll();setInterval(()=>{if(connected&&!busy)poll();},250);
