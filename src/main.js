import './rounds.css';
import {renderRoundTimeline} from './rounds.js';
import './style.css';
import './research.css';
import {initResearch,renderResearch,renderRunVideo,renderRunTimer,reloadResearchCatalog} from './research.js';
import {initBenchmarks,renderBenchmarks} from './benchmarks.js';
import {initBehavior,renderBehavior,behaviorScore} from './behavior.js';
import {initRsi,renderRsi,reloadRsiCatalog} from './rsi.js';
import {initHarnessPAI,renderHarnessPAI} from './harnesspai.js';
import {initSimulators,renderSimulator} from './simulators.js';
import {initEnvironment,renderEnvironment} from './environment.js';
import {initLayout} from './layout.js';
import {initRepair,renderRepair} from './repair.js';
import {initMultiarm,renderMultiarm,updateModelCatalog} from './multiarm.js';
import {setTaskDraft} from './task_drafts.js';
import {harnessNames,providerNames,effectiveHarness,harnessCompatibility,harnessReasoningError} from './agent_harness.js';

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
let renderedModelContext='';
let catalogSession='',catalogLoading=false,catalogPromise=null;
function reloadWorkspaceCatalog(){
  if(catalogPromise)return catalogPromise;
  catalogLoading=true;
  catalogPromise=(async()=>{
    const session=state?.session_id,engine=state?.backend?.engine_id;
    try{
      const response=await fetch('/api/models');if(!response.ok)throw new Error('模型目录读取失败');
      const models=await response.json();
      if(session!==state?.session_id||engine!==state?.backend?.engine_id)return false;
      const refreshed=await Promise.all([reloadResearchCatalog(engine),reloadRsiCatalog(engine)]);
      if(refreshed.some(result=>result===false)||session!==state?.session_id||engine!==state?.backend?.engine_id)return false;
      updateModelCatalog(models,state?.model);catalogSession=session||'';return true;
    }catch(error){toast(error.message);return false;}
    finally{catalogLoading=false;catalogPromise=null;enableControls();}
  })();
  return catalogPromise;
}
const frame=$('#sim-frame');
const defaultViews=[{id:'perspective',label:'透视'},{id:'top',label:'俯视'},{id:'front',label:'正视'},{id:'wrist',label:'腕部'}];
const cameraLabels={front:'前方相机',side:'侧方相机',wrist:'腕部相机',head:'头部相机',left_wrist:'左腕相机',right_wrist:'右腕相机'};
let viewPresetSignature='',cameraGridSignature='';
function viewPresets(current){
  if(!current?.backend?.capabilities?.includes('view'))return [];
  return Array.isArray(current.view_presets)?current.view_presets.filter(item=>item&&typeof item.id==='string'&&typeof item.label==='string'):defaultViews.filter(item=>item.id!=='wrist'||current.has_wrist_camera);
}
function renderViewPresets(current){
  const presets=viewPresets(current),signature=JSON.stringify(presets),root=$('.view-switch');
  if(signature!==viewPresetSignature){
    viewPresetSignature=signature;root.replaceChildren();
    for(const preset of presets){const button=document.createElement('button');button.type='button';button.dataset.view=preset.id;button.textContent=preset.label;button.setAttribute('aria-pressed','false');button.addEventListener('click',()=>operate({action:'view',view:preset.id}));root.append(button);}
  }
  root.hidden=!presets.length;
  $('.viewport-bottom > span').hidden=!current?.backend?.capabilities?.includes('camera');
  $('#view-reset').hidden=!presets.length;
  $('#view-reset').title=presets.length?'恢复 '+presets[0].label:'当前环境没有可选视角';
  return presets;
}
function renderObservationCameras(names,sourceName,source,valid){
  const cameras=[...new Set(names.filter(name=>typeof name==='string'&&name.length>0))];
  const signature=JSON.stringify(cameras),root=$('.camera-grid');
  if(signature!==cameraGridSignature){
    cameraGridSignature=signature;root.replaceChildren();
    for(const [index,name] of cameras.entries()){
      const figure=document.createElement('figure'),box=document.createElement('div'),img=document.createElement('img'),placeholder=document.createElement('span');
      img.dataset.camera=name;img.alt='Agent '+(cameraLabels[name]||name)+'观察';placeholder.textContent='等待观察';box.append(img,placeholder);
      const caption=document.createElement('figcaption'),number=document.createElement('b'),label=document.createElement('span'),nativeName=document.createElement('small');
      number.textContent=String(index+1).padStart(2,'0');label.textContent=cameraLabels[name]||name;nativeName.textContent=name.toUpperCase();caption.append(number,label,nativeName);figure.append(box,caption);root.append(figure);
    }
  }
  root.classList.toggle('single-camera',cameras.length===1);
  for(const img of root.querySelectorAll('img')){img.hidden=!valid;img.nextElementSibling.hidden=valid;if(valid)img.src=`/api/${sourceName}/camera/${encodeURIComponent(img.dataset.camera)}?observation=${source.observation_id}`;else img.removeAttribute('src');}
}
const environmentBudget=document.createElement('label');environmentBudget.className='environment-budget';environmentBudget.hidden=true;
environmentBudget.innerHTML='环境动作预算 <input id="agent-environment-steps" type="number" min="1" max="10000" step="1" value="120" required> 步';
$('#agent-record-video').closest('label').after(environmentBudget);
const simulationBudget=document.createElement('label');simulationBudget.className='environment-budget';
simulationBudget.innerHTML='任务仿真预算 <input id="agent-sim-budget" type="number" min="1" value="120" required> 秒 · 推理时物理暂停';
environmentBudget.after(simulationBudget);
function toast(text){$('#toast').textContent=text;$('#toast').classList.add('visible');clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('#toast').classList.remove('visible'),3500);}
async function api(payload){
  const response=await fetch('/api/command',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  let data;
  try{data=await response.json();}catch{throw new Error('仿真服务没有返回有效响应，请检查本地服务。');}
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
  const enabled=connected&&!busy&&!state?.backend_switching;
  document.querySelectorAll('.sim-action, [data-model], #reset, #view-reset, #pause, #gripper-open, #gripper-close, #stop, #send, #command, #speed, #apply-scene, #scene-select, #task-select, #scene-seed, #use-task, .viewer-options input, #joint-controls input, [data-view]').forEach(el=>el.disabled=!enabled);
  $('#step').disabled=!enabled||!state?.paused;
  const active=['running','settling'].includes(state?.task?.status);
  document.querySelectorAll('[data-action="home"],[data-action="demo"]').forEach(el=>el.disabled=!enabled||active);
  const takeover=!!(state?.agent?.active||state?.policy?.active||state?.comparison?.active||state?.research?.active||state?.experiment?.active||(state?.benchmark?.active||state?.harnesspai?.active));
  $('#agent-record-video').disabled=!enabled||takeover;
  $('#agent-environment-steps').disabled=!enabled||takeover;
  $('#agent-stop').disabled=!state?.agent?.active;
  $('#agent-one-round').disabled=!enabled||takeover;
  $('#capture-observation').disabled=!enabled||takeover;
  const agentConfigEditable=state?.agent?.status==='repair_paused'&&!state?.policy?.active&&!state?.comparison?.active&&!state?.research?.active&&!state?.experiment?.active&&!(state?.benchmark?.active||state?.harnesspai?.active);
  $('#agent-settings').disabled=(takeover&&!agentConfigEditable)||busy;
  if(takeover)document.querySelectorAll('[data-model],#joint-controls input,.sim-action,#gripper-open,#gripper-close,#command,#send,#speed,#gravity,#step,#apply-scene,#scene-select,#task-select,#scene-seed').forEach(el=>el.disabled=true);
  const capabilities=state?.backend?.capabilities;
  if(capabilities){for(const el of document.querySelectorAll('[data-action]'))if(!capabilities.includes(el.dataset.action))el.disabled=true;
    const presets=[...document.querySelectorAll('.sim-action')];
    for(const preset of presets)preset.hidden=!capabilities.includes(preset.dataset.action);
    const presetHeading=$('.tasks-heading');if(presetHeading)presetHeading.hidden=!presets.some(preset=>!preset.hidden);
    for(const name of ['contacts','joints','collision'])if(!capabilities.includes('flags'))$(`#show-${name}`).disabled=true;
    if(!capabilities.includes('gravity'))$('#gravity').disabled=true;
    if(!capabilities.includes('instruction'))for(const id of ['#gripper-open','#gripper-close'])$(id).disabled=true;
    for(const [id,action] of [['#reset','reset'],['#pause','pause'],['#speed','speed'],['#step','step'],['#view-reset','view']])if(!capabilities.includes(action))$(id).disabled=true;
    if(!capabilities.includes('control'))document.querySelectorAll('#joint-controls input').forEach(el=>el.disabled=true);
    document.querySelectorAll('[data-view]').forEach(el=>{if(!capabilities.includes('view'))el.disabled=true;});}
  const discrete=['text','discrete'].includes(state?.backend?.action_mode);
  $('#agent-environment-steps').disabled=!enabled||takeover||!discrete;
  $('#agent-sim-budget').disabled=!enabled||takeover||discrete;
  $('#task-budget-apply').disabled=!enabled||takeover||discrete;
  $('#policy-tab').disabled=discrete;$('#joints-tab').disabled=discrete;
  if(discrete)$('#pause').disabled=true;
  if(catalogLoading||state?.session_id!==catalogSession)document.querySelectorAll('[data-model],#apply-scene,#scene-select,#task-select,#scene-seed,#use-task').forEach(el=>el.disabled=true);
}
async function operate(payload,{message=false}={}){
  if(!connected||busy)return;
  try{
    busy=true;enableControls();
    if(payload.action==='load'){$('#scene-error').hidden=false;$('#scene-error').textContent='正在编译模型与加载原生渲染资源…';}
    const result=await api(payload);await poll();if(message)toast(result.message);return result;
  }catch(error){toast(error.message);return null;}
  finally{busy=false;enableControls();if(connected)$('#scene-error').hidden=true;}
}
const angular=a=>!a.unit||a.unit==='rad';
const percentage=a=>['command','open_fraction'].includes(a.unit);
const unitText=a=>percentage(a)?'%':angular(a)?'°':['normalized','normalized_command','normalized_controller_command'].includes(a.unit)?'':` ${a.unit}`;
const toDisplay=(a,value)=>a.unit==='command'?value/255*100:a.unit==='open_fraction'?value*100:angular(a)?value*180/Math.PI:value;
const fromDisplay=(a,value)=>a.unit==='command'?value/100*255:a.unit==='open_fraction'?value/100:angular(a)?value*Math.PI/180:value;
const targetText=(a,value)=>`${toDisplay(a,value).toFixed(angular(a)||percentage(a)?1:2)}${unitText(a)}`;
const actualText=a=>typeof a.actual!=='number'||!Number.isFinite(a.actual)?'—':a.unit==='command'?`${(a.actual*2*1000).toFixed(1)} mm`:targetText(a,a.actual);
function activeGripper(current){
  const grippers=(current.actuators||[]).filter(a=>a.is_gripper);
  return grippers.find(a=>a.robot_id===current.active_robot)||grippers.find(a=>!a.robot_id)||(!current.active_robot&&grippers.length===1?grippers[0]:null);
}
function renderGripper(current){
  const gripper=activeGripper(current);$('#gripper-status').textContent=gripper?actualText(gripper):'—';
  $('#gripper-detail').textContent=!gripper||['normalized_command','normalized_controller_command'].includes(gripper.unit)?'夹爪测量未提供':gripper.unit==='open_fraction'?'实际开度 · 0% 闭合 / 100% 张开':gripper.unit==='command'?'两指间距（实际）':angular(gripper)?'夹爪关节（实际）':'夹爪位置（实际）';
}

function buildJoints(){
  const root=$('#joint-controls');root.replaceChildren();
  state.actuators.forEach((a,localIndex)=>{
    const row=document.createElement('div');row.className='joint-control';
    const header=document.createElement('div');header.className='joint-header';
    const label=document.createElement('label');label.htmlFor=`joint-${a.index}`;
    const finger=Object.entries({thumb:'拇指',index:'食指',middle:'中指',ring:'无名指',little:'小指'}).find(([key])=>a.name.includes(key));
    const nativeLabel={arm_dx:'末端 ΔX',arm_dy:'末端 ΔY',arm_dz:'末端 ΔZ',arm_drx:'末端旋转 X',arm_dry:'末端旋转 Y',arm_drz:'末端旋转 Z',gripper_command:'夹爪命令',base_vx_command:'底盘前后',base_vy_command:'底盘侧移',base_yaw_command:'底盘旋转',torso_command:'躯干升降',base_mode:'底盘模式'};
    const jointLabel=['normalized_command','normalized_controller_command'].includes(a.unit)?(nativeLabel[a.name]||a.name):finger?`${finger[1]} · ${a.name.split('_').at(-1)}`:a.is_gripper?'夹爪':`关节 ${localIndex+1}`;
    label.innerHTML=`<b>${a.is_gripper?'G':String(a.index+1).padStart(2,'0')}</b><span>${jointLabel}<small>${a.name}</small></span>`;
    const output=document.createElement('output');output.id=`target-${a.index}`;output.textContent=targetText(a,a.target);header.append(label,output);
    const input=document.createElement('input');input.type='range';input.id=`joint-${a.index}`;input.min=String(Math.ceil(toDisplay(a,a.min)*100)/100);input.max=String(Math.floor(toDisplay(a,a.max)*100)/100);input.step=angular(a)||percentage(a)?'.1':'.01';input.value=String(toDisplay(a,a.target));input.setAttribute('aria-label',a.is_gripper?'夹爪目标':`关节 ${a.index+1} 目标`);
    input.setAttribute('aria-label',`${jointLabel} ${angular(a)?'目标角度':'目标值'}`);
    input.addEventListener('input',()=>{output.textContent=targetText(a,fromDisplay(a,Number(input.value)));input.dataset.editing='true';});
    input.addEventListener('change',async()=>{input.dataset.editing='false';await operate({action:'control',index:a.index,value:fromDisplay(a,Number(input.value))});});
    const meta=document.createElement('div');meta.className='joint-meta';meta.innerHTML=`<span>${toDisplay(a,a.min).toFixed(angular(a)?0:2)}${unitText(a)}</span><span id="actual-${a.index}">实际 ${actualText(a)}</span><span>${toDisplay(a,a.max).toFixed(angular(a)?0:2)}${unitText(a)}</span>`;
    row.append(header,input,meta);root.append(row);
  });
  $('#engine-version').textContent=`${state.engine} ${state.version}`;$('#watermark-version').textContent=state.version;
  $('#viewer-engine').textContent=`${state.engine.toUpperCase()} VIEWER`;
  frame.alt=`${state.engine} 原生仿真实时画面`;
  const watermark=$('#watermark-engine');if(watermark)watermark.textContent=state.engine;
  $('#actuator-count').textContent=`${state.nu} 个`;$('#mesh-count').textContent=state.meshes==null?'—':`${state.meshes} 个`;$('#timestep').textContent=['text','discrete'].includes(state.backend?.action_mode)?'按环境动作推进':`${(state.timestep*1000).toFixed(1)} ms`;
    $('#model-source').href=state.source;$('#model-source').textContent='官方模型来源 ↗';$('#scene-name').textContent=state.name;
  document.querySelectorAll('[data-model]').forEach(el=>{const selected=el.dataset.model===state.model;el.classList.toggle('selected',selected);el.setAttribute('aria-pressed',String(selected));});
  $('#gripper-label').textContent=(state.robots||[]).length>1?'当前臂夹爪':percentage(activeGripper(state)||{})?'夹爪开度':'夹爪角度';lastTaskKey='';
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
  if(state.backend?.action_mode==='native'){status.complete='✓ 本次原生动作已执行。'+(state.backend?.engine_id==='omnigibson'?behaviorScore(state.evaluation):'任务成功以环境判定为准。');status.settling='等待原生物理步骤执行完成';}
  $('#task-result').textContent=task.status==='failed'&&task.message?task.message:status[task.status]||task.status;
}
function renderState(){
  const modelContext=JSON.stringify([state.session_id,state.backend?.engine_id,state.model]);
  if(state.generation!==generation||state.active_robot!==activeRobot||modelContext!==renderedModelContext){generation=state.generation;activeRobot=state.active_robot;renderedModelContext=modelContext;lastEvent=0;$('#event-log').replaceChildren();buildJoints();}
  const discrete=['text','discrete'].includes(state.backend?.action_mode);
  const textEnvironment=state.backend?.action_mode==='text';
  $('.scene-column').dataset.actionMode=state.backend?.action_mode||'tcp_pose';
  $('.viewport-panel').hidden=textEnvironment;$('.telemetry').hidden=textEnvironment;
  frame.alt=textEnvironment?'原生文字观察预览（用于视频与回放）':`${state.engine} 原生相机画面`;
  const presets=renderViewPresets(state);
  frame.title=state.backend?.capabilities?.includes('camera')?'左键拖动旋转；右键、中键或 Shift + 左键拖动平移；滚轮缩放。点击预设视角可复位。':textEnvironment?'任务观察与可执行动作显示在环境交互面板。':discrete?'使用环境交互中的原生导航命令改变视角。':presets.length?'通过下方按钮切换真实相机；此环境不提供自由拖拽视角。':'当前为原生固定相机画面。';
  environmentBudget.hidden=!discrete;simulationBudget.hidden=discrete;
  $('#connection').replaceChildren(document.createElement('i'),document.createTextNode(`${state.engine} 已连接`));$('#fps').textContent=`${state.fps.toFixed(0)} FPS`;$('#sim-time').textContent=discrete?`第 ${state.steps} 步`:`t = ${state.time.toFixed(3)} s`;
  const active=['running','settling'].includes(state.task.status);
  $('#scene-status').textContent=state.paused?'仿真已暂停':active?(state.task.steps[state.task.step]||'执行中'):'仿真运行中';
  $('#status-dot').classList.toggle('running',!state.paused);$('#robot-status').innerHTML=`<i></i>${state.paused?'已暂停':active?'执行中':'运行中'}`;
  $('#physics-rate').textContent=discrete?`按原生命令推进 · ${state.steps} 步`:`${(1/state.timestep).toFixed(0)} Hz · ${state.steps} 步`;
  $('#position').textContent=state.tcp.map(v=>v.toFixed(3)).join(' / ');$('#contact-count').textContent=state.contacts??'—';
  const contactLabel=$('#contact-label'),contactScope=$('#contact-scope');
  if(contactLabel)contactLabel.textContent=state.contact_scope==='active_robot'?'机械臂接触':'接触数量';
  if(contactScope)contactScope.textContent=state.contact_scope==='not_reported'?'当前 SDK 未提供接触总数':state.contact_scope==='active_robot'?'当前机械臂的接触点':'全场接触点 · 由物理引擎计算';
  renderGripper(state);
  for(const id of ['#position','#gripper-status','#contact-count'])$(id).closest('.telemetry-cell').hidden=discrete;
  $('.viewer-options').hidden=discrete||!state.backend?.capabilities?.some(name=>['flags','gravity'].includes(name));$('.playback-controls').hidden=discrete;$('.viewport-bottom').hidden=discrete||(!presets.length&&!state.backend?.capabilities?.includes('camera'));
  $('.speed-label').hidden=!!state.backend?.capabilities&&!state.backend.capabilities.includes('speed');
  $('#joints-tab').hidden=discrete;$('#policy-tab').hidden=discrete;
  $('#joints-tab').textContent=state.actuators.some(a=>a.unit==='normalized_command')?'原生命令':'关节控制';
  if(discrete&&$('#agent-tab').getAttribute('aria-selected')!=='true')selectTab('agent');
  $('.agent-contract p').textContent=discrete?'环境观察 → 模型选择原生命令 → 环境反馈与独立验收':state.backend?.action_mode==='native'?'原生相机与整机状态 → 模型输出底盘、躯干和双臂控制 → 物理反馈与 BDDL 验收。动作维度和控制单位由当前机器人声明；交互研究与官方成绩分开记录。':state.backend?.engine_id==='robocasa365'?'视觉 Agent 使用绝对末端位姿并保持底盘；厨房导航与完整移动操作请使用 12 维原生 VLA/WAM 策略或手动命令。':state.backend?.engine_id==='robodojo'?'RoboDojo 交互研究：原生三相机 → 当前控制臂动作 → 官方任务判据。此处交互实验不计作官方完整评测成绩。':'图像 + 机械臂状态 → 绝对末端位姿；本地 IK → 执行器 → 新一轮观察';
  $('[data-stage="observing"]').textContent=discrete?'环境观察':'相机观察';$('[data-stage="executing"]').textContent=discrete||state.backend?.action_mode==='native'?'原生控制':'IK 执行';
  $('#frame-info').textContent=`渲染与物理帧 ${state.render_ms.toFixed(0)} ms`;
  $('#pause').innerHTML=icon(state.paused?'play':'pause')+(state.paused?'继续':'暂停');
  $('#speed').value=String(state.speed);$('#gravity').checked=state.gravity;
  for(const key of ['contacts','joints','collision'])$(`#show-${key}`).checked=state.flags[key];
  const viewNames={perspective:'透视视图',top:'俯视视图',front:'正视视图',wrist:'腕部相机',custom:'自由视图'};$('#view-tag').textContent=presets.find(item=>item.id===state.view)?.label||viewNames[state.view]||'原生相机';
  document.querySelectorAll('[data-view]').forEach(el=>{const selected=el.dataset.view===state.view;el.classList.toggle('selected',selected);el.setAttribute('aria-pressed',String(selected));});
  state.actuators.forEach(a=>{const input=$(`#joint-${a.index}`);if(input.dataset.editing!=='true'){input.value=String(toDisplay(a,a.target));$(`#target-${a.index}`).textContent=targetText(a,a.target);}$(`#actual-${a.index}`).textContent=`实际 ${actualText(a)}`;});
  if(state.events.length&&state.events.at(-1).id!==lastEvent){lastEvent=state.events.at(-1).id;const log=$('#event-log');log.replaceChildren();state.events.slice(-6).reverse().forEach(e=>{const row=document.createElement('p');row.className=e.kind;row.textContent=e.message;log.append(row);});}
  renderTask();renderAgent();enableControls();renderResearch(state);renderRsi(state);renderHarnessPAI(state);renderRepair(state);renderMultiarm(state,{disabled:!connected||busy});renderSimulator(state);renderEnvironment(state);renderBenchmarks(state);renderBehavior(state);
  if(state.session_id&&state.session_id!==catalogSession&&!catalogLoading)reloadWorkspaceCatalog();
}
function renderAgent(){
  const agent=state.agent;if(!agent)return;
  const names={idle:'等待任务',observing:'采集观察',thinking:'模型决策中',executing:'正在执行',repairing:'故障暂停 · 正在诊断',repair_paused:'故障暂停 · 等待处理',retrying:'重试当前轮',verifying:'环境核验中',replan:'未通过核验 · 重新决策',unverified:'未通过环境验收',verified:'评价器确认成功',model_done:'模型自报完成（非验收）',blocked:'模型无法继续',failed:'环境判定任务失败',error:'调用或执行失败',stopped:'接管已停止',round_complete:'本轮已结束',limit:'已达到运行预算'};
  $('#agent-status').textContent=names[agent.status]||agent.status;
  $('#agent-message').textContent=agent.message;
  const roundLimit=Object.hasOwn(agent,'max_rounds')?agent.max_rounds:agentConfig?.max_rounds;
  $('#agent-round').textContent=roundLimit==null?`${agent.round} 轮 · 不限轮数`:`${agent.round} / ${roundLimit} 轮`;
  renderRunTimer($('#agent-run-time'),agent);
  renderRunVideo($('#agent-video'),agent.run_id,agent.video);
  document.querySelectorAll('[data-stage]').forEach(el=>el.classList.toggle('active',el.dataset.stage===agent.status));
  const action=agent.action;$('#agent-action').hidden=!action;
  if(action){const vector=Array.isArray(action.action),native=vector||typeof action.action==='string';$('#agent-action dl').hidden=native;
    $('#agent-action .section-caption').textContent=vector?'下一步整机原生动作':native?'下一步原生环境动作':'下一步绝对末端位姿 · WORLD';
    if(!native){$('#action-position').textContent=action.position_m.map(x=>x.toFixed(4)).join(' / ');$('#action-quaternion').textContent=action.quaternion_wxyz.map(x=>x.toFixed(3)).join(' / ');$('#action-gripper').textContent=`张开 ${(action.gripper_open*100).toFixed(0)}% · ${action.duration_s.toFixed(1)} s`;}
    $('#action-explanation').textContent=vector?`${action.action.map((value,index)=>`${state.action_space?.names?.[index]||index}: ${value.toFixed(4)}`).join(' · ')}\n执行 ${action.steps} 个控制周期\n${action.explanation}`:native?`${action.action}\n${action.explanation}`:action.explanation;}
  renderRoundTimeline($('#agent-history'),agent);
  const sourceName=state.policy?.active||($('#policy-tab').getAttribute('aria-selected')==='true'&&state.policy?.observation_time!=null)?'policy':'agent';
  const source=state[sourceName],observationKey=`${state.session_id}:${state.backend?.engine_id}:${sourceName}:${source?.observation_id}:${source?.observation_robot}`;
  if(observationKey!==lastObservation||source?.observation_robot!==state.model){
    lastObservation=observationKey;const valid=source?.observation_time!=null&&source.observation_robot===state.model;
    $('#observation-time').textContent=valid?`${['text','discrete'].includes(state.backend?.action_mode)?`第 ${source.observation_time} 步`:`t = ${source.observation_time.toFixed(3)} s`} · 第 ${source.observation_id} 次采集`:'尚未采集当前环境';
    const mode=state.backend?.action_mode;
    const names=source?.observation_cameras?.length?source.observation_cameras:(Array.isArray(state.observation_cameras)?state.observation_cameras:mode==='text'?[]:mode==='discrete'?['front']:['front','side','wrist']);
    $('#camera-title').textContent=`Agent 的原生相机观察 · ${names.length} 路`;
    $('#camera-description').textContent=mode==='discrete'?'发送当前 Unity 原生相机图像与公开环境观察。':'每轮发送同一物理时刻的原生相机图像与机械臂状态。相机名称保留所属位置，画面在下一轮观察时更新。';
    $('#camera-panel').hidden=mode==='text';renderObservationCameras(names,sourceName,source,valid);
  }
}
function disconnected(message,{serviceReachable=false}={}){
  const status=serviceReachable?'仿真环境未就绪':'仿真服务未连接';
  connected=false;state={...state,ready:false};$('#connection').textContent=status;$('#scene-error').hidden=false;$('#scene-error').replaceChildren();
  $('.viewport-panel').hidden=false;
  const title=document.createElement('strong');title.textContent=status;const detail=document.createElement('p');detail.textContent=message;
  const help=document.createElement('p');help.textContent=serviceReachable?'仿真服务仍在线。可在左侧“仿真环境”中点击“重新加载环境”，或选择其他已安装环境。':'请启动项目中的仿真服务，再点击下方按钮重新连接。';const retry=document.createElement('button');retry.textContent=serviceReachable?'刷新环境状态':'重新连接';retry.className='secondary';retry.addEventListener('click',()=>{frameStarted=false;poll();});$('#scene-error').append(title,detail,help,retry);
  frame.removeAttribute('src');frameStarted=false;$('#robot-status').textContent=serviceReachable?'未就绪':'离线';$('#scene-status').textContent=serviceReachable?'引擎未就绪':'引擎离线';$('#status-dot').classList.remove('running');enableControls();
  renderResearch(state);renderRsi(state);renderHarnessPAI(state);renderMultiarm(state,{disabled:true});renderEnvironment(state);renderSimulator(state);renderBenchmarks(state);renderBehavior(state);
}
let pollPromise=null,lastPollAt=0,lastPollError='';
function poll(){
  if(pollPromise)return pollPromise;
  pollPromise=pollState().finally(()=>{pollPromise=null;});return pollPromise;
}
async function pollState(){
  lastPollAt=Date.now();
  try{
    const response=await fetch('/api/state',{signal:AbortSignal.timeout(4000)});if(!response.ok)throw new Error(`本地服务返回 ${response.status}`);
    const next=await response.json();
    if(!next.ready){
      state=next;renderSimulator(next);
      if(next.model&&next.task&&next.actuators){renderAgent();renderResearch(next);renderRsi(next);renderHarnessPAI(next);renderRepair(next);}
      lastPollError=next.error||'环境仍在初始化，请等待完成。';
      disconnected(lastPollError,{serviceReachable:true});return;
    }
    state=next;connected=true;if(!frameStarted){frame.src=`/api/stream?session=${Date.now()}`;frameStarted=true;}
    $('#scene-error').hidden=true;renderState();lastPollError='';
  }catch(error){lastPollError=error.message;disconnected(error.message);}
}
document.querySelectorAll('[data-model]').forEach(el=>el.addEventListener('click',()=>{if(state?.model!==el.dataset.model)operate({action:'load',model:el.dataset.model});}));
document.querySelectorAll('.sim-action').forEach(el=>el.addEventListener('click',()=>operate({action:el.dataset.action})));
$('#reset').addEventListener('click',()=>operate({action:'reset'}));$('#pause').addEventListener('click',()=>operate({action:'pause',value:!state.paused}));$('#step').addEventListener('click',()=>operate({action:'step'}));$('#stop').addEventListener('click',()=>operate({action:'stop'}));
$('#speed').addEventListener('change',e=>operate({action:'speed',value:Number(e.target.value)}));$('#gravity').addEventListener('change',e=>operate({action:'gravity',value:e.target.checked}));
for(const name of ['contacts','joints','collision'])$(`#show-${name}`).addEventListener('change',e=>operate({action:'flags',flags:{[name]:e.target.checked}}));
$('#view-reset').addEventListener('click',()=>{const preset=viewPresets(state)[0];if(preset)operate({action:'view',view:preset.id});});
$('#gripper-open').addEventListener('click',()=>operate({action:'instruction',text:'张开夹爪'}));$('#gripper-close').addEventListener('click',()=>operate({action:'instruction',text:'闭合夹爪'}));
function selectTab(name){for(const item of ['joints','agent','policy']){$(`#${item}-tab`).setAttribute('aria-selected',String(item===name));$(`#${item}-tab`).classList.toggle('selected',item===name);$(`#${item}-panel`).hidden=item!==name;}}
$('#joints-tab').addEventListener('click',()=>selectTab('joints'));$('#agent-tab').addEventListener('click',()=>selectTab('agent'));
async function submit(event,oneRound=false){
  event?.preventDefault();if(!connected||busy)return;
  const text=$('#command').value.trim();if(!text){toast('请输入一个任务。');return;}
  const maxEnvironmentSteps=Number($('#agent-environment-steps').value),maxSimTime=Number($('#agent-sim-budget').value);
  if(!['text','discrete'].includes(state?.backend?.action_mode)&&(!Number.isFinite(maxSimTime)||maxSimTime<1)){toast('仿真预算须为不小于 1 的有限秒数。');return;}
  if(['text','discrete'].includes(state?.backend?.action_mode)&&(!Number.isInteger(maxEnvironmentSteps)||maxEnvironmentSteps<1||maxEnvironmentSteps>10000)){toast('环境动作预算须为 1–10000 的整数。');return;}
  busy=true;enableControls();
  try{await agentApi('start',{task:text,one_round:oneRound,record_video:$('#agent-record-video').checked,...(['text','discrete'].includes(state?.backend?.action_mode)?{max_environment_steps:maxEnvironmentSteps}:{max_sim_time_s:maxSimTime})});await poll();}
  catch(error){toast(error.message);}
  finally{busy=false;enableControls();}
}
$('#command-form').addEventListener('submit',submit);$('#command').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey&&!e.isComposing){e.preventDefault();submit();}});
$('#help').addEventListener('click',()=>$('#help-dialog').showModal());$('#close-help').addEventListener('click',()=>$('#help-dialog').close());
$('#agent-one-round').addEventListener('click',()=>submit(null,true));
$('#example-task').addEventListener('click',()=>{setTaskDraft($('#command'),state?.evaluation?.instruction||'');$('#command').focus();});
$('#agent-stop').addEventListener('click',async()=>{try{await agentApi('stop',{});await poll();}catch(error){toast(error.message);}});
$('#capture-observation').addEventListener('click',async()=>{busy=true;enableControls();try{await agentApi('observe',{});await poll();}catch(error){toast(error.message);}finally{busy=false;enableControls();}});
function updateConfigSummary(){if(!agentConfig)return;$('#agent-model-name').textContent=agentConfig.model;$('#agent-provider-name').textContent='模型连接：'+providerNames[agentConfig.provider];$('#agent-harness-name').textContent='Harness：'+harnessNames[effectiveHarness(agentConfig)];$('#agent-run-budget').textContent=agentConfig.max_rounds==null?'不限制轮数':`最多 ${agentConfig.max_rounds} 轮`;}
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
  const provider=$('#config-provider').value,harness=$('#config-harness').value||effectiveHarness({provider}),external=harness==='http';
  const mismatch=harnessCompatibility(harness,provider)||harnessReasoningError(harness,selectedReasoning(),{provider,baseUrl:$('#config-base-url').value,thinking:$('#config-deepseek-thinking').value});
  $('#harness-compatibility').textContent=mismatch;$('#harness-compatibility').hidden=!mismatch;
  $('#apply-agent-config').disabled=!!mismatch||capabilityLoading;
  $('#harness-http-fields').hidden=!external;
  $('#harness-http-fields').querySelectorAll('input').forEach(input=>input.disabled=!external);
  $('#config-forward-model-key').disabled=!external||provider==='custom';
  if(!external||provider==='custom')$('#config-forward-model-key').checked=false;
  $('#config-harness-url').required=external;
  $('#harness-help').textContent={
    workbench:'工作台组织观察、模型请求、动作执行与反馈。模型 API 直接由本机后端调用。',
    codex_cli:'每轮通过 Codex CLI 的临时只读会话生成结构化动作，禁用 shell、插件和多代理工具。物理执行、停止与成功验收仍由工作台管理。',
    claude_code:'调用官方 Claude Code CLI 的 bare 模式，使用三路图像及结构化输出；不加载个人工具、插件或订阅登录。每次返回一个动作，内部结构化输出可能包含多次模型请求，CLI 汇总用量单独记录。',
    dsh:'调用 DeepSeek 官方 Harness 的 SDK Agent，支持图像及独立模型 API。使用隔离配置，不启用本机文件、终端或其他外部工具；动作执行与任务验收由工作台管理。',
    http:'外部 harness 接收 embodied-agent-v1 请求，可在一次请求内编排自己的工具与模型，返回一个符合当前环境契约的动作。物理执行与任务验收仍由工作台管理。'
  }[harness];
  $('#harness-forward-help').textContent=$('#config-forward-model-key').checked?`执行实验时，模型 API Key 将随请求发送到 ${$('#config-harness-url').value||'所填 harness 地址'}；harness 鉴权使用另一份密钥。`:'默认只发送模型协议、地址、名称和推理参数；模型密钥由 harness 服务自行配置。';
  let deepseek=false;
  try{deepseek=provider==='compatible'&&new URL($('#config-base-url').value).hostname==='api.deepseek.com';}catch{}
  const managed=['codex','custom'].includes(provider),cli=harness==='codex_cli';
  $('#api-config-fields').hidden=managed;
  $('#api-budget-fields').hidden=cli;
  $('#api-config-fields').querySelectorAll('input').forEach(input=>input.disabled=managed);
  $('#config-output-tokens').disabled=cli;
  $('#output-budget-help').hidden=cli;
  $('#deepseek-thinking-field').hidden=!deepseek;
  $('#reasoning-field').hidden=false;
  $('#config-reasoning').disabled=capabilityLoading||(deepseek&&$('#config-deepseek-thinking').value==='disabled');
  reasoningFields();
  $('#provider-help').textContent=provider==='codex'?'使用本机 Codex 的现有登录，无需 API Key；模型名称仍可独立选择，每轮计入 Codex 订阅用量。':provider==='custom'?'模型名称和参数交给外部 harness；实际模型连接与凭据由该服务管理，工作台不会用模型地址直接推理。':'填写模型服务的准确模型 ID、API 地址与 API Key，可接入支持相应协议的路由服务。密钥仅存在服务内存中。';
  if(deepseek)$('#provider-help').textContent='deepseek-flash 支持图像输入。思考模式和推理强度会传给 DeepSeek 官方接口。API Key 仅存于本机服务内存，重启后需重新填写。';
  else if(provider==='compatible')$('#provider-help').textContent='适用于支持多图输入和 JSON 输出的 OpenAI 兼容模型服务。模型 ID 可自定义；密钥仅保存在本机服务内存。';
}
$('#config-base-url').addEventListener('input',()=>{providerFields();scheduleModelCapabilities();});
$('#config-harness').addEventListener('change',providerFields);
$('#config-harness-url').addEventListener('input',()=>{$('#config-forward-model-key').checked=false;providerFields();});
$('#config-forward-model-key').addEventListener('change',providerFields);
$('#config-model').addEventListener('input',scheduleModelCapabilities);
$('#config-api-key').addEventListener('change',scheduleModelCapabilities);
$('#config-reasoning').addEventListener('change',providerFields);
$('#config-custom-reasoning').addEventListener('input',providerFields);
$('#refresh-model-capabilities').addEventListener('click',()=>refreshModelCapabilities());
$('#config-deepseek-thinking').addEventListener('change',providerFields);
$('#config-provider').addEventListener('change',()=>{const provider=$('#config-provider').value,current=$('#config-base-url').value;const defaults=['https://api.openai.com/v1','https://api.anthropic.com','https://api.deepseek.com'];if(!current||defaults.includes(current))$('#config-base-url').value=provider==='anthropic'?'https://api.anthropic.com':provider==='compatible'?'https://api.deepseek.com':'https://api.openai.com/v1';$('#config-api-key').value='';$('#config-model').placeholder=provider==='custom'?'填写服务约定的模型标识':'填写服务实际提供的模型 ID';providerFields();refreshModelCapabilities(null);});
$('#agent-settings').addEventListener('click',async()=>{try{
  agentConfig=await agentApi('config');const c=agentConfig;
  $('#config-harness').value=effectiveHarness(c);$('#config-harness-url').value=c.harness_url||(c.provider==='custom'?c.base_url:'http://127.0.0.1:8020/agent');
  $('#config-harness-api-key').value='';$('#config-harness-api-key').placeholder=c.has_harness_api_key||(!c.harness&&c.provider==='custom'&&c.has_api_key)?'当前 harness 已有密钥，留空保留':'本机接口可不填；远程接口使用独立密钥';
  $('#config-forward-model-key').checked=c.forward_model_key===true;
  $('#config-provider').value=c.provider;$('#config-model').value=c.model;$('#config-base-url').value=c.base_url;
  $('#config-deepseek-thinking').value=c.deepseek_thinking??'enabled';$('#config-output-tokens').value=c.max_output_tokens??16384;
  $('#config-rounds').value=c.max_rounds??'';$('#config-timeout').value=c.timeout_s;$('#config-api-key').value='';
  $('#config-api-key').placeholder=c.has_api_key?'当前模型地址已有密钥，留空保留':'仅保留在本机服务内存中';
  $('#config-error').hidden=true;providerFields();$('#agent-config-dialog').showModal();refreshModelCapabilities(c.reasoning);
}catch(error){toast(error.message);}});
$('#close-agent-config').addEventListener('click',()=>$('#agent-config-dialog').close());
$('#agent-config-form').addEventListener('submit',async event=>{event.preventDefault();try{
  if(capabilityLoading)throw new Error('模型能力正在读取，请稍候再应用。');
  const harness=$('#config-harness').value,provider=$('#config-provider').value,error=harnessCompatibility(harness,provider)||harnessReasoningError(harness,selectedReasoning(),{provider,baseUrl:$('#config-base-url').value,thinking:$('#config-deepseek-thinking').value});if(error)throw new Error(error);
  agentConfig=await agentApi('config',{harness,provider,model:$('#config-model').value.trim(),base_url:$('#config-base-url').value.trim(),api_key:$('#config-api-key').value,
    harness_url:harness==='http'?$('#config-harness-url').value.trim():'',harness_api_key:harness==='http'?$('#config-harness-api-key').value:'',forward_model_key:harness==='http'&&$('#config-forward-model-key').checked,
    reasoning:selectedReasoning(),reasoning_native:true,deepseek_thinking:$('#config-deepseek-thinking').value,
    max_output_tokens:$('#config-output-tokens').disabled?(agentConfig?.max_output_tokens??16384):Number($('#config-output-tokens').value),max_rounds:Number($('#config-rounds').value)||null,timeout_s:Number($('#config-timeout').value)});
  $('#config-api-key').value='';$('#config-harness-api-key').value='';updateConfigSummary();$('#agent-config-dialog').close();toast('Harness 与模型配置已应用。');
}catch(error){$('#config-error').textContent=error.message;$('#config-error').hidden=false;}});
agentApi('config').then(config=>{agentConfig=config;updateConfigSummary();}).catch(()=>{});
initBenchmarks({toast});
initBehavior({toast});
initResearch({operate,toast,selectTab,getState:()=>state,catalogReady:()=>!catalogLoading&&state?.session_id===catalogSession});
initRsi({operate,toast,selectTab,getState:()=>state,catalogReady:()=>!catalogLoading&&state?.session_id===catalogSession});
initHarnessPAI({toast,getState:()=>state});
initRepair({toast});
initMultiarm({operate});
initEnvironment({execute:async payload=>{if(!connected||busy)throw new Error('工作台正在处理其他操作，请稍后重试。');busy=true;enableControls();try{const result=await api(payload);await poll();return result;}finally{busy=false;enableControls();}}});
initLayout({toast});
initSimulators({onOpenBenchmark:()=>document.querySelector('[data-page="benchmarks"]').click(),onChanged:async result=>{
  generation=-1;lastObservation=-1;await poll();
  if(state?.session_id!==result.session_id)await poll();
  if(!connected||state?.session_id!==result.session_id)throw new Error(lastPollError||'新环境的状态尚未同步，请重新连接');
  if(!await reloadWorkspaceCatalog())throw new Error('机器人与场景目录读取失败');
}});
// Camera gestures modify the active engine's camera. There is no client-side 3D simulation.
let pointer=null,cameraPending={dx:0,dy:0,zoom:0,pan_dx:0,pan_dy:0},cameraSending=false;
frame.title='左键拖动旋转；右键、中键或 Shift + 左键拖动平移；滚轮缩放。点击预设视角可复位。';
async function flushCamera(){
  if(state?.backend?.capabilities&&!state.backend.capabilities.includes('camera')){cameraPending={dx:0,dy:0,zoom:0,pan_dx:0,pan_dy:0};return;}
  if(cameraSending||!connected||busy||!Object.values(cameraPending).some(Boolean))return;
  cameraSending=true;const payload=cameraPending;cameraPending={dx:0,dy:0,zoom:0,pan_dx:0,pan_dy:0};
  try{await api({action:'camera',dx:Math.max(-180,Math.min(180,payload.dx)),dy:Math.max(-180,Math.min(180,payload.dy)),zoom:Math.max(-1,Math.min(1,payload.zoom)),pan_dx:Math.max(-2,Math.min(2,payload.pan_dx)),pan_dy:Math.max(-2,Math.min(2,payload.pan_dy))});}
  catch(error){toast(error.message);}
  finally{cameraSending=false;if(Object.values(cameraPending).some(Boolean))flushCamera();}
}
frame.addEventListener('contextmenu',e=>e.preventDefault());
frame.addEventListener('pointerdown',e=>{if(!connected||state?.backend?.capabilities&&!state.backend.capabilities.includes('camera')||![0,1,2].includes(e.button))return;e.preventDefault();pointer={id:e.pointerId,x:e.clientX,y:e.clientY,pan:e.button!==0||e.shiftKey};frame.setPointerCapture(e.pointerId);});
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
frame.addEventListener('wheel',e=>{if(!connected||state?.backend?.capabilities&&!state.backend.capabilities.includes('camera'))return;e.preventDefault();cameraPending.zoom+=e.deltaY*.001;flushCamera();},{passive:false});
frame.addEventListener('error',()=>{frameStarted=false;if(connected){$('#scene-error').hidden=false;$('#scene-error').textContent='原生画面连接中断，正在重新连接。';}});
poll();setInterval(()=>{if(!busy&&(connected||Date.now()-lastPollAt>2000))poll();},250);
