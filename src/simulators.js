const $=selector=>document.querySelector(selector);
let context={},catalog=null,lastState=null,activeId='',loading=false,refreshing=false,initialized=false,lastReadiness='';

const isBenchmark=backend=>backend?.id==='robodojo'&&backend.kind==='benchmark';
const statusName=backend=>isBenchmark(backend)?'独立评测入口':backend.status==='available'?'可加载':backend.status==='unavailable'?'运行环境异常':backend.status==='not_ready'?'待准备':backend.status==='not_installed'?'未安装':backend.status==='planned'?'尚未适配':'状态未知';
const canLoad=backend=>backend?.status==='available'&&backend.implemented!==false&&backend.installed!==false;
// Unavailable interactive adapters can still be inspected without loading them.
const canSelect=backend=>!!backend&&(backend.implemented!==false||isBenchmark(backend));
const isRunning=state=>!!(state?.backend_switching||state?.agent?.active||state?.policy?.active||state?.comparison?.active||state?.research?.active||state?.experiment?.active||(state?.benchmark?.active||state?.harnesspai?.active)||['running','settling'].includes(state?.task?.status));
const asText=value=>Array.isArray(value)?value.join(' · '):value==null?'':String(value);
const unavailable=()=>lastState?.ready===false;

function notice(message,error=false,diagnostic=''){
  const node=$('#simulator-message');
  node.textContent=message;node.classList.toggle('error',error);node.setAttribute('role',error?'alert':'status');
  const details=$('#simulator-diagnostic');
  details.hidden=!diagnostic;details.open=false;$('#simulator-diagnostic-text').textContent=diagnostic;
}

function reportFailure(prefix,error,suffix=''){
  const diagnostic=error.diagnostic||error.message||String(error);
  const beforeTrace=diagnostic.split('Traceback (most recent call last):')[0].trim();
  const firstLine=(beforeTrace||error.message||'服务返回错误，请查看完整诊断。').split(/\r?\n/)[0];
  const summary=firstLine.length>260?firstLine.slice(0,260)+'…':firstLine;
  notice(`${prefix}${summary}${suffix}`,true,diagnostic);
}

async function request(path,options){
  const response=await fetch(path,options);
  let payload;
  try{payload=await response.json();}catch{throw new Error(`仿真服务返回 ${response.status}，未提供有效响应。`);}
  if(!response.ok){
    const detail=payload.detail||payload.error||payload.message;
    const error=new Error(typeof detail==='string'?detail:detail?.message||`仿真服务返回 ${response.status}。`);
    error.diagnostic=typeof detail==='string'?detail:detail?JSON.stringify(detail,null,2):error.message;
    throw error;
  }
  return payload;
}

function docsLink(url,label){
  if(!url)return null;
  let parsed;
  try{parsed=new URL(url);}catch{return null;}
  if(!['https:','http:'].includes(parsed.protocol))return null;
  const link=document.createElement('a');link.href=parsed.href;link.textContent=label;link.target='_blank';link.rel='noopener noreferrer';
  return link;
}

function integrationDetails(backend){
  const root=document.createElement('div');root.className='simulator-integration';
  const status=document.createElement('p');
  const integration=backend.integration||{};
  const validated={runtime_preflight_only:'依赖预检完成 · 原生运行待验收',not_recorded:'尚未记录原生验收'};
  status.textContent=validated[integration.native_validation]||'原生验收范围尚未报告';root.append(status);
  if(integration.mode==='interactive_research'){const mode=document.createElement('p');mode.textContent='交互研究环境';root.append(mode);}
  if(integration.note){const note=document.createElement('p');note.textContent=asText(integration.note);root.append(note);}
  const capabilities=backend.capabilities||[];
  const labels={observe:'观察',move_tcp:'末端动作',native_action:'整机原生动作',policy_chunk:'策略动作',environment_action:'环境命令',verify_task:'原生评价',checkpoint:'状态记录',restore:'状态恢复',replay_frame:'轨迹回放',camera:'自由视角',view:'预设相机'};
  const supported=Object.entries(labels).filter(([key])=>capabilities.includes(key)).map(([,label])=>label);
  if(supported.length){const features=document.createElement('p');features.textContent='接口支持：'+supported.join(' · ');root.append(features);}
  const limitations=Array.isArray(integration.limitations)?integration.limitations:[];
  if(limitations.length){const list=document.createElement('ul');for(const text of limitations){const item=document.createElement('li');item.textContent=String(text);list.append(item);}root.append(list);}
  return root;
}

function selectedReadiness(backend){
  const signature=JSON.stringify([backend,!!context.onOpenBenchmark]);if(signature===lastReadiness)return;lastReadiness=signature;
  const benchmark=$('#simulator-benchmark');benchmark.hidden=backend?.id!=='robodojo';benchmark.disabled=!context.onOpenBenchmark;
  const root=$('#simulator-readiness');root.replaceChildren();root.hidden=!backend;if(!backend)return;
  root.append(integrationDetails(backend));
  const readiness=backend.readiness;
  if(readiness?.message){const message=document.createElement('p');message.textContent=asText(readiness.message);root.append(message);}
  if(Array.isArray(readiness?.checks)){
    const list=document.createElement('ul');list.className='simulator-readiness-checks';
    for(const check of readiness.checks){const item=document.createElement('li');item.className=check.status==='passed'?'passed':'missing';item.textContent=`${check.status==='passed'?'✓':'待准备'} ${asText(check.label)}${check.detail?'：'+asText(check.detail):''}`;list.append(item);}
    root.append(list);
  }
}

function renderCatalog(){
  const select=$('#simulator-select'),previous=select.value;
  select.replaceChildren();$('#simulator-catalog').replaceChildren();
  const backends=catalog?.backends||[];
  for(const backend of backends){
    const option=document.createElement('option');option.value=backend.id;
    option.textContent=`${backend.name||backend.id} · ${backend.id===activeId?'当前环境':statusName(backend)}`;
    option.disabled=!canSelect(backend);select.append(option);
    const row=document.createElement('article');row.className='simulator-catalog-item';
    const heading=document.createElement('div'),name=document.createElement('strong'),badge=document.createElement('span');
    name.textContent=backend.name||backend.id;badge.textContent=statusName(backend);badge.className=`simulator-status simulator-status-${['available','not_installed','planned'].includes(backend.status)?backend.status:'unknown'}`;
    heading.append(name,badge);row.append(heading);
    const description=document.createElement('p');description.textContent=asText(backend.description);row.append(description);
    if(backend.requirements){const requirement=document.createElement('p');requirement.className='simulator-requirements';requirement.textContent=asText(backend.requirements);row.append(requirement);}
    if(backend.error){const error=document.createElement('p');error.className='simulator-message error';error.textContent=asText(backend.error);row.append(error);}
    row.append(integrationDetails(backend));
    const docs=docsLink(backend.docs,'官方文档 ↗');if(docs)row.append(docs);
    $('#simulator-catalog').append(row);
  }
  select.value=backends.some(item=>item.id===previous&&canSelect(item))?previous:activeId;
  if(!select.value&&backends.length)select.value=backends[0].id;
  updateControls();
}

function updateControls(){
  const select=$('#simulator-select'),backend=catalog?.backends.find(item=>item.id===select.value);
  const blocked=isRunning(lastState),switching=loading||!!lastState?.backend_switching,pending=switching||refreshing;
  select.disabled=pending||blocked||!catalog?.backends.length;
  $('#simulator-load').disabled=pending||blocked||!(isBenchmark(backend)?context.onOpenBenchmark:canLoad(backend))||(backend?.id===activeId&&!unavailable());
  $('#simulator-refresh').disabled=pending;
  $('#simulator-card').setAttribute('aria-busy',String(pending));
  $('#simulator-load').textContent=switching?'正在加载…':isBenchmark(backend)?'打开官方评测':backend?.id===activeId?(unavailable()?'重新加载环境':'当前环境'):canLoad(backend)?'加载交互环境':'运行环境未就绪';
  const current=catalog?.backends.find(item=>item.id===activeId);
  $('#simulator-current').textContent=current?`${current.name||current.id}${current.version?` ${current.version}`:''}`:activeId||'等待读取';
  $('#simulator-description').textContent=backend?asText(backend.description):'从服务读取已适配环境及安装状态。';
  selectedReadiness(backend);
  $('#simulator-lock').hidden=!blocked;
  $('#simulator-lock').textContent=switching?'正在切换环境，请等待加载完成。':'任务执行中，停止后可切换环境。';
}

async function refreshCatalog({announce=true}={}){
  if(loading||refreshing)return;
  refreshing=true;updateControls();
  if(announce)notice('正在检查仿真环境与依赖…');
  try{
    const next=await request('/api/simulators');
    if(!Array.isArray(next.backends)||!next.backends.every(item=>item&&typeof item.id==='string'))throw new Error('仿真服务未返回有效的环境目录。');
    catalog=next;
    activeId=next.active||lastState?.backend?.engine_id||activeId;
    renderCatalog();
    notice(announce?'环境目录已刷新。':'选择环境后点击加载；切换会重置当前仿真。');
  }catch(error){reportFailure('读取环境目录失败：',error);}
  finally{refreshing=false;updateControls();}
}

async function selectSimulator(){
  if(loading||refreshing)return;
  if(lastState?.backend_switching){notice('正在切换环境，请等待加载完成。');return;}
  if(isRunning(lastState)){notice('当前任务仍在执行，请停止任务后切换仿真环境。',true);return;}
  const target=$('#simulator-select').value,backend=catalog?.backends.find(item=>item.id===target);
  if(isBenchmark(backend)){
    if(context.onOpenBenchmark){context.onOpenBenchmark();notice('已打开 RoboDojo 独立评测流程；当前仿真环境保持不变。');}
    return;
  }
  if(!canLoad(backend)||(target===activeId&&!unavailable()))return;
  loading=true;let loaded=false;updateControls();
  notice(target==='genesis'?`正在加载 ${backend.name||target}；首次 CPU 内核编译可能需要数分钟，请等待完成…`:`正在加载 ${backend.name||target}，请等待完成…`);
  try{
    const result=await request('/api/simulators/select',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({engine_id:target})});
    activeId=result.backend?.engine_id||result.active||target;
    loaded=true;
    if(catalog)catalog.active=activeId;
    renderCatalog();$('#simulator-select').value=activeId;
    notice(`已加载 ${backend.name||activeId}，正在同步机器人与场景…`);
    await context.onChanged?.(result);
    notice(`已加载 ${backend.name||activeId}，机器人与场景范围已更新。`);
  }catch(error){
    $('#simulator-select').value=activeId;
    reportFailure(loaded?'环境已加载，但界面同步失败：':'环境切换未完成：',error,loaded?'。请刷新页面重新同步。':'。请查看完整诊断，刷新目录后重试。');
  }finally{loading=false;updateControls();}
}

export function initSimulators(callbacks={}){
  context=callbacks;
  if(!initialized){
    initialized=true;
    $('#simulator-select').addEventListener('change',updateControls);
    $('#simulator-refresh').addEventListener('click',()=>refreshCatalog());
    $('#simulator-load').addEventListener('click',selectSimulator);
    $('#simulator-benchmark').addEventListener('click',()=>context.onOpenBenchmark?.());
  }
  return refreshCatalog({announce:false});
}

export function renderSimulator(state){
  lastState=state;
  const next=state?.backend?.engine_id||state?.engine_id||state?.engine;
  if(typeof next==='string'&&next!==activeId){
    activeId=next;
    if(catalog){catalog.active=next;renderCatalog();$('#simulator-select').value=next;}
  }
  updateControls();
}
