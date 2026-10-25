const $=selector=>document.querySelector(selector);
let context={},catalog=null,lastState=null,activeId='',loading=false,refreshing=false,initialized=false;

const statusName=backend=>backend.status==='available'?'可加载':backend.status==='not_installed'?'未安装':backend.status==='planned'?'尚未适配':'状态未知';
const canLoad=backend=>backend?.status==='available'&&backend.implemented!==false&&backend.installed!==false;
const isRunning=state=>!!(state?.backend_switching||state?.agent?.active||state?.policy?.active||state?.comparison?.active||state?.research?.active||state?.experiment?.active||state?.benchmark?.active||['running','settling'].includes(state?.task?.status));
const asText=value=>Array.isArray(value)?value.join(' · '):value==null?'':String(value);
const unavailable=()=>lastState?.ready===false;

function notice(message,error=false){
  const node=$('#simulator-message');
  node.textContent=message;node.classList.toggle('error',error);node.setAttribute('role',error?'alert':'status');
}

async function request(path,options){
  const response=await fetch(path,options);
  let payload;
  try{payload=await response.json();}catch{throw new Error(`仿真服务返回 ${response.status}，未提供有效响应。`);}
  if(!response.ok){
    const detail=payload.detail||payload.error||payload.message;
    throw new Error(typeof detail==='string'?detail:detail?.message||`仿真服务返回 ${response.status}。`);
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

function renderCatalog(){
  const select=$('#simulator-select'),previous=select.value;
  select.replaceChildren();$('#simulator-catalog').replaceChildren();
  const backends=catalog?.backends||[];
  for(const backend of backends){
    const option=document.createElement('option');option.value=backend.id;
    option.textContent=`${backend.name||backend.id} · ${backend.id===activeId?'当前环境':statusName(backend)}`;
    option.disabled=!canLoad(backend);select.append(option);
    const row=document.createElement('article');row.className='simulator-catalog-item';
    const heading=document.createElement('div'),name=document.createElement('strong'),badge=document.createElement('span');
    name.textContent=backend.name||backend.id;badge.textContent=statusName(backend);badge.className=`simulator-status simulator-status-${['available','not_installed','planned'].includes(backend.status)?backend.status:'unknown'}`;
    heading.append(name,badge);row.append(heading);
    const description=document.createElement('p');description.textContent=asText(backend.description);row.append(description);
    if(backend.requirements){const requirement=document.createElement('p');requirement.className='simulator-requirements';requirement.textContent=asText(backend.requirements);row.append(requirement);}
    const docs=docsLink(backend.docs,'官方文档 ↗');if(docs)row.append(docs);
    $('#simulator-catalog').append(row);
  }
  select.value=backends.some(item=>item.id===previous&&canLoad(item))?previous:activeId;
  if(!select.value&&backends.length)select.value=backends[0].id;
  updateControls();
}

function updateControls(){
  const select=$('#simulator-select'),backend=catalog?.backends.find(item=>item.id===select.value);
  const blocked=isRunning(lastState),switching=loading||!!lastState?.backend_switching,pending=switching||refreshing;
  select.disabled=pending||blocked||!catalog?.backends.length;
  $('#simulator-load').disabled=pending||blocked||!canLoad(backend)||(backend.id===activeId&&!unavailable());
  $('#simulator-refresh').disabled=pending;
  $('#simulator-card').setAttribute('aria-busy',String(pending));
  $('#simulator-load').textContent=switching?'正在加载…':backend?.id===activeId?(unavailable()?'重新加载环境':'当前环境'):'加载环境';
  const current=catalog?.backends.find(item=>item.id===activeId);
  $('#simulator-current').textContent=current?`${current.name||current.id}${current.version?` ${current.version}`:''}`:activeId||'等待读取';
  $('#simulator-description').textContent=backend?asText(backend.description):'从服务读取已适配环境及安装状态。';
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
  }catch(error){notice(`读取环境目录失败：${error.message}`,true);}
  finally{refreshing=false;updateControls();}
}

async function selectSimulator(){
  if(loading||refreshing)return;
  if(lastState?.backend_switching){notice('正在切换环境，请等待加载完成。');return;}
  if(isRunning(lastState)){notice('当前任务仍在执行，请停止任务后切换仿真环境。',true);return;}
  const target=$('#simulator-select').value,backend=catalog?.backends.find(item=>item.id===target);
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
    notice(loaded?`环境已加载，但界面同步失败：${error.message}。请刷新页面重新同步。`:`环境切换未完成：${error.message}。请刷新目录确认当前环境后重试。`,true);
  }finally{loading=false;updateControls();}
}

export function initSimulators(callbacks={}){
  context=callbacks;
  if(!initialized){
    initialized=true;
    $('#simulator-select').addEventListener('change',updateControls);
    $('#simulator-refresh').addEventListener('click',()=>refreshCatalog());
    $('#simulator-load').addEventListener('click',selectSimulator);
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
