const $=selector=>document.querySelector(selector);
const escape=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const names={agent:'视觉 Agent',smolvla:'SmolVLA',vla:'自定义 VLA',wam:'自定义 WAM'};
const statuses={idle:'等待任务',loading:'加载模型',preparing:'准备实验',observing:'采集观察',thinking:'策略推理',executing:'执行动作',verified:'评价器确认成功',model_done:'模型报告完成',blocked:'无法继续',error:'运行错误',failed:'执行失败',stopped:'已停止',cancelled:'已取消',canceled:'已取消',interrupted:'服务中断',returned:'已返回',round_complete:'本轮结束',limit:'达到上限',running:'运行中',complete:'实验完成',unfinished:'未结束'};
let context, catalog, configs={}, lastGeneration=-1, lastRows='', lastPolicy='', currentPage='workspace', runs=[], selectedRun='',resultsOffset=0,resultsHasMore=true,analysisCatalog=[],refreshingResults=false,lastResultRefresh=0,lastRunStates='';
let selectedResult=null,resultsGeneration=0,detailGeneration=0,analysisGeneration=0,analysisCatalogGeneration=0,hasAnalysisResults=false,resultMutationPending=false;
const checkedResults=new Set();
let batchDeleteIds=[];
let pricingState=null,pricingLoading=false,pricingRefreshing=false,lastPricingCheck=0,pricingGeneration=0,pricingError='',pricingErrorIsManual=false;
const RESULTS_PAGE_SIZE=100;

async function request(path,payload,method=payload===undefined?'GET':'POST'){
  const response=await fetch(`/api/${path}`,{method,...(payload===undefined?{}:{headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)})});
  const data=await response.json();
  if(!response.ok)throw new Error(typeof data.detail==='string'?data.detail:Array.isArray(data.detail)?data.detail.map(e=>e.msg).join('；'):'请求失败');
  return data;
}
export function runVideoMarkup(runId,video,compact=false){
  if(!video)return '';
  if(runId&&video.status==='saved'&&video.path&&Number(video.frame_count)>0){
    const url=`/api/runs/${encodeURIComponent(runId)}/artifact/${String(video.path).split('/').map(encodeURIComponent).join('/')}`;
    const content=`<div class="run-video-panel"><div><strong>轨迹视频</strong><span>${Number(video.frame_count)} 帧${Number(video.fps)>0?` · ${Number(video.fps).toFixed(0)} FPS`:''}</span><a class="text-button" href="${escape(url)}" download>下载视频 ↗</a></div><video controls playsinline preload="${compact?'none':'metadata'}" src="${escape(url)}">当前浏览器无法直接播放此视频，请使用下载链接。</video></div>`;
    return compact?`<details class="run-video-inline"><summary>播放轨迹视频</summary>${content}</details>`:content;
  }
  if(video.status==='disabled')return compact?'':'<p class="video-record-note">本次未开启轨迹视频。</p>';
  const message=video.status==='failed'?`视频保存失败：${video.error||'编码器未返回视频文件。'}`:video.status==='saved'?'没有可用的视频帧或文件记录。':video.status==='recording'?'正在录制轨迹视频，结束后可播放。':'视频等待录制，结束后可播放。';
  return `<p class="video-record-note${video.status==='failed'?' error':''}">${escape(message)}</p>`;
}
export function renderRunVideo(container,runId,video){
  if(!container)return;const key=JSON.stringify([runId,video]);if(container.dataset.videoKey===key)return;container.dataset.videoKey=key;container.hidden=!video;container.innerHTML=runVideoMarkup(runId,video);
}
const metric=value=>typeof value==='number'&&Number.isFinite(value)?value:null;
const numberText=value=>metric(value)==null?'—':value.toLocaleString('zh-CN',{maximumFractionDigits:2});
const secondsText=value=>metric(value)==null?'—':`${Number(value).toFixed(2)} s`;
export function renderRunTimer(container,run){
  if(!container)return;const timing=run?.timing||{},wall=metric(timing.elapsed_wall_s)??metric(run?.elapsed_s)??metric(run?.metrics?.wall_time_s),inference=metric(timing.inference_elapsed_s);
  container.hidden=wall==null&&inference==null;if(container.hidden)return;
  const current=metric(timing.current_inference_s),activeCalls=metric(timing.active_calls);
  container.innerHTML=`<span>墙钟 <strong>${secondsText(wall)}</strong></span><span title="包含网络、排队与服务处理时间">累计调用 <strong>${secondsText(inference)}</strong></span>${current==null?'':activeCalls>0?`<span>当前请求等待 <strong>${secondsText(current)}</strong>${activeCalls>1?` · ${activeCalls} 个并发请求`:''}</span>`:'<span>暂无推理请求</span>'}${run?.active?'<span class="timer-active">运行中</span>':''}`;
}
export function metricChartMarkup(points,title,xLabel='轮次 / 调用'){
  const values=points.map(point=>metric(point.value));if(!points.length||values.every(value=>value==null))return '<p class="empty-state">没有返回可绘制的真实数值。</p>';
  const width=640,height=265,left=65,right=20,top=20,bottom=58,plotWidth=width-left-right,plotHeight=height-top-bottom,max=Math.max(...values.filter(value=>value!=null),0)||1;
  const x=index=>left+(points.length===1?plotWidth/2:index/(points.length-1)*plotWidth),y=value=>top+plotHeight-value/max*plotHeight;
  let grid='',paths='',dots='',labels='',segment=[];
  for(let tick=0;tick<=4;tick++){const value=max*tick/4,py=y(value),label=value===0?'0':Number(value.toPrecision(3)).toLocaleString('zh-CN');grid+=`<line x1="${left}" y1="${py}" x2="${width-right}" y2="${py}"/><text x="${left-10}" y="${py+4}" text-anchor="end">${escape(label)}</text>`;}
  const flush=()=>{if(segment.length>1)paths+=`<polyline points="${segment.join(' ')}"/>`;segment=[];};
  values.forEach((value,index)=>{if(value==null){flush();return;}segment.push(`${x(index)},${y(value)}`);dots+=`<circle cx="${x(index)}" cy="${y(value)}" r="4"><title>${escape(points[index].label)}：${escape(numberText(value))}</title></circle>`;});flush();
  const ticks=[...new Set(Array.from({length:Math.min(6,points.length)},(_,index)=>Math.round(index*(points.length-1)/Math.max(1,Math.min(6,points.length)-1))))];
  for(const index of ticks)labels+=`<text x="${x(index)}" y="${height-bottom+23}" text-anchor="middle">${escape(points[index].shortLabel||points[index].label)}</text>`;
  return `<svg class="metric-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="${escape(title)}"><title>${escape(title)}</title><g class="chart-grid">${grid}</g><g class="chart-series">${paths}${dots}</g><g class="chart-labels">${labels}<text x="${left+plotWidth/2}" y="${height-8}" text-anchor="middle">${escape(xLabel)}</text></g></svg>`;
}
function options(select,items,value){select.replaceChildren(...items.map(item=>{const option=document.createElement('option');option.value=item.id;option.textContent=item.name;return option;}));if(value)select.value=value;}
function tasksFor(sceneId,select,taskId){const scene=catalog.scenes.find(s=>s.id===sceneId);options(select,scene.tasks.map(id=>({id,name:catalog.tasks[id].name})),scene.tasks.includes(taskId)?taskId:scene.tasks[0]);}
function updateSceneInfo(){if(!catalog)return;const scene=catalog.scenes.find(s=>s.id===$('#scene-select').value);$('#scene-description').textContent=scene.description;}
function updateCompareCriterion(){if(catalog)$('#comparison-criterion').textContent=catalog.tasks[$('#compare-task').value]?.criterion||'';}
function fillTask(){if(!catalog)return;const instruction=catalog.tasks[$('#task-select').value].instruction;$('#command').value=instruction;$('#policy-task').value=instruction;}
function updatePolicySummary(){const kind=$('#policy-kind').value;const config=configs[kind];$('#policy-model-name').textContent=config?.model||names[kind];$('#policy-model-note').textContent=kind==='smolvla'?'官方 SmolVLA 直接预测动作块。默认映射为 SO-101，基础权重未在当前场景微调。':`${names[kind]} 通过你配置的推理服务预测动作。${config?.endpoint?'接口已配置。':'尚未配置接口。'}`;}
function changePage(page){currentPage=page;for(const name of ['workspace','rsi','comparison','logs'])$(`#${name}-view`).hidden=name!==page;document.querySelectorAll('[data-page]').forEach(button=>button.classList.toggle('selected',button.dataset.page===page));if(page==='logs'){refreshLogs();refreshPricingStatus();}}

export async function initResearch(callbacks){
  context=callbacks;
  document.querySelectorAll('[data-page]').forEach(button=>button.addEventListener('click',()=>changePage(button.dataset.page)));
  $('#policy-tab').addEventListener('click',()=>context.selectTab('policy'));
  $('#scene-select').addEventListener('change',()=>{tasksFor($('#scene-select').value,$('#task-select'));updateSceneInfo();});
  $('#compare-scene').addEventListener('change',()=>{tasksFor($('#compare-scene').value,$('#compare-task'));updateCompareCriterion();});
  $('#compare-task').addEventListener('change',updateCompareCriterion);
  $('#use-task').addEventListener('click',fillTask);
  $('#apply-scene').addEventListener('click',async()=>{await context.operate({action:'scene',scene_id:$('#scene-select').value,task_id:$('#task-select').value,seed:Number($('#scene-seed').value)});fillTask();});
  $('#policy-kind').addEventListener('change',updatePolicySummary);
  $('#policy-form').addEventListener('submit',event=>{event.preventDefault();startPolicy(false);});
  $('#policy-one-chunk').addEventListener('click',()=>startPolicy(true));
  $('#policy-stop').addEventListener('click',async()=>{try{await request('policy/stop',{});}catch(error){context.toast(error.message);}});
  $('#policy-settings').addEventListener('click',openPolicySettings);
  $('#close-policy-config').addEventListener('click',()=>$('#policy-config-dialog').close());
  $('#policy-config-form').addEventListener('submit',savePolicySettings);
  $('#comparison-form').addEventListener('submit',async event=>{
    event.preventDefault();
    try{
      const seedText=$('#compare-seeds').value.trim();
      if(!/^\d+(\s*[,，]\s*\d+)*$/.test(seedText))throw new Error('请填写整数种子，用逗号分隔。');
      await request('comparison/start',{robot:$('#compare-robot').value,scene_id:$('#compare-scene').value,task_id:$('#compare-task').value,
        seeds:seedText.split(/[,，]/).map(Number),methods:[...document.querySelectorAll('input[name="method"]:checked')].map(x=>x.value),max_sim_time_s:Number($('#compare-limit').value),record_video:$('#compare-record-video').checked});
    }catch(error){context.toast(error.message);$('#comparison-message').textContent=error.message;}
  });
  $('#compare-stop').addEventListener('click',async()=>{try{await request('comparison/stop',{});}catch(error){context.toast(error.message);}});
  $('#refresh-logs').addEventListener('click',()=>refreshLogs({reset:true}));for(const selector of ['#log-filter','#log-status-filter'])$(selector).addEventListener('change',()=>{checkedResults.clear();renderRunList();});
  $('#log-record-scope').addEventListener('change',()=>{checkedResults.clear();runs=[];resultsOffset=0;resultsHasMore=true;clearResult();renderRunList();refreshLogs({reset:true,force:true});});
  $('#load-more-results').addEventListener('click',loadMoreResults);
  $('#select-visible-results').addEventListener('change',event=>{for(const run of filteredResults())if(!resultIsActive(run)&&!run.deleted_at){if(event.target.checked)checkedResults.add(run.run_id);else checkedResults.delete(run.run_id);}renderRunList();});
  $('#delete-selected-results').addEventListener('click',openBatchResultDelete);
  $('#result-batch-delete-form').addEventListener('submit',deleteSelectedResults);
  $('#refresh-pricing').addEventListener('click',()=>refreshPricingStatus(true));
  $('#rename-result').addEventListener('click',openResultRename);
  $('#delete-result').addEventListener('click',openResultDelete);
  $('#restore-result').addEventListener('click',restoreResult);
  $('#result-name-form').addEventListener('submit',saveResultName);
  $('#result-auto-name').addEventListener('click',()=>{$('#result-name-input').value='';$('#result-name-input').focus();});
  $('#result-delete-form').addEventListener('submit',deleteResult);
  for(const action of ['name','delete','batch-delete']){const dialog=$(`#result-${action}-dialog`);$(`#cancel-result-${action}`).addEventListener('click',()=>dialog.close());dialog.addEventListener('cancel',event=>{if(resultMutationPending)event.preventDefault();});}
  for(const tab of ['records','analysis'])$(`#results-${tab}-tab`).addEventListener('click',()=>{for(const name of ['records','analysis']){$(`#results-${name}-tab`).setAttribute('aria-selected',String(name===tab));$(`#results-${name}-tab`).classList.toggle('selected',name===tab);$(`#results-${name}-panel`).hidden=name!==tab;}if(tab==='analysis')refreshAnalysisCatalog();});
  $('#results-analysis-form').addEventListener('submit',event=>{event.preventDefault();runAnalysis();});
  $('#analysis-model').addEventListener('change',()=>fillSelect($('#analysis-task'),analysisCatalog.filter(run=>run.model===$('#analysis-model').value).map(run=>run.task),'请选择同一任务'));
  $('#result-children').addEventListener('click',event=>{const button=event.target.closest('[data-result-run]');if(button)showRun(runs.find(run=>run.run_id===button.dataset.resultRun)||{run_id:button.dataset.resultRun});});
  try{
    [catalog,configs]=await Promise.all([request('scenes'),request('policy/config')]);
    for(const selector of ['#scene-select','#compare-scene'])options($(selector),catalog.scenes,'tabletop');
    tasksFor('tabletop',$('#task-select'));tasksFor('tabletop',$('#compare-task'));
    updateSceneInfo();updateCompareCriterion();updatePolicySummary();
    if(!$('#policy-task').value)$('#policy-task').value=catalog.tasks.lift_red.instruction;
  }catch(error){context.toast(error.message);}
}

async function startPolicy(oneChunk){
  try{await request('policy/start',{kind:$('#policy-kind').value,task:$('#policy-task').value.trim(),one_chunk:oneChunk,max_sim_time_s:Number($('#policy-time-limit').value),record_video:$('#policy-record-video').checked});}catch(error){context.toast(error.message);$('#policy-message').textContent=error.message;}
}

async function openPolicySettings(){
  try{
    configs=await request('policy/config');const kind=$('#policy-kind').value,c=configs[kind];
    $('#policy-config-model').value=c.model;$('#policy-config-endpoint').value=c.endpoint;
    $('#policy-config-key').value='';$('#policy-config-key').placeholder=c.has_api_key?'当前接口已有密钥，留空保留':'可选，仅保存在服务内存';
    $('#policy-config-device').value=c.device;$('#policy-config-horizon').value=c.horizon;$('#policy-config-dt').value=c.dt_s;
    $('#policy-config-chunks').value=c.max_chunks;$('#policy-config-timeout').value=c.timeout_s;
    $('#policy-endpoint-field').hidden=kind==='smolvla';$('#policy-key-field').hidden=kind==='smolvla';$('#policy-device-field').hidden=kind!=='smolvla';
    $('#policy-config-help').textContent=kind==='smolvla'?'默认使用官方 lerobot/smolvla_base，也可填写匹配 SO-101 的微调权重。动作映射和限幅结果会保存在日志中。':`填写实际运行 ${names[kind]} 的服务地址。接口接收三路图像、机械臂状态和任务，返回关节动作块或末端位姿；协议见项目 POLICY_PROTOCOL.md。`;
    $('#policy-config-error').hidden=true;$('#policy-config-dialog').showModal();
  }catch(error){context.toast(error.message);}
}
async function savePolicySettings(event){
  event.preventDefault();const kind=$('#policy-kind').value;
  try{
    configs[kind]=await request('policy/config',{kind,model:$('#policy-config-model').value.trim(),endpoint:kind==='smolvla'?'':$('#policy-config-endpoint').value.trim(),api_key:$('#policy-config-key').value,
      device:$('#policy-config-device').value,horizon:Number($('#policy-config-horizon').value),dt_s:Number($('#policy-config-dt').value),max_chunks:Number($('#policy-config-chunks').value),timeout_s:Number($('#policy-config-timeout').value)});
    $('#policy-config-key').value='';$('#policy-config-dialog').close();updatePolicySummary();context.toast('策略配置已应用。');
  }catch(error){$('#policy-config-error').textContent=error.message;$('#policy-config-error').hidden=false;}
}

export function renderResearch(state){
  if(!context)return;
  if(catalog&&state.generation!==lastGeneration){
    lastGeneration=state.generation;$('#scene-select').value=state.scene_id;$('#scene-seed').value=state.seed;
    tasksFor(state.scene_id,$('#task-select'),state.evaluation?.task_id);updateSceneInfo();
    $('#model-note').hidden=!state.model_note;$('#model-note').textContent=state.model_note;
  }
  const evaluation=state.evaluation;
  if(evaluation){$('#evaluation-name').textContent=evaluation.name;$('#evaluation-criterion').textContent=evaluation.criterion;$('#evaluation-result').textContent=evaluation.success?'已达成':'尚未达成';$('#evaluation-result').classList.toggle('success',evaluation.success);}
  if(state.model==='rm65_gaia'){$('#gripper-label').textContent='灵巧手';$('#gripper-status').textContent='15 关节';$('#gripper-detail').textContent='可在关节控制中逐指调节';}
  const policy=state.policy||{},comparison=state.comparison||{},locked=!!(state.agent?.active||policy.active||comparison.active||state.research?.active||state.experiment?.active);
  $('#policy-status').textContent=statuses[policy.status]||policy.status||'等待任务';$('#policy-message').textContent=policy.message||'';
  $('#policy-round').textContent=`${policy.round||0} / ${policy.max_rounds||configs[$('#policy-kind').value]?.max_chunks||60} 次`;
  renderRunTimer($('#policy-run-time'),policy);renderRunTimer($('#comparison-run-time'),comparison);
  $('#policy-stop').disabled=!policy.active;$('#policy-start').disabled=locked;$('#policy-one-chunk').disabled=locked;$('#policy-settings').disabled=policy.active||comparison.active;$('#policy-kind').disabled=locked;
  $('#policy-record-video').disabled=locked;$('#compare-record-video').disabled=locked;renderRunVideo($('#policy-video'),policy.run_id,policy.video);
  for(const selector of ['#apply-scene','#scene-select','#task-select','#scene-seed'])$(selector).disabled=locked||$('#reset').disabled;
  $('#compare-start').disabled=locked;$('#compare-stop').disabled=!comparison.active;
  document.querySelectorAll('#comparison-form input,#comparison-form select').forEach(input=>input.disabled=comparison.active);
  $('#policy-log-link').hidden=!policy.run_id;if(policy.run_id)$('#policy-log-link').href=`/api/runs/${policy.run_id}/download`;
  const key=JSON.stringify(policy.history||[]);
  if(key!==lastPolicy){lastPolicy=key;$('#policy-history').innerHTML=(policy.history||[]).slice(-5).reverse().map(row=>`<div class="round-record"><strong>动作块 ${row.round} · ${row.frames} 帧</strong><p>推理 ${row.latency_s.toFixed(2)} s · 执行目标限幅 ${row.limited_targets} 项</p></div>`).join('');}
  $('#comparison-message').textContent=comparison.message||'尚未运行。';
  const rowsKey=JSON.stringify(comparison.rows||[]);
  if(rowsKey!==lastRows){lastRows=rowsKey;renderComparison(comparison.rows||[]);}
  const keys=['agent','policy','comparison','research','experiment'],runStates=JSON.stringify(keys.map(key=>[state[key]?.run_id,state[key]?.status,state[key]?.active]));
  if(currentPage==='logs'&&!refreshingResults&&Date.now()-lastResultRefresh>3000&&(runStates!==lastRunStates||keys.some(key=>state[key]?.active))){lastRunStates=runStates;refreshLogs();}
  if(currentPage==='logs'&&Date.now()-lastPricingCheck>(pricingState?.refreshing?3000:60000))refreshPricingStatus();
}

function renderComparison(rows){
  const root=$('#comparison-rows');
  if(!rows.length){root.innerHTML='<tr><td colspan="8" class="empty-state">选择方法并开始实验，真实运行结果会出现在这里。</td></tr>';$('#comparison-cards').replaceChildren();return;}
  const seconds=value=>value==null?'—':Number(value).toFixed(2)+' s';
  root.innerHTML=rows.map(row=>{const m=row.metrics||{},valid=!!row.metrics&&!['error','stopped'].includes(row.status);return `<tr><td><strong>${escape(names[row.method])}</strong><small>种子 ${row.seed}</small></td><td><code>${escape(row.initial_state_id)}</code></td><td title="${escape(row.message)}">${escape(statuses[row.status]||row.status)}</td><td>${valid?(m.success?'<span class="result-pill success">成功</span>':'<span class="result-pill">未达成</span>'):'—'}</td><td>${seconds(m.sim_time_s)}</td><td>${seconds(m.mean_inference_s)}</td><td>${m.inference_count??'—'}</td><td>${row.run_id?`<a href="/api/runs/${escape(row.run_id)}/download">JSONL ↗</a>${runVideoMarkup(row.run_id,row.video,true)}`:'—'}</td></tr>`;}).join('');
  const methods=[...new Set(rows.map(row=>row.method))];
  $('#comparison-cards').innerHTML=methods.map(method=>{
    const group=rows.filter(row=>row.method===method),last=group.at(-1),success=group.filter(row=>row.metrics?.success&&!['error','stopped'].includes(row.status)).length;
    return `<article class="panel comparison-card"><div class="comparison-card-heading"><strong>${escape(names[method])}</strong><span>${escape(statuses[last.status]||last.status)}</span></div>${last.run_id?`<img src="/api/runs/${escape(last.run_id)}/frame" alt="${escape(names[method])} 运行最终画面">`:'<div class="comparison-placeholder">正在运行，结束后显示真实画面</div>'}<div class="comparison-card-metrics"><div><small>成功 / 全部分配试验</small><strong>${success} / ${group.length}</strong></div><div><small>错误 / 停止</small><strong>${group.filter(row=>row.status==='error').length} / ${group.filter(row=>row.status==='stopped').length}</strong></div></div><p>${escape(last.message||'等待执行结果。')}</p>${last.run_id?runVideoMarkup(last.run_id,last.video):''}</article>`;
  }).join('');
}

function resultError(error){$('#results-error').textContent=error.message;$('#results-error').hidden=false;context.toast(error.message);}
function category(run){if(run.metrics?.success||run.success||run.status==='verified')return 'success';if(['stopped','cancelled','canceled'].includes(run.status))return 'stopped';if(run.active||['unfinished','interrupted','running','preparing','observing','thinking','executing','loading','planning','deciding'].includes(run.status))return 'unfinished';return 'error';}
function resultModel(run){return run.model||run.config?.model||run.recipe?.agent?.model||run.recipe?.policy?.model||'';}
function resultName(run){return run.display_name||run.default_name||run.run_id||'实验记录';}
function resultIsActive(run){return !!run.active||run.status==='running';}
function resultScope(){return $('#log-record-scope').value==='deleted';}
function resultListPath(offset){return `runs?limit=${RESULTS_PAGE_SIZE}&offset=${offset}&deleted=${resultScope()}`;}
function filteredResults(){return runs.filter(run=>($('#log-filter').value==='all'||run.kind===$('#log-filter').value)&&($('#log-status-filter').value==='all'||category(run)===$('#log-status-filter').value));}
function runKinds(kind){return {agent:'视觉 Agent',policy:'策略模型',composition:'方法版本运行',comparison:'对比实验',research_experiment:'RSI 配对评测',repair:'故障修复 Agent'}[kind]||kind;}
function fillSelect(select,items,placeholder){const previous=select.value;select.replaceChildren();const first=document.createElement('option');first.value='';first.textContent=placeholder;select.append(first);for(const value of [...new Set(items.filter(Boolean))].sort()){const option=document.createElement('option');option.value=value;option.textContent=value;select.append(option);}if([...select.options].some(option=>option.value===previous))select.value=previous;}
function mergeRunPages(...pages){const records=new Map();for(const page of pages)for(const run of page)if(!records.has(run.run_id))records.set(run.run_id,run);return [...records.values()].sort((a,b)=>a.run_id===b.run_id?0:a.run_id>b.run_id?-1:1);}
function resultPagingState(){const button=$('#load-more-results');button.disabled=refreshingResults||!resultsHasMore;button.textContent=refreshingResults?'正在读取结果…':resultsHasMore?'加载更多结果':'已读取全部结果';}
async function refreshLogs({reset=false,force=false}={}){
  if(refreshingResults&&!force)return;const generation=++resultsGeneration;refreshingResults=true;lastResultRefresh=Date.now();$('#results-error').hidden=true;
  resultPagingState();
  try{
    const page=await request(resultListPath(0));if(generation!==resultsGeneration)return;const known=new Set(runs.map(run=>run.run_id));
    if(reset||!runs.length||page.length<RESULTS_PAGE_SIZE||!page.some(run=>known.has(run.run_id))){runs=page;resultsOffset=page.length;resultsHasMore=page.length===RESULTS_PAGE_SIZE;}
    else{resultsOffset+=page.filter(run=>!known.has(run.run_id)).length;runs=mergeRunPages(page,runs);}
    renderRunList();const chosen=runs.find(run=>run.run_id===selectedRun);if(chosen)await showRun(chosen);else if(!selectedRun&&runs.length)await showRun(runs[0]);else if(selectedRun&&reset)await showRun(selectedResult||{run_id:selectedRun});else if(!runs.length)clearResult();
  }catch(error){if(generation===resultsGeneration)resultError(error);}finally{if(generation===resultsGeneration){refreshingResults=false;resultPagingState();}}
}
async function loadMoreResults(){
  if(refreshingResults||!resultsHasMore)return;const generation=++resultsGeneration;refreshingResults=true;lastResultRefresh=Date.now();$('#results-error').hidden=true;resultPagingState();
  try{const page=await request(resultListPath(resultsOffset));if(generation!==resultsGeneration)return;resultsOffset+=page.length;resultsHasMore=page.length===RESULTS_PAGE_SIZE;runs=mergeRunPages(runs,page);renderRunList();}
  catch(error){if(generation===resultsGeneration)resultError(error);}finally{if(generation===resultsGeneration){refreshingResults=false;resultPagingState();}}
}
function renderRunList(){
  const filtered=filteredResults(),eligible=new Set(filtered.filter(run=>!resultIsActive(run)&&!run.deleted_at).map(run=>run.run_id));
  for(const id of checkedResults)if(!eligible.has(id))checkedResults.delete(id);
  renderBatchControls();
  $('#results-count').textContent=`显示 ${filtered.length} / 已读取 ${runs.length} 条${resultScope()?'已删除记录，可恢复。':'运行，包含错误与取消。'}`;
  const root=$('#run-list');root.replaceChildren();if(!filtered.length){root.innerHTML='<p class="empty-state">筛选条件下没有运行记录。</p>';return;}
  for(const run of filtered){const row=document.createElement('div');row.className='run-select-row';if(!resultScope()){const choice=document.createElement('label');choice.className='run-select-control';const checkbox=document.createElement('input');checkbox.type='checkbox';checkbox.checked=checkedResults.has(run.run_id);checkbox.disabled=resultIsActive(run)||resultMutationPending;checkbox.setAttribute('aria-label',`选择实验 ${resultName(run)}`);choice.title=resultIsActive(run)?'实验正在运行，结束后可选择删除。':'选择这条实验记录';checkbox.addEventListener('change',()=>{if(checkbox.checked)checkedResults.add(run.run_id);else checkedResults.delete(run.run_id);renderBatchControls();});choice.append(checkbox);row.append(choice);}const button=document.createElement('button');button.className='run-list-item'+(selectedRun===run.run_id?' selected':'');const wall=metric(run.elapsed_s)??metric(run.metrics?.wall_time_s)??metric(run.wall_time_s);button.innerHTML=`<span>${escape(runKinds(run.kind))}<em>${escape(statuses[run.status]||run.status)}</em></span><strong>${escape(resultName(run))}</strong><p>${escape(run.task||'未记录任务指令')}</p><p>${escape(resultModel(run)||'模型见详情')} · 墙钟 ${secondsText(wall)}</p><small>${escape(run.run_id)}</small>`;button.addEventListener('click',()=>showRun(run));row.append(button);root.append(row);}
}
function renderBatchControls(){
  const eligible=filteredResults().filter(run=>!resultIsActive(run)&&!run.deleted_at),chosen=eligible.filter(run=>checkedResults.has(run.run_id)).length;
  $('#result-batch-controls').hidden=resultScope();const select=$('#select-visible-results');select.checked=eligible.length>0&&chosen===eligible.length;select.indeterminate=chosen>0&&chosen<eligible.length;select.disabled=resultMutationPending||!eligible.length;
  $('#selected-results-count').textContent=`已选择 ${checkedResults.size} 条${checkedResults.size>1000?'，每批最多 1000 条，请减少选择。':'；全选仅覆盖已显示的可删除记录。'}`;$('#delete-selected-results').disabled=resultMutationPending||!checkedResults.size||checkedResults.size>1000;
}
async function showRun(run){
  const id=run.run_id,generation=++detailGeneration;selectedRun=id;selectedResult=run;renderRunList();$('#results-error').hidden=true;renderResultActions(run);
  try{const [analysis,events]=await Promise.all([request(`runs/${encodeURIComponent(id)}/analysis`),request(`runs/${encodeURIComponent(id)}/events`)]);if(selectedRun!==id||generation!==detailGeneration)return;
    selectedResult={...run,...analysis};renderResultActions(selectedResult);
    $('#log-download').hidden=false;$('#log-download').href=`/api/runs/${encodeURIComponent(id)}/download`;$('#log-detail').textContent=events.map(event=>JSON.stringify(event,null,2)).join('\n\n');
    renderRunVideo($('#log-video'),id,analysis.video||run.artifacts?.video||run.video);renderResultAnalysis(analysis,run);
  }catch(error){if(selectedRun===id&&generation===detailGeneration)resultError(error);}
}
function clearResult(){
  selectedRun='';selectedResult=null;detailGeneration++;$('#log-detail-title').textContent='选择一条实验结果';$('#log-detail-id').textContent='';$('#log-download').hidden=true;$('#result-actions').hidden=true;$('#result-management-note').hidden=true;
  for(const selector of ['#result-summary','#result-calls','#result-children','#result-task-text'])$(selector).replaceChildren();
  for(const selector of ['#result-round-charts','#result-chart-note','#result-outcome-message','#result-data-issues','#result-cost-panel'])$(selector).hidden=true;
  renderRunVideo($('#log-video'),'',null);$('#result-description').textContent='接口未返回的用量与推理文本会标记缺失，不推测补齐。';$('#log-detail').textContent='运行记录将在这里展示。';
}
function renderResultActions(run){
  $('#log-detail-title').textContent=resultName(run);$('#log-detail-id').textContent=run.run_id;$('#result-actions').hidden=false;
  $('#rename-result').disabled=resultMutationPending;$('#delete-result').hidden=!!run.deleted_at;$('#delete-result').disabled=resultIsActive(run)||resultMutationPending;$('#restore-result').hidden=!run.deleted_at;$('#restore-result').disabled=resultMutationPending;
  const note=run.deleted_at?`已于 ${run.deleted_at} 移入回收站，已从跨实验分析中移除。原始日志和轨迹视频保留，可恢复。`:resultIsActive(run)?'实验正在运行，结束后可删除记录。':'默认名称由主要模型与开始时间生成；重命名不修改原始日志。';
  $('#result-management-note').textContent=note;$('#result-management-note').hidden=false;
}
function openResultRename(){
  if(!selectedResult||resultMutationPending)return;const dialog=$('#result-name-dialog');dialog.dataset.runId=selectedResult.run_id;$('#result-name-input').value=selectedResult.custom_name||'';$('#result-name-input').placeholder=selectedResult.default_name||resultName(selectedResult);$('#result-name-default').textContent=`自动名称：${selectedResult.default_name||resultName(selectedResult)}`;$('#result-name-error').textContent='';dialog.showModal();$('#result-name-input').focus();
}
function openResultDelete(){
  if(!selectedResult||resultIsActive(selectedResult)||resultMutationPending)return;const dialog=$('#result-delete-dialog');dialog.dataset.runId=selectedResult.run_id;$('#result-delete-name').textContent=resultName(selectedResult);$('#result-delete-id').textContent=selectedResult.run_id;$('#result-delete-error').textContent='';dialog.showModal();
}
function setResultMutationPending(pending){
  resultMutationPending=pending;for(const selector of ['#result-name-input','#save-result-name','#result-auto-name','#cancel-result-name','#confirm-result-delete','#cancel-result-delete','#confirm-result-batch-delete','#cancel-result-batch-delete'])$(selector).disabled=pending;
  renderRunList();
  if(selectedResult)renderResultActions(selectedResult);
}
async function refreshManagedResults(){
  analysisGeneration++;
  await refreshLogs({reset:true,force:true});await refreshAnalysisCatalog();
  if(hasAnalysisResults){if($('#analysis-model').value&&$('#analysis-task').value)await runAnalysis();else clearAnalysis('实验记录已更新，请重新选择模型与任务。');}
}
async function saveResultName(event){
  event.preventDefault();if(resultMutationPending)return;const dialog=$('#result-name-dialog'),id=dialog.dataset.runId;setResultMutationPending(true);$('#result-name-error').textContent='';
  try{await request(`runs/${encodeURIComponent(id)}`,{name:$('#result-name-input').value},'PATCH');dialog.close();await refreshManagedResults();context.toast('实验名称已保存。');}
  catch(error){$('#result-name-error').textContent=error.message;resultError(error);}finally{setResultMutationPending(false);}
}
async function deleteResult(event){
  event.preventDefault();if(resultMutationPending)return;const dialog=$('#result-delete-dialog'),id=dialog.dataset.runId;setResultMutationPending(true);$('#result-delete-error').textContent='';
  try{await request(`runs/${encodeURIComponent(id)}`,undefined,'DELETE');dialog.close();clearResult();await refreshManagedResults();context.toast('实验记录已移入回收站，可随时恢复。');}
  catch(error){$('#result-delete-error').textContent=error.message;resultError(error);}finally{setResultMutationPending(false);}
}
async function restoreResult(){
  if(!selectedResult||resultMutationPending)return;const id=selectedResult.run_id;setResultMutationPending(true);
  try{await request(`runs/${encodeURIComponent(id)}/restore`,{});clearResult();await refreshManagedResults();context.toast('实验记录已恢复，可在实验记录列表查看。');}
  catch(error){resultError(error);}finally{setResultMutationPending(false);}
}
function openBatchResultDelete(){
  if(resultMutationPending||resultScope()||!checkedResults.size||checkedResults.size>1000)return;
  batchDeleteIds=[...checkedResults];$('#result-batch-delete-count').textContent=`将所选 ${batchDeleteIds.length} 条实验记录移入回收站。`;$('#result-batch-delete-list').replaceChildren();
  for(const id of batchDeleteIds){const item=document.createElement('li');item.textContent=`${resultName(runs.find(run=>run.run_id===id)||{run_id:id})} · ${id}`;$('#result-batch-delete-list').append(item);}
  $('#result-batch-delete-error').textContent='';$('#result-batch-delete-dialog').showModal();
}
async function deleteSelectedResults(event){
  event.preventDefault();if(resultMutationPending||!batchDeleteIds.length)return;const ids=[...batchDeleteIds],namesById=new Map(runs.map(run=>[run.run_id,resultName(run)]));setResultMutationPending(true);$('#result-batch-delete-error').textContent='';
  try{const response=await request('runs/batch-delete',{run_ids:ids}),deleted=response.deleted||[],failed=response.failed||[];$('#result-batch-delete-dialog').close();
    for(const id of deleted)checkedResults.delete(id);if(deleted.includes(selectedRun))clearResult();await refreshManagedResults();
    const message=`已移入回收站 ${deleted.length} 条${failed.length?`，未删除 ${failed.length} 条。`:'。'}`;$('#results-batch-outcome').hidden=false;$('#results-batch-outcome-summary').textContent=message;$('#results-batch-failures').replaceChildren();
    for(const failure of failed){const item=document.createElement('li');item.textContent=`${namesById.get(failure.run_id)||failure.run_id}（${failure.run_id}）：${failure.detail}`;$('#results-batch-failures').append(item);}
    context.toast(failed.length?`${message}具体原因见结果列表上方。`:message);
  }catch(error){$('#result-batch-delete-error').textContent=error.message;resultError(error);}finally{setResultMutationPending(false);}
}
function moneyText(value){
  const amount=metric(value);if(amount==null)return '不可估算';
  return amount>0&&amount<1e-8?`USD ${amount.toExponential(3)}`:`USD ${amount.toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:8})}`;
}
function localPriceTime(value){if(!value)return '未记录';const date=new Date(value);return Number.isNaN(date.getTime())?String(value):date.toLocaleString('zh-CN',{hour12:false});}
function priceSourceLink(url,label='官方价格页 ↗'){
  if(typeof url!=='string'||!/^https?:\/\//i.test(url))return '<span>未提供来源链接</span>';
  return `<a class="text-button" href="${escape(url)}" target="_blank" rel="noopener noreferrer">${escape(label)}</a>`;
}
function costAmountMarkup(cost){
  const amount=metric(cost?.amount_usd),known=metric(cost?.known_amount_usd),reference=metric(cost?.reference_amount_usd),minimum=metric(cost?.amount_min_usd),maximum=metric(cost?.amount_max_usd);
  const cash=amount!=null?moneyText(amount):minimum!=null&&maximum!=null?`${moneyText(minimum)} ～ ${moneyText(maximum)}`:cost?.status==='partial'&&known>0?`${moneyText(known)}（已知部分）`:'不可完整估算';
  return `<div><small>API 费用估算</small><strong>${escape(cash)}</strong></div>${reference==null&&!cost?.has_reference&&cost?.status!=='reference'?'':`<div class="cost-reference"><small>Codex 订阅 · API 等价参考</small><strong>${escape(moneyText(reference))}</strong><span>用于对照 API 用量成本，不代表订阅实际扣费。</span></div>`}`;
}
function costBreakdownMarkup(rows){
  if(!rows?.length)return '<p class="research-note">没有可展示的计价组成。</p>';
  const components={input:'输入',input_tokens:'输入',uncached_input:'未缓存输入',input_uncached:'未缓存输入',cached_input:'缓存命中输入',input_cached:'缓存命中输入',cache_read:'缓存读取',cache_creation:'缓存写入',cache_write:'缓存写入',output:'输出',output_tokens:'输出',reasoning:'推理'};
  return `<div class="result-table-wrap"><table class="result-table cost-breakdown-table"><thead><tr><th>组成</th><th>Tokens</th><th>USD / 百万 tokens</th><th>估算金额 / USD</th></tr></thead><tbody>${rows.map(row=>`<tr><td>${escape(components[row.component]||row.component||'未记录')}</td><td>${numberText(row.tokens)}</td><td>${escape(metric(row.rate_per_million)==null?'未记录':String(row.rate_per_million))}</td><td>${escape(moneyText(row.amount_usd))}</td></tr>`).join('')}</tbody></table></div>`;
}
function costDiagnosisText(cost){
  if(metric(cost?.amount_min_usd)!=null&&metric(cost?.amount_max_usd)!=null)return '金额上下界来自尚未确定的计价条件，例如未覆盖年份的节假日日历；这不代表接口的 Token 用量一定缺失。';
  if(cost?.status==='partial'||cost?.status==='unavailable')return '费用未完整估算的具体原因见下方：可能缺少接口实际用量、缓存明细或适用单价。取消前未返回的 usage 不会按 0 补算。';
  if(cost?.calendar_sources?.length)return '已按有来源、限定年份的国务院全国放假安排核对峰谷条件；周末即使调休上班，也不作为周一至周五的高峰时段。';
  return '';
}
function renderResultCost(cost){
  $('#result-cost-panel').hidden=false;const state={complete:'全部调用已估算',partial:'费用范围或部分估算',unavailable:'费用不可估算',reference:'API 等价参考'}[cost?.status]||'费用未记录';
  $('#result-cost-amounts').innerHTML=costAmountMarkup(cost);
  const basis={recorded_snapshot:'按每次调用保存的价格快照计算。',current_price_estimate:'历史记录未保存价格快照，按当前可用价格估算；不代表实验当时的价格。',mixed:'包含已保存的价格快照与按当前价格补算的历史调用。'}[cost?.basis]||'尚无可用计价依据。';
  const types=[['billable_calls','API 调用'],['reference_calls','订阅参考调用'],['local_calls','本地策略调用']].filter(([key])=>metric(cost?.[key])!=null).map(([key,label])=>`${label} ${numberText(cost[key])} 次`).join(' / ');
  const calls=cost?.call_count_incomplete?`已估算 ${numberText(cost?.priced_calls)} 次 / 调用总数不完整（已知至少 ${numberText(cost?.total_calls)} 次） · 存在无法读取的调用记录`:`已确定单价与完整用量 ${numberText(cost?.priced_calls)} / ${numberText(cost?.total_calls)} 次调用${metric(cost?.missing_calls)==null?'':` · 尚未完整估算 ${numberText(cost.missing_calls)} 次`}`;
  $('#result-cost-note').textContent=`${state} · ${calls}。${types?`${types}。`:''}${basis}${costDiagnosisText(cost)}费用按官方标价与接口返回用量估算，实际账单以服务商为准。`;
  $('#result-cost-warnings').replaceChildren();for(const warning of cost?.warnings||[]){const item=document.createElement('li');item.textContent=warning;$('#result-cost-warnings').append(item);}$('#result-cost-warnings').hidden=!cost?.warnings?.length;
  $('#result-cost-breakdown').innerHTML=costBreakdownMarkup(cost?.breakdown);
  $('#result-price-snapshots').innerHTML=cost?.price_sources?.length?cost.price_sources.map(source=>`<p class="research-note">${escape(source.provider||'接口未记录')} / ${escape(source.model||'模型未记录')} · 价格更新时间（本地）${escape(localPriceTime(source.updated_at))} · ${priceSourceLink(source.source_url)}</p>`).join(''):'<p class="research-note">本实验没有可追溯的价格来源记录。</p>';
  if(cost?.calendar_sources?.length)$('#result-price-snapshots').insertAdjacentHTML('beforeend',cost.calendar_sources.map(source=>`<p class="research-note">节假日日历适用 ${escape(source.coverage||source.year)} 年 · 官方文件日期 ${escape(source.document_date)} · ${priceSourceLink(source.source_url,'国务院年度放假安排依据 ↗')}</p>`).join(''));
  renderPricingStatus();
}
function callCostMarkup(cost){
  if(!cost)return '<div class="call-cost"><strong>费用未记录</strong></div>';
  const status={complete:'费用估算',partial:'费用范围或部分估算',unavailable:'费用不可估算',reference:'API 等价参考'}[cost.status]||'费用估算';
  const pricing=cost.pricing||{},basis={recorded_snapshot:'调用时价格快照',current_price_estimate:'按当前可用价格补算',mixed:'快照与当前价格混合'}[cost.basis]||'';
  return `<div class="call-cost"><strong>${escape(status)}</strong><div class="cost-amounts">${costAmountMarkup(cost)}</div>${basis||pricing.updated_at?`<p>${escape(basis)}${pricing.updated_at?` · 价格更新时间（本地）${escape(localPriceTime(pricing.updated_at))}`:''}</p>`:''}<p>${escape(costDiagnosisText(cost))}</p>${pricing.source_url?priceSourceLink(pricing.source_url):''}${cost.calendar_resolution?.source_url?priceSourceLink(cost.calendar_resolution.source_url,'节假日判定依据 ↗'):''}${cost.warnings?.length?`<ul>${cost.warnings.map(warning=>`<li>${escape(warning)}</li>`).join('')}</ul>`:''}</div>`;
}
function renderPricingStatus(){
  const busy=pricingRefreshing||pricingState?.refreshing;$('#refresh-pricing').disabled=!!busy;$('#refresh-pricing').textContent=busy?'正在更新官方价格…':'刷新官方价格';
  const interval=metric(pricingState?.refresh_interval_s),schedule=interval==null?'价格更新周期由服务端管理':`服务端每 ${numberText(interval/60)} 分钟自动更新`;
  $('#pricing-status').textContent=pricingLoading&&!pricingState?'正在读取价格来源状态…':`${schedule} · 最近价格更新（本地）${localPriceTime(pricingState?.updated_at)}${busy?' · 更新中':''}`;
  $('#pricing-error').textContent=pricingError;$('#pricing-error').hidden=!pricingError;
  $('#pricing-providers').innerHTML=pricingState?.providers?.length?pricingState.providers.map(source=>`<article class="pricing-provider"><div><strong>${escape(({openai:'OpenAI',anthropic:'Anthropic',deepseek:'DeepSeek'})[source.provider]||source.provider)}</strong><span class="result-pill${source.status==='ready'?' success':''}">${escape({ready:'价格可用',stale:'保留上次价格',unavailable:'价格不可用'}[source.status]||source.status)}</span></div><p>${priceSourceLink(source.source_url)} · ${numberText(source.model_count)} 个模型</p><p>价格更新（本地）${escape(localPriceTime(source.updated_at))}<br>最近尝试（本地）${escape(localPriceTime(source.last_attempt_at))}</p>${source.error?`<p class="pricing-source-error">${escape(source.error)}</p>`:''}</article>`).join(''):'<p class="research-note">尚未取得官方价格来源状态。无法匹配的模型或用量会保留为费用缺失。</p>';
}
async function refreshPricingStatus(manual=false){
  if(pricingRefreshing||(!manual&&pricingLoading))return;const generation=++pricingGeneration;lastPricingCheck=Date.now();if(manual){pricingRefreshing=true;pricingError='';pricingErrorIsManual=false;}else pricingLoading=true;renderPricingStatus();
  try{const data=await request(manual?'pricing/refresh':'pricing',manual?{}:undefined);if(generation!==pricingGeneration)return;pricingState=data;if(manual||!pricingErrorIsManual)pricingError='';renderPricingStatus();
    if(manual){if(selectedResult)await showRun(selectedResult);const incomplete=data.providers?.some(source=>source.status!=='ready');context.toast(incomplete?'价格刷新结束，部分来源保留旧价格或暂不可用，详见来源状态。':'官方价格已刷新，当前实验费用已重新读取。');}
  }catch(error){if(generation===pricingGeneration){if(manual||!pricingErrorIsManual){pricingError=`价格状态更新失败：${error.message}${pricingState?'；仍显示上次取得的来源状态。':''}`;pricingErrorIsManual=manual;}if(manual)context.toast(pricingError);}}
  finally{if(generation===pricingGeneration){pricingLoading=false;pricingRefreshing=false;renderPricingStatus();}}
}
function renderResultAnalysis(analysis,run){
  const total=analysis.total||{},rounds=analysis.rounds||[],wall=metric(analysis.elapsed_s)??metric(run.metrics?.wall_time_s),state=statuses[analysis.status]||analysis.status;
  $('#result-description').textContent=`${runKinds(analysis.kind||run.kind)} · ${analysis.model||resultModel(run)||'模型未记录'} · 推理强度 ${analysis.reasoning||'未记录'} · ${state||'状态未记录'}${analysis.started_at?` · 开始 ${analysis.started_at}`:''}`;
  $('#result-task-text').textContent=analysis.task||run.task||'';const message=analysis.message||analysis.error||run.message||'';$('#result-outcome-message').textContent=typeof message==='string'?message:JSON.stringify(message);$('#result-outcome-message').hidden=!message;$('#result-outcome-message').classList.toggle('error',['error','failed','blocked'].includes(analysis.status||run.status));
  $('#result-data-issues').textContent=(analysis.issues||[]).length?`记录读取提示：${analysis.issues.join('；')}`:'';$('#result-data-issues').hidden=!analysis.issues?.length;
  $('#result-summary').innerHTML=[['完整墙钟',secondsText(wall)],['模型调用次数',numberText(total.calls)],['总 tokens',numberText(total.total_tokens)],['输出 tokens',numberText(total.output_tokens)],['累计调用耗时',secondsText(total.inference_s)],['用量记录',total.usage_complete?'完整':'存在缺失']].map(([label,value])=>`<div><small>${escape(label)}</small><strong>${escape(value)}</strong></div>`).join('');
  renderResultCost(analysis.cost);
  const points=rounds.map((row,index)=>({label:`第 ${row.round??'?'} 轮 / 调用 ${row.call_id??index+1}`,shortLabel:`${row.round??'?'} / ${index+1}`,value:row.output_tokens}));
  $('#result-round-charts').hidden=false;$('#result-chart-note').hidden=false;$('#result-token-chart').innerHTML=metricChartMarkup(points,'每轮接口输出 tokens');$('#result-time-chart').innerHTML=metricChartMarkup(points.map((point,index)=>({...point,value:rounds[index].inference_s})),'每轮模型调用耗时');
  const open=new Set([...document.querySelectorAll('#result-calls details[open]')].map(details=>details.dataset.call));
  $('#result-calls').replaceChildren();
  if(!rounds.length)$('#result-calls').innerHTML='<p class="empty-state">此运行没有模型调用记录。批量实验可打开下方子试验；接口调用前失败或取消也会保留整次运行。</p>';
  for(const [index,row] of rounds.entries()){
    const detail=document.createElement('details');detail.className='result-call';detail.dataset.call=String(row.call_id??index);detail.open=open.has(detail.dataset.call);
    const phase={agent:'视觉 Agent 调用',agent_inference:'视觉 Agent 调用',policy:'策略模型调用',policy_inference:'策略模型调用',planning:'主 Agent 规划',decision:'决策器调用',legacy_provider_output:'历史接口输出'}[row.phase]||row.phase||'模型调用',callStatus={complete:'调用完成',returned:'已返回',running:'调用中',cancelled:'已取消',error:'调用失败',interrupted:'服务中断'}[row.status]||row.status||'状态未记录';
    detail.innerHTML=`<summary><strong>第 ${escape(row.round??'?')} 轮 · ${escape(phase)}</strong><span>${secondsText(row.inference_s)} · 输出 ${numberText(row.output_tokens)} tokens · ${escape(callStatus)}</span></summary><div class="call-meta">${escape(row.provider||'接口未记录')} / ${escape(row.model||'模型未记录')} · 推理强度 ${escape(row.reasoning_effort||analysis.reasoning||'未记录')} · 输入 ${numberText(row.input_tokens)} / 输出 ${numberText(row.output_tokens)} / 推理 ${numberText(row.reasoning_tokens)} tokens</div><h4>接口公开输出 / 摘要</h4><pre>${escape(typeof row.public_text==='string'?row.public_text:row.public_text==null?'服务未返回可展示的公开文本。':JSON.stringify(row.public_text,null,2))}</pre><h4>服务返回的推理文字 / 摘要</h4><pre>${escape(row.reasoning?.availability==='returned'?row.reasoning.text||'返回字段为空。':'服务未返回可展示的推理文字或摘要。')}</pre>${row.error?`<p class="call-error">${escape(typeof row.error==='string'?row.error:JSON.stringify(row.error))}</p>`:''}`;
    detail.insertAdjacentHTML('beforeend',callCostMarkup(row.cost));$('#result-calls').append(detail);
  }
  $('#result-children').replaceChildren();if(analysis.children?.length){const label=document.createElement('p');label.className='research-note';label.textContent='本实验的子试验结果（包括未生成独立运行编号的失败或取消）：';$('#result-children').append(label);for(const child of analysis.children){const row=typeof child==='string'?{run_id:child}:child||{},entry=document.createElement('article');entry.className='result-child';const title=document.createElement('strong');title.textContent=[row.display_name||row.method_name||names[row.method]||row.method||row.task||'子试验',row.seed==null?'':`种子 ${row.seed}`,statuses[row.status||row.analysis_status]||row.status||row.analysis_status||'',row.deleted_at?'已移入回收站':''].filter(Boolean).join(' · ');entry.append(title);const messages=[row.message,row.error,row.analysis_error].filter(Boolean);if(messages.length){const reason=document.createElement('p');reason.textContent=messages.map(value=>typeof value==='string'?value:JSON.stringify(value)).join('；');entry.append(reason);}if(row.run_id){const button=document.createElement('button');button.type='button';button.className='text-button';button.dataset.resultRun=row.run_id;button.textContent=`查看 ${row.run_id}`;entry.append(button);}else{const note=document.createElement('small');note.textContent='未生成独立运行编号，状态和错误由本批实验保留。';entry.append(note);}$('#result-children').append(entry);}}
}
async function refreshAnalysisCatalog(){
  const generation=++analysisCatalogGeneration;
  try{const response=await request('experiments/analysis?limit=1000');if(generation!==analysisCatalogGeneration)return;analysisCatalog=response.groups?.length?response.groups:response.runs||[];fillSelect($('#analysis-model'),analysisCatalog.map(run=>run.model),'请选择同一模型');fillSelect($('#analysis-task'),analysisCatalog.filter(run=>!$('#analysis-model').value||run.model===$('#analysis-model').value).map(run=>run.task),'请选择同一任务');}catch(error){if(generation===analysisCatalogGeneration)resultError(error);}
}
function clearAnalysis(message){
  analysisGeneration++;hasAnalysisResults=false;$('#refresh-analysis').disabled=false;$('#analysis-message').textContent=message;$('#analysis-issues').hidden=true;$('#analysis-issues-text').textContent='';$('#analysis-token-chart').replaceChildren();$('#analysis-time-chart').replaceChildren();$('#analysis-rows').innerHTML='<tr><td colspan="6" class="empty-state">分析结果将在这里展示。</td></tr>';
}
async function runAnalysis(){
  const model=$('#analysis-model').value,task=$('#analysis-task').value,kind=$('#analysis-kind').value;if(!model||!task){resultError(new Error('请先选择相同模型与相同任务，再比较推理强度。'));return;}
  const generation=++analysisGeneration;hasAnalysisResults=true;$('#results-error').hidden=true;$('#refresh-analysis').disabled=true;$('#analysis-message').textContent='正在读取真实调用记录与缺失情况…';
  try{const query=new URLSearchParams({model,task,kind,limit:$('#analysis-limit').value});const response=await request(`experiments/analysis?${query}`);if(generation!==analysisGeneration)return;const order={disabled:0,none:0,off:0,minimal:1,low:2,medium:3,high:4,xhigh:5,ultra:6};
    const groups=(response.groups||[]).filter(group=>group.model===model&&group.task===task&&group.kind===kind).sort((a,b)=>(order[a.reasoning]??99)-(order[b.reasoning]??99)||String(a.reasoning).localeCompare(String(b.reasoning)));
    const points=groups.map(group=>({label:group.reasoning||'未记录',shortLabel:group.reasoning||'未记录',value:group.mean_output_tokens}));
    $('#analysis-token-chart').innerHTML=metricChartMarkup(points,'按推理强度比较单次实验累计输出 tokens 均值','推理强度');$('#analysis-time-chart').innerHTML=metricChartMarkup(points.map((point,index)=>({...point,value:groups[index].mean_inference_s})),'按推理强度比较单次实验调用总耗时均值','推理强度');
    $('#analysis-rows').innerHTML=groups.length?groups.map(group=>`<tr><td><strong>${escape(group.reasoning||'未记录')}</strong><small>${escape(group.model)}</small></td><td>${numberText(group.runs)}<small>输出样本 ${numberText(group.output_samples)} / 时间样本 ${numberText(group.inference_samples)}</small></td><td>${numberText(group.output_tokens)}<small>均值 ${numberText(group.mean_output_tokens)}</small></td><td>${secondsText(group.inference_s)}<small>均值 ${secondsText(group.mean_inference_s)}</small></td><td>${numberText(group.total_tokens)}<small>完整样本 ${numberText(group.token_samples)}</small></td><td>输出 ${numberText(group.missing_output_runs)}<small>调用耗时 ${numberText(group.missing_inference_runs)}</small></td></tr>`).join(''):'<tr><td colspan="6" class="empty-state">筛选范围内没有可比较的运行。</td></tr>';
    $('#analysis-message').textContent=`${groups.length} 个推理强度组 · 匹配 ${response.matched_runs??response.runs?.length??0} 条运行，列表读取 ${response.returned_runs??response.runs?.length??0} 条。${response.aggregation_scope==='leaf_runs_only'?'统计仅累计实际执行的子试验，排除父实验重复汇总。':''}${response.excluded_active_runs?`进行中的 ${response.excluded_active_runs} 条未纳入终态成本统计。`:''}缺失用量不作为 0，均值仅使用对应字段完整的实验。`;
    $('#analysis-issues').hidden=!response.issues?.length;$('#analysis-issues-text').textContent=(response.issues||[]).map(issue=>typeof issue==='string'?issue:JSON.stringify(issue)).join('\n');
  }catch(error){if(generation===analysisGeneration){resultError(error);$('#analysis-message').textContent=error.message;}}finally{if(generation===analysisGeneration)$('#refresh-analysis').disabled=false;}
}
