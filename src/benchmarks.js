import './benchmarks.css';

const API='/api/benchmarks/robodojo';
const $=selector=>document.querySelector(selector);
const DIMENSIONS={generalization:'泛化',memory:'记忆',precision:'精密操作','long-horizon':'长程任务',open:'开放任务'};
const RUN_STATES={idle:'等待评测',not_ready:'待运行环境',preparing:'正在检查运行环境',starting:'正在启动',running:'官方评测运行中',stopping:'正在停止',stopped:'已停止',completed:'执行结束',complete:'执行结束',failed:'执行失败',error:'执行错误',imported:'已导入官方结果'};
let context={},nodes=null,platformState=null,snapshot=null,initialized=false,reachable=false,refreshing=false,busy='',stopPending=false,lastFetch=0,planUrl='',lastCatalog='',lastHistory='',lastReadiness='',initialSelection=false;
let fileVersion=0,actionVersion=0;
let planPayload=null,scriptUrl='';
const selected=new Set();

const idOf=value=>typeof value==='string'?value:String(value?.id??value?.name??'');
const list=value=>Array.isArray(value)?value:!value||typeof value!=='object'?[]:Object.entries(value).map(([id,item])=>typeof item==='object'&&!Array.isArray(item)?{id,...item}:{id,name:typeof item==='string'?item:id});
const catalog=()=>snapshot?.catalog||snapshot||{};
const tasks=()=>list(catalog().tasks).map(item=>typeof item==='string'?{id:item,name:item}:{...item,id:idOf(item)}).filter(item=>item.id);
const asText=value=>value==null?'':typeof value==='string'?value:JSON.stringify(value);
const finite=value=>typeof value==='number'&&Number.isFinite(value)?value:null;

function element(tag,className,text){const node=document.createElement(tag);if(className)node.className=className;if(text!==undefined)node.textContent=text;return node;}
function field(label,id,type='text',value=''){
  const root=element('label','benchmark-field',label),input=element(type==='select'?'select':'input');
  if(type!=='select')input.type=type;input.id=id;input.value=value;root.append(input);return{root,input};
}
function button(label,className='secondary'){const node=element('button',className,label);node.type='button';return node;}
function note(text,error=false){if(!nodes)return;nodes.message.textContent=text;nodes.message.classList.toggle('error',error);nodes.message.setAttribute('role',error?'alert':'status');}
function report(error){note(error?.message||String(error),true);context.toast?.(error?.message||String(error));}
function controllerBusy(){return !!(platformState?.backend_switching||['agent','policy','comparison','research','composition','experiment','repair'].some(key=>platformState?.[key]?.active));}

async function request(path='',payload){
  const response=await fetch(API+path,payload===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  let data;try{data=await response.json();}catch{throw new Error(`RoboDojo 接口返回 ${response.status}，没有有效 JSON。`);}
  if(!response.ok){const detail=data.detail??data.error??data.message;throw new Error(typeof detail==='string'?detail:Array.isArray(detail)?detail.map(item=>item.msg||asText(item)).join('；'):`RoboDojo 请求失败（${response.status}）。`);}
  return data;
}

function options(select,items,preferred){
  const previous=select.value;select.replaceChildren();
  for(const item of items){const option=element('option','',typeof item==='string'?item:item.label||item.name||idOf(item));option.value=idOf(item);select.append(option);}
  select.value=items.some(item=>idOf(item)===previous)?previous:items.some(item=>idOf(item)===preferred)?preferred:idOf(items[0]);
}
function filteredTasks(){const dimension=nodes.dimension.value,query=nodes.search.value.trim().toLowerCase();return tasks().filter(task=>(!dimension||task.dimension===dimension)&&(!query||`${task.id} ${task.name||''}`.toLowerCase().includes(query)));}
function renderTaskList(){
  const visible=filteredTasks();nodes.tasks.replaceChildren();
  for(const task of visible){
    const row=element('label','benchmark-task-option'),input=element('input');input.type='checkbox';input.value=task.id;input.checked=selected.has(task.id);
    input.addEventListener('change',()=>{if(input.checked)selected.add(task.id);else selected.delete(task.id);invalidatePlan();updateControls();});
    const text=element('span'),name=element('strong','',task.name&&task.name!==task.id?task.name:task.id),meta=element('small','',`${DIMENSIONS[task.dimension]||task.dimension||'未分类'}${task.variant==='random'?' · 随机变体':''}${name.textContent!==task.id?` · ${task.id}`:''}`);
    text.append(name,meta);row.append(input,text);nodes.tasks.append(row);
  }
  nodes.emptyTasks.hidden=visible.length>0;updateControls();
}
function renderCatalog(){
  const value=catalog(),items=tasks(),key=JSON.stringify(value);if(key===lastCatalog)return;lastCatalog=key;
  const valid=new Set(items.map(task=>task.id));for(const task of selected)if(!valid.has(task))selected.delete(task);
  if(!initialSelection&&items.length){selected.add(items.some(task=>task.id==='stack_bowls')?'stack_bowls':items[0].id);initialSelection=true;}
  const dimensions=list(value.dimensions).map(item=>typeof item==='string'?{id:item,name:DIMENSIONS[item]||item}:{...item,id:idOf(item),name:DIMENSIONS[idOf(item)]||item.name||idOf(item)});
  options(nodes.dimension,[{id:'',name:'全部能力维度'},...dimensions]);
  options(nodes.embodiment,list(value.embodiments),'arx_x5');options(nodes.importEmbodiment,list(value.embodiments),'arx_x5');
  options(nodes.importTask,items.map(item=>({id:item.id,name:item.name||item.id})),'stack_bowls');
  nodes.revision.textContent=`${value.benchmark_id||'RoboDojo'} · ${value.version||'版本未记录'} · ${value.revision?String(value.revision).slice(0,12):'提交未记录'}`;
  nodes.catalogCount.textContent=`${items.length} 个任务配置 · ${dimensions.length} 个能力维度`;
  if(planPayload){try{if(JSON.stringify(planInput())!==JSON.stringify(planPayload))invalidatePlan();}catch{invalidatePlan();}}
  renderTaskList();
}
function renderReadiness(){
  const readiness=snapshot?.readiness||{},key=JSON.stringify([readiness,reachable]);if(key===lastReadiness)return;lastReadiness=key;
  const ready=reachable&&readiness.ready===true;
  nodes.readinessTitle.textContent=ready?'静态依赖检查通过':'运行环境待接入';
  nodes.readinessMessage.textContent=!reachable?'无法读取工作台服务；目录与检查项可能是上次缓存，请刷新重试。':ready?'原生加载与任务结果以实际运行日志为准。':readiness.message||'可以先准备评测计划、导入已有官方结果；当前未启动仿真。';
  nodes.readiness.classList.toggle('ready',ready);nodes.checks.replaceChildren();
  for(const check of Array.isArray(readiness.checks)?readiness.checks:[]){
    const row=element('li','benchmark-check'),status=String(check.status??'unknown');
    const good=['pass','passed','ok','ready','available'].includes(status),bad=['fail','failed','error','missing'].includes(status);
    const badge=element('span',good?'check-pass':bad?'check-fail':'check-pending',good?'通过':bad?'未就绪':'待确认');
    const detail=element('div'),label=element('strong','',check.label||check.id||'运行检查'),description=element('p','',asText(check.detail));detail.append(label,description);row.append(badge,detail);nodes.checks.append(row);
  }
}
function renderJob(){
  const job=snapshot?.state||{},active=job.active===true;
  nodes.jobStatus.textContent=RUN_STATES[job.status]||job.status||'尚未运行';nodes.jobStatus.classList.toggle('active',active);
  nodes.jobMessage.textContent=job.message||(active?'正在执行官方 RoboDojo 流程。':'生成计划与导入结果都不会启动仿真。');
  const elapsed=finite(job.timing?.elapsed_wall_s)??finite(job.timing?.elapsed_s)??finite(job.elapsed_s);
  nodes.jobTime.textContent=elapsed==null?'墙钟时间：未运行':`运行墙钟：${elapsed.toFixed(1)} s`;
  const completed=finite(job.completed_jobs),total=finite(job.total_jobs);
  nodes.jobProgress.hidden=total==null&&!job.current_task;
  nodes.jobProgress.textContent=[total==null?'':`已归档任务 × 种子：${completed??'—'} / ${total}（单项可能仅含部分回合）`,job.current_task?`${active?'当前':'最后'}任务：${job.current_task} · seed ${job.current_seed??'—'}`:''].filter(Boolean).join('\n');
  nodes.jobRun.textContent=job.run_id?`运行记录：${job.run_id}`:'';
  if(nodes.jobLinks.dataset.run!==(job.run_id||'')){nodes.jobLinks.dataset.run=job.run_id||'';nodes.jobLinks.replaceChildren();if(job.run_id)nodes.jobLinks.append(runLinks(job.run_id));}
}
function ownArtifactUrl(value){
  if(typeof value!=='string'||!value)return null;
  try{const url=new URL(value,window.location.origin);return url.origin===window.location.origin&&url.pathname.startsWith('/api/runs/')?url.pathname+url.search+url.hash:null;}catch{return null;}
}
function nativeVideos(container,items){
  const videos=(Array.isArray(items)?items:[]).map(item=>typeof item==='string'?{url:item}:item).filter(item=>ownArtifactUrl(item?.url));if(!videos.length)return;
  const details=element('details','benchmark-videos'),summary=element('summary','',`官方轨迹视频 · ${videos.length} 路`),grid=element('div','benchmark-video-grid');
  for(const item of videos){const figure=element('figure'),label=element('figcaption','',item.camera||item.name||'官方视频'),video=element('video');video.controls=true;video.playsInline=true;video.preload='none';video.src=ownArtifactUrl(item.url);figure.append(label,video);grid.append(figure);}
  details.append(summary,grid);container.append(details);
}
function runLinks(runId){
  const row=element('div','benchmark-run-links');if(typeof runId!=='string'||!runId)return row;
  const id=encodeURIComponent(runId),open=button('查看实验详情','text-button');
  open.addEventListener('click',()=>window.dispatchEvent(new CustomEvent('embodiworks:open-result',{detail:{run_id:runId}})));
  row.append(open);
  for(const [path,label] of [['analysis','结果 JSON ↗'],['download','下载日志'],['bundle','下载归档']]){const link=element('a','',label);link.href=`/api/runs/${id}/${path}`;if(path!=='analysis')link.download='';else{link.target='_blank';link.rel='noopener noreferrer';}row.append(link);}
  return row;
}
function renderHistory(){
  const history=Array.isArray(snapshot?.history)?snapshot.history:[],key=JSON.stringify(history);if(key===lastHistory)return;lastHistory=key;nodes.history.replaceChildren();
  nodes.historyEmpty.hidden=history.length>0;
  for(const run of history){
    const row=element('article','benchmark-history-row'),benchmark=typeof run.benchmark==='object'&&run.benchmark?run.benchmark:{},metrics=run.metrics||{};
    const heading=element('div','benchmark-history-heading'),title=element('strong','',run.display_name||run.name||benchmark.task_id||run.task_id||run.task||run.run_id||'RoboDojo 结果'),kind=element('span','',RUN_STATES[run.status]||run.status||'本地结果');heading.append(title,kind);row.append(heading);
    const meta=element('p','benchmark-history-meta',[benchmark.policy||run.policy||run.model,benchmark.embodiment||run.embodiment,benchmark.seed!=null?`seed ${benchmark.seed}`:run.seed!=null?`seed ${run.seed}`:'',run.run_id].filter(Boolean).join(' · '));row.append(meta);
    const count=finite(metrics.episodes)??finite(metrics.eval_time)??finite(metrics.evaluation_episodes),sr=finite(metrics.success_rate),score=finite(metrics.score)??finite(metrics.graded_score);
    const numbers=element('div','benchmark-history-metrics');
    for(const [label,value] of [['有效回合',count==null?'未提供':String(count)],['本地任务 SR',sr==null?'未提供':`${(sr*100).toFixed(1)}%`],['本地过程得分',score==null?'未提供':`${score.toFixed(1)} / 100`]]){const metric=element('div');metric.append(element('small','',label),element('strong','',value));numbers.append(metric);}
    row.append(numbers,element('p','benchmark-history-note','本地运行或导入结果；不代表官方完整基准分数。'),runLinks(run.run_id));nativeVideos(row,run.native_videos||run.artifacts?.native_videos);nodes.history.append(row);
  }
}
function updateControls(){
  if(!nodes)return;
  const active=snapshot?.state?.active===true,pending=!!busy||stopPending,hasTasks=selected.size>0&&tasks().length>0;
  nodes.plan.disabled=!reachable||pending||refreshing||!hasTasks;
  nodes.start.disabled=!reachable||pending||refreshing||!hasTasks||snapshot?.readiness?.ready!==true||active||controllerBusy();
  nodes.stop.disabled=stopPending||(!active&&busy!=='start');nodes.refresh.disabled=pending||refreshing;
  nodes.importSubmit.disabled=!reachable||pending||refreshing||!tasks().length;
  nodes.exportScript.disabled=!reachable||pending||refreshing||!planPayload;
  for(const input of nodes.editable)input.disabled=pending;
  for(const input of nodes.tasks.querySelectorAll('input'))input.disabled=pending;
  nodes.selectedCount.textContent=`已选择 ${selected.size} 项 · 当前显示 ${filteredTasks().length} 项`;
  nodes.plan.textContent=busy==='plan'?'正在生成…':'生成评测计划';nodes.start.textContent=busy==='start'?'正在启动…':'运行官方评测';nodes.stop.textContent=stopPending?'正在停止…':'停止评测';nodes.importSubmit.textContent=busy==='import'?'正在导入…':'导入官方结果';
  nodes.exportScript.textContent=busy==='export'?'正在生成 Linux 脚本…':'生成 Linux 运行脚本';
  nodes.startReason.textContent=controllerBusy()?'工作台有自动任务运行，请先停止后再启动基准。':snapshot?.readiness?.ready!==true?'当前可生成计划和导入结果；运行按钮等待官方运行环境就绪。':active?'官方评测正在运行，可在此停止。':'运行使用外部 XPolicyLab WebSocket 策略，不会自动调用工作台 HTTP 模型或 Codex。';
  nodes.root.setAttribute('aria-busy',String(pending||refreshing));
}
async function refresh({announce=false}={}){
  if(refreshing||busy||stopPending)return;
  refreshing=true;updateControls();
  try{
    const value=await request();const data=value.catalog||value;
    if(!Array.isArray(data.tasks))throw new Error('RoboDojo 服务未提供有效任务目录。');
    snapshot=value;reachable=true;renderCatalog();renderReadiness();renderJob();renderHistory();if(announce)note('官方目录与运行状态已刷新。');
  }catch(error){reachable=false;renderReadiness();report(error);}
  finally{lastFetch=Date.now();refreshing=false;updateControls();}
}
function seeds(value){
  const text=value.trim();if(!/^\d+(\s*[,，]\s*\d+)*$/.test(text))throw new Error('种子须为非负整数，用逗号分隔。');
  const result=[...new Set(text.split(/[,，]/).map(Number))];if(result.some(value=>!Number.isSafeInteger(value)||![0,1,2].includes(value)))throw new Error('官方评测种子为 0、1、2。');return result;
}
function planInput(){
  if(!selected.size)throw new Error('请至少选择一个官方任务。');
  const episodeText=nodes.episodes.value.trim(),episodes=episodeText==='native'?'native':/^\d+$/.test(episodeText)?Number(episodeText):null;
  if(episodes!=='native'&&(!Number.isSafeInteger(episodes)||episodes<1||episodes>50))throw new Error('回合数须为 1–50 的整数或 native。');
  const port=Number(nodes.port.value);if(!Number.isInteger(port)||port<1||port>65535)throw new Error('策略服务端口须为 1–65535。');
  const timeText=nodes.wallTime.value.trim(),wallTime=Number(timeText);if(!/^\d+$/.test(timeText)||!Number.isSafeInteger(wallTime)||wallTime<10||wallTime>86400)throw new Error('整次评测墙钟上限须为 10–86400 秒的整数。');
  const policy=nodes.policy.value.trim(),host=nodes.host.value.trim(),checkpoint=nodes.checkpoint.value.trim();
  if(!policy||!host||!checkpoint||!nodes.embodiment.value)throw new Error('请填写策略、本体配置、服务地址和检查点标签。');
  return{task_ids:tasks().filter(task=>selected.has(task.id)).map(task=>task.id),seeds:seeds(nodes.seeds.value),embodiment:nodes.embodiment.value,
    policy_name:policy,policy_host:host,policy_port:port,checkpoint,action_type:nodes.actionType.value,episodes,mode:nodes.mode.value,copy_videos:nodes.recordVideos.checked,max_wall_time_s:wallTime};
}
function invalidatePlan(){
  const hadPlan=planPayload!==null;
  planPayload=null;
  if(planUrl){URL.revokeObjectURL(planUrl);planUrl='';}
  if(scriptUrl){URL.revokeObjectURL(scriptUrl);scriptUrl='';}
  nodes.planDownload.hidden=true;nodes.scriptDownload.hidden=true;
  if(!nodes.planOutput.hidden)nodes.planTitle.textContent='配置已变化 · 请重新生成计划';
  if(hadPlan)note('配置已变化，原计划与脚本已失效，请重新生成评测计划。');
  updateControls();
}
function showPlan(plan,payload){
  invalidatePlan();planPayload=JSON.parse(JSON.stringify(payload));
  nodes.planOutput.hidden=false;nodes.planCommands.replaceChildren();nodes.planWarnings.replaceChildren();
  nodes.planTitle.textContent=`计划 ${plan.plan_id||''} · 尚未执行`;
  for(const warning of Array.isArray(plan.warnings)?plan.warnings:[])nodes.planWarnings.append(element('li','',asText(warning)));
  for(const command of Array.isArray(plan.commands)?plan.commands:[]){const item=element('details','benchmark-command'),heading=element('summary','',`${command.task_id||'任务'} · seed ${command.seed??'—'}`),pre=element('pre','',JSON.stringify(command.argv??command,null,2));item.append(heading,pre);nodes.planCommands.append(item);}
  planUrl=URL.createObjectURL(new Blob([JSON.stringify(plan,null,2)],{type:'application/json;charset=utf-8'}));nodes.planDownload.hidden=false;
  nodes.planDownload.href=planUrl;nodes.planDownload.download=`robodojo-plan-${String(plan.plan_id||'preview').replace(/[^A-Za-z0-9_-]/g,'_').slice(0,80)}.json`;
}
async function createPlan(){
  if(busy||stopPending||refreshing||!reachable)return;
  let payload;try{payload=planInput();}catch(error){report(error);return;}
  busy='plan';updateControls();note('正在读取官方入口并生成参数计划；不会启动仿真。');
  try{showPlan(await request('/plan',payload),payload);note('计划已生成，尚未执行。可下载 JSON，或生成在 Linux 服务器执行的脚本。');}catch(error){report(error);}
  finally{busy='';updateControls();}
}
async function exportScript(){
  if(busy||stopPending||refreshing||!reachable||!planPayload)return;
  try{if(JSON.stringify(planInput())!==JSON.stringify(planPayload)){invalidatePlan();note('配置已变化，请重新生成计划后导出脚本。',true);return;}}catch(error){invalidatePlan();report(error);return;}
  const payload=planPayload;busy='export';updateControls();note('正在生成 Linux 运行脚本，不会连接服务器或启动评测。');
  try{
    const exported=await request('/export',payload);if(payload!==planPayload)return;
    if(typeof exported.script!=='string'||!exported.script.trim())throw new Error('服务未返回有效的 Linux 运行脚本。');
    if(scriptUrl)URL.revokeObjectURL(scriptUrl);scriptUrl=URL.createObjectURL(new Blob([exported.script],{type:'text/x-shellscript;charset=utf-8'}));
    nodes.scriptDownload.href=scriptUrl;nodes.scriptDownload.download=String(exported.filename||'robodojo-plan.sh').replace(/[^A-Za-z0-9_.-]/g,'_');nodes.scriptDownload.hidden=false;
    note('Linux 脚本已生成。先在已准备好的官方源码根目录运行 bash 文件名.sh --dry-run；核对后去掉 --dry-run 执行。');
  }catch(error){report(error);}
  finally{busy='';updateControls();}
}
async function start(){
  if(busy||stopPending||refreshing||!reachable)return;
  if(snapshot?.readiness?.ready!==true||snapshot?.state?.active||controllerBusy()){note('运行条件尚未就绪，或已有任务正在执行。可以先生成计划。',true);return;}
  let payload;try{payload=planInput();}catch(error){report(error);return;}
  const version=++actionVersion;busy='start';updateControls();note('正在提交官方评测任务…');
  try{await request('/start',payload);if(version===actionVersion)note('官方评测任务已提交，运行情况以下方服务状态为准。');}catch(error){if(version===actionVersion)report(error);}
  finally{busy='';updateControls();await refresh();}
}
async function stop(){
  if(stopPending||(!snapshot?.state?.active&&busy!=='start'))return;
  actionVersion++;stopPending=true;updateControls();note('正在停止官方评测并保留已产出的记录…');
  try{await request('/stop',{});note('停止请求已处理；请查看服务返回的最终状态。');}catch(error){report(error);}
  finally{stopPending=false;updateControls();await refresh();}
}
async function importResult(){
  if(busy||stopPending||refreshing||!reachable)return;
  nodes.importResult.replaceChildren();
  let result,payload;
  try{
    try{result=JSON.parse(nodes.importJson.value);}catch{throw new Error('结果 JSON 无法解析，请选择官方 _result.json 或粘贴其完整内容。');}
    if(!result||Array.isArray(result)||typeof result!=='object')throw new Error('官方结果须为 JSON 对象。');
    const seedText=nodes.importSeed.value.trim(),seed=Number(seedText);
    if(!/^\d+$/.test(seedText)||!Number.isSafeInteger(seed)||![0,1,2].includes(seed))throw new Error('导入种子须为 0、1、2。');
    if(!nodes.importTask.value||!nodes.importPolicy.value.trim()||!nodes.importEmbodiment.value)throw new Error('请填写结果对应的任务、策略与本体配置。');
    payload={task_id:nodes.importTask.value,seed,policy:nodes.importPolicy.value.trim(),embodiment:nodes.importEmbodiment.value,payload:result,copy_videos:nodes.copyVideos.checked};
    if(nodes.artifactDir.value.trim())payload.artifact_dir=nodes.artifactDir.value.trim();
  }catch(error){report(error);return;}
  busy='import';updateControls();note('正在校验并归档官方结果；不会启动仿真或调用模型。');
  try{const imported=await request('/import',payload);nodes.importResult.replaceChildren(element('p','',`已导入记录：${imported.run_id||'见历史记录'}。这是结果归档，不是本次现场运行证明。`),runLinks(imported.run_id));nativeVideos(nodes.importResult,imported.native_videos);note('官方结果已归档，可在实验结果与分析中管理。');}
  catch(error){report(error);}
  finally{busy='';updateControls();await refresh();}
}
async function loadFile(){
  const version=++fileVersion,file=nodes.importFile.files?.[0];if(!file)return;
  nodes.importJson.value='';nodes.importResult.replaceChildren();
  if(file.size>10*1024*1024){note('请选择单任务官方 _result.json（不超过 10 MB），不要选择数据集或视频。',true);return;}
  try{const value=await file.text();if(version!==fileVersion)return;JSON.parse(value);nodes.importJson.value=value;note(`已读取 ${file.name}，尚未导入。请核对任务、种子和策略。`);}
  catch(error){if(version===fileVersion)note(`读取结果文件失败：${error.message}`,true);}
}

export function initBenchmarks(callbacks={}){
  context=callbacks;if(initialized)return;const nav=$('.workspace-nav'),main=$('main');if(!nav||!main)return;
  initialized=true;
  const navigation=button('官方基准');navigation.dataset.page='benchmarks';navigation.append(element('span','','ROBODOJO'));const navDetail=nav.querySelector('.nav-detail');if(navDetail)nav.insertBefore(navigation,navDetail);else nav.append(navigation);
  const root=element('section','benchmarks-view');root.id='benchmarks-view';root.hidden=true;root.setAttribute('aria-label','RoboDojo 官方基准');
  const heading=element('div','benchmark-page-heading'),headingText=element('div');headingText.append(element('div','eyebrow','OFFICIAL BENCHMARK / ROBODOJO'),element('h2','','RoboDojo 评测'),element('p','','配置官方任务、连接外部策略、归档可追溯结果。'));
  const refreshButton=button('刷新目录与状态');heading.append(headingText,refreshButton);root.append(heading);
  const boundary=element('div','benchmark-boundary');boundary.append(element('strong','','选择任务 → 生成计划 → 运行或导入结果'),element('p','','本机可以准备评测计划与导入结果。原生仿真需要独立 Linux GPU、Isaac Sim 5.1 和 XPolicyLab 策略服务；可导出 Linux 脚本到准备好的服务器执行。Smoke 检查链路，任务结果取官方环境判据；部分任务不计作官方完整基准分数。'));root.append(boundary);
  const message=element('p','benchmark-message');message.setAttribute('role','status');root.append(message);
  const readiness=element('section','panel benchmark-readiness'),readinessTitle=element('h3','','运行环境待接入'),readinessMessage=element('p',''),checks=element('ul','benchmark-checks');const setupGuide=element('a','text-button','查看运行环境与策略接入指南 ↗');setupGuide.href='https://github.com/shank000/embodied-agent-studio/blob/main/docs/robodojo.md';setupGuide.target='_blank';setupGuide.rel='noopener noreferrer';readiness.append(readinessTitle,readinessMessage,setupGuide,checks);root.append(readiness);
  const layout=element('div','benchmark-layout'),setup=element('section','panel benchmark-setup'),form=element('form','benchmark-form');
  const setupHeading=element('div','benchmark-section-heading'),catalogCount=element('span','','正在读取官方目录'),revision=element('p','benchmark-revision');setupHeading.append(element('h3','','评测计划'),catalogCount);setup.append(setupHeading,revision);
  const filters=element('div','benchmark-filters'),dimension=field('能力维度','benchmark-dimension','select'),search=field('筛选任务','benchmark-search');search.input.placeholder='任务名称';filters.append(dimension.root,search.root);form.append(filters);
  const selection=element('div','benchmark-selection'),selectVisible=button('选择当前显示'),clear=button('清空选择'),selectedCount=element('span','');selection.append(selectVisible,clear,selectedCount);form.append(selection);
  const taskList=element('div','benchmark-tasks'),emptyTasks=element('p','benchmark-muted','没有符合筛选条件的官方任务。');taskList.setAttribute('aria-label','选择官方评测任务');form.append(taskList,emptyTasks);
  const fields=element('div','benchmark-fields'),embodiment=field('本体配置','benchmark-embodiment','select'),seedField=field('种子（逗号分隔）','benchmark-seeds','text','0'),policy=field('XPolicyLab 策略目录名','benchmark-policy','text','demo_policy'),host=field('XPolicyLab 策略主机 / IP','benchmark-host','text','127.0.0.1'),port=field('WebSocket 端口','benchmark-port','number','9999'),checkpoint=field('检查点 / 标签','benchmark-checkpoint','text','external'),actionType=field('动作类型标签','benchmark-action-type','select'),mode=field('评测类型','benchmark-mode','select'),episodes=field('每任务回合数','benchmark-episodes','text','1'),wallTime=field('整次评测墙钟上限 / 秒','benchmark-wall-time','number','3600');wallTime.input.min='10';wallTime.input.max='86400';wallTime.input.step='1';
  options(actionType.input,[{id:'ee',name:'ee · 绝对末端目标'},{id:'joint',name:'joint · 关节目标'}],'ee');options(mode.input,[{id:'smoke',name:'Smoke · 连接与执行检查'},{id:'benchmark',name:'Benchmark · 正式协议候选'}],'smoke');episodes.input.placeholder='正整数或 native';port.input.min='1';port.input.max='65535';
  for(const item of [embodiment,seedField,policy,host,port,checkpoint,actionType,mode,episodes,wallTime])fields.append(item.root);form.append(fields,element('p','benchmark-muted','本体/策略名对应官方配置。native 使用任务规定回合数；整数会受任务原生数量限制。动作标签须与外部策略输出一致，不会自动转换动作。demo_policy 是输出零动作的连通样例，不具备任务策略能力。主机/IP 和端口应指向 XPolicyLab WebSocket 策略服务；工作台主控制模型须先通过 XPolicyLab 部署适配，不能直接填写模型 API 地址或 Codex 模型名。'));
  const recordingLabel=element('label','benchmark-copy-videos'),recordVideos=element('input');recordVideos.type='checkbox';recordVideos.id='benchmark-record-videos';recordingLabel.append(recordVideos,element('span','','将运行产生的原生视频复制到实验归档'));form.append(recordingLabel,element('p','benchmark-muted','墙钟预算覆盖所选全部任务与种子，超时后停止并保留已完成记录。Linux 导出脚本将视频留在官方结果目录，可在下方导入。'));
  const actions=element('div','benchmark-actions'),plan=button('生成评测计划'),startButton=button('运行官方评测','primary-button'),stopButton=button('停止评测','secondary benchmark-stop');actions.append(plan,startButton,stopButton);form.append(actions);const startReason=element('p','benchmark-muted');form.append(startReason);setup.append(form);
  const side=element('div','benchmark-side'),job=element('section','panel benchmark-job'),jobStatus=element('strong','','尚未运行'),jobMessage=element('p',''),jobTime=element('span',''),jobProgress=element('p','benchmark-job-progress'),jobRun=element('p','benchmark-revision'),jobLinks=element('div','benchmark-job-links');job.append(element('h3','','当前评测'),jobStatus,jobMessage,jobTime,jobProgress,jobRun,jobLinks);side.append(job);
  const planOutput=element('section','panel benchmark-plan-output'),planHeading=element('div','benchmark-section-heading'),planTitle=element('h3','','计划预览'),planDownload=element('a','','下载计划 JSON'),scriptButton=button('生成 Linux 运行脚本'),scriptDownload=element('a','','下载 Linux 脚本');scriptDownload.hidden=true;planOutput.hidden=true;planHeading.append(planTitle,planDownload);const planWarnings=element('ul','benchmark-plan-warnings'),planCommands=element('div','benchmark-plan-commands');const exports=element('div','benchmark-actions');exports.append(scriptButton,scriptDownload);planOutput.append(planHeading,element('p','benchmark-muted','参数数组使用本机路径；在 Linux 服务器运行请下载脚本。生成和下载都不会执行命令，修改配置后需重新生成。'),exports,planWarnings,planCommands);side.append(planOutput);layout.append(setup,side);root.append(layout);
  const importer=element('section','panel benchmark-import'),importForm=element('form','benchmark-import-form');importer.append(element('h3','','导入官方结果'),element('p','benchmark-muted','可在没有仿真服务器时归档已有 _result.json。eval_time 表示有效回合数，不是秒；SR 与过程得分独立保存。'));
  const importFields=element('div','benchmark-import-fields'),importTask=field('结果任务','benchmark-import-task','select'),importSeed=field('结果种子','benchmark-import-seed','number','0'),importPolicy=field('策略名称','benchmark-import-policy','text','demo_policy'),importEmbodiment=field('结果本体','benchmark-import-embodiment','select');for(const item of [importTask,importSeed,importPolicy,importEmbodiment])importFields.append(item.root);importForm.append(importFields);
  const file=field('读取官方 _result.json','benchmark-import-file','file');file.input.accept='.json,application/json';const jsonLabel=element('label','benchmark-field','或粘贴完整官方结果 JSON'),importJson=element('textarea','benchmark-json');importJson.id='benchmark-import-json';importJson.rows=7;importJson.spellcheck=false;jsonLabel.append(importJson);importForm.append(file.root,jsonLabel);
  const artifactDir=field('本机官方结果目录（可选）','benchmark-artifact-dir');artifactDir.input.placeholder='目录用于关联原生视频，不会执行其中的文件';const copyLabel=element('label','benchmark-copy-videos'),copyVideos=element('input');copyVideos.type='checkbox';copyVideos.id='benchmark-copy-videos';copyVideos.checked=false;copyLabel.append(copyVideos,element('span','','复制原生视频到实验归档（默认不复制）'));const importSubmit=button('导入官方结果','primary-button'),importResultContainer=element('div','benchmark-import-result');importForm.append(artifactDir.root,copyLabel,importSubmit);importer.append(importForm,importResultContainer);root.append(importer);
  const historySection=element('section','panel benchmark-history');historySection.append(element('h3','','RoboDojo 实验记录'),element('p','benchmark-muted','每条记录可进入现有实验管理或下载原始归档。部分任务结果不会汇总成官方完整分数。'),element('p','benchmark-muted','外部策略未上报的 token、推理耗时和费用记为未知，不按零费用计算。'));const history=element('div','benchmark-history-list'),historyEmpty=element('p','benchmark-empty','尚无 RoboDojo 记录。可先生成计划，或导入已有官方结果。');historySection.append(history,historyEmpty);root.append(historySection);
  main.insertBefore(root,main.querySelector('footer'));
  nodes={root,readiness,readinessTitle,readinessMessage,checks,revision,catalogCount,dimension:dimension.input,search:search.input,tasks:taskList,emptyTasks,selectedCount,embodiment:embodiment.input,seeds:seedField.input,policy:policy.input,host:host.input,port:port.input,checkpoint:checkpoint.input,actionType:actionType.input,mode:mode.input,episodes:episodes.input,wallTime:wallTime.input,recordVideos,plan,start:startButton,stop:stopButton,refresh:refreshButton,startReason,jobStatus,jobMessage,jobTime,jobProgress,jobRun,jobLinks,planOutput,planTitle,planDownload,exportScript:scriptButton,scriptDownload,planWarnings,planCommands,importTask:importTask.input,importSeed:importSeed.input,importPolicy:importPolicy.input,importEmbodiment:importEmbodiment.input,importFile:file.input,importJson,artifactDir:artifactDir.input,copyVideos,importSubmit,importResult:importResultContainer,history,historyEmpty,message,
    editable:[dimension.input,search.input,embodiment.input,seedField.input,policy.input,host.input,port.input,checkpoint.input,actionType.input,mode.input,episodes.input,wallTime.input,recordVideos,importTask.input,importSeed.input,importPolicy.input,importEmbodiment.input,file.input,importJson,artifactDir.input,copyVideos,selectVisible,clear]};
  dimension.input.addEventListener('change',renderTaskList);search.input.addEventListener('input',renderTaskList);selectVisible.addEventListener('click',()=>{for(const task of filteredTasks())selected.add(task.id);invalidatePlan();renderTaskList();});clear.addEventListener('click',()=>{selected.clear();invalidatePlan();renderTaskList();});
  for(const input of [embodiment.input,seedField.input,policy.input,host.input,port.input,checkpoint.input,actionType.input,mode.input,episodes.input,wallTime.input,recordVideos]){input.addEventListener('input',invalidatePlan);input.addEventListener('change',invalidatePlan);}
  scriptButton.addEventListener('click',exportScript);
  form.addEventListener('submit',event=>{event.preventDefault();return createPlan();});plan.addEventListener('click',createPlan);startButton.addEventListener('click',start);stopButton.addEventListener('click',stop);refreshButton.addEventListener('click',()=>refresh({announce:true}));navigation.addEventListener('click',()=>refresh());
  importForm.addEventListener('submit',event=>{event.preventDefault();return importResult();});
  // The result container and action handler deliberately have different names.
  importSubmit.addEventListener('click',importResult);file.input.addEventListener('change',loadFile);importJson.addEventListener('input',()=>{fileVersion++;});
  updateControls();return refresh();
}

export function renderBenchmarks(state){
  platformState=state;if(!nodes)return;updateControls();
  if(!nodes.root.hidden&&!refreshing&&!busy&&!stopPending&&Date.now()-lastFetch>4000)void refresh();
}
