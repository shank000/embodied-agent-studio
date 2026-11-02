// BEHAVIOR's official evaluator has its own lifecycle and protocol.
const $=selector=>document.querySelector(selector);
let panel,provider,legacy,platform={},info=null,busy=false,fetching=false,lastFetch=0,toast=()=>{};
const API='/api/benchmarks/behavior';
export function behaviorPlan(values){
  const indices=String(values.instances||'0').split(/[，,\s]+/).filter(Boolean).map(Number);
  if(!indices.length||indices.some(n=>!Number.isInteger(n)||n<0)||new Set(indices).size!==indices.length)throw new Error('实例编号须为不重复的非负整数。');
  return {task_name:values.task.trim(),instance_indices:indices,mode:values.mode,policy_host:values.host.trim(),
    policy_port:Number(values.port),policy_scheme:values.scheme,policy_name:values.policy.trim(),
    num_rollouts:Number(values.rollouts),max_steps:values.maxSteps.trim()?Number(values.maxSteps):null,
    record_video:values.video,max_wall_time_s:Number(values.wall)};
}
export function behaviorScore(evaluation){
  if(!evaluation)return '尚无原生评价';
  const q=typeof evaluation.q_score==='number'?evaluation.q_score:evaluation.q_score?.final;
  return `完全成功：${evaluation.success===true?'是':evaluation.success===false?'否':'未判定'} · 目标完成 Q：${Number.isFinite(q)?q.toFixed(3):'未返回'}`;
}
function renderHistory(){
  const root=$('#behavior-history');root.replaceChildren();
  if(!info?.history?.length){root.textContent='尚无 BEHAVIOR 记录。运行或导入结果后，可在这里进入实验分析。';return;}
  for(const item of info.history){
    const row=document.createElement('p'),button=document.createElement('button');button.type='button';button.className='secondary';
    button.textContent=item.display_name||`${item.model||'策略'} · ${item.run_id}`;
    button.onclick=()=>window.dispatchEvent(new CustomEvent('embodiworks:open-result',{detail:{run_id:item.run_id}}));
    const metrics=item.metrics||{},description=document.createElement('span');
    description.textContent=` ${item.status||'未知'} · ${metrics.episodes??0} 个回合 · 成功率 ${Number.isFinite(metrics.success_rate)?(100*metrics.success_rate).toFixed(1)+'%':'未知'} · Q ${Number.isFinite(metrics.mean_reported_q)?metrics.mean_reported_q.toFixed(3):'未知'}`;
    row.append(button,description);root.append(row);
  }
}
async function request(path='',payload){
  const response=await fetch(API+path,payload===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  const data=await response.json();if(!response.ok)throw new Error(typeof data.detail==='string'?data.detail:JSON.stringify(data.detail||data));return data;
}
function message(value,error=false){$('#behavior-message').textContent=value;$('#behavior-message').classList.toggle('config-error',error);}
function input(){const v=id=>$('#behavior-'+id).value;return behaviorPlan({task:v('task'),instances:v('instances'),mode:v('mode'),host:v('host'),port:v('port'),scheme:v('scheme'),policy:v('policy'),rollouts:v('rollouts'),maxSteps:v('max-steps'),video:$('#behavior-video').checked,wall:v('wall')});}
function locks(){
  if(!panel)return;
  const own=info?.state?.active===true;
  const occupied=platform.backend_switching||['agent','policy','comparison','research','experiment','benchmark','harnesspai','repair'].some(k=>platform[k]?.active);
  $('#behavior-start').disabled=busy||!info?.readiness?.ready||own||occupied;
  $('#behavior-stop').disabled=busy||!own;
  $('#behavior-plan').disabled=busy||own;$('#behavior-import').disabled=busy||own;
  panel.querySelectorAll('input,select').forEach(node=>node.disabled=busy||own);
}
async function action(fn){if(busy)return;busy=true;locks();try{await fn();await refresh();}catch(error){message(error.message,true);toast(error.message);}finally{busy=false;locks();}}
async function refresh(){
  if(fetching||!panel)return;fetching=true;
  try{
    info=await request();lastFetch=Date.now();
    $('#behavior-readiness').textContent=info.readiness?.message||'运行环境未确认';
    const checks=$('#behavior-checks');checks.replaceChildren();
    for(const check of info.readiness?.checks||[]){const li=document.createElement('li');li.textContent=`${check.status==='passed'?'✓':'待准备'} ${check.label}：${check.detail||''}`;checks.append(li);}
    const state=info.state||{},elapsed=state.timing?.elapsed_wall_s;$('#behavior-job').textContent=(state.message||'尚未启动官方评估器。')+(Number.isFinite(elapsed)?` · 墙钟 ${elapsed.toFixed(1)} 秒`:'');
    const run=$('#behavior-run');run.hidden=!state.run_id;if(state.run_id){run.href=`/api/runs/${encodeURIComponent(state.run_id)}/bundle`;run.textContent='下载本次评测记录与原生文件';}
    $('#behavior-results').textContent=JSON.stringify(state.results||[],null,2);
    renderHistory();
    const list=$('#behavior-tasks');list.replaceChildren();
    for(const task of info.tasks||[]){const option=document.createElement('option');option.value=typeof task==='string'?task:task.id||task.name;list.append(option);}
  }catch(error){info=null;message(`BEHAVIOR 状态读取失败：${error.message}`,true);}finally{fetching=false;locks();}
}
export function initBehavior(callbacks={}){
  const root=$('#benchmarks-view');if(!root||panel)return;toast=callbacks.toast||toast;
  legacy=document.createElement('div');legacy.id='robodojo-benchmark-content';while(root.firstChild)legacy.append(root.firstChild);root.append(legacy);
  const label=document.createElement('label');label.className='benchmark-field';label.htmlFor='benchmark-provider';label.textContent='评测环境';provider=document.createElement('select');provider.id='benchmark-provider';provider.setAttribute('aria-label','评测环境');provider.innerHTML='<option value="robodojo">RoboDojo</option><option value="behavior">BEHAVIOR / OmniGibson</option>';label.append(provider);root.prepend(label);
  root.setAttribute('aria-label','原生基准评测');const nav=$('[data-page="benchmarks"] span');if(nav)nav.textContent='ROBODOJO / BEHAVIOR';
  panel=document.createElement('section');panel.id='behavior-benchmark-panel';panel.className='panel benchmark-setup';panel.hidden=true;
  panel.innerHTML=`<div class="eyebrow">BEHAVIOR / OMNIGIBSON</div><h2>长序列家务任务评测</h2><p>交互研究在“仿真环境”中加载 OmniGibson；这里运行上游评估器并连接外部 WebSocket 策略服务。原生 Q、成功率和视频直接归档，不将接口测试算作任务成绩。</p>
    <p>固定适配版本 v3.9.4；官方赛事版本要求须另行核对。此评估器采用官方默认 assisted 抓取，交互研究使用 physical 抓取，两种结果须分开比较。外部策略未报告的 Token 与费用保持未知。运行前请切回 MuJoCo，以释放已加载的重型仿真。</p>
    <details><summary id="behavior-readiness">读取准备条件…</summary><ul id="behavior-checks"></ul><p>部署说明：docs/omnigibson.md。可以先生成计划；准备条件满足后才可启动。</p></details>
    <form id="behavior-form" class="benchmark-form"><div class="benchmark-filters">
      <label class="benchmark-field">任务名称<input id="behavior-task" list="behavior-tasks" value="turning_on_radio" required><datalist id="behavior-tasks"></datalist></label>
      <label class="benchmark-field">实例编号<input id="behavior-instances" value="0" required placeholder="0,1,2"></label>
      <label class="benchmark-field">划分<select id="behavior-mode"><option value="public_test">Public test</option><option value="train">Train</option></select></label>
      <label class="benchmark-field">策略名称<input id="behavior-policy" value="external-policy" required></label>
      <label class="benchmark-field">策略主机<input id="behavior-host" value="127.0.0.1" required></label>
      <label class="benchmark-field">策略端口<input id="behavior-port" type="number" min="1" max="65535" value="8001" required></label>
      <label class="benchmark-field">传输<select id="behavior-scheme"><option value="ws">ws</option><option value="wss">wss</option></select></label>
      <label class="benchmark-field">每实例回合数<input id="behavior-rollouts" type="number" min="1" value="1" required></label>
      <label class="benchmark-field">原生步骤上限<input id="behavior-max-steps" type="number" min="1" placeholder="留空使用官方默认"></label>
      <label class="benchmark-field">墙钟预算 / 秒<input id="behavior-wall" type="number" min="1" value="3600" required></label>
    </div><label class="record-video-option"><input id="behavior-video" type="checkbox" checked>保存官方轨迹视频</label>
    <div class="rsi-actions"><button id="behavior-plan" type="button" class="secondary">预览执行计划</button><button id="behavior-start" type="submit" class="primary-button" disabled>运行原生评估器</button><button id="behavior-stop" type="button" class="secondary stop" disabled>停止评测</button><button id="behavior-refresh" type="button" class="secondary">刷新准备状态</button></div></form>
    <p id="behavior-message" role="status"></p><p id="behavior-job"></p><a id="behavior-run" class="text-button" hidden>下载记录</a>
    <details><summary>执行计划与版本</summary><pre id="behavior-plan-output" style="white-space:pre-wrap;overflow-wrap:anywhere"></pre></details>
    <details><summary>原生评测结果</summary><pre id="behavior-results" style="white-space:pre-wrap;overflow-wrap:anywhere"></pre></details>
    <details><summary>导入已有官方 JSON 结果</summary><p>导入保留原始指标并标记外部来源，不声称由本机执行。使用上方策略名称与划分。</p><input id="behavior-result-file" type="file" accept=".json,application/json"><button id="behavior-import" class="secondary" type="button">导入结果</button></details>
    <details><summary>BEHAVIOR 实验记录</summary><p>点击记录名称进入现有实验管理，查看视频、原生指标和成本。部分任务结果不代表整个挑战成绩。</p><div id="behavior-history"></div></details>`;
  root.append(panel);
  provider.onchange=()=>{legacy.hidden=provider.value==='behavior';panel.hidden=!legacy.hidden;if(!panel.hidden)refresh();};
  $('#behavior-plan').onclick=()=>action(async()=>{$('#behavior-plan-output').textContent=JSON.stringify(await request('/plan',input()),null,2);message('计划已生成；尚未启动原生仿真。');});
  $('#behavior-form').onsubmit=event=>{event.preventDefault();action(async()=>{await request('/start',input());message('评测已提交；任务结果以官方评估器输出为准。');});};
  $('#behavior-stop').onclick=()=>action(async()=>{await request('/stop',{});message('停止请求已处理，已有结果保留。');});
  $('#behavior-refresh').onclick=()=>refresh();
  $('#behavior-import').onclick=()=>action(async()=>{const file=$('#behavior-result-file').files[0];if(!file)throw new Error('请选择官方 JSON 结果文件。');const result=await request('/import',{payload:JSON.parse(await file.text()),policy_name:$('#behavior-policy').value.trim(),mode:$('#behavior-mode').value});message(`已导入外部结果：${result.run_id||'完成'}`);});
}
export function renderBehavior(state){platform=state||{};if(!panel)return;locks();if(!panel.hidden&&!panel.closest('[hidden]')&&Date.now()-lastFetch>3000)refresh();}
