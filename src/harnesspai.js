import './harnesspai.css';

const $ = selector => document.querySelector(selector);
const escape = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let info, live = {}, busy = false, callbacks, initialized = false, lastRows = '', renderedRun, renderedRows = [];
let renderedBatches = [], batchRun;
const labels = {idle:'等待程序',preparing:'准备验证',running:'验证中',verified:'全部通过',unverified:'未全部通过',error:'执行错误',stopped:'已停止'};
const artifact = (id, path) => `/api/runs/${encodeURIComponent(id)}/artifact/${path.split('/').map(encodeURIComponent).join('/')}`;

async function request(path, payload) {
  const response = await fetch(`/api/harnesspai${path}`, payload === undefined ? {} : {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  const data = await response.json();
  if (response.status === 404 && !path) throw new Error('当前后端尚未加载 HarnessPAI 接口，请重启工作台服务后刷新此页。');
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : (data.detail || []).map(e => e.msg).join('；') || '请求失败');
  return data;
}

export function programResultsMarkup(runId, rows = []) {
  return rows.map(row => {
    const base = `${row.artifact_dir || `episodes/seed-${row.seed}`}/`, video = row.video;
    const checks = row.evaluation?.checks || [];
    return `<details class="harnesspai-trial"><summary>种子 ${escape(row.seed)} · ${row.success === true ? '环境验收通过' : row.process_ok === false ? '执行错误' : '未通过环境验收'} · ${row.elapsed_s == null ? '未记录耗时' : Number(row.elapsed_s).toFixed(2)+' s'} · ${Number(row.steps || 0)} 步</summary>
      ${row.message ? `<p class="error">${escape(row.message)}</p>` : ''}
      <p>${escape(row.evaluation?.criterion || '缺少原生验收结果。程序返回值不作为成功依据。')}</p>
      <ul>${checks.map(check => `<li>${check.passed ? '✓' : '×'} ${escape(check.label)}：${escape(typeof check.actual === 'number' ? Number(check.actual.toFixed(6)) : check.actual)} ${escape(check.unit)}（要求 ${escape(check.expected)}）</li>`).join('')}</ul>
      <p class="harnesspai-links"><a href="${artifact(runId, base + 'stdout.log')}" target="_blank" rel="noopener">标准输出</a><a href="${artifact(runId, base + 'stderr.log')}" target="_blank" rel="noopener">错误日志</a>${row.record_path ? `<a href="${artifact(runId, base + row.record_path + '/events.jsonl')}" download>逐步观察、动作与反馈</a>` : ''}</p>
      ${video?.status === 'saved' ? `<video controls playsinline preload="none" src="${artifact(runId, base + video.path)}"></video>` : `<p>${video?.status === 'failed' ? '视频保存失败：'+escape(video.error) : '本轮无已完成的视频。'}</p>`}
    </details>`;
  }).join('');
}

export function evolutionResultsMarkup(runId, result = {}) {
  if (!['auto','evolve'].includes(result.mode) && !result.batches?.length) return '';
  return `<h4>程序演化 · 模型权重保持冻结</h4><p>编码与诊断调用 ${Number(result.model_calls || 0)} 次；动作模型与工具调用 ${Number(result.primitive_calls || 0)} 次。演化过程使用的种子与最终验证分开；本地 MuJoCo 适配结果不等同于论文 LIBERO＋π0.5 分数。</p>
    ${modelResponseMarkup(result.last_model_response)}${evolutionBatchesMarkup(runId, result.batches || [])}
    ${result.skill_writeback ? `<p class="${result.skill_writeback.ok ? 'rsi-help' : 'error'}">经验保存：${result.skill_writeback.ok ? '已写入，可供后续演化检索' : escape(result.skill_writeback.error || '未完成；程序验证结果仍保留')}</p>` : ''}`;
}

function evolutionBatchesMarkup(runId, batches, offset=0) {
  return batches.map((batch, index) => {
    const round = batch.path?.match(/round_(\d+)/)?.[1];
    const label = batch.stage === 'baseline' ? '初始程序' : round != null ? `演化第 ${Number(round)+1} 轮 · ${batch.path.includes('/probe') ? '小批次检查' : '完整演化批次'}` : `演化批次 ${index+offset}`;
    return `<details class="harnesspai-trial"><summary>${label} · 通过 ${Math.round((batch.success_rate || 0)*100)}% · ${Number(batch.elapsed_s || 0).toFixed(2)} s</summary><p>${escape(batch.path)}</p>${programResultsMarkup(runId, batch.rows)}</details>`;
  }).join('');
}

function modelResponseMarkup(response) {
  if (!response) return '';
  return `<details class="harnesspai-trial" ${response.role === 'edit' && !response.file_count ? 'open' : ''}><summary>最近模型调用 ${Number(response.id)} · ${{edit:'程序改写',diagnosis:'失败分析',memory:'经验整理'}[response.role] || '模型反馈'} · 返回 ${Number(response.file_count || 0)} 个文件</summary><p style="white-space:pre-wrap">${escape(response.explanation)}</p></details>`;
}

export function renderProgramAnalysis(analysis) {
  let root = $('#result-harnesspai');
  if (!root) { root = document.createElement('section'); root.id = 'result-harnesspai'; $('#result-summary').after(root); }
  root.hidden = !analysis.harnesspai;
  if (root.hidden) { root.replaceChildren(); return; }
  const result = analysis.harnesspai;
  root.innerHTML = `<h3>HarnessPAI 程序验证</h3><p>完成 ${result.completed_seeds} / ${result.requested_seeds?.length || 0} 个最终验证种子。成功由 MuJoCo 原生判据确认。编码模型、已配置动作模型和工具的调用计入本实验；自定义程序自行发起的外部请求尚未计量。</p><p class="harnesspai-links"><a href="${artifact(analysis.run_id, 'candidate/workbench_candidate/program.py')}" download>下载最终冻结程序</a><a href="/api/runs/${encodeURIComponent(analysis.run_id)}/bundle" download>下载完整实验</a></p>${automationMarkup(result)}${evolutionResultsMarkup(analysis.run_id, result)}${programResultsMarkup(analysis.run_id, result.rows)}${analysis.status === 'verified' && result.rows?.length >= 2 && !result.archive?.ok ? `<button class="secondary" type="button" data-program-promote="${escape(analysis.run_id)}">保存为已验证程序</button>` : ''}`;
}

function tasks() {
  const scene = $('#hp-scene').value, select = $('#hp-task');
  select.innerHTML = (info.scenes[scene]?.tasks || []).map(id => `<option value="${escape(id)}">${escape(info.tasks[id].name)}</option>`).join('');
  taskDescription();
}

function taskDescription() {
  const task = info?.tasks?.[$('#hp-task').value];
  const b=task?.task_spec?.recommended_budget;
  $('#hp-task-description').textContent = task ? `${task.instruction} 环境验收：${task.criterion}${b?` 建议${b.sim_seconds}仿真秒、${b.max_actions}动作；下方预算请按需调整。`:''}` : '';
  $('#hp-use-budget').hidden=!b;
}

function modeFields() {
  const mode = $('#hp-mode').value;
  $('#hp-evolve-settings').hidden = mode === 'validate';
  $('#hp-program-editor').hidden = mode === 'auto';
  $('#hp-source').required = mode !== 'auto';
  $('#hp-start').textContent = mode === 'auto' ? '开始任务 · 自动复用与演化' : mode === 'evolve' ? '开始演化与独立验证' : '冻结并验证程序';
}

export function automationMarkup(result={}) {
  const origin = result.source_origin, archive = result.archive;
  return `${origin ? `<p class="rsi-help">${origin.kind === 'retrieved' ? '已自动匹配任务、场景和工具接口相同的程序，先复测，通过后直接执行最终验证。已有版本的复测不代表全新的未见种子评测。' : origin.kind === 'generated' ? '未找到匹配程序，由编码模型自动构建候选；通过环境验收后归档。' : '使用高级模式提供的候选程序。'}</p>` : ''}${archive ? `<p class="${archive.ok ? 'rsi-help' : 'error'}">${archive.ok ? (archive.reused ? '已复用原程序版本，无需重复注册。' : '已自动保存为可复用程序，下次相同任务自动检索。') : '程序通过验证，但自动归档失败：'+escape(archive.error)}</p>` : ''}`;
}

function addTool() {
  if (document.querySelectorAll('.hp-tool-row').length >= 8) return;
  $('#hp-tools').insertAdjacentHTML('beforeend', `<div class="hp-tool-row harnesspai-trial"><div class="rsi-fields"><label>工具名称 · 英文<input data-tool="name" required pattern="[a-z][a-z0-9_]{0,39}" placeholder="vision"></label><label>用途<select data-tool="kind"><option value="perception">视觉感知</option><option value="decision">决策</option></select></label><label>模型名称<input data-tool="model" required></label><label>推理地址<input data-tool="endpoint" type="url" required></label><label>API Key<input data-tool="api_key" type="password" autocomplete="off"></label></div><label>能力与返回内容说明<textarea data-tool="description" rows="2" required placeholder="例如：输入目标描述，返回像素坐标与置信度。"></textarea></label><button class="secondary" type="button" data-remove-tool>移除工具</button></div>`);
}

async function refresh() {
  let next;
  try { next = await request(''); }
  catch (error) { info = {ready:false}; $('#hp-install').textContent = error.message; throw error; }
  info = next;
  // Older running servers may serve the latest built UI from the same directory.
  // Read the actual API schema before offering an unsupported request shape.
  const schemaResponse = await fetch('/openapi.json');
  if (!schemaResponse.ok) throw new Error('无法读取当前服务支持的功能，请刷新运行环境。');
  const apiSchema = await schemaResponse.json();
  info.evolution_ready = apiSchema.components?.schemas?.ProgramValidation?.properties?.mode?.enum?.includes('evolve') === true;
  info.auto_ready = apiSchema.components?.schemas?.ProgramValidation?.properties?.mode?.enum?.includes('auto') === true;
  $('#hp-mode').querySelector('option[value="auto"]').disabled = !info.auto_ready;
  $('#hp-mode').querySelector('option[value="evolve"]').disabled = !info.evolution_ready;
  if (!info.auto_ready && $('#hp-mode').value === 'auto') $('#hp-mode').value = info.evolution_ready ? 'evolve' : 'validate';
  if (!info.evolution_ready) $('#hp-mode').value = 'validate';
  modeFields();
  $('#hp-primitives').disabled = !info.auto_ready;
  $('#hp-install').textContent = info.ready ? `${info.scope} 上游版本 ${info.revision.slice(0, 8)}。` : info.error;
  if (info.ready && !info.evolution_ready) $('#hp-install').textContent += ' 此服务尚未加载自动演化更新，当前可运行固定程序验证。';
  else if (info.ready && !info.auto_ready) $('#hp-install').textContent += ' 此服务尚未加载自动任务更新；重启工作台后即可自动检索、生成与归档。当前保留高级程序模式。';
  if (!initialized) {
    $('#hp-source').value = info.example;
    $('#hp-scene').innerHTML = Object.entries(info.scenes).map(([id, scene]) => `<option value="${escape(id)}">${escape(scene.name)}</option>`).join('');
    tasks(); $('#hp-task').value = info.auto_ready ? 'lift_red' : 'reach_up'; taskDescription(); initialized = true;
  }
  const versions = $('#hp-library');
  versions.innerHTML = '<option value="">选择已验证程序…</option>' + info.versions.map(v => `<option value="${escape(v.run_id)}">${escape(v.name)} · ${escape(v.run_id)}</option>`).join('');
  live = info.state;
  renderHarnessPAI(callbacks?.getState?.() || {});
}

async function action(work) {
  if (busy) return;
  busy = true; $('#hp-error').textContent = '';
  renderHarnessPAI(callbacks?.getState?.() || {});
  try { await work(); }
  catch (error) { $('#hp-error').textContent = error.message; callbacks?.toast?.(error.message); }
  finally { busy = false; renderHarnessPAI(callbacks?.getState?.() || {}); }
}

export function initHarnessPAI(context) {
  callbacks = context;
  const root = $('#harnesspai-panel'); if (!root) return;
  root.innerHTML = `<div class="rsi-card-heading"><div><span class="eyebrow">HARNESSPAI / TASK EXECUTION</span><h3>选择任务，自动复用与演化。</h3></div><span id="hp-status" class="result-pill">检查运行环境</span></div>
    <p id="hp-install" class="rsi-help" role="status">正在读取运行环境…</p><p class="rsi-help">在独立的 MuJoCo / Panda 场景中验证。自动演化复用官方调度：改写候选程序、失败反馈、重试验证、保留经验。模型权重始终冻结，不启动训练。原版 LIBERO＋π0.5、SAM3、Molmo2 实验仍需独立模型服务。</p>
    <form id="hp-form"><fieldset id="hp-inputs"><div class="rsi-fields"><label>场景<select id="hp-scene" required></select></label><label>任务<select id="hp-task" required></select></label></div><p id="hp-task-description" class="rsi-help"></p><button id="hp-use-budget" type="button" class="text-button" hidden>使用任务建议预算 ↗</button>
    <label>运行方式<select id="hp-mode"><option value="auto">自动任务 · 检索、生成、验证与归档</option><option value="validate">高级：固定程序验证</option><option value="evolve">高级：从指定程序演化</option></select></label>
    <div id="hp-evolve-settings"><h4>编码与失败诊断模型</h4><div class="rsi-fields"><label>接入方式<select id="hp-provider"><option value="codex">本机 Codex 登录</option><option value="openai">OpenAI Responses API</option><option value="compatible">Chat Completions 兼容 API</option><option value="anthropic">Anthropic API</option><option value="custom">自定义模型服务</option></select></label><label>模型名称<input id="hp-model" value="gpt-6-astra"></label><label>服务地址<input id="hp-base-url" value="https://api.openai.com/v1"></label><label>API Key · 仅在内存保留<input id="hp-api-key" type="password" autocomplete="off"></label></div><p class="rsi-help">已有程序通过复测时不会调用编码模型。新任务或执行失败时才生成、修改程序；模型权重保持冻结。Codex 登录无需填写密钥。</p></div>
    <details class="harnesspai-trial"><summary>动作模型与视觉／决策工具 · 可选</summary><fieldset id="hp-primitives"><p class="rsi-help">默认使用图像处理与 MuJoCo IK。可接入适配 Panda 的 VLA / WAM；编码模型负责组织程序，动作模型负责输出动作，两者可以分别替换。</p><label>动作能力<select id="hp-policy-kind"><option value="">直接控制 · MuJoCo IK</option><option value="vla">VLA 策略服务</option><option value="wam">WAM 策略服务</option></select></label><div id="hp-policy-settings" class="rsi-fields" hidden><label>策略模型名称<input id="hp-policy-model"></label><label>推理地址<input id="hp-policy-endpoint" type="url" placeholder="https://service.example/predict"></label><label>API Key<input id="hp-policy-key" type="password" autocomplete="off"></label></div><p class="rsi-help">策略接口沿用 embodied-policy-v1，须返回绝对 TCP 位姿或具名执行器动作块；默认 SmolVLA 的 SO-101 映射不能直接用于 Panda。外部工具使用 embodied-tool-v1。未配置的服务不会自动下载或假装运行。</p><div id="hp-tools"></div><button id="hp-add-tool" type="button" class="secondary">添加视觉／决策工具</button></fieldset></details>
    <details class="harnesspai-trial"><summary>验证种子与运行预算</summary><div class="rsi-fields"><label>最终验证种子<input id="hp-seeds" value="42,43" required></label><label>开发／演化种子<input id="hp-evolution-seeds" value="0,1,2"></label><label>最多演化轮数<input id="hp-rounds" type="number" min="5" max="50" value="5"></label><label>每个种子超时 / 秒<input id="hp-timeout" type="number" value="120" min="10" max="3600" required></label><label>最多动作<input id="hp-steps" type="number" value="20" min="1" max="1000" required></label><label>仿真时间 / 秒<input id="hp-sim" type="number" value="30" min="1" max="300" required></label></div><p class="rsi-help">两个种子集合须分开；最终验证结果不用于本次程序修改。自动任务至少需要两个最终验证种子，成功后自动归档。未知任务需先配置环境验收标准；当前提供的任务都有原生判据。</p></details>
    <div id="hp-program-editor" hidden><div class="rsi-fields"><label>程序名称<input id="hp-name" value="末端抬高程序" maxlength="100"></label><label>载入程序版本<select id="hp-library"><option value="">暂无程序</option></select></label></div><label>候选程序 · def run(ctx)<textarea id="hp-source" rows="15" spellcheck="false" aria-describedby="hp-code-help"></textarea></label><p id="hp-code-help" class="rsi-help">高级模式原样验证或演化这份代码。自动任务无需选择、编辑或注册程序。ctx.observe() 获取图像与机器人状态；ctx.move_tcp() 执行动作；ctx.policy() 与 ctx.tool() 调用已配置能力。</p></div>
    <label class="record-video-option"><input id="hp-video" type="checkbox" checked>为每个种子保存轨迹视频</label><label class="record-video-option"><input id="hp-consent" type="checkbox" required>允许在本机执行本次检索或生成的 Python 程序（拥有本机用户权限），并保存通过验证的版本。</label></fieldset>
    <div class="rsi-actions"><button id="hp-refresh" class="secondary" type="button">刷新运行环境</button><button id="hp-stop" class="secondary stop" type="button" disabled>停止验证</button><button id="hp-start" class="primary-button" type="submit" disabled>冻结并验证程序</button></div></form>
    <p id="hp-error" class="error" role="alert"></p><p id="hp-message" class="rsi-help" role="status"></p><a id="hp-runner-log" class="text-button" hidden target="_blank" rel="noopener">查看完整执行日志 ↗</a><p id="hp-time" class="rsi-help"></p><div id="hp-automation"></div><p id="hp-evolution-meta" class="rsi-help"></p><div id="hp-model-response"></div><div id="hp-evolution-results"></div><div id="hp-results"></div><div class="rsi-actions"><a id="hp-bundle" class="text-button" hidden download>下载完整程序与验证材料 ↗</a><button id="hp-promote" class="secondary" type="button" hidden>保存为已验证程序</button></div>`;
  $('#hp-mode').addEventListener('change', modeFields);
  $('#hp-task').addEventListener('change',taskDescription);
  $('#hp-use-budget').addEventListener('click',()=>{const b=info?.tasks?.[$('#hp-task').value]?.task_spec?.recommended_budget;if(b){$('#hp-sim').value=b.sim_seconds;$('#hp-steps').value=b.max_actions;}});
  $('#hp-policy-kind').addEventListener('change',()=>{
    const enabled=!!$('#hp-policy-kind').value;
    $('#hp-policy-settings').hidden=!enabled;
    $('#hp-policy-model').required=enabled; $('#hp-policy-endpoint').required=enabled;
  });
  $('#hp-add-tool').addEventListener('click',addTool);
  $('#hp-tools').addEventListener('click',event=>{if(event.target.closest('[data-remove-tool]'))event.target.closest('.hp-tool-row').remove();});
  $('#hp-scene').addEventListener('change', tasks);
  $('#hp-refresh').addEventListener('click', () => action(refresh));
  $('#hp-stop').addEventListener('click', () => action(async () => { live = await request('/stop', {}); }));
  $('#hp-promote').addEventListener('click', () => promote(live.run_id));
  $('#hp-library').addEventListener('change', () => action(async () => {
    if (!$('#hp-library').value) return;
    const program = await request(`/programs/${encodeURIComponent($('#hp-library').value)}`);
    $('#hp-source').value = program.source; $('#hp-name').value = program.manifest.name;
    $('#hp-scene').value = program.manifest.scene_id; tasks(); $('#hp-task').value = program.manifest.task_id; taskDescription();
    $('#hp-seeds').value = program.manifest.seeds.join(','); $('#hp-consent').checked = false;
  }));
  $('#hp-form').addEventListener('submit', event => { event.preventDefault(); action(async () => {
    const raw = $('#hp-seeds').value.trim();
    if (!/^\d+(\s*[,，]\s*\d+)*$/.test(raw)) throw new Error('种子须为逗号分隔的非负整数。');
    const mode=$('#hp-mode').value, evolving = mode !== 'validate', evolutionSeeds = $('#hp-evolution-seeds').value.trim();
    if (evolving && !/^\d+(\s*[,，]\s*\d+)*$/.test(evolutionSeeds)) throw new Error('演化种子须为逗号分隔的非负整数。');
    const tools = [...document.querySelectorAll('.hp-tool-row')].map(row=>Object.fromEntries([...row.querySelectorAll('[data-tool]')].map(field=>[field.dataset.tool,field.value.trim()])));
    live = await request('/start', {mode,...(mode==='auto' ? {} : {name:$('#hp-name').value.trim(),source:$('#hp-source').value}),
      ...(info.auto_ready ? {tools,action_model:$('#hp-policy-kind').value ? {kind:$('#hp-policy-kind').value,model:$('#hp-policy-model').value.trim(),endpoint:$('#hp-policy-endpoint').value.trim(),api_key:$('#hp-policy-key').value} : null} : {}),
      ...(evolving ? {evolution_rounds:Number($('#hp-rounds').value), evolution_seeds:evolutionSeeds.split(/[,，]/).map(Number),
        coder:{provider:$('#hp-provider').value,model:$('#hp-model').value.trim(),base_url:$('#hp-base-url').value.trim(),api_key:$('#hp-api-key').value,timeout_s:300}} : {}),
      scene_id:$('#hp-scene').value,task_id:$('#hp-task').value,seeds:raw.split(/[,，]/).map(Number),
      timeout_s:Number($('#hp-timeout').value),max_steps:Number($('#hp-steps').value),sim_seconds:Number($('#hp-sim').value),
      record_video:$('#hp-video').checked,allow_local_code_execution:$('#hp-consent').checked});
  }); });
  document.addEventListener('click', event => { const button = event.target.closest('[data-program-promote]'); if (button) promote(button.dataset.programPromote); });
  action(refresh);
}

async function promote(id) {
  await action(async () => { const saved = await request('/promote', {run_id:id}); callbacks?.toast?.(`已保存：${saved.name}`); await refresh(); });
}

export function renderHarnessPAI(state) {
  if (!$('#hp-form')) return;
  if (state.harnesspai && (!live.run_id || state.harnesspai.run_id === live.run_id || state.harnesspai.active)) live = state.harnesspai;
  const otherActive = ['agent','policy','comparison','research','experiment','benchmark','repair'].some(key => state[key]?.active);
  const locked = busy || live.active || otherActive || state.backend_switching || state.ready === false;
  $('#hp-inputs').disabled = !!locked;
  $('#hp-start').disabled = !!locked || !info?.ready;
  $('#hp-stop').disabled = busy || !live.active;
  $('#hp-refresh').disabled = busy || live.active;
  $('#hp-status').textContent = labels[live.status] || live.status || '等待程序';
  $('#hp-message').textContent = live.message || '先使用示例检查运行链路，再修改程序验证新任务。';
  $('#hp-runner-log').hidden = !live.run_id || live.status !== 'error';
  $('#hp-runner-log').href = live.run_id ? artifact(live.run_id,'runner.log') : '';
  $('#hp-time').textContent = live.elapsed_s == null ? '' : `累计墙钟 ${Number(live.elapsed_s).toFixed(2)} s · 已完成 ${live.rows?.length || 0} 个最终验证种子`;
  const batches = live.batches || [];
  $('#hp-automation').innerHTML = automationMarkup(live);
  const response = $('#hp-model-response'), responseKey = JSON.stringify([live.run_id,live.last_model_response]);
  if (response.dataset?.key !== responseKey) {
    response.innerHTML = modelResponseMarkup(live.last_model_response);
    if (response.dataset) response.dataset.key = responseKey;
  }
  $('#hp-evolution-meta').textContent = batches.length || live.primitive_calls ? `程序演化 · 编码与诊断调用 ${live.model_calls || 0} 次 · 动作模型与工具调用 ${live.primitive_calls || 0} 次 · 未训练模型${live.skill_writeback ? ' · 经验保存：'+(live.skill_writeback.ok ? '已完成' : live.skill_writeback.error || '未完成') : ''}` : '';
  if (batchRun !== live.run_id || JSON.stringify(renderedBatches) !== JSON.stringify(batches)) {
    const append = batchRun === live.run_id && JSON.stringify(batches.slice(0,renderedBatches.length)) === JSON.stringify(renderedBatches);
    if (append) $('#hp-evolution-results').insertAdjacentHTML('beforeend', evolutionBatchesMarkup(live.run_id, batches.slice(renderedBatches.length), renderedBatches.length));
    else $('#hp-evolution-results').innerHTML = evolutionBatchesMarkup(live.run_id, batches);
    batchRun = live.run_id; renderedBatches = batches;
  }
  const key = JSON.stringify([live.run_id, live.rows]);
  if (key !== lastRows) {
    const rows = live.rows || [];
    const append = renderedRun === live.run_id && renderedRows.length <= rows.length && JSON.stringify(rows.slice(0,renderedRows.length)) === JSON.stringify(renderedRows);
    if (append) $('#hp-results').insertAdjacentHTML('beforeend', programResultsMarkup(live.run_id, rows.slice(renderedRows.length)));
    else $('#hp-results').innerHTML = programResultsMarkup(live.run_id, rows);
    renderedRun = live.run_id; renderedRows = rows; lastRows = key;
  }
  $('#hp-bundle').hidden = !live.run_id || live.active; $('#hp-bundle').href = `/api/runs/${encodeURIComponent(live.run_id)}/bundle`;
  const saved = info?.versions?.some(version => version.run_id === live.run_id);
  $('#hp-promote').textContent = saved ? '本次验证程序已入库' : '保存本次已验证的程序快照';
  $('#hp-promote').hidden = !live.can_promote || !!live.archive?.ok; $('#hp-promote').disabled = !!locked || !!saved;
}
