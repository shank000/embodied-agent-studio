// Pure DOM-state contracts. No browser, network, model, or simulator is started.
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';
import {setTaskDraft,syncTaskDraft} from '../src/task_drafts.js';

class Element {
  constructor(tag='div', id='') { this.tagName=tag;this.id=id;this.children=[];this.dataset={};this.value='';this.hidden=false;this.disabled=false;this.open=false;this.scrollTop=0;this.scrollLeft=0;this.classList={toggle(){}}; }
  get options(){return this.children.filter(child=>child.tagName==='option');}
  set textContent(value) { this.replaceChildren();this.text=String(value); }
  get textContent() { return (this.text||'')+this.children.map(child=>child.textContent).join(''); }
  set innerHTML(value) {
    this.replaceChildren();this.html=value;
    // These tests inspect only the two text scrollers within each call card.
    for(const match of value.matchAll(/<pre>([\s\S]*?)<\/pre>/g)){const pre=new Element('pre');pre.textContent=match[1];this.append(pre);}
  }
  get innerHTML() { return this.html||''; }
  replaceChildren(...nodes) { for(const child of this.children)child.parentElement=null;this.children=[];this.text='';this.html='';this.append(...nodes); }
  append(...nodes) { for(const node of nodes)this.insertBefore(node,null);if(this.tagName==='select'&&!this.options.some(option=>option.value===this.value))this.value=this.options[0]?.value||''; }
  insertBefore(node,before) { node.remove();const index=before?this.children.indexOf(before):this.children.length;this.children.splice(index,0,node);node.parentElement=this; }
  remove() { if(this.parentElement){const list=this.parentElement.children;list.splice(list.indexOf(this),1);this.parentElement=null;} }
  querySelectorAll(selector) {
    const matches=node=>selector.split(',').some(part=>{const [tag,cls]=part.split('.');return (!tag||node.tagName===tag)&&(!cls||(node.className||'').split(' ').includes(cls));});
    return this.children.flatMap(node=>[...(matches(node)?[node]:[]),...node.querySelectorAll(selector)]);
  }
  querySelector(selector) { return this.querySelectorAll(selector)[0]||null; }
  insertAdjacentHTML(_,value) { this.html=(this.html||'')+value; }
  setAttribute() {}
  addEventListener(type,listener) { (this.listeners??={})[type]=listener; }
  focus() { this.focused=true; }
  after() {}
}

function moduleContext(file,extra={}) {
  const nodes=new Map(),element=id=>{if(!nodes.has(id))nodes.set(id,new Element('div',id));return nodes.get(id);};
  const scope=vm.createContext({renderRoundTimeline(){},renderExperimentCosts(){},console,URLSearchParams,setTaskDraft,syncTaskDraft,window:{addEventListener(){}},...extra,document:{querySelector:selector=>element(selector.slice(1)),querySelectorAll:()=>[],createElement:tag=>new Element(tag)},Option:function(name,value){const option=new Element('option');option.textContent=name;option.value=value;return option;}});
  const code=readFileSync(new URL('../src/'+file,import.meta.url),'utf8').replace(/^import .*;\s*$/gm,'').replace(/\bexport /g,'');
  vm.runInContext(code,scope);
  return {scope,element,run:code=>vm.runInContext(code,scope)};
}

function resultsContext() {
  const setup=moduleContext('research.js');
  setup.run('context={toast(){}};renderRunList=()=>{};renderResultCost=()=>{};');
  return setup;
}

test('failed selection clears the previous record body, video, and download link',async()=>{
  const {scope,element,run}=resultsContext();
  scope.analysis={run_id:'A',display_name:'Record A',rounds:[],total:{},video:{status:'saved',path:'trajectory.mp4',frame_count:20,fps:20}};
  run("request=async path=>path.endsWith('/analysis')?analysis:[];");
  await run("showRun({run_id:'A'})");
  assert.equal(element('log-download').hidden,false);
  assert.equal(element('log-video').hidden,false);
  run("request=async()=>{throw new Error('Record B unavailable');};");
  await run("showRun({run_id:'B',display_name:'Record B'})");
  assert.equal(element('log-detail-title').textContent,'Record B');
  assert.equal(element('log-download').hidden,true);
  assert.equal(element('log-video').hidden,true);
  assert.equal(element('result-summary').children.length,0);
  assert.equal(element('result-summary').innerHTML,'');
  assert.equal(element('result-calls').children.length,0);
  assert.equal(element('result-cost-panel').hidden,true);
  assert.equal(element('results-error').textContent,'Record B unavailable');
});

test('unchanged result polling keeps the original detail DOM',async()=>{
  const {scope,element,run}=resultsContext();
  scope.analysis={run_id:'A',rounds:[{call_id:1,public_text:'Long output'}],total:{}};
  run("request=async path=>path.endsWith('/analysis')?analysis:[];");
  await run("showRun({run_id:'A'})");
  const original=element('result-calls').children[0];original.open=true;original.querySelector('pre').scrollTop=120;
  await run("showRun({run_id:'A'})");
  assert.equal(element('result-calls').children[0],original);
  assert.equal(original.open,true);
  assert.equal(original.querySelector('pre').scrollTop,120);
});

test('new calls preserve existing cards; changed text preserves its scroll and expansion',()=>{
  const {scope,element,run}=resultsContext();
  scope.analysis={rounds:[{call_id:1,public_text:'Long output'}],total:{}};
  run("renderResultAnalysis(analysis,{run_id:'A'})");
  const card=element('result-calls').children[0],text=card.querySelector('pre');card.open=true;text.scrollTop=120;text.scrollLeft=15;
  scope.analysis.rounds.push({call_id:2,public_text:'Another call'});
  run("renderResultAnalysis(analysis,{run_id:'A'})");
  assert.equal(element('result-calls').children.length,2);
  assert.equal(card.querySelector('pre'),text);
  scope.analysis.rounds[0].public_text+=' with more output';
  run("renderResultAnalysis(analysis,{run_id:'A'})");
  assert.equal(element('result-calls').children[0],card);
  assert.equal(card.open,true);
  assert.equal(card.querySelector('pre').scrollTop,120);
  assert.equal(card.querySelector('pre').scrollLeft,15);
});

test('historical experiment timer is not overwritten by a different live run',()=>{
  const timers=new Map();
  const {scope,element,run}=moduleContext('rsi.js',{renderRunTimer:(node,run)=>timers.set(node.id,run),renderRunVideo(){}});
  run('context={};renderLocks=()=>{};renderRunFeedback=()=>{};');
  scope.archived={run_id:'A',elapsed_s:40,rows:[]};scope.live={experiment:{run_id:'B',active:true,elapsed_s:5},research:{}};
  element('rsi-experiment-history').value='A';run('renderExperiment(archived);renderRsi(live);');
  assert.equal(timers.get('rsi-experiment-time').run_id,'A');
  element('rsi-experiment-history').value='';run('renderRsi(live);');
  assert.equal(timers.get('rsi-experiment-time').run_id,'B');
});

test('inactive configuration controls are disabled and regain their entered values when visible',()=>{
  const {element,run}=moduleContext('rsi.js');
  for(const [parent,children] of Object.entries({
    'rsi-agent-config-fields':['rsi-agent-source','rsi-agent-model','rsi-agent-url'],
    'rsi-agent-json-field':['rsi-agent-json'],'rsi-policy-json-field':['rsi-policy-json'],
    'rsi-policy-kind-field':['rsi-policy-kind'],'rsi-single-decision-config':['rsi-decision-url','rsi-decision-timeout'],
    'rsi-multi-decision-config':['rsi-decisions-json'],
    'rsi-decision-fields':['rsi-single-decision-config','rsi-multi-decision-config','rsi-decision-strategy','rsi-hybrid-policy']
  }))for(const id of children){const node=element(id);if(!id.endsWith('-config'))node.tagName='input';element(parent).append(node);}
  element('rsi-agent-json').value='{"provider":"compatible","model":"test","base_url":"invalid-url"}';element('rsi-policy-json').value='{}';
  element('rsi-decision-url').value='also-invalid';element('rsi-decision-strategy').value='single';element('rsi-mode').value='policy';run('updateMode();');
  assert.equal(element('rsi-agent-url').disabled,true);
  assert.equal(element('rsi-decision-url').disabled,true);
  assert.equal(element('rsi-policy-json').disabled,false);
  element('rsi-mode').value='hybrid';element('rsi-decision-strategy').value='weighted_vote';run('updateMode();');
  assert.equal(element('rsi-decisions-json').disabled,false);
  assert.equal(element('rsi-decision-url').disabled,true);
  element('rsi-decision-strategy').value='single';run('updateMode();');
  assert.equal(element('rsi-decision-url').disabled,false);
  assert.equal(element('rsi-decision-url').value,'also-invalid');
  assert.equal(element('rsi-decisions-json').disabled,true);
  element('rsi-agent-json').value='{"provider":"codex","model":"test"}';run('updateMode();');
  assert.equal(element('rsi-agent-url').disabled,true);
});

test('model cards and comparison entries follow the active engine catalog',()=>{
  const {scope,element,run}=moduleContext('multiarm.js');run('initMultiarm({operate(){}});');
  scope.models=['panda','dual_panda','triple_panda'].map(id=>({id,name:id,arm_dof:7,hand_dof:1}));run("updateModelCatalog(models,'dual_panda');");
  for(const id of ['compare-robot','rsi-protocol-robot'])assert.deepEqual(element(id).children.map(option=>option.value),['panda','dual_panda','triple_panda']);
  scope.models=[scope.models[0]];run("updateModelCatalog(models,'panda');");
  for(const id of ['compare-robot','rsi-protocol-robot'])assert.deepEqual(element(id).children.map(option=>option.value),['panda']);
  assert.deepEqual(element('model-cards').children.map(card=>card.dataset.model),['panda']);
});

for(const [stateKey,path,button] of [['research','research/stop','rsi-run-stop'],['experiment','research/experiments/stop','rsi-experiment-stop']]){
  test(`${stateKey} stop remains available during archive refresh and ignores duplicate submissions`,async()=>{
    const {scope,element,run}=moduleContext('rsi.js');
    let rejectRead,finishStop,stopCalls=0;
    const read=new Promise((_,reject)=>rejectRead=reject),stop=new Promise(resolve=>finishStop=resolve);
    scope.fixtureRequest=path=>path.endsWith('/stop')?(stopCalls++,stop):read;
    scope.fixtureState={[stateKey]:{active:true}};
    run('context={toast(){}};currentState=fixtureState;request=fixtureRequest;');
    const refreshing=run('action(refresh)');
    assert.equal(element('rsi-refresh').disabled,true);
    assert.equal(element(button).disabled,false);
    const stopping=run(`stopRun('${path}')`);
    assert.equal(element(button).disabled,true);
    await run(`stopRun('${path}')`);
    assert.equal(stopCalls,1);
    finishStop({});await stopping;
    assert.equal(element('rsi-refresh').disabled,true);
    assert.equal(element(button).disabled,false);
    rejectRead(new Error('Archive refresh failed'));await refreshing;
    assert.equal(element('rsi-error').hidden,false);
    assert.equal(element('rsi-error').textContent,'Archive refresh failed');
    assert.equal(element('rsi-refresh').disabled,false);
  });
}

test('Codex repair mode ignores stale invalid API fields in validation and payload',()=>{
  const {element,run}=moduleContext('repair.js');
  for(const id of ['repair-base','repair-key']){element(id).tagName='input';element('repair-api-fields').append(element(id));}
  element('repair-base').value='invalid-url';element('repair-key').value='stale-key';element('repair-tokens').value='invalid-number';
  element('repair-provider').value='codex';element('repair-model').value='fixture';element('repair-timeout').value='60';element('repair-recovery-budget').value='2';
  run('config={max_output_tokens:32768};providerFields();');
  assert.equal(element('repair-base').disabled,true);
  assert.equal(element('repair-key').disabled,true);
  assert.equal(element('repair-tokens').disabled,true);
  const sent=run('payload()');
  assert.equal(sent.base_url,'https://api.openai.com/v1');
  assert.equal(sent.api_key,'');
  assert.equal(sent.max_output_tokens,32768);
  element('repair-provider').value='openai';run('providerFields();');
  assert.equal(element('repair-base').disabled,false);
  assert.equal(element('repair-tokens').disabled,false);
});

test('automatic retry alone is visibly inactive until automatic diagnosis is enabled',()=>{
  const {element,run}=moduleContext('repair.js');
  run('config={auto_analyze:false,auto_recover:true,max_recovery_attempts:2};renderRecovery();');
  assert.match(element('repair-automatic-status').textContent,/自动分析：关闭/);
  element('repair-auto').checked=false; element('repair-auto-recover').checked=true;
  run('automaticFields()');
  assert.equal(element('repair-auto-recover').disabled,true);
  assert.match(element('repair-auto-note').textContent,/不会启动修复 Agent/);
  element('repair-auto').checked=true;
  run('automaticFields()');
  assert.equal(element('repair-auto-recover').disabled,false);
  assert.equal(element('repair-auto-recover').checked,true);
  run('current.automatic={auto_analyze:true,auto_recover:true,max_recovery_attempts:2};renderRecovery();');
  assert.match(element('repair-automatic-status').textContent,/自动诊断并重试/);
  run('current.automatic.auto_recover=false;renderRecovery();');
  assert.match(element('repair-automatic-status').textContent,/重试需手动确认/);
});

test('skipped recovery shows its reason and cannot resume a finished round',()=>{
  const {element,run}=moduleContext('repair.js');
  run("current={recovery:{failure_id:'failure',source_run_id:'run',status:'skipped',phase:'agent_inference',owner:'agent',round:2,message:'自动分析已关闭',can_resume:false}};renderRecovery();");
  assert.equal(element('repair-recovery').hidden,false);
  assert.equal(element('repair-resume').disabled,true);
  assert.equal(element('repair-card-status').textContent,'未启动故障修复');
  assert.match(element('repair-card-message').textContent,/第 2 轮/);
  assert.match(element('repair-recovery-message').textContent,/自动分析已关闭/);
});

test('Codex partial cost displays the known reference amount without an empty API bill',()=>{
  const {scope,run}=resultsContext();
  scope.cost={status:'partial',has_reference:true,reference_calls:33,billable_calls:0,local_calls:0,
    reference_amount_usd:null,known_reference_amount_usd:9.12345,amount_usd:null,known_amount_usd:0};
  const markup=run('costAmountMarkup(cost)');
  assert.match(markup,/9\.12345/);
  assert.match(markup,/已知部分/);
  assert.match(markup,/API 等价参考/);
  assert.doesNotMatch(markup,/API 费用估算|不可估算/);
  scope.cost.reference_amount_min_usd=9;scope.cost.reference_amount_max_usd=10;
  assert.match(run('costAmountMarkup(cost)'),/～/);
  assert.match(run('costDiagnosisText(cost)'),/金额上下界/);
  scope.cost.billable_calls=1;scope.cost.known_amount_usd=.123;
  assert.match(run('costAmountMarkup(cost)'),/API 费用估算/);
});

test('partial token totals retain known usage without claiming a complete total',()=>{
  const {run}=resultsContext();
  assert.equal(run("tokenTotalText({total_tokens:null,known_total_tokens:698356},'total_tokens')"),'698,356（已知）');
  assert.equal(run("tokenTotalText({total_tokens:0,known_total_tokens:0},'total_tokens')"),'0');
  assert.equal(run("tokenTotalText({total_tokens:null,known_total_tokens:0},'total_tokens')"),'—');
});

test('Agent provider fields disable hidden token inputs and restore API validation',()=>{
  const {element,run}=moduleContext('repair.js',{URL});
  const source=readFileSync(new URL('../src/main.js',import.meta.url),'utf8');
  run('const capabilityLoading=false;function reasoningFields(){};'+source.slice(source.indexOf('function providerFields(){'),source.indexOf("\n$('#config-base-url').addEventListener")));
  for(const id of ['config-base-url','config-api-key']){element(id).tagName='input';element('api-config-fields').append(element(id));}
  element('config-base-url').value='invalid-url';element('config-output-tokens').value='invalid-number';
  for(const provider of ['codex','custom','openai']){
    element('config-provider').value=provider;run('providerFields();');
    assert.equal(element('config-output-tokens').disabled,provider!=='openai');
    assert.equal(element('config-base-url').disabled,provider==='codex');
  }
});

test('experiment categories use environment metrics and never promote model completion claims',()=>{
  const {scope,run}=resultsContext();
  for(const [record,expected] of [
    [{status:'model_done',success:true,metrics:{success:false}},'error'],
    [{status:'verified',metrics:{success:false}},'error'],
    [{status:'verified',success:true},'error'],
    [{status:'model_done'},'error'],
    [{status:'complete',metrics:{success:'true'}},'error'],
    [{status:'verified',metrics:{success:true}},'success'],
    [{status:'complete',metrics:{success:true}},'success'],
    [{status:'error',metrics:{success:true}},'error'],
    [{status:'stopped',metrics:{success:true}},'stopped'],
    [{active:true,status:'verified',metrics:{success:true}},'unfinished'],
    [{status:'verifying',metrics:{success:false}},'unfinished'],
    [{status:'replan',metrics:{success:false}},'unfinished']
  ]){scope.record=record;assert.equal(run('category(record)'),expected,JSON.stringify(record));}
});

test('drawer evaluation shows physical measurements, thresholds, hold time, and rejected conditions',()=>{
  const {scope,element,run}=resultsContext();
  scope.evaluation={task_id:'open_drawer',name:'打开抽屉',criterion:'打开距离 ≥ 0.18 m 并稳定 0.3 s',source:'mujoco_privileged_state',version:'task-evaluation-v2',success:false,currently_satisfied:false,hold_s:0,required_hold_s:0.3,evaluated_at_sim_s:2.5,checks:[{key:'drawer_open',label:'抽屉打开距离',actual:0,expected:'≥ 0.18 m',unit:'m',passed:false}],failure_reasons:['抽屉尚未打开']};
  run('renderSceneEvaluation(evaluation)');
  assert.equal(element('evaluation-result').textContent,'未通过环境验收');
  const evidence=element('evaluation-evidence').innerHTML;
  for(const text of ['MuJoCo 特权状态','task-evaluation-v2','抽屉打开距离','0 m','≥ 0.18 m','0.000 / 0.300 s','抽屉尚未打开','failed'])assert.ok(evidence.includes(text),text);
  scope.evaluation.currently_satisfied=true;scope.evaluation.hold_s=0.1;
  run('renderSceneEvaluation(evaluation)');
  assert.equal(element('evaluation-result').textContent,'未通过环境验收');
  assert.match(element('evaluation-evidence').innerHTML,/正在等待稳定性验收/);
  scope.evaluation.success=true;scope.evaluation.hold_s=0.3;scope.evaluation.checks[0].passed=true;
  run('renderSceneEvaluation(evaluation)');
  assert.equal(element('evaluation-result').textContent,'环境验收通过');
});

test('record details retain rejected model claims and never replace failed final environment metrics',async()=>{
  const {scope,element,run}=resultsContext();
  scope.evaluation={source:'mujoco_privileged_state',version:'task-evaluation-v2',success:false,checks:[{label:'抽屉打开距离',actual:0,unit:'m',expected:'≥ 0.18 m',passed:false}],failure_reasons:['抽屉未打开 <unsafe>'],hold_s:0,required_hold_s:0.3};
  scope.analysis={run_id:'A',status:'unverified',metrics:{success:false,evaluation:scope.evaluation},completion_verifications:[{round:1,accepted:false,model_claim:'抽屉已打开 <unsafe>',message:'模型自报完成，环境未通过',timestamp:'2026-10-23T00:00:00Z',evaluation:scope.evaluation}],rounds:[],total:{}};
  run("request=async path=>path.endsWith('/analysis')?analysis:[];");
  await run("showRun({run_id:'A',status:'verified',success:true})");
  assert.equal(element('result-evaluation-result').textContent,'未通过环境验收');
  assert.match(element('result-evaluation-evidence').innerHTML,/0 m/);
  assert.match(element('result-evaluation-evidence').innerHTML,/&lt;unsafe&gt;/);
  assert.match(element('result-completion-verifications').innerHTML,/第 1 轮 · 模型完成声明/);
  assert.match(element('result-completion-verifications').innerHTML,/环境拒绝/);
  assert.match(element('result-completion-verifications').innerHTML,/模型声明：抽屉已打开 &lt;unsafe&gt;/);
  assert.match(element('result-completion-note').textContent,/逐条保留/);
  assert.doesNotMatch(element('result-completion-verifications').innerHTML,/环境接受/);
  run("request=async()=>{throw new Error('Not found');};");
  await run("showRun({run_id:'B'})");
  assert.equal(element('result-evaluation').hidden,true);
});

test('historical model_done records visibly remain unverified when evidence is absent',()=>{
  const {scope,element,run}=resultsContext();scope.analysis={status:'model_done',rounds:[],total:{}};
  run("renderResultAnalysis(analysis,{run_id:'legacy',status:'model_done'})");
  assert.match(element('result-description').textContent,/模型自报完成（非验收）/);
  assert.equal(element('result-evaluation-result').textContent,'未记录环境验收');
  assert.match(element('result-evaluation-evidence').innerHTML,/未记录环境评价证据/);
  assert.match(element('result-completion-note').textContent,/最近 50 条/);
});

test('task shortcut uses the selected environment instruction without changing the scene or criterion',()=>{
  const {element,run}=resultsContext();
  run("let state={evaluation:{task_id:'open_drawer',instruction:'抓住把手并将抽屉拉开。'}};");
  const source=readFileSync(new URL('../src/main.js',import.meta.url),'utf8'),statement=source.split('\n').find(line=>line.startsWith("$('#example-task').addEventListener"));
  run(statement);element('example-task').listeners.click();
  assert.equal(element('command').value,'抓住把手并将抽屉拉开。');
  assert.equal(element('command').focused,true);
  assert.equal(run('state.evaluation.task_id'),'open_drawer');
});

test('live policy delegates full round records while retaining completion rejection status',()=>{
  const {scope,element,run}=resultsContext();const forwarded=[];
  scope.renderRoundTimeline=(root,controller)=>forwarded.push({root,controller});
  scope.fixtureState={policy:{run_id:'run',active:true,status:'replan',round:2},comparison:{}};
  run('renderResearch(fixtureState)');
  assert.equal(forwarded[0].root,element('policy-history'));
  assert.equal(forwarded[0].controller,scope.fixtureState.policy);
  assert.equal(element('policy-status').textContent,'未通过核验 · 重新决策');
});

test('a newly loaded native environment remains connected while its old scene catalog is replaced',()=>{
  const {scope,element,run}=resultsContext();
  run("catalog={scenes:[{id:'tabletop',tasks:['lift_red']}],tasks:{lift_red:{name:'Lift'}}};lastGeneration=1;");
  scope.fixtureState={generation:2,scene_id:'alfworld_unseen',model:'alfworld_agent',backend:{action_mode:'text'},policy:{},comparison:{}};
  assert.doesNotThrow(()=>run('renderResearch(fixtureState)'));
  assert.equal(run('lastGeneration'),1,'Do not consume the new generation before its scene catalog arrives.');
  assert.equal(element('policy-start').disabled,true);
  assert.equal(element('compare-start').disabled,true);
  assert.match(element('comparison-method-note').textContent,/离散环境/);
  scope.fixtureState.backend.action_mode='tcp_pose';
  run('renderResearch(fixtureState)');
  assert.doesNotMatch(element('comparison-method-note').textContent,/此文字或离散环境/);
});

test('opening a workbench already in a native environment initializes its own tasks without tabletop defaults',async()=>{
  const {scope,element,run}=resultsContext();
  scope.nativeCatalog={scenes:[{id:'alfworld_unseen',name:'Household',description:'Official text environment',tasks:['put']}],tasks:{put:{name:'Put',instruction:'Put the vase in the safe',criterion:'Native won'}}};
  scope.bootstrapErrors=[];
  run("request=async path=>path==='scenes'?nativeCatalog:{};");
  await run('initResearch({toast:message=>bootstrapErrors.push(message),operate(){},selectTab(){}})');
  assert.deepEqual(scope.bootstrapErrors,[]);
  assert.equal(element('scene-select').value,'alfworld_unseen');
  assert.equal(element('compare-scene').value,'alfworld_unseen');
  assert.equal(element('policy-task').value,'Put the vase in the safe');
});

test('composition rejection history renders without an action or selected candidate',()=>{
  const {scope,element,run}=moduleContext('rsi.js');
  scope.fixtureRun={run_id:'composition',step:4,history:[{step:4,result:{status:'completion_rejected',message:'环境尚未验收，请重新规划。'}}]};
  run('renderRunFeedback(fixtureRun)');
  const html=element('rsi-run-history').innerHTML;
  assert.match(html,/第 4 轮 · 完成声明被环境拒绝/);
  assert.match(html,/环境尚未验收，请重新规划。/);
  assert.doesNotMatch(html,/undefined|NaN|completion_rejected|位置误差|姿态误差/);
});

test('normalized mobile commands retain native units and missing measurements do not become angles',()=>{
  const source=readFileSync(new URL('../src/main.js',import.meta.url),'utf8');
  const code=source.slice(source.indexOf('const angular='),source.indexOf('function buildJoints()'));
  const scope=vm.createContext({});vm.runInContext(code,scope);
  assert.equal(vm.runInContext("targetText({unit:'normalized_command'},.5)",scope),'0.50');
  assert.equal(vm.runInContext("fromDisplay({unit:'normalized_command'},.5)",scope),.5);
  assert.equal(vm.runInContext("actualText({unit:'normalized_command',actual:null})",scope),'—');
  assert.equal(vm.runInContext("targetText({unit:'rad'},Math.PI/2)",scope),'90.0°');
  assert.equal(vm.runInContext("targetText({unit:'command'},255)",scope),'100.0%');
});

test('external benchmark archives label archive time and keep external inference usage missing',()=>{
  const {scope,element,run}=resultsContext();
  scope.analysis={run_id:'fixture',kind:'benchmark',record_origin:'external_import',status:'imported',elapsed_s:.01,
    rounds:[],total:{calls:null,total_tokens:null,inference_s:null},benchmark:{summary:{eval_time:2,success_rate:.5,score:62.5},
    coverage:{scope:'partial'},denominator:{native_valid_completed:2}}};
  run('renderResultAnalysis(analysis,{},[])');
  assert.match(element('result-summary').innerHTML,/本次归档耗时/);
  assert.doesNotMatch(element('result-summary').innerHTML,/完整墙钟/);
  assert.match(element('result-native-benchmark').textContent,/原实验墙钟.*未上报/);
  run('clearResult()');assert.equal(element('result-native-benchmark').hidden,true);
});

test('known empty cancelled runs show no calls while unknown historical usage remains missing',()=>{
  const {scope,element,run}=resultsContext();
  scope.analysis={run_id:'cancelled-fixture',status:'stopped',rounds:[],total:{calls:0,input_tokens:0,output_tokens:0,usage_complete:false}};
  run('renderResultAnalysis(analysis,{},[])');
  assert.match(element('result-summary').innerHTML,/无模型调用/);
  scope.analysis.total={calls:0,input_tokens:null,output_tokens:null,usage_complete:false};
  run('renderResultAnalysis(analysis,{},[])');
  assert.match(element('result-summary').innerHTML,/存在缺失/);
  assert.doesNotMatch(element('result-summary').innerHTML,/无模型调用/);
});

test('generated task instructions follow the active environment while handwritten tasks survive changes',()=>{
  const {scope,element,run}=resultsContext();
  scope.active={session_id:'robot-session',generation:1,model:'panda',backend:{engine_id:'mujoco'},evaluation:{task_id:'drawer',instruction:'Open the drawer.'},policy:{},comparison:{}};
  run('renderResearch(active)');
  assert.equal(element('command').value,'Open the drawer.');
  assert.equal(element('policy-task').value,'Open the drawer.');
  element('policy-task').value='My custom research task';
  scope.active={...scope.active,session_id:'text-session',generation:2,model:'alfworld_agent',backend:{engine_id:'alfworld',action_mode:'text'},evaluation:{task_id:'put',instruction:'Put the mug in the sink.'}};
  run('renderResearch(active)');
  assert.equal(element('command').value,'Put the mug in the sink.');
  assert.equal(element('policy-task').value,'My custom research task');
});

test('a rejected scene application does not replace task drafts with an unapplied candidate',async()=>{
  const {scope,element,run}=resultsContext();
  scope.nativeCatalog={engine_id:'fixture',scenes:[{id:'room',name:'Room',tasks:['one','two']}],tasks:{one:{name:'One',instruction:'First task'},two:{name:'Two',instruction:'Second task'}}};
  run("request=async path=>path==='scenes'?nativeCatalog:{}");
  await run('initResearch({toast(){},operate:async()=>null,selectTab(){}})');
  element('task-select').value='two';element('command').value='Keep this task';element('policy-task').value='Keep policy task';
  await element('apply-scene').listeners.click();
  assert.equal(element('command').value,'Keep this task');assert.equal(element('policy-task').value,'Keep policy task');
  run('context.operate=async()=>({ok:true})');await element('apply-scene').listeners.click();
  assert.equal(element('command').value,'Second task');assert.equal(element('policy-task').value,'Second task');
});

for(const [file,reload] of [['research.js','reloadResearchCatalog'],['rsi.js','reloadRsiCatalog']])test(`${file} rejects an old engine catalog that arrives after the replacement`,async()=>{
  const {scope,element,run}=moduleContext(file);
  if(file==='rsi.js')element('rsi-protocol-scene').tagName='select';
  scope.first={engine_id:'first',scenes:[{id:'first-room',name:'First',tasks:['a']}],tasks:{a:{name:'A'}}};
  scope.second={engine_id:'second',scenes:[{id:'second-room',name:'Second',tasks:['b']}],tasks:{b:{name:'B'}}};
  run("let pending=[];let live={backend:{engine_id:'first'}};context={getState:()=>live};request=()=>new Promise(resolve=>pending.push(resolve));");
  const old=run(`${reload}('first')`);
  run("live={backend:{engine_id:'second'}}");const current=run(`${reload}('second')`);
  run('pending[1](second)');assert.equal(await current,true);
  run('pending[0](first)');assert.equal(await old,false);
  assert.equal(run('catalog.engine_id'),'second');
  assert.equal(element(file==='research.js'?'scene-select':'rsi-protocol-scene').value,'second-room');
});

test('workspace model responses cannot mark an old catalog as belonging to a new session',async()=>{
  const source=readFileSync(new URL('../src/main.js',import.meta.url),'utf8');
  const code=source.slice(source.indexOf('function reloadWorkspaceCatalog()'),source.indexOf('const frame='));
  const updates=[],scope=vm.createContext({updates,toast(){},enableControls(){},updateModelCatalog:models=>updates.push(models),reloadResearchCatalog:async()=>true,reloadRsiCatalog:async()=>true});
  vm.runInContext("let state={session_id:'old',backend:{engine_id:'old'}};let catalogSession='',catalogLoading=false,catalogPromise=null;let pending=[];let fetch=()=>new Promise(resolve=>pending.push(resolve));",scope);
  vm.runInContext(code,scope);const first=vm.runInContext('reloadWorkspaceCatalog()',scope);
  vm.runInContext("state={session_id:'new',backend:{engine_id:'new'}};pending[0]({ok:true,json:async()=>[{id:'old-robot'}]})",scope);
  assert.equal(await first,false);assert.equal(updates.length,0);assert.equal(vm.runInContext('catalogSession',scope),'');
  const next=vm.runInContext('reloadWorkspaceCatalog()',scope);vm.runInContext("pending[1]({ok:true,json:async()=>[{id:'new-robot'}]})",scope);
  assert.equal(await next,true);assert.equal(updates.length,1);assert.equal(updates[0][0].id,'new-robot');assert.equal(vm.runInContext('catalogSession',scope),'new');
});

test('environments without camera controls retain normal page scrolling and do not capture pointer gestures',()=>{
  const source=readFileSync(new URL('../src/main.js',import.meta.url),'utf8'),listeners={},calls=[];
  const scope=vm.createContext({frame:{addEventListener:(type,fn)=>listeners[type]=fn,setPointerCapture:id=>calls.push(id)},flushCamera:()=>calls.push('camera'),calls});
  vm.runInContext("let connected=true,state={backend:{capabilities:['observe']}},pointer=null,cameraPending={zoom:0};",scope);
  for(const line of source.split('\n').filter(line=>line.startsWith("frame.addEventListener('pointerdown'")||line.startsWith("frame.addEventListener('wheel'")))vm.runInContext(line,scope);
  const event={button:0,pointerId:1,deltaY:10,preventDefault:()=>calls.push('prevented')};listeners.pointerdown(event);listeners.wheel(event);
  assert.deepEqual(calls,[]);
  vm.runInContext("state.backend.capabilities=['camera']",scope);listeners.pointerdown(event);listeners.wheel(event);
  assert.deepEqual(calls,['prevented',1,'prevented','camera']);
});

test('a replacement server session rebuilds robot controls even when generation and active arm repeat',()=>{
  const source=readFileSync(new URL('../src/main.js',import.meta.url),'utf8'),start=source.indexOf('function renderState(){')+'function renderState(){'.length;
  const body=source.slice(start,source.indexOf('const discrete=',start)),counts={build:0,clear:0};
  const scope=vm.createContext({counts,buildJoints:()=>counts.build++,$:()=>({replaceChildren:()=>counts.clear++})});
  vm.runInContext("let state={session_id:'first',backend:{engine_id:'robocasa365'},model:'panda_omron',generation:1,active_robot:'arm1'};let generation=-1,activeRobot='',renderedModelContext='',lastEvent=0;",scope);
  vm.runInContext(`function renderControls(){${body}}`,scope);vm.runInContext('renderControls();renderControls()',scope);
  assert.equal(counts.build,1);
  vm.runInContext("state={session_id:'second',backend:{engine_id:'mujoco'},model:'panda',generation:1,active_robot:'arm1'};renderControls()",scope);
  assert.equal(counts.build,2);assert.equal(counts.clear,2);
});

test('initialization failures without robot fields still disable controls, and task helpers unlock after catalog sync',()=>{
  const source=readFileSync(new URL('../src/main.js',import.meta.url),'utf8'),body=source.slice(source.indexOf('function enableControls(){'),source.indexOf('async function operate('));
  const nodes=new Map(),element=id=>{if(!nodes.has(id))nodes.set(id,{disabled:false});return nodes.get(id);},useTask=element('#use-task');
  const scope=vm.createContext({$:element,document:{querySelectorAll:selector=>selector.includes('#use-task')?[useTask]:[]}});
  vm.runInContext("let state={ready:false,error:'Native initialization failed'},connected=false,busy=false,catalogLoading=false,catalogSession='';",scope);vm.runInContext(body,scope);
  assert.doesNotThrow(()=>vm.runInContext('enableControls()',scope));assert.equal(useTask.disabled,true);
  vm.runInContext("state={ready:true,session_id:'new',task:{status:'idle'},paused:true};connected=true;catalogLoading=true;enableControls()",scope);assert.equal(useTask.disabled,true);
  vm.runInContext("catalogLoading=false;catalogSession='new';enableControls()",scope);assert.equal(useTask.disabled,false);
});

for(const file of ['research.js','rsi.js'])test(`${file} locks execution after a network disconnect and unlocks after the next ready state`,()=>{
  const source=readFileSync(new URL('../src/main.js',import.meta.url),'utf8');
  const disconnected=source.slice(source.indexOf('function disconnected('),source.indexOf('let pollPromise='));
  const {scope,element,run}=moduleContext(file,{renderRunTimer(){},renderRunVideo(){}});
  const methods=[{value:'agent',disabled:false},{value:'vla',disabled:false}];
  scope.document.querySelectorAll=selector=>selector==='input[name="method"]:checked'?methods:[];
  const render=file==='research.js'?'renderResearch':'renderRsi';
  const other=file==='research.js'?'renderRsi':'renderResearch';
  scope[other]=()=>{};
  Object.assign(scope,{enableControls(){},renderMultiarm(){},renderEnvironment(){},renderSimulator(){},renderBenchmarks(){},frame:{removeAttribute(){}},poll(){}});
  element('status-dot').classList.remove=()=>{};
  scope.live={ready:true,session_id:'session',generation:1,model:'panda',backend:{engine_id:'mujoco'},
    agent:{active:false},policy:{active:false,run_id:'previous-run'},comparison:{active:false},research:{},experiment:{}};
  run('let state=live,connected=true,frameStarted=true;context={toast(){},catalogReady:()=>true,getState:()=>state};');
  if(file==='rsi.js')run("methods=[{id:'one'},{id:'two'}];protocols=[{id:'protocol'}]");
  element('policy-kind').value='vla';element('command').value='Keep my task';element('policy-task').value='Keep policy task';
  run(`${render}(state)`);
  const controls=file==='research.js'?['policy-start','policy-one-chunk','compare-start']:['rsi-run-start','rsi-experiment-start','rsi-save-protocol'];
  for(const id of controls)assert.equal(element(id).disabled,false,id+' initially ready');
  run(disconnected);run("disconnected('Connection refused')");
  for(const id of controls)assert.equal(element(id).disabled,true,id+' offline');
  assert.equal(run('state.ready'),false);assert.equal(run('state.policy.run_id'),'previous-run');
  assert.equal(element('command').value,'Keep my task');assert.equal(element('policy-task').value,'Keep policy task');
  assert.match(element('scene-error').textContent,/Connection refused/);
  run(`state=live;connected=true;${render}(state)`);
  for(const id of controls)assert.equal(element(id).disabled,false,id+' reconnected');
  run("state=null;disconnected('Startup unavailable')");
  for(const id of controls)assert.equal(element(id).disabled,true,id+' initial failure');
});

test('polling distinguishes a reachable unready environment from an unreachable service without replacing the real error',async()=>{
  const source=readFileSync(new URL('../src/main.js',import.meta.url),'utf8');
  const start=source.indexOf('function disconnected(');
  const code=source.slice(start,source.indexOf("document.querySelectorAll('[data-model]')",start));
  for(const scenario of [
    {state:{ready:false,error:'ALFWorld process exited: fixture cause'},status:'仿真环境未就绪',detail:'ALFWorld process exited: fixture cause',button:'刷新环境状态'},
    {state:{ready:false},status:'仿真环境未就绪',detail:'环境仍在初始化，请等待完成。',button:'刷新环境状态'},
    {error:'Failed to fetch',status:'仿真服务未连接',detail:'Failed to fetch',button:'重新连接'}
  ]){
    const nodes=new Map(),element=id=>{if(!nodes.has(id))nodes.set(id,new Element('div',id));return nodes.get(id);};
    const rendered=[],urls=[];element('#status-dot').classList.remove=()=>{};
    const scope=vm.createContext({AbortSignal,$:element,document:{createElement:tag=>new Element(tag)},
      frame:{removeAttribute(){}},enableControls(){},renderAgent(){},renderRepair(){},renderState(){},
      renderResearch:state=>rendered.push(state),renderRsi(){},renderMultiarm(){},renderEnvironment(){},renderSimulator(){},renderBenchmarks(){},
      fetch:async url=>{urls.push(url);if(scenario.error)throw new TypeError(scenario.error);return {ok:true,json:async()=>scenario.state};}});
    vm.runInContext('let state=null,connected=false,frameStarted=false;'+code,scope);
    await vm.runInContext('pollState()',scope);
    assert.equal(element('#connection').textContent,scenario.status);
    assert.equal(element('#robot-status').textContent,scenario.state?'未就绪':'离线');
    const error=element('#scene-error');assert.match(error.textContent,new RegExp(scenario.detail.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')));
    assert.equal(error.querySelector('button').textContent,scenario.button);
    if(scenario.state){assert.match(error.textContent,/左侧.*重新加载环境/);assert.doesNotMatch(error.textContent,/请启动项目|服务未连接/);}
    else assert.match(error.textContent,/请启动项目中的仿真服务/);
    assert.equal(rendered.at(-1).ready,false);assert.deepEqual(urls,['/api/state']);
  }
});
