// Pure DOM-state contracts. No browser, network, model, or simulator is started.
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

class Element {
  constructor(tag='div', id='') { this.tagName=tag;this.id=id;this.children=[];this.dataset={};this.value='';this.hidden=false;this.disabled=false;this.open=false;this.scrollTop=0;this.scrollLeft=0;this.classList={toggle(){}}; }
  set textContent(value) { this.replaceChildren();this.text=String(value); }
  get textContent() { return (this.text||'')+this.children.map(child=>child.textContent).join(''); }
  set innerHTML(value) {
    this.replaceChildren();this.html=value;
    // These tests inspect only the two text scrollers within each call card.
    for(const match of value.matchAll(/<pre>([\s\S]*?)<\/pre>/g)){const pre=new Element('pre');pre.textContent=match[1];this.append(pre);}
  }
  get innerHTML() { return this.html||''; }
  replaceChildren(...nodes) { for(const child of this.children)child.parentElement=null;this.children=[];this.text='';this.html='';this.append(...nodes); }
  append(...nodes) { for(const node of nodes)this.insertBefore(node,null); }
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
  const scope=vm.createContext({console,URLSearchParams,...extra,document:{querySelector:selector=>element(selector.slice(1)),querySelectorAll:()=>[],createElement:tag=>new Element(tag)},Option:function(name,value){const option=new Element('option');option.textContent=name;option.value=value;return option;}});
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

test('multiarm models are available in both comparison entry points',()=>{
  const {element,run}=moduleContext('multiarm.js');run('initMultiarm({operate(){}});');
  for(const id of ['compare-robot','rsi-protocol-robot'])assert.deepEqual(element(id).children.map(option=>option.value),['dual_panda','triple_panda']);
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

test('live policy rejection history renders without action-block fields or fabricated metrics',()=>{
  const {scope,element,run}=resultsContext();
  scope.fixtureState={policy:{active:true,status:'replan',round:2,history:[{round:2,result:{status:'completion_rejected',message:'环境尚未验收，请继续尝试 <retry>。'}}]},comparison:{}};
  run('renderResearch(fixtureState)');
  const html=element('policy-history').innerHTML;
  assert.match(html,/第 2 轮 · 完成声明被环境拒绝/);
  assert.match(html,/环境尚未验收，请继续尝试 &lt;retry&gt;。/);
  assert.doesNotMatch(html,/undefined|NaN|推理|帧|执行目标限幅/);
  assert.equal(element('policy-status').textContent,'未通过核验 · 重新决策');
  scope.fixtureState.policy.history.push({round:3,frames:12,latency_s:1.25,limited_targets:0});
  run('renderResearch(fixtureState)');
  assert.match(element('policy-history').innerHTML,/动作块 3 · 12 帧/);
  assert.match(element('policy-history').innerHTML,/推理 1.25 s · 执行目标限幅 0 项/);
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
