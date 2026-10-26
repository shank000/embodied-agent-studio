import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';
import {setTaskDraft,syncTaskDraft} from '../src/task_drafts.js';

class Element{
  constructor(tag='div'){this.tagName=tag;this.children=[];this.value='';this.disabled=false;this.hidden=false;this.attributes={};this.listeners={};this.classes=new Set();this.classList={toggle:(name,on)=>on?this.classes.add(name):this.classes.delete(name)};}
  set textContent(value){this.text=String(value);this.children=[];}
  get textContent(){return(this.text||'')+this.children.map(child=>child.textContent).join('');}
  append(...nodes){this.children.push(...nodes);}
  replaceChildren(...nodes){this.children=nodes;this.text='';}
  setAttribute(name,value){this.attributes[name]=value;}
  addEventListener(type,callback){this.listeners[type]=callback;}
}

const catalog=(genesisStatus='available')=>({active:'mujoco',session_id:'session-a',backends:[
  {id:'mujoco',name:'MuJoCo',version:'3.13.0',status:'available',installed:true,implemented:true,description:'Current engine'},
  {id:'genesis',name:'Genesis',version:'1.4.5',status:genesisStatus,installed:genesisStatus==='available',implemented:true,description:'Panda tabletop'},
  {id:'future',name:'Future engine',status:'planned',installed:false,implemented:false,description:'<img src=x onerror=bad()>',requirements:'Separate runtime',docs:'javascript:bad()'},
]});

function setup(responses,onChanged=async()=>{},file='simulators.js'){
  const nodes=new Map(),calls=[];
  const element=id=>{if(!nodes.has(id))nodes.set(id,new Element());return nodes.get(id);};
  const scope=vm.createContext({URL,URLSearchParams,console,setTaskDraft,syncTaskDraft,callbacks:{onChanged},document:{querySelector:selector=>element(selector.slice(1)),createElement:tag=>new Element(tag)},fetch:async(url,options={})=>{
    calls.push({url,...options});
    const next=responses.shift();
    if(next instanceof Error)throw next;
    if(typeof next==='function')return next(url,options);
    assert.ok(next,'Unexpected request');
    return{ok:next.status==null||next.status<400,status:next.status||200,json:async()=>structuredClone(next.body)};
  }});
  const source=readFileSync(new URL('../src/'+file,import.meta.url),'utf8').replace(/^import .*;\s*$/gm,'').replace(/\bexport /g,'');
  vm.runInContext(source,scope);
  return{element,calls,run:code=>vm.runInContext(code,scope),scope};
}

test('simulator availability comes from catalog; unsafe descriptions remain text and unsafe links are excluded',async()=>{
  const {element,run}=setup([{body:catalog('not_installed')}]);
  await run('initSimulators(callbacks)');
  assert.equal(element('simulator-select').value,'mujoco');
  assert.deepEqual(element('simulator-select').children.map(option=>[option.value,option.disabled]),[['mujoco',false],['genesis',true],['future',true]]);
  assert.equal(element('simulator-load').disabled,true);
  const future=element('simulator-catalog').children[2];
  assert.ok(future.textContent.includes('<img src=x onerror=bad()>'));
  assert.equal(future.children.some(node=>node.tagName==='a'),false);
});

test('RoboDojo opens its independent benchmark from the environment selector without switching physics',async()=>{
  const value=catalog();value.backends.push({id:'robodojo',name:'RoboDojo',kind:'benchmark',status:'external_runtime',installed:null,implemented:true,description:'Isaac Sim 5.1 independent rollout'});
  let opened=0;
  const {element,calls,run,scope}=setup([{body:value}]);scope.callbacks.onOpenBenchmark=()=>opened++;
  await run('initSimulators(callbacks)');
  const option=element('simulator-select').children.find(item=>item.value==='robodojo');
  assert.equal(option.disabled,false);assert.ok(option.textContent.includes('独立评测入口'));
  element('simulator-select').value='robodojo';element('simulator-select').listeners.change();
  assert.equal(element('simulator-load').textContent,'打开 RoboDojo');assert.equal(element('simulator-load').disabled,false);
  await element('simulator-load').listeners.click();
  assert.equal(opened,1);assert.equal(calls.length,1);assert.ok(element('simulator-current').textContent.startsWith('MuJoCo'));
  assert.ok(element('simulator-message').textContent.includes('当前仿真环境保持不变'));
  run("renderSimulator({backend:{engine_id:'mujoco'}})");
  assert.equal(element('simulator-select').value,'robodojo');
});

test('broken interpreter is unavailable with an actionable diagnostic instead of claiming loadable',async()=>{
  const value=catalog('unavailable');value.backends[1].error='Python home missing; repair pyvenv.cfg <literal>';
  const {element,run}=setup([{body:value}]);await run('initSimulators(callbacks)');
  const option=element('simulator-select').children.find(item=>item.value==='genesis');
  assert.equal(option.disabled,true);assert.ok(option.textContent.includes('运行环境异常'));
  assert.ok(element('simulator-catalog').textContent.includes(value.backends[1].error));
});

test('RoboDojo cannot be posted as an interactive backend when navigation is unavailable',async()=>{
  const value=catalog();value.backends.push({id:'robodojo',name:'RoboDojo',kind:'benchmark',status:'external_runtime'});
  const {element,calls,run}=setup([{body:value}]);await run('initSimulators(callbacks)');
  element('simulator-select').value='robodojo';element('simulator-select').listeners.change();
  assert.equal(element('simulator-load').disabled,true);
  await element('simulator-load').listeners.click();assert.equal(calls.length,1);
});

test('active experiment locks selection and prevents even a queued load handler from making a request',async()=>{
  const {element,calls,run}=setup([{body:catalog()}]);
  await run('initSimulators(callbacks)');
  element('simulator-select').value='genesis';
  run("renderSimulator({backend:{engine_id:'mujoco'},experiment:{active:true}})");
  assert.equal(element('simulator-select').disabled,true);
  assert.equal(element('simulator-load').disabled,true);
  assert.equal(element('simulator-lock').hidden,false);
  await element('simulator-load').listeners.click();
  assert.equal(calls.length,1);
  assert.ok(element('simulator-message').textContent.includes('当前任务仍在执行'));
});

test('loading is single flight and changes active engine only after a successful server response',async()=>{
  const pending=Promise.withResolvers();let changed=0;
  const {element,calls,run}=setup([{body:catalog()},()=>pending.promise],async()=>changed++);
  await run('initSimulators(callbacks)');
  element('simulator-select').value='genesis';
  const first=element('simulator-load').listeners.click(),second=element('simulator-load').listeners.click();
  assert.equal(calls.length,2);
  assert.equal(element('simulator-refresh').disabled,true);
  assert.equal(element('simulator-select').disabled,true);
  assert.ok(element('simulator-current').textContent.startsWith('MuJoCo'));
  assert.equal(JSON.parse(calls[1].body).engine_id,'genesis');
  pending.resolve({ok:true,status:200,json:async()=>({backend:{engine_id:'genesis'},session_id:'session-b'})});
  await Promise.all([first,second]);
  assert.equal(changed,1);
  assert.ok(element('simulator-current').textContent.startsWith('Genesis'));
  assert.equal(element('simulator-select').value,'genesis');
  assert.equal(element('simulator-refresh').disabled,false);
  assert.equal(element('simulator-load').disabled,true);
});

test('native backend loading is shown as a pending switch instead of a running experiment',async()=>{
  const {element,calls,run}=setup([{body:catalog()}]);
  await run('initSimulators(callbacks)');
  element('simulator-select').value='genesis';
  run("renderSimulator({backend:{engine_id:'mujoco'},backend_switching:true})");
  assert.equal(element('simulator-select').disabled,true);
  assert.equal(element('simulator-refresh').disabled,true);
  assert.equal(element('simulator-card').attributes['aria-busy'],'true');
  assert.equal(element('simulator-load').textContent,'正在加载…');
  assert.equal(element('simulator-lock').hidden,false);
  assert.equal(element('simulator-lock').textContent,'正在切换环境，请等待加载完成。');
  await element('simulator-load').listeners.click();
  assert.equal(calls.length,1);
  assert.equal(element('simulator-message').textContent,'正在切换环境，请等待加载完成。');
  run("renderSimulator({backend:{engine_id:'mujoco'},backend_switching:false})");
  assert.equal(element('simulator-select').disabled,false);
  assert.equal(element('simulator-refresh').disabled,false);
  assert.equal(element('simulator-lock').hidden,true);
  assert.equal(element('simulator-card').attributes['aria-busy'],'false');
});

test('a rejected switch preserves active engine and exposes the server reason',async()=>{
  const {element,run}=setup([{body:catalog()},{status:409,body:{detail:'An experiment started in another tab'}}]);
  await run('initSimulators(callbacks)');
  element('simulator-select').value='genesis';
  await element('simulator-load').listeners.click();
  assert.equal(element('simulator-select').value,'mujoco');
  assert.ok(element('simulator-current').textContent.startsWith('MuJoCo'));
  assert.ok(element('simulator-message').textContent.includes('An experiment started in another tab'));
  assert.equal(element('simulator-message').attributes.role,'alert');
  assert.equal(element('simulator-refresh').disabled,false);
});

test('long switch failures show a bounded summary and retain exact diagnostics as text',async()=>{
  const raw='切换失败，当前仿真保持不变：ai2thor ConnectionClosedError: Display connection closed by server: [Errno 104] Connection reset by peer\nTraceback (most recent call last):\n  File "/runtime.py", line 1\n'+('stack details\n'.repeat(100))+'<img src=x onerror=bad()>\nXlib.error.ConnectionClosedError: Display connection closed';
  const {element,run}=setup([{body:catalog()},{status:500,body:{detail:raw}},{body:catalog()}]);
  await run('initSimulators(callbacks)');element('simulator-select').value='genesis';
  await element('simulator-load').listeners.click();
  assert.ok(element('simulator-message').textContent.includes('ConnectionClosedError'));
  assert.equal(element('simulator-message').textContent.includes('Traceback'),false);
  assert.ok(element('simulator-message').textContent.length<400);
  assert.equal(element('simulator-diagnostic').hidden,false);assert.equal(element('simulator-diagnostic').open,false);
  assert.equal(element('simulator-diagnostic-text').textContent,raw);assert.equal(element('simulator-diagnostic-text').children.length,0);
  assert.equal(element('simulator-select').value,'mujoco');
  await element('simulator-refresh').listeners.click();
  assert.equal(element('simulator-diagnostic').hidden,true);assert.equal(element('simulator-diagnostic-text').textContent,'');
});

test('a failed engine can be explicitly reloaded while healthy same-engine loads stay disabled',async()=>{
  const {element,calls,run}=setup([{body:catalog()},{body:{backend:{engine_id:'mujoco'},session_id:'recovered'}}]);
  await run('initSimulators(callbacks)');
  run("renderSimulator({ready:true,backend:{engine_id:'mujoco'}})");
  assert.equal(element('simulator-load').disabled,true);
  run("renderSimulator({ready:false,backend:{engine_id:'mujoco'},error:'Native process exited'})");
  assert.equal(element('simulator-load').disabled,false);
  assert.equal(element('simulator-load').textContent,'重新加载环境');
  await run('selectSimulator()');
  assert.equal(calls[1].url,'/api/simulators/select');
  assert.equal(JSON.parse(calls[1].body).engine_id,'mujoco');
});

test('refresh detects a newly installed backend without switching or running a simulation',async()=>{
  const {element,calls,run}=setup([{body:catalog('not_installed')},{body:catalog()}]);
  await run('initSimulators(callbacks)');
  assert.equal(element('simulator-select').children[1].disabled,true);
  await element('simulator-refresh').listeners.click();
  assert.equal(element('simulator-select').children[1].disabled,false);
  assert.equal(element('simulator-select').value,'mujoco');
  assert.deepEqual(calls.map(call=>call.url),['/api/simulators','/api/simulators']);
});

test('a UI synchronization error reports that the engine already loaded',async()=>{
  const {element,run}=setup([{body:catalog()},{body:{backend:{engine_id:'genesis'},session_id:'session-b'}}],async()=>{throw new Error('Scene catalog failed');});
  await run('initSimulators(callbacks)');element('simulator-select').value='genesis';
  await element('simulator-load').listeners.click();
  assert.equal(element('simulator-select').value,'genesis');
  assert.ok(element('simulator-message').textContent.includes('环境已加载，但界面同步失败'));
  assert.ok(element('simulator-current').textContent.startsWith('Genesis'));
});

test('reasoning charts require one physical context and exclude other engines from the same model and task',async()=>{
  const shared={model:'same-model',task:'same task',kind:'agent',reasoning:'high',robot:'panda',scene_id:'tabletop',task_id:'lift_red',runs:2};
  const mujoco={...shared,engine_id:'mujoco',engine_version:'3.13.0',evaluator_source:'mujoco_privileged_state',evaluator_version:'task-evaluation-v2',mean_output_tokens:2,mean_inference_s:.5};
  const genesis={...shared,engine_id:'genesis',engine_version:'1.4.5',evaluator_source:'genesis_privileged_state',evaluator_version:'genesis-task-v1',mean_output_tokens:100,mean_inference_s:20};
  const {element,calls,run,scope}=setup([{body:{groups:[mujoco,genesis],aggregation_scope:'leaf_runs_only'}}],undefined,'research.js');
  scope.selected=mujoco;run('context={toast(){}};metricChartMarkup=(points)=>JSON.stringify(points);');
  element('analysis-context').value=run('analysisContextKey(selected)');
  element('analysis-model').value='same-model';element('analysis-task').value='same task';element('analysis-kind').value='agent';element('analysis-limit').value='100';
  await run('runAnalysis()');
  assert.equal(calls.length,1);
  assert.deepEqual(JSON.parse(element('analysis-token-chart').innerHTML).map(point=>point.value),[2]);
  assert.ok(element('analysis-rows').innerHTML.includes('MuJoCo 3.13.0'));
  assert.equal(element('analysis-rows').innerHTML.includes('Genesis'),false);
  assert.ok(element('analysis-message').textContent.includes('当前物理环境内 2 条'));
});

test('Panda cannot execute the SO-101 SmolVLA mapping; selecting SO-101 restores the retained preference',async()=>{
  const {element,calls,run}=setup([],undefined,'research.js'),select=element('policy-kind');
  const smolvla=new Element('option');smolvla.value='smolvla';select.options=[smolvla];select.value='smolvla';
  run("context={toast(){}};policyRobot='panda';updatePolicyAvailability('panda',false)");
  assert.equal(smolvla.disabled,true);assert.equal(select.value,'smolvla');
  assert.equal(element('policy-start').disabled,true);assert.equal(element('policy-one-chunk').disabled,true);
  assert.ok(element('policy-model-note').textContent.includes('仅适配 SO-101'));
  await run('startPolicy(false)');assert.equal(calls.length,0);
  select.value='vla';run("updatePolicyAvailability('panda',false)");
  assert.equal(select.value,'vla');assert.equal(element('policy-start').disabled,false);
  select.value='smolvla';run("updatePolicyAvailability('so101',false)");
  assert.equal(smolvla.disabled,false);assert.equal(select.value,'smolvla');
  assert.equal(element('policy-start').disabled,false);assert.equal(element('policy-one-chunk').disabled,false);
});

test('comparison filters unsupported checked methods by the requested robot and retains checkbox preferences',()=>{
  const {element,run,scope}=setup([],undefined,'research.js');
  const agent=new Element('input');agent.value='agent';agent.checked=true;
  const smolvla=new Element('input');smolvla.value='smolvla';smolvla.checked=true;
  scope.document.querySelector=selector=>selector==='input[name="method"][value="smolvla"]'?smolvla:element(selector.slice(1));
  scope.document.querySelectorAll=selector=>selector==='input[name="method"]:checked'?[agent,smolvla].filter(input=>input.checked):[];
  element('compare-robot').value='panda';run("policyRobot='so101';updateComparisonAvailability(false)");
  assert.equal(smolvla.disabled,true);assert.equal(smolvla.checked,true);
  assert.equal(run('JSON.stringify(availableComparisonMethods())'),'["agent"]');
  assert.equal(element('compare-start').disabled,true);
  assert.ok(element('comparison-method-note').textContent.includes('至少选择两种'));
  element('compare-robot').value='so101';run("policyRobot='panda';updateComparisonAvailability(false)");
  assert.equal(smolvla.disabled,false);assert.equal(smolvla.checked,true);
  assert.equal(run('JSON.stringify(availableComparisonMethods())'),'["agent","smolvla"]');
  assert.equal(element('compare-start').disabled,false);
});
