import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

class Element{
  constructor(tag='div'){
    this.tagName=tag;this.children=[];this.listeners={};this.attributes={};this.value='';this.disabled=false;this.hidden=false;this.classes=new Set();
    this.classList={add:name=>this.classes.add(name),toggle:(name,on)=>on?this.classes.add(name):this.classes.delete(name)};
  }
  set textContent(value){this.text=String(value);this.children=[];}
  get textContent(){return(this.text||'')+this.children.map(child=>child.textContent).join('');}
  append(...nodes){this.children.push(...nodes);}
  replaceChildren(...nodes){this.children=nodes;this.text='';}
  setAttribute(name,value){this.attributes[name]=value;}
  addEventListener(type,callback){this.listeners[type]=callback;}
}

const base=(extra={})=>({ready:true,session_id:'session-a',generation:4,backend:{engine_id:'alfworld',action_mode:'text'},
  environment:{task_name:'Put a mug in the sink.',observation_text:'You see a table.',admissible_commands:['look','go to table 1']},task:{status:'idle'},...extra});

function setup(execute=async()=>({ok:true,message:{last_action_success:true}})){
  const root=new Element(),calls=[];
  const scope=vm.createContext({console,callbacks:{execute:async payload=>{calls.push(structuredClone(payload));return execute(payload);}},document:{
    querySelector:selector=>selector==='#environment-panel'?root:null,createElement:tag=>new Element(tag)}});
  const source=readFileSync(new URL('../src/environment.js',import.meta.url),'utf8').replace(/^import .*;\s*$/gm,'').replace(/\bexport /g,'');
  vm.runInContext(source,scope);vm.runInContext('initEnvironment(callbacks)',scope);
  const find=className=>{const walk=node=>node.className?.split(' ').includes(className)?node:node.children.map(walk).find(Boolean);return walk(root);};
  const render=state=>{scope.nextState=structuredClone(state);vm.runInContext('renderEnvironment(nextState)',scope);};
  return{root,calls,find,render};
}

test('only discrete environments show task, native observation and string action buttons; SDK text stays text',()=>{
  const ui=setup(),hostile='<img src=x onerror=bad()>',state=base();
  state.environment={task_name:hostile,observation_text:hostile,admissible_commands:[hostile,'look',null,3]};
  ui.render(state);
  assert.equal(ui.root.hidden,false);
  assert.equal(ui.find('environment-task').textContent,hostile);
  assert.equal(ui.find('environment-observation').textContent,hostile);
  assert.equal(ui.find('environment-observation').children.length,0);
  assert.equal(ui.find('environment-commands').children.length,2);
  assert.ok(ui.find('environment-hint').textContent.includes('文字排版'));
  ui.render(base({backend:{engine_id:'mujoco',action_mode:'tcp_pose'}}));
  assert.equal(ui.root.hidden,true);
});

test('manual and native command buttons pass exactly one string and current generation',async()=>{
  const ui=setup();ui.render(base());
  await ui.find('environment-commands').children[1].listeners.click();
  assert.deepEqual(ui.calls,[{action:'environment_action',environment_action:'go to table 1',generation:4}]);
  ui.find('environment-input').value='  look  ';
  await ui.find('environment-form').listeners.submit({preventDefault(){}});
  assert.equal(ui.calls[1].environment_action,'look');
});

test('single flight disables manual and listed actions until the request completes',async()=>{
  const pending=Promise.withResolvers(),ui=setup(()=>pending.promise);ui.render(base());
  const button=ui.find('environment-commands').children[0],first=button.listeners.click();
  await button.listeners.click();
  assert.equal(ui.calls.length,1);
  assert.equal(ui.find('environment-input').disabled,true);
  assert.ok(ui.find('environment-commands').children.every(node=>node.disabled));
  pending.resolve({ok:true,message:{last_action_success:true}});await first;
  assert.equal(ui.find('environment-input').disabled,false);
  assert.ok(ui.find('environment-status').textContent.includes('请求已处理'));
});

test('all automated controllers, backend switch and offline states lock manual interactions',async()=>{
  for(const extra of [{ready:false},{backend_switching:true},...['agent','policy','comparison','research','composition','experiment','repair'].map(key=>({[key]:{active:true}}))]){
    const ui=setup();ui.render(base(extra));
    assert.equal(ui.find('environment-input').disabled,true,JSON.stringify(extra));
    await ui.find('environment-commands').children[0].listeners.click();
    assert.equal(ui.calls.length,0);
    assert.equal(ui.find('environment-lock').hidden,false);
  }
});

test('native rejection and thrown errors preserve the real cause rather than claiming success',async()=>{
  const ui=setup(async()=>({ok:true,message:{last_action_success:false,error:'Target is occluded <script>bad()</script>'}}));ui.render(base());
  await ui.find('environment-commands').children[0].listeners.click();
  assert.ok(ui.find('environment-status').textContent.includes('Target is occluded <script>bad()</script>'));
  assert.equal(ui.find('environment-status').attributes.role,'alert');
  assert.equal(ui.find('environment-status').children.length,0);
  const network=setup(async()=>{throw new Error('Unity process disconnected');});network.render(base());
  await network.find('environment-commands').children[0].listeners.click();
  assert.ok(network.find('environment-status').textContent.includes('Unity process disconnected'));
  assert.equal(network.find('environment-input').disabled,false);
});

test('old queued buttons and late responses cannot operate or overwrite a newly loaded episode',async()=>{
  const pending=Promise.withResolvers(),ui=setup(()=>pending.promise);ui.render(base());
  const old=ui.find('environment-commands').children[0];
  ui.render(base({generation:5}));
  await old.listeners.click();assert.equal(ui.calls.length,0);
  const running=ui.find('environment-commands').children[0].listeners.click();
  ui.render(base({generation:6}));
  pending.resolve({ok:true,message:{last_action_success:false,error:'Old episode failed'}});await running;
  assert.equal(ui.find('environment-status').textContent,'');
  assert.equal(ui.find('environment-input').disabled,false);
});

test('empty input makes no request, and unchanged polls retain the current action nodes',async()=>{
  const ui=setup();ui.render(base());const button=ui.find('environment-commands').children[0];
  ui.render(base());assert.equal(ui.find('environment-commands').children[0],button);
  await ui.find('environment-form').listeners.submit({preventDefault(){}});
  assert.equal(ui.calls.length,0);
  assert.ok(ui.find('environment-status').textContent.includes('请输入动作'));
});
