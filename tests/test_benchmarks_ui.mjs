import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

class Element{
  constructor(tag='div'){
    this.tagName=tag;this.children=[];this.listeners={};this.attributes={};this.dataset={};this.value='';this.disabled=false;this.hidden=false;this.checked=false;this.files=[];this.classes=new Set();
    this.classList={toggle:(name,on)=>on?this.classes.add(name):this.classes.delete(name)};
  }
  set textContent(value){this.text=String(value);this.children=[];}
  get textContent(){return(this.text||'')+this.children.map(child=>child.textContent).join('');}
  append(...nodes){this.children.push(...nodes);}
  insertBefore(node,before){const index=this.children.indexOf(before);if(index<0)this.append(node);else this.children.splice(index,0,node);}
  replaceChildren(...nodes){this.children=nodes;this.text='';}
  setAttribute(name,value){this.attributes[name]=value;}
  addEventListener(type,callback){this.listeners[type]=callback;}
  querySelectorAll(selector){const found=[];for(const child of this.children){if(child.tagName===selector)found.push(child);found.push(...child.querySelectorAll(selector));}return found;}
  querySelector(selector){const match=node=>selector.startsWith('.')?node.className?.split(' ').includes(selector.slice(1)):node.tagName===selector;for(const child of this.children){if(match(child))return child;const nested=child.querySelector(selector);if(nested)return nested;}return null;}
}

const catalog=(extra={})=>({benchmark_id:'robodojo',version:'0.2.0',revision:'f57e1633ffed59761bc8e568cfe7530c4b33381c',
  dimensions:[{id:'generalization',name:'Generalization'},{id:'precision',name:'Precision'}],
  tasks:[{id:'stack_bowls',name:'stack bowls',dimension:'generalization'},{id:'insert_key',name:'insert key',dimension:'precision'}],
  embodiments:[{id:'arx_x5',name:'Dual ARX X5'}],
  readiness:{ready:false,message:'Waiting for independent Linux GPU runtime.',checks:[{id:'source',label:'Source',status:'passed',detail:'pinned'},{id:'runtime',label:'Runtime',status:'missing',detail:'not installed'}]},
  state:{active:false,status:'idle'},history:[],...extra});

const result={success_rate:0.5,eval_time:2,score:62.5,details:{0:{layout_id:1,success:true,score:1},1:{layout_id:2,success:false,score:0.25}}};

function setup(initial=catalog(),respond=async()=>({})){
  const nav=new Element('nav'),main=new Element('main'),calls=[],toasts=[],events=[],blobs=[],revoked=[];const navDetail=new Element();navDetail.className='nav-detail';nav.append(navDetail);main.append(new Element('footer'));
  let nextInfo=initial;
  const document={createElement:tag=>new Element(tag),querySelector:selector=>selector==='.workspace-nav'?nav:selector==='main'?main:null};
  class MockURL extends URL{}
  MockURL.createObjectURL=blob=>{blobs.push(blob);return`blob:plan-${blobs.length}`;};MockURL.revokeObjectURL=value=>revoked.push(value);
  const scope=vm.createContext({console,document,URL:MockURL,Blob,window:{location:{origin:'http://127.0.0.1:8000'},dispatchEvent:event=>events.push(event)},CustomEvent:class{constructor(type,options){this.type=type;this.detail=options.detail;}},
    callbacks:{toast:text=>toasts.push(text)},fetch:async(url,options={})=>{
      const call={url,method:options.method||'GET',body:options.body?JSON.parse(options.body):undefined};calls.push(call);
      const response=call.method==='GET'?nextInfo:await respond(call);
      if(response instanceof Error)throw response;
      return{ok:response?.httpError!==true,status:response?.httpError?422:200,json:async()=>structuredClone(response)};
    }});
  const source=readFileSync(new URL('../src/benchmarks.js',import.meta.url),'utf8').replace(/^import .*;\s*$/gm,'').replace(/\bexport /g,'');
  vm.runInContext(source,scope);
  const all=(predicate,node=main)=>[...(predicate(node)?[node]:[]),...node.children.flatMap(child=>all(predicate,child))];
  const find=className=>all(node=>node.className?.split(' ').includes(className))[0];
  const id=id=>all(node=>node.id===id)[0];
  const button=text=>all(node=>node.tagName==='button'&&node.textContent===text)[0];
  const click=node=>node.listeners.click({preventDefault(){}});
  const init=()=>vm.runInContext('initBenchmarks(callbacks)',scope);
  const render=state=>{scope.nextState=structuredClone(state);return vm.runInContext('renderBenchmarks(nextState)',scope);};
  return{nav,main,calls,toasts,events,blobs,revoked,all,find,id,button,click,init,render,setInfo:value=>{nextInfo=value;}};
}

test('adds one benchmark page; unready runtime still permits planning and result import',async()=>{
  const ui=setup();await ui.init();await ui.init();
  assert.equal(ui.nav.children.length,2);assert.equal(ui.nav.children[0].dataset.page,'benchmarks');assert.equal(ui.nav.children[1].className,'nav-detail');
  assert.equal(ui.id('benchmarks-view').hidden,true);
  assert.equal(ui.main.children.at(-1).tagName,'footer');
  assert.equal(ui.button('运行官方评测').disabled,true);
  assert.equal(ui.button('生成评测计划').disabled,false);
  assert.equal(ui.button('导入官方结果').disabled,false);
  assert.equal(ui.button('停止评测').disabled,true);
  assert.ok(ui.find('benchmark-checks').textContent.includes('通过'));
  assert.ok(ui.main.textContent.includes('不按零费用计算'));
  assert.match(ui.main.textContent,/demo_policy 是输出零动作的连通样例/);
  assert.match(ui.main.textContent,/XPolicyLab WebSocket 策略服务/);
  assert.match(ui.main.textContent,/不能直接填写模型 API 地址或 Codex 模型名/);
  assert.equal(ui.calls.length,1);
});

test('plan export retains official argv and warnings as text without starting simulation',async()=>{
  const hostile='<img src=x onerror=bad()>',plan={plan_id:'preview/1',ready:false,commands:[{task_id:hostile,seed:0,argv:['bash','scripts/robodojo.sh','client','--task',hostile]}],warnings:[hostile]};
  const ui=setup(catalog(),async()=>plan);await ui.init();await ui.click(ui.button('生成评测计划'));
  assert.deepEqual(ui.calls[1],{url:'/api/benchmarks/robodojo/plan',method:'POST',body:{task_ids:['stack_bowls'],seeds:[0],embodiment:'arx_x5',policy_name:'demo_policy',policy_host:'127.0.0.1',policy_port:9999,checkpoint:'external',action_type:'ee',episodes:1,mode:'smoke',copy_videos:false,max_wall_time_s:3600}});
  assert.equal(ui.find('benchmark-plan-output').hidden,false);
  assert.equal(ui.find('benchmark-plan-warnings').children[0].textContent,hostile);
  assert.equal(ui.find('benchmark-plan-warnings').children[0].children.length,0);
  assert.ok(ui.find('benchmark-plan-commands').textContent.includes(hostile));
  assert.equal(ui.blobs.length,1);assert.deepEqual(JSON.parse(await ui.blobs[0].text()),plan);
  assert.ok(ui.main.textContent.includes('尚未执行'));
  assert.equal(ui.calls.some(call=>call.url.endsWith('/start')),false);
  await ui.click(ui.button('生成评测计划'));assert.deepEqual(ui.revoked,['blob:plan-1']);
});

test('filters retain selections and official seeds/native episodes are transmitted',async()=>{
  const ui=setup(catalog(),async()=>({plan_id:'filtered',commands:[]}));await ui.init();
  ui.id('benchmark-dimension').value='precision';await ui.id('benchmark-dimension').listeners.change();
  assert.equal(ui.find('benchmark-tasks').children.length,1);
  await ui.click(ui.button('选择当前显示'));ui.id('benchmark-seeds').value='0, 1，2,0';ui.id('benchmark-episodes').value='native';ui.id('benchmark-mode').value='benchmark';
  await ui.click(ui.button('生成评测计划'));
  assert.deepEqual(ui.calls[1].body.task_ids,['stack_bowls','insert_key']);assert.deepEqual(ui.calls[1].body.seeds,[0,1,2]);
  assert.equal(ui.calls[1].body.episodes,'native');assert.equal(ui.calls[1].body.mode,'benchmark');
  await ui.click(ui.button('清空选择'));assert.equal(ui.button('生成评测计划').disabled,true);
});

test('invalid seeds, episode counts and ports preserve the error and make no plan request',async()=>{
  for(const [id,value,reason] of [['benchmark-seeds','42','0、1、2'],['benchmark-episodes','51','1–50'],['benchmark-episodes','1.5','1–50'],['benchmark-port','0','1–65535'],['benchmark-wall-time','9','10–86400'],['benchmark-wall-time','1.5','10–86400']]){
    const ui=setup();await ui.init();ui.id(id).value=value;await ui.click(ui.button('生成评测计划'));
    assert.equal(ui.calls.length,1);assert.ok(ui.find('benchmark-message').textContent.includes(reason));assert.equal(ui.find('benchmark-message').attributes.role,'alert');
  }
});

test('ready start is single flight, and automated workbench controllers block new native jobs',async()=>{
  const pending=Promise.withResolvers(),ui=setup(catalog({readiness:{ready:true,checks:[]}}),()=>pending.promise);await ui.init();
  assert.ok(ui.find('benchmark-readiness').textContent.includes('静态依赖检查通过'));
  assert.ok(ui.find('benchmark-readiness').textContent.includes('原生加载与任务结果以实际运行日志为准'));
  const start=ui.button('运行官方评测'),first=ui.click(start);await ui.click(start);
  assert.equal(ui.calls.filter(call=>call.url.endsWith('/start')).length,1);
  assert.equal(ui.id('benchmark-policy').disabled,true);assert.equal(ui.button('停止评测').disabled,false);
  pending.resolve({active:true,status:'running'});await first;assert.equal(ui.id('benchmark-policy').disabled,false);
  for(const extra of [{backend_switching:true},...['agent','policy','comparison','research','composition','experiment','repair'].map(key=>({[key]:{active:true}}))]){
    ui.render(extra);assert.equal(start.disabled,true);await ui.click(start);
  }
  assert.equal(ui.calls.filter(call=>call.url.endsWith('/start')).length,1);
});

test('stop during pending start rejects duplicate stop and reads canonical final state',async()=>{
  const startReply=Promise.withResolvers(),stopReply=Promise.withResolvers();
  const ui=setup(catalog({readiness:{ready:true,checks:[]}}),call=>call.url.endsWith('/start')?startReply.promise:stopReply.promise);await ui.init();
  const start=ui.click(ui.button('运行官方评测')),stopButton=ui.button('停止评测'),stop=ui.click(stopButton);await ui.click(stopButton);
  assert.equal(ui.calls.filter(call=>call.url.endsWith('/stop')).length,1);
  ui.setInfo(catalog({readiness:{ready:true,checks:[]},state:{active:false,status:'stopped',message:'Native process stopped.',timing:{elapsed_wall_s:12.5}}}));
  stopReply.resolve({});await stop;startReply.resolve({});await start;
  assert.ok(ui.find('benchmark-job').textContent.includes('已停止'));assert.ok(ui.find('benchmark-job').textContent.includes('12.5 s'));
  assert.ok(!ui.find('benchmark-message').textContent.includes('任务已提交'));
});

test('unready result import archives explicit payload with videos off and uses separate episode/SR/score metrics',async()=>{
  const run={run_id:'external-1',display_name:'Imported benchmark',status:'imported',metrics:{episodes:2,success_rate:0.5,score:62.5},benchmark:{policy:'custom',embodiment:'arx_x5',seed:0}};
  const ui=setup(catalog(),async call=>{assert.ok(call.url.endsWith('/import'));ui.setInfo(catalog({history:[run]}));return run;});await ui.init();
  ui.id('benchmark-import-json').value=JSON.stringify(result);ui.id('benchmark-import-policy').value='custom';await ui.click(ui.button('导入官方结果'));
  assert.deepEqual(ui.calls[1].body,{task_id:'stack_bowls',seed:0,policy:'custom',embodiment:'arx_x5',payload:result,copy_videos:false});
  assert.ok(ui.find('benchmark-import-result').textContent.includes('external-1'));
  const history=ui.find('benchmark-history-list');assert.ok(history.textContent.includes('50.0%'));assert.ok(history.textContent.includes('62.5 / 100'));assert.ok(history.textContent.includes('有效回合2'));
  assert.ok(!history.textContent.includes('2 s'));assert.ok(history.textContent.includes('不代表官方完整基准分数'));
  await ui.click(ui.all(node=>node.tagName==='button'&&node.textContent==='查看实验详情')[0]);assert.equal(ui.events[0].type,'embodiworks:open-result');assert.equal(ui.events[0].detail.run_id,'external-1');
  const links=ui.all(node=>node.tagName==='a').map(node=>node.href);assert.ok(links.includes('/api/runs/external-1/bundle'));assert.ok(links.includes('/api/runs/external-1/analysis'));
});

test('invalid imports and real server errors never claim import or execution success',async()=>{
  for(const value of ['not json','[]','null']){const ui=setup();await ui.init();ui.id('benchmark-import-json').value=value;await ui.click(ui.button('导入官方结果'));assert.equal(ui.calls.length,1);assert.equal(ui.find('benchmark-message').attributes.role,'alert');}
  const ui=setup(catalog(),async()=>({httpError:true,detail:'Official score disagrees with episode details.'}));await ui.init();ui.id('benchmark-import-json').value=JSON.stringify(result);await ui.click(ui.button('导入官方结果'));
  assert.equal(ui.find('benchmark-message').textContent,'Official score disagrees with episode details.');assert.equal(ui.find('benchmark-import-result').children.length,0);assert.equal(ui.button('导入官方结果').disabled,false);
});

test('history renders all SDK strings as text and plays only archived same-origin videos',async()=>{
  const hostile='<script>bad()</script>',videos=[{url:'/api/runs/r1/artifact/benchmark/videos/episode_head.mp4',camera:hostile},{url:'https://evil.invalid/api/runs/r1/video.mp4'},{url:'javascript:bad()'},{url:'/api/settings'}];
  const ui=setup(catalog({history:[{run_id:'r1',display_name:hostile,status:'imported',metrics:{},native_videos:videos}]}));await ui.init();
  const history=ui.find('benchmark-history-list'),players=history.querySelectorAll('video');assert.equal(players.length,1);assert.equal(players[0].controls,true);assert.equal(players[0].preload,'none');
  assert.ok(history.textContent.includes(hostile));assert.equal(history.querySelectorAll('script').length,0);assert.ok(history.textContent.includes('未提供'));
});

test('result file loads without implicit import and large/invalid files remain explicit errors',async()=>{
  const ui=setup();await ui.init();const file=ui.id('benchmark-import-file');file.files=[{name:'_result.json',size:100,text:async()=>JSON.stringify(result)}];await file.listeners.change();assert.equal(ui.calls.length,1);assert.deepEqual(JSON.parse(ui.id('benchmark-import-json').value),result);
  assert.ok(ui.find('benchmark-message').textContent.includes('尚未导入'));
  file.files=[{name:'huge.json',size:11*1024*1024,text:async()=>{throw Error('must not read');}}];await file.listeners.change();assert.ok(ui.find('benchmark-message').textContent.includes('不超过 10 MB'));assert.equal(ui.id('benchmark-import-json').value,'');
  file.files=[{name:'bad.json',size:2,text:async()=>'no'}];await file.listeners.change();assert.ok(ui.find('benchmark-message').textContent.includes('读取结果文件失败'));
});

test('offline service disables mutations and refresh restores planning without claiming runtime readiness',async()=>{
  const ui=setup(new Error('Workbench offline'));await ui.init();assert.equal(ui.button('生成评测计划').disabled,true);assert.equal(ui.button('导入官方结果').disabled,true);assert.ok(ui.find('benchmark-message').textContent.includes('Workbench offline'));
  ui.setInfo(catalog());await ui.click(ui.button('刷新目录与状态'));assert.equal(ui.button('生成评测计划').disabled,false);assert.equal(ui.button('运行官方评测').disabled,true);
});

test('failed later refresh and import cannot reuse stale ready or successful-import messages',async()=>{
  let fail=false;
  const ui=setup(catalog({readiness:{ready:true,message:'Runtime passed.',checks:[]}}),async()=>fail?{httpError:true,detail:'Invalid new result.'}:{run_id:'first'});await ui.init();
  ui.id('benchmark-import-json').value=JSON.stringify(result);await ui.click(ui.button('导入官方结果'));assert.ok(ui.find('benchmark-import-result').textContent.includes('first'));
  fail=true;await ui.click(ui.button('导入官方结果'));assert.equal(ui.find('benchmark-import-result').children.length,0);assert.equal(ui.find('benchmark-message').textContent,'Invalid new result.');
  ui.setInfo(new Error('Connection lost'));await ui.click(ui.button('刷新目录与状态'));assert.equal(ui.button('运行官方评测').disabled,true);assert.ok(ui.find('benchmark-readiness').textContent.includes('上次缓存'));assert.ok(!ui.find('benchmark-readiness').textContent.includes('Runtime passed.'));
});

test('Linux export uses reviewed plan settings and leaves runtime idle',async()=>{
  const script='#!/usr/bin/env bash\nprintf "official fixture"\n';
  const ui=setup(catalog(),async call=>call.url.endsWith('/export')?{filename:'robodojo-plan-fixture.sh',script}:{plan_id:'fixture',commands:[]});await ui.init();
  ui.id('benchmark-wall-time').value='7200';ui.id('benchmark-record-videos').checked=true;
  assert.equal(ui.button('生成 Linux 运行脚本').disabled,true);
  await ui.click(ui.button('生成评测计划'));
  assert.equal(ui.button('生成 Linux 运行脚本').disabled,false);
  await ui.click(ui.button('生成 Linux 运行脚本'));
  const sent=ui.calls.find(call=>call.url.endsWith('/export'));
  assert.deepEqual(sent.body,ui.calls.find(call=>call.url.endsWith('/plan')).body);
  assert.equal(sent.body.max_wall_time_s,7200);assert.equal(sent.body.copy_videos,true);
  const link=ui.all(node=>node.tagName==='a'&&node.textContent==='下载 Linux 脚本')[0];
  assert.equal(link.hidden,false);assert.equal(link.download,'robodojo-plan-fixture.sh');
  assert.equal(await ui.blobs.at(-1).text(),script);
  assert.match(ui.find('benchmark-message').textContent,/--dry-run/);
  assert.equal(ui.calls.some(call=>call.url.endsWith('/start')),false);
});

test('changing planned configuration revokes stale JSON and script downloads',async()=>{
  const ui=setup(catalog(),async call=>call.url.endsWith('/export')?{filename:'plan.sh',script:'#!/bin/bash\n'}:{plan_id:'fixture',commands:[]});await ui.init();
  await ui.click(ui.button('生成评测计划'));await ui.click(ui.button('生成 Linux 运行脚本'));
  ui.id('benchmark-policy').value='different_policy';ui.id('benchmark-policy').listeners.input();
  assert.deepEqual(ui.revoked,['blob:plan-1','blob:plan-2']);
  for(const label of ['下载计划 JSON','下载 Linux 脚本'])assert.equal(ui.all(node=>node.tagName==='a'&&node.textContent===label)[0].hidden,true);
  assert.match(ui.find('benchmark-plan-output').textContent,/配置已变化/);
  assert.match(ui.find('benchmark-message').textContent,/原计划与脚本已失效/);
  assert.doesNotMatch(ui.find('benchmark-message').textContent,/Linux 脚本已生成/);
  const before=ui.calls.length;await ui.click(ui.button('生成 Linux 运行脚本'));assert.equal(ui.calls.length,before);
  await ui.click(ui.button('生成评测计划'));assert.equal(ui.calls.at(-1).body.policy_name,'different_policy');
  const input=ui.find('benchmark-tasks').querySelectorAll('input')[1];input.checked=true;input.listeners.change();
  assert.equal(ui.button('生成 Linux 运行脚本').disabled,true);
});

test('export is single flight and server failure restores controls without claiming a download',async()=>{
  const pending=Promise.withResolvers();let exports=0;
  const ui=setup(catalog(),async call=>call.url.endsWith('/export')?(exports++,pending.promise):{plan_id:'fixture',commands:[]});await ui.init();await ui.click(ui.button('生成评测计划'));
  const exportButton=ui.button('生成 Linux 运行脚本'),first=ui.click(exportButton);await ui.click(exportButton);
  assert.equal(exports,1);assert.equal(exportButton.disabled,true);assert.equal(ui.id('benchmark-wall-time').disabled,true);
  pending.resolve({httpError:true,detail:'Export unavailable'});await first;
  assert.equal(exportButton.disabled,false);assert.equal(ui.id('benchmark-wall-time').disabled,false);
  assert.equal(ui.find('benchmark-message').textContent,'Export unavailable');
  assert.equal(ui.all(node=>node.tagName==='a'&&node.textContent==='下载 Linux 脚本')[0].hidden,true);
  assert.equal(ui.blobs.length,1);
});

test('manual result edits supersede a pending file read',async()=>{
  const ui=setup();await ui.init();const pending=Promise.withResolvers();
  const file=ui.id('benchmark-import-file');file.files=[{name:'_result.json',size:100,text:()=>pending.promise}];
  const loading=file.listeners.change();ui.id('benchmark-import-json').value='{"manual":true}';ui.id('benchmark-import-json').listeners.input();
  pending.resolve(JSON.stringify(result));await loading;
  assert.equal(ui.id('benchmark-import-json').value,'{"manual":true}');assert.equal(ui.calls.length,1);
});

test('active native job exposes the original result record while it is running',async()=>{
  const ui=setup(catalog({state:{active:true,status:'running',run_id:'native-run',elapsed_s:12,current_task:'stack_bowls',current_seed:0,completed_jobs:1,total_jobs:3}}));await ui.init();
  const job=ui.find('benchmark-job-links');assert.match(job.textContent,/查看实验详情/);
  await ui.click(job.querySelector('button'));assert.equal(ui.events[0].detail.run_id,'native-run');
  const progress=ui.find('benchmark-job-progress');assert.equal(progress.hidden,false);
  assert.match(progress.textContent,/已归档任务 × 种子：1 \/ 3/);
  assert.match(progress.textContent,/单项可能仅含部分回合/);
  assert.match(progress.textContent,/当前任务：stack_bowls · seed 0/);
});
