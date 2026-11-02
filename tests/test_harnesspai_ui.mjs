import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';

function setup() {
  const nodes = new Map();
  const node = id => {if(!nodes.has(id))nodes.set(id,{hidden:false,disabled:false,innerHTML:'',textContent:'',value:'',href:'',insertAdjacentHTML(_,html){this.innerHTML+=html;}});return nodes.get(id);};
  const scope = vm.createContext({document:{querySelector:selector=>node(selector.slice(1))},console});
  const source = readFileSync(new URL('../src/harnesspai.js',import.meta.url),'utf8').replace(/^import .*;\s*$/gm,'').replace(/\bexport /g,'');
  vm.runInContext(source,scope);
  return {node,scope,run:source=>vm.runInContext(source,scope)};
}

test('program results distinguish process exit from native success and escape error messages',()=>{
  const {run}=setup();
  const html=run(`programResultsMarkup('run', [{seed:42,process_ok:true,success:false,elapsed_s:3,steps:2,message:'<script>bad</script>',evaluation:{criterion:'drawer must open',checks:[{passed:false,label:'drawer slide',actual:0,expected:'> .12'}]}}])`);
  assert.match(html,/未通过环境验收/);assert.match(html,/3.00 s/);assert.match(html,/drawer must open/);
  assert.match(html,/&lt;script&gt;/);assert.doesNotMatch(html,/<script>/);
  assert.match(html,/标准输出/);assert.match(html,/错误日志/);
});

test('active program locks edits, keeps stop available, and retains video nodes on polling',()=>{
  const {run,node}=setup();
  run(`info={ready:true};renderHarnessPAI({ready:true,harnesspai:{run_id:'run',active:true,status:'running',rows:[{seed:42,success:false}]}})`);
  assert.equal(node('hp-inputs').disabled,true);assert.equal(node('hp-start').disabled,true);assert.equal(node('hp-stop').disabled,false);
  node('hp-results').innerHTML='video still playing';
  run(`renderHarnessPAI({ready:true,harnesspai:{run_id:'run',active:true,status:'running',rows:[{seed:42,success:false}],elapsed_s:10}})`);
  assert.equal(node('hp-results').innerHTML,'video still playing');
  run(`renderHarnessPAI({ready:true,harnesspai:{run_id:'run',active:true,status:'running',rows:[{seed:42,success:false},{seed:43,success:true}]}})`);
  assert.ok(node('hp-results').innerHTML.startsWith('video still playing'));
  assert.equal(node('hp-bundle').hidden,true);
  run(`renderHarnessPAI({ready:true,harnesspai:{run_id:'run',active:false,status:'verified',rows:[],can_promote:true}})`);
  assert.equal(node('hp-start').disabled,false);assert.equal(node('hp-stop').disabled,true);assert.equal(node('hp-promote').hidden,false);
  run(`renderHarnessPAI({ready:false})`);
  assert.equal(node('hp-start').disabled,true);
});

test('another controller blocks program launch and a stale idle poll cannot hide a newly started run',()=>{
  const {run,node}=setup();
  run(`info={ready:true};live={run_id:'new',active:true,status:'running'};renderHarnessPAI({ready:true,harnesspai:{run_id:'old',active:false,status:'verified'}})`);
  assert.equal(node('hp-status').textContent,'验证中');
  run(`live={active:false,status:'idle'};renderHarnessPAI({ready:true,agent:{active:true}})`);
  assert.equal(node('hp-start').disabled,true);
});

test('evolution preserves a playing baseline video when calls or batches are appended',()=>{
  const {run,node}=setup();
  run(`info={ready:true};renderHarnessPAI({ready:true,harnesspai:{run_id:'evo',active:true,batches:[{stage:'baseline',success_rate:0,rows:[]}],model_calls:1}})`);
  node('hp-evolution-results').innerHTML='baseline video playing';
  run(`renderHarnessPAI({ready:true,harnesspai:{run_id:'evo',active:true,batches:[{stage:'baseline',success_rate:0,rows:[]}],model_calls:2}})`);
  assert.equal(node('hp-evolution-results').innerHTML,'baseline video playing');
  run(`renderHarnessPAI({ready:true,harnesspai:{run_id:'evo',active:true,batches:[{stage:'baseline',success_rate:0,rows:[]},{stage:'evolution',success_rate:1,elapsed_s:4,rows:[]}],model_calls:2}})`);
  assert.ok(node('hp-evolution-results').innerHTML.startsWith('baseline video playing'));
  assert.match(node('hp-evolution-results').innerHTML,/100%/);
  const html=run(`evolutionResultsMarkup('evo',{mode:'evolve',skill_writeback:{ok:false,error:'<bad>'}})`);
  assert.match(html,/&lt;bad&gt;/);assert.match(html,/不等同于论文/);
});

test('no-edit explanation and actual evolution round distinguish probes from final validation',()=>{
  const {run,node}=setup();
  const html=run(`evolutionResultsMarkup('run',{mode:'evolve',last_model_response:{id:1,role:'edit',file_count:0,explanation:'<no change>'},batches:[{path:'evolution/runs/round_001/probe_0',rows:[]},{path:'evolution/runs/round_001/rollout',rows:[]}]})`);
  assert.match(html,/返回 0 个文件/);assert.match(html,/&lt;no change&gt;/);
  assert.match(html,/演化第 2 轮 · 小批次检查/);assert.match(html,/演化第 2 轮 · 完整演化批次/);
  run(`info={ready:true};renderHarnessPAI({ready:true,harnesspai:{run_id:'run',elapsed_s:5,rows:[],last_model_response:{id:1,role:'edit',file_count:0,explanation:'no change'}}})`);
  assert.match(node('hp-time').textContent,/0 个最终验证种子/);
  assert.match(node('hp-model-response').innerHTML,/no change/);
});

test('automatic mode hides manual registration and reports reuse in historical results',()=>{
  const {run,node}=setup();
  node('hp-mode').value='auto';run('modeFields()');
  assert.equal(node('hp-program-editor').hidden,true);
  assert.equal(node('hp-source').required,false);
  assert.match(node('hp-start').textContent,/自动复用/);
  run(`renderProgramAnalysis({run_id:'auto',status:'verified',harnesspai:{mode:'auto',rows:[{seed:42},{seed:43}],source_origin:{kind:'retrieved'},archive:{ok:true,reused:true}}})`);
  assert.match(node('result-harnesspai').innerHTML,/无需重复注册/);
  assert.doesNotMatch(node('result-harnesspai').innerHTML,/data-program-promote/);
  node('hp-mode').value='validate';run('modeFields()');
  assert.equal(node('hp-program-editor').hidden,false);
  assert.equal(node('hp-source').required,true);
  assert.match(run(`automationMarkup({archive:{ok:false,error:'<storage error>'}})`),/&lt;storage error&gt;/);
});

test('a failed evolution exposes its full execution log alongside the cause',()=>{
  const {run,node}=setup();
  run(`info={ready:true};renderHarnessPAI({ready:true,harnesspai:{run_id:'failed-run',active:false,status:'error',message:'PermissionError: file occupied'}})`);
  assert.match(node('hp-message').textContent,/PermissionError/);
  assert.equal(node('hp-runner-log').hidden,false);
  assert.equal(node('hp-runner-log').href,'/api/runs/failed-run/artifact/runner.log');
  run(`renderHarnessPAI({ready:true,harnesspai:{run_id:'failed-run',status:'running',active:true}})`);
  assert.equal(node('hp-runner-log').hidden,true);
});
