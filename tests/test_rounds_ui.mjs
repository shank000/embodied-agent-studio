import test from 'node:test';
import assert from 'node:assert/strict';
import {roundMarkup,renderRoundTimeline} from '../src/rounds.js';

test('round shows duration, all returned text, calls, recovery and native evidence safely',()=>{
  const html=roundMarkup({round:7,status:'complete',elapsed_s:4.2,phase_s:{inference:3,execution:1,observation:.2},
    calls:[{model:'routed-model',inference_s:3,status:'error',public_text:'<script>bad</script>',reasoning:{text:'returned summary'},input_tokens:4,output_tokens:2,error:'timeout'}],
    events:[{event:'recovery_retry',data:{attempt:1}},{event:'execution',data:{result:'feedback'}},{event:'completion_verification',data:{accepted:false}}],reward:0},'run');
  assert.match(html,/第 7 轮/);assert.match(html,/4.20 s/);assert.match(html,/执行与核验 1.00 s/);
  assert.match(html,/returned summary/);assert.match(html,/恢复后重试/);assert.match(html,/完成核验/);
  assert.match(html,/&lt;script&gt;/);assert.doesNotMatch(html,/<script>/);
  assert.match(html,/输入 4 \/ 输出 2/);
});

test('failed round without action and legacy missing metrics are still readable',()=>{
  const html=roundMarkup({round:1,status:'error',elapsed_s:null,error:'失败 <retry>',events:[]},'run');
  assert.match(html,/整轮 未记录/);assert.match(html,/失败 &lt;retry&gt;/);
  assert.doesNotMatch(html,/NaN|undefined/);
});

test('Claude CLI aggregate usage is visibly a CLI reference rather than a computed API bill',()=>{
  const html=roundMarkup({round:2,status:'complete',calls:[{harness:'claude_code',provider:'anthropic',model:'route-model',
    status:'complete',input_tokens:1000,output_tokens:100,reasoning_effort:'high',
    cli_num_turns:3,cli_model_usage:{'route-model':{inputTokens:1000,outputTokens:100}},cli_total_cost_usd:.0042,
    cost:{status:'unavailable',amount_usd:null,cli_reported_cost_usd:.0042},public_text:'Action'}]},'fixture');
  assert.match(html,/Claude Code 内部轮次 3/);
  assert.match(html,/用量为 CLI 汇总/);
  assert.match(html,/CLI 费用参考 USD 0\.004200/);
  assert.match(html,/实际费用以模型服务账单为准/);
  assert.match(html,/输入 1,000 \/ 输出 100/);
  assert.match(html,/推理强度 high/);
});

test('missing and invalid CLI fee references never render NaN, fake zero bills or unsafe markup',()=>{
  for(const value of [undefined,null,-1,Infinity,NaN,'<script>bad</script>']){
    const html=roundMarkup({round:1,status:'error',calls:[{harness:'claude_code',cli_total_cost_usd:value}]},'fixture');
    assert.match(html,/用量为 CLI 汇总/);
    assert.doesNotMatch(html,/CLI 费用参考 USD|NaN|Infinity|<script>|undefined/);
  }
  const zero=roundMarkup({round:1,status:'complete',calls:[{harness:'claude_code',cli_num_turns:1,cli_total_cost_usd:0}]},'fixture');
  assert.match(zero,/CLI 费用参考 USD 0\.000000/);
  for(const harness of ['workbench','dsh','codex_cli','http']){
    const html=roundMarkup({round:1,status:'complete',calls:[{harness,cli_total_cost_usd:.0042,reasoning_effort:'off'}]},'fixture');
    assert.doesNotMatch(html,/Claude Code 内部轮次|CLI 费用参考/);
    assert.match(html,/推理强度 off/);
  }
});

test('an empty run timeline shows a zero count without an inverted page range',async()=>{
  const count={textContent:''},error={textContent:''},buttons=[{},{}];
  const list={children:[],querySelectorAll:()=>[],replaceChildren(...nodes){this.children=nodes;}};
  const root={replaceChildren(){},closest:()=>null,querySelectorAll:()=>buttons,
    querySelector:selector=>({'.round-list':list,'.round-list-count':count,'.round-list-error':error}[selector])};
  const originalFetch=globalThis.fetch;
  globalThis.fetch=async()=>({ok:true,json:async()=>({total:0,rounds:[]})});
  try{
    renderRoundTimeline(root,{run_id:'empty-run',status:'observing',active:true});
    await new Promise(resolve=>setImmediate(resolve));
    assert.equal(count.textContent,'共 0 轮');
    assert.equal(error.textContent,'等待第一轮观察。');
    assert.equal(buttons[0].disabled,true);assert.equal(buttons[1].disabled,true);
  }finally{globalThis.fetch=originalFetch;}
});
