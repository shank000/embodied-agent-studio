import test from 'node:test';
import assert from 'node:assert/strict';
import {roundMarkup} from '../src/rounds.js';

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
