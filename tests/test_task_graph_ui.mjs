import test from 'node:test';
import assert from 'node:assert/strict';
import {taskGraphMarkup,routingMarkup} from '../src/task_graph.js';
import {costCoverageMarkup} from '../src/experiment_costs.js';

test('composite stages expose invalidation and terminal failure without hiding cause',()=>{
  const html=taskGraphMarkup({version:'v1',stages:[{id:'first',label:'取料',status:'achieved',satisfied:true},{id:'last',label:'交付',status:'revoked',after:['first']}],terminal_failure:true,failure_reason:'<physical violation>',events:[{id:1,type:'revoked',stage_id:'last',time:2}]});
  assert.match(html,/取料/);assert.match(html,/条件已失效/);assert.match(html,/任务已进入失败终态/);assert.match(html,/&lt;physical violation&gt;/);assert.match(html,/2.00 s/);
});
test('unknown routing probabilities and inclusive costs stay explicit',()=>{
  const html=routingMarkup({trace_id:'run',call_id:'call',requested_model:'router',model:'chosen',routing:{availability:'returned',selected_capability:'plan',router_policy_version:'v1',selection_probability:null}});
  assert.match(html,/router → 实际 chosen/);assert.match(html,/选择概率 未返回/);
  const cost=costCoverageMarkup({coverage_details:{calls:3,routing_reported_calls:1,excluded_inclusive_calls:2,unmetered_categories:['compute']},evidence_totals:{subscription_reference:{known_reference_amount_usd:.02}},non_model_evidence_totals:{actual_invoice:.1}});
  assert.match(cost,/未重复计入/);assert.match(cost,/订阅 API 等价参考/);assert.match(cost,/已登记账单/);assert.match(cost,/尚未登记：计算/);assert.doesNotMatch(cost,/undefined|NaN/);
});
