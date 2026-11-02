import assert from 'node:assert/strict';
import test from 'node:test';
import {behaviorPlan,behaviorScore} from '../src/behavior.js';

const form={task:' turning_on_radio ',instances:'0, 1，2',mode:'public_test',host:'127.0.0.1',port:'8001',
  scheme:'ws',policy:'fixture',rollouts:'1',maxSteps:'',video:true,wall:'3600'};
test('benchmark form uses native evaluator fields and preserves official default step budget',()=>{
  const result=behaviorPlan(form);
  assert.deepEqual(result.instance_indices,[0,1,2]);assert.equal(result.task_name,'turning_on_radio');
  assert.equal(result.max_steps,null);assert.equal(result.record_video,true);assert.equal(result.policy_port,8001);
  assert.equal(result.max_wall_time_s,3600);
});
test('instance typos and duplicate episodes are rejected before submission',()=>{
  for(const instances of ['0,0','-1','foo','1.5'])assert.throws(()=>behaviorPlan({...form,instances}));
});
test('partial Q is distinct from full success and missing score is unknown',()=>{
  assert.match(behaviorScore({success:false,q_score:.8}),/完全成功：否.*0.800/);
  assert.match(behaviorScore({success:true,q_score:{final:1}}),/完全成功：是.*1.000/);
  assert.match(behaviorScore({}),/未判定.*未返回/);
});
