// Public configuration handlers; no browser, model call or simulator is started.
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';
import {harnessNames,providerNames,effectiveHarness,harnessCompatibility,harnessReasoningError} from '../src/agent_harness.js';

class Element {
  constructor(tag='div'){this.tagName=tag;this.children=[];this.value='';this.disabled=false;this.hidden=false;this.checked=false;this.listeners={};this.classList={toggle(){}};}
  get options(){return this.children.filter(node=>node.tagName==='option');}
  set textContent(value){this.text=String(value);this.children=[];}
  get textContent(){return this.text||'';}
  append(node){this.children.push(node);if(this.tagName==='select'&&this.children.length===1)this.value=node.value;}
  replaceChildren(){this.children=[];this.value='';}
  querySelectorAll(selector){return this.children.filter(node=>node.tagName===selector);}
  addEventListener(type,listener){this.listeners[type]=listener;}
  showModal(){this.open=true;}
  close(){this.open=false;}
}

const defaults={provider:'openai',harness:'workbench',model:'routing-model',base_url:'https://router.example/v1',
  harness_url:'http://127.0.0.1:8020/agent',has_api_key:true,has_harness_api_key:false,
  reasoning:'high',max_rounds:10,timeout_s:120,max_output_tokens:16384};

function setup(config={},capability={levels:['low','high'],source:'service_metadata'}){
  const nodes=new Map(),calls=[],toasts=[];
  const element=id=>{if(!nodes.has(id))nodes.set(id,new Element());return nodes.get(id);};
  const html=readFileSync(new URL('../index.html',import.meta.url),'utf8');
  for(const match of html.matchAll(/<(input|select|button|form|dialog|p|em)[^>]*id="([^"]+)"([^>]*)>/g)){
    const node=element(match[2]);node.tagName=match[1];node.value=match[3].match(/value="([^"]*)"/)?.[1]||'';
  }
  for(const match of html.matchAll(/<select[^>]*id="([^"]+)"[^>]*>([\s\S]*?)<\/select>/g)){
    for(const item of match[2].matchAll(/<option value="([^"]*)"[^>]*>([^<]*)<\/option>/g)){
      const option=new Element('option');option.value=item[1];option.textContent=item[2];element(match[1]).append(option);
    }
  }
  for(const id of ['config-base-url','config-api-key'])element('api-config-fields').append(element(id));
  for(const id of ['config-harness-url','config-harness-api-key','config-forward-model-key'])element('harness-http-fields').append(element(id));
  let saved=structuredClone(config);
  const api=async(path,body)=>{
    calls.push({path,body:body===undefined?undefined:structuredClone(body)});
    if(path==='model-capabilities')return structuredClone(capability);
    assert.equal(path,'config');
    if(body){saved={...body,
      has_api_key:!!body.api_key||(body.provider===saved.provider&&body.base_url===saved.base_url&&!!saved.has_api_key),
      has_harness_api_key:body.harness==='http'&&(!!body.harness_api_key||(saved.harness==='http'&&body.harness_url===saved.harness_url&&!!saved.has_harness_api_key))};
      delete saved.api_key;delete saved.harness_api_key;}
    return structuredClone(saved);
  };
  const scope=vm.createContext({console,URL,harnessNames,providerNames,effectiveHarness,harnessCompatibility,harnessReasoningError,
    agentApi:api,toast:value=>toasts.push(value),setTimeout:()=>1,clearTimeout(){},
    document:{querySelector:selector=>element(selector.slice(1)),createElement:tag=>new Element(tag)}});
  vm.runInContext('const $=s=>document.querySelector(s);let agentConfig=null;',scope);
  const source=readFileSync(new URL('../src/main.js',import.meta.url),'utf8');
  vm.runInContext(source.slice(source.indexOf('function updateConfigSummary(){'),source.indexOf("agentApi('config').then")),scope);
  const trigger=async(id,event='change')=>{await element(id).listeners[event]({preventDefault(){}});await new Promise(resolve=>setImmediate(resolve));};
  return{element,calls,toasts,trigger,open:()=>trigger('agent-settings','click'),submit:()=>trigger('agent-config-form','submit')};
}

test('five harness choices are separate from model protocols and switching preserves model connection',async()=>{
  const ui=setup(defaults);await ui.open();
  assert.deepEqual(ui.element('config-harness').options.map(node=>node.value),['workbench','codex_cli','claude_code','dsh','http']);
  assert.equal(ui.element('config-harness').value,'workbench');
  ui.element('config-api-key').value='model-fixture-key';
  for(const harness of ['codex_cli','claude_code','dsh','http','workbench']){
    ui.element('config-harness').value=harness;await ui.trigger('config-harness');
    assert.equal(ui.element('config-model').value,'routing-model');
    assert.equal(ui.element('config-base-url').value,'https://router.example/v1');
    assert.equal(ui.element('config-api-key').value,'model-fixture-key');
    assert.equal(ui.element('harness-http-fields').hidden,harness!=='http');
    assert.equal(ui.element('config-harness-api-key').disabled,harness!=='http');
  }
});

test('Claude Code and dsh save and reopen independent model connection and token settings',async()=>{
  for(const [harness,provider] of [['claude_code','anthropic'],['dsh','openai'],['dsh','compatible'],['dsh','anthropic']]){
    const ui=setup({...defaults,harness,provider,model:'route/robot-model',max_output_tokens:32768});await ui.open();
    assert.equal(ui.element('apply-agent-config').disabled,false,`${harness}/${provider}`);
    assert.equal(ui.element('api-config-fields').hidden,false);
    assert.equal(ui.element('config-output-tokens').disabled,false);
    ui.element('config-api-key').value='new-model-key';
    await ui.submit();
    const sent=ui.calls.find(call=>call.path==='config'&&call.body).body;
    assert.equal(sent.harness,harness);assert.equal(sent.provider,provider);
    assert.equal(sent.model,'route/robot-model');assert.equal(sent.base_url,defaults.base_url);
    assert.equal(sent.api_key,'new-model-key');assert.equal(sent.max_output_tokens,32768);
    assert.equal(sent.forward_model_key,false);assert.equal(sent.harness_api_key,'');
    assert.equal(ui.element('agent-config-dialog').open,false);
    assert.match(ui.element('agent-harness-name').textContent,harness==='dsh'?/DeepSeek Harness/:/Claude Code/);
    await ui.open();
    assert.equal(ui.element('config-harness').value,harness);assert.equal(ui.element('config-provider').value,provider);
    assert.equal(ui.element('config-model').value,'route/robot-model');assert.equal(ui.element('config-base-url').value,defaults.base_url);
    assert.equal(ui.element('config-api-key').value,'');assert.match(ui.element('config-api-key').placeholder,/已有密钥/);
  }
});

test('new CLI harnesses block mismatched protocols without changing entered model fields',async()=>{
  for(const [harness,provider] of [['claude_code','openai'],['claude_code','compatible'],['claude_code','codex'],['dsh','codex'],['dsh','custom']]){
    const ui=setup({...defaults,harness:'workbench',provider});await ui.open();
    ui.element('config-harness').value=harness;await ui.trigger('config-harness');
    assert.equal(ui.element('apply-agent-config').disabled,true,`${harness}/${provider}`);
    assert.equal(ui.element('harness-compatibility').hidden,false);
    await ui.submit();
    assert.equal(ui.calls.filter(call=>call.path==='config'&&call.body).length,0);
    assert.equal(ui.element('config-provider').value,provider);
    assert.equal(ui.element('config-model').value,defaults.model);
    assert.equal(ui.element('config-base-url').value,defaults.base_url);
    assert.equal(ui.element('agent-config-dialog').open,true);
  }
});

test('CLI unsupported effort blocks saving and service default restores a valid form',async()=>{
  for(const [harness,provider,effort] of [['claude_code','anthropic','minimal'],['dsh','openai','ultra']]){
    const ui=setup({...defaults,harness,provider,reasoning:effort},{levels:['high',effort],source:'service_metadata'});await ui.open();
    assert.equal(ui.element('config-reasoning').value,effort);
    assert.equal(ui.element('apply-agent-config').disabled,true);
    assert.match(ui.element('harness-compatibility').textContent,/推理强度/);
    await ui.submit();assert.equal(ui.calls.filter(call=>call.path==='config'&&call.body).length,0);
    ui.element('config-reasoning').value='';await ui.trigger('config-reasoning');
    assert.equal(ui.element('apply-agent-config').disabled,false);
    await ui.submit();assert.equal(ui.calls.find(call=>call.path==='config'&&call.body).body.reasoning,null);
  }
});

test('disabled DeepSeek thinking ignores stale effort while preserving it for an explicit re-enable',async()=>{
  const ui=setup({...defaults,harness:'dsh',provider:'compatible',base_url:'https://api.deepseek.com',
    deepseek_thinking:'disabled',reasoning:'ultra'},{levels:['high','ultra'],source:'service_metadata'});await ui.open();
  assert.equal(ui.element('config-reasoning').disabled,true);
  assert.equal(ui.element('apply-agent-config').disabled,false);
  ui.element('config-deepseek-thinking').value='enabled';await ui.trigger('config-deepseek-thinking');
  assert.equal(ui.element('config-reasoning').disabled,false);
  assert.equal(ui.element('apply-agent-config').disabled,true);
  await ui.submit();assert.equal(ui.calls.filter(call=>call.path==='config'&&call.body).length,0);
  ui.element('config-deepseek-thinking').value='disabled';await ui.trigger('config-deepseek-thinking');
  await ui.submit();
  const saved=ui.calls.find(call=>call.path==='config'&&call.body).body;
  assert.equal(saved.deepseek_thinking,'disabled');
  assert.equal(ui.element('agent-config-dialog').open,false);
});

test('unsupported harness/model pair shows the reason and never saves or silently changes protocol',async()=>{
  const ui=setup({...defaults,provider:'anthropic',harness:'workbench'});await ui.open();
  ui.element('config-harness').value='codex_cli';await ui.trigger('config-harness');
  assert.equal(ui.element('apply-agent-config').disabled,true);
  assert.equal(ui.element('harness-compatibility').hidden,false);
  assert.match(ui.element('harness-compatibility').textContent,/不直接支持/);
  await ui.submit();
  assert.equal(ui.calls.filter(call=>call.path==='config'&&call.body).length,0);
  assert.equal(ui.element('config-provider').value,'anthropic');
  assert.equal(ui.element('agent-config-dialog').open,true);
});

test('HTTP harness endpoint and credential are independent from the selected model and clear after saving',async()=>{
  const ui=setup({...defaults,harness:'http'});await ui.open();
  ui.element('config-harness-url').value='https://harness.example/agent';
  ui.element('config-harness-api-key').value='harness-fixture-key';
  ui.element('config-api-key').value='model-fixture-key';
  await ui.submit();
  const sent=ui.calls.find(call=>call.path==='config'&&call.body).body;
  assert.equal(sent.harness,'http');assert.equal(sent.provider,'openai');
  assert.equal(sent.harness_url,'https://harness.example/agent');
  assert.equal(sent.harness_api_key,'harness-fixture-key');
  assert.equal(sent.base_url,'https://router.example/v1');assert.equal(sent.api_key,'model-fixture-key');
  assert.equal(sent.forward_model_key,false);
  assert.equal(ui.element('config-api-key').value,'');assert.equal(ui.element('config-harness-api-key').value,'');
  assert.equal(ui.element('agent-config-dialog').open,false);
  assert.match(ui.element('agent-harness-name').textContent,/自定义 HTTP/);
  assert.match(ui.element('agent-provider-name').textContent,/Responses API/);
});

test('model key forwarding is opt-in for one harness destination and URL edits revoke it',async()=>{
  const ui=setup({...defaults,harness:'http',forward_model_key:true});await ui.open();
  assert.equal(ui.element('config-forward-model-key').checked,true);
  ui.element('config-harness-url').value='https://another-harness.example/agent';await ui.trigger('config-harness-url','input');
  assert.equal(ui.element('config-forward-model-key').checked,false);
  assert.match(ui.element('harness-forward-help').textContent,/默认只发送模型协议/);
  ui.element('config-forward-model-key').checked=true;await ui.trigger('config-forward-model-key');
  assert.match(ui.element('harness-forward-help').textContent,/another-harness.example/);
  await ui.submit();assert.equal(ui.calls.find(call=>call.path==='config'&&call.body).body.forward_model_key,true);
});

test('legacy Codex and custom Agent settings render their actual execution paths',async()=>{
  for(const provider of ['codex','custom']){
    const legacy={...defaults,provider,base_url:provider==='custom'?'https://legacy.example/agent':defaults.base_url};
    delete legacy.harness;delete legacy.harness_url;delete legacy.has_harness_api_key;
    const ui=setup(legacy);await ui.open();
    assert.equal(ui.element('config-harness').value,provider==='codex'?'codex_cli':'http');
    assert.equal(ui.element('api-config-fields').hidden,true);
    if(provider==='custom'){
      assert.equal(ui.element('config-harness-url').value,'https://legacy.example/agent');
      assert.match(ui.element('config-harness-api-key').placeholder,/当前 harness 已有密钥/);
    }
  }
});

test('capability lookup uses model credentials only, never the external harness credential',async()=>{
  const ui=setup({...defaults,harness:'http'});await ui.open();
  ui.element('config-api-key').value='model-fixture-key';ui.element('config-harness-api-key').value='harness-fixture-key';
  await ui.trigger('refresh-model-capabilities','click');
  const lookup=ui.calls.filter(call=>call.path==='model-capabilities').at(-1).body;
  assert.deepEqual(lookup,{provider:'openai',model:'routing-model',base_url:'https://router.example/v1',api_key:'model-fixture-key'});
  assert.doesNotMatch(JSON.stringify(lookup),/harness-fixture-key|harness_url|harness_api_key/);
});
