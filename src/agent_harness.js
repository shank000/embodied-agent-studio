export const harnessNames={workbench:'工作台视觉闭环',codex_cli:'Codex CLI · 受限单轮',claude_code:'Claude Code · 受限单轮',dsh:'DeepSeek Harness · dsh',http:'自定义 HTTP harness'};
export const providerNames={codex:'Codex 本机登录',openai:'Responses API',compatible:'Chat Completions API',anthropic:'Claude Messages API',custom:'由 harness 服务管理模型'};
export function harnessReasoningError(harness,effort,{provider,baseUrl,thinking}={}){
  if(harness==='dsh'&&provider==='compatible'&&thinking==='disabled'){
    try{if(new URL(baseUrl).hostname==='api.deepseek.com')return '';}catch{}
  }
  const levels={claude_code:['low','medium','high','xhigh','max'],dsh:['off','minimal','low','medium','high','xhigh','max']}[harness];
  return levels&&effort&&!levels.includes(effort)?`${harnessNames[harness]} 当前可传入的推理强度为 ${levels.join(' / ')}，还需符合模型本身的支持范围；也可选择服务默认。`:'';
}
export function effectiveHarness(config={}){return config.harness||(config.provider==='codex'?'codex_cli':config.provider==='custom'?'http':'workbench');}
export function harnessCompatibility(harness,provider){
  if(harness==='workbench'&&!['openai','compatible','anthropic'].includes(provider))return '工作台视觉闭环需要 Responses、Chat Completions 或 Claude Messages 模型接口。';
  if(harness==='codex_cli'&&!['codex','openai'].includes(provider))return 'Codex CLI 支持本机 Codex 登录或 Responses API；不直接支持 Chat Completions 和 Claude Messages。';
  if(harness==='claude_code'&&provider!=='anthropic')return 'Claude Code 当前使用 Messages API 或对应路由服务；bare 模式不复用订阅登录，请单独填写模型 API Key。';
  if(harness==='dsh'&&!['openai','compatible','anthropic'].includes(provider))return 'dsh 通过独立模型 API 调用；支持 Responses、Chat Completions 或 Messages，不能复用 Codex 登录。';
  if(harness==='http'&&provider==='codex')return '自定义 HTTP harness 不能直接复用本机 Codex 登录；请选择模型 API，或由 harness 服务管理模型。';
  return '';
}
