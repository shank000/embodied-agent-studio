import './environment.css';

const $=selector=>document.querySelector(selector);
let context={},lastState=null,busy=false,initialized=false,lastCommands='',lastSession='';
let nodes=null;

const discrete=state=>['text','discrete'].includes(state?.backend?.action_mode);
const session=state=>`${state?.backend?.engine_id||''}:${state?.session_id||''}:${state?.generation??''}`;

function lockReason(state){
  if(!state||state.ready!==true)return '环境未连接，请先加载环境。';
  if(state.backend_switching)return '正在切换环境，请等待加载完成。';
  if(['agent','policy','comparison','research','composition','experiment','repair','benchmark'].some(key=>state[key]?.active)
      ||['running','settling','verifying'].includes(state.task?.status))return '自动控制正在执行，请停止后再进行手动交互。';
  if(!discrete(state))return '当前环境使用机械臂控制。';
  return '';
}

function message(text,error=false){
  nodes.status.textContent=text;
  nodes.status.classList.toggle('error',error);
  nodes.status.setAttribute('role',error?'alert':'status');
}

function updateControls(){
  if(!nodes)return;
  const reason=lockReason(lastState),disabled=busy||!!reason;
  nodes.input.disabled=disabled;nodes.submit.disabled=disabled;
  for(const button of nodes.commands.children)button.disabled=disabled;
  nodes.lock.hidden=!reason;nodes.lock.textContent=reason;
  nodes.root.setAttribute('aria-busy',String(busy));
  nodes.submit.textContent=busy?'执行中…':'执行环境动作';
}

async function executeCommand(command,expectedSession=session(lastState)){
  if(busy)return;
  if(expectedSession!==session(lastState)){message('环境已切换，请从最新动作列表重新选择。',true);return;}
  const reason=lockReason(lastState);
  if(reason){message(reason,true);return;}
  if(typeof command!=='string'||!command.trim()){message('请输入动作，或选择当前环境提供的可执行命令。',true);return;}
  if(typeof context.execute!=='function'){message('环境交互接口尚未连接。',true);return;}
  const currentSession=session(lastState);
  busy=true;message('正在执行原生环境动作…');updateControls();
  try{
    const response=await context.execute({action:'environment_action',environment_action:command.trim(),generation:lastState.generation});
    if(session(lastState)!==currentSession)return;
    const result=typeof response?.message==='object'&&response.message!==null?response.message:response?.result??response;
    if(response?.ok===false||result?.last_action_success===false||result?.lastActionSuccess===false){
      const reason=result?.error||result?.errorMessage||result?.message||(typeof response?.message==='string'?response.message:'原生环境拒绝了此动作，请查看最新观察后调整。');
      message(`环境执行失败：${String(reason)}`,true);
    }else if(result?.error){
      message(`环境执行返回错误：${String(result.error)}`,true);
    }else{
      message('动作请求已处理；请查看环境最新观察。');
    }
  }catch(error){
    if(session(lastState)===currentSession)message(`环境动作未完成：${error?.message||String(error)}`,true);
  }finally{busy=false;updateControls();}
}

function element(tag,className,text){
  const node=document.createElement(tag);if(className)node.className=className;
  if(text!==undefined)node.textContent=text;return node;
}

export function initEnvironment(callbacks={}){
  context=callbacks;
  if(initialized)return;
  const root=$('#environment-panel');if(!root)return;
  initialized=true;root.classList.add('environment-panel');
  const heading=element('div','environment-heading'),title=element('h3','','环境交互'),mode=element('span','environment-mode');
  heading.append(title,mode);
  const task=element('p','environment-task'),hint=element('p','environment-hint');
  const observation=element('pre','environment-observation');observation.setAttribute('aria-label','当前原生环境观察');
  const label=element('h4','','当前可执行动作'),commands=element('div','environment-commands');
  const empty=element('p','environment-empty','当前环境尚未提供动作列表。');
  const form=element('form','environment-form'),inputLabel=element('label','','手动环境动作');
  const input=element('textarea','environment-input');input.id='environment-command';input.rows=2;
  input.setAttribute('aria-label','手动环境动作');input.spellcheck=false;
  const submit=element('button','primary-button','执行环境动作');submit.type='submit';inputLabel.append(input);form.append(inputLabel,submit);
  const lock=element('p','environment-lock'),status=element('p','environment-status');status.setAttribute('role','status');
  root.replaceChildren(heading,task,hint,observation,label,commands,empty,form,lock,status);
  nodes={root,mode,task,hint,observation,commands,empty,input,submit,lock,status};
  form.addEventListener('submit',event=>{event.preventDefault();return executeCommand(input.value);});
  root.hidden=true;updateControls();
}

export function renderEnvironment(state){
  lastState=state;if(!nodes)return;
  nodes.root.hidden=!discrete(state);
  if(!discrete(state)){lastCommands='';lastSession='';nodes.commands.replaceChildren();return;}
  const nextSession=session(state);
  if(nextSession!==lastSession){lastSession=nextSession;lastCommands='';message('');nodes.input.value='';}
  const environment=state.environment||{};
  nodes.mode.textContent=state.backend.action_mode==='text'?'文字环境 · 逐步动作':'视觉环境 · 离散动作';
  nodes.task.textContent=typeof environment.task_name==='string'?environment.task_name:'等待读取当前任务';
  nodes.hint.textContent=state.backend.action_mode==='text'?'以下是官方文字观察；场景画面为文字排版，没有三路相机或机械臂 TCP。':'以下是原生第一人称观察与动作反馈；请使用导航或物体交互动作。';
  nodes.observation.textContent=typeof environment.observation_text==='string'?environment.observation_text:'等待原生环境观察…';
  const commands=Array.isArray(environment.admissible_commands)?environment.admissible_commands.filter(command=>typeof command==='string'):[];
  const signature=JSON.stringify(commands);
  if(signature!==lastCommands){
    lastCommands=signature;nodes.commands.replaceChildren();
    for(const command of commands){
      const button=element('button','environment-action',command);button.type='button';button.title=command;
      button.addEventListener('click',()=>executeCommand(command,nextSession));nodes.commands.append(button);
    }
  }
  nodes.empty.hidden=commands.length>0;
  updateControls();
}
