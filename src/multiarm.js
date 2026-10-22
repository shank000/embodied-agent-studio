const $=selector=>document.querySelector(selector);
let context,lastRobots='';
export function initMultiarm(callbacks){
  context=callbacks;
  for(const [id,name,count] of [['dual_panda','双 Panda',2],['triple_panda','三 Panda',3]]){
    const button=document.createElement('button');button.className='model-card';button.dataset.model=id;button.setAttribute('aria-pressed','false');
    button.innerHTML=`<img src="/api/models/${id}/image" alt="${name}模型"><span><strong>${name}</strong><small>${count} 台真实机械臂 · 共享物理工作区</small></span><b class="multiarm-count">×${count}</b>`;
    button.addEventListener('click',()=>context.operate({action:'load',model:id}));$('#model-cards').append(button);
    $('#rsi-protocol-robot').append(new Option(name,id));
  }
  const label=document.createElement('label');label.className='active-robot-field';label.innerHTML='当前控制臂<select id="active-robot" aria-label="当前控制机械臂"></select><small>其他臂保持各自目标并继续参与物理仿真。Agent 可用 robot_id 在轮间选择控制臂。</small>';
  $('.agent-heading').after(label);label.hidden=true;
  $('#active-robot').addEventListener('change',()=>context.operate({action:'select_robot',robot_id:$('#active-robot').value}));
}
export function renderMultiarm(state,{disabled=false}={}){
  const select=$('#active-robot');if(!select)return;const robots=state.robots||[];
  select.parentElement.hidden=robots.length<2;
  const signature=JSON.stringify(robots.map(robot=>[robot.id,robot.name]));if(signature!==lastRobots){lastRobots=signature;select.replaceChildren(...robots.map(robot=>new Option(`${robot.name} · ${robot.id}`,robot.id)));}
  select.value=state.active_robot||'arm1';
  select.disabled=disabled||!!(state.agent?.active||state.policy?.active||state.comparison?.active||state.research?.active||state.experiment?.active);
}
