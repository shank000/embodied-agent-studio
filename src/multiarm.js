const $=selector=>document.querySelector(selector);
let context,lastRobots='';
export function updateModelCatalog(models,currentModel){
  const count=$('#model-catalog-count');if(count)count.textContent=`${String(models.length).padStart(2,'0')} MODELS`;
  const root=$('#model-cards');root.replaceChildren();
  for(const item of models){
    const button=document.createElement('button');button.className='model-card';button.dataset.model=item.id;button.setAttribute('aria-pressed',String(item.id===currentModel));button.classList.toggle('selected',item.id===currentModel);
    const img=document.createElement('img');img.src=item.image;img.alt=item.name;
    const label=document.createElement('span'),title=document.createElement('strong'),note=document.createElement('small');title.textContent=item.name;note.textContent=item.arm_dof?`${item.arm_dof} 轴 · ${item.hand_dof} 夹爪/手关节`:'原生环境智能体';label.append(title,note);button.append(img,label);
    if(item.robot_count>1){const count=document.createElement('b');count.className='multiarm-count';count.textContent=`×${item.robot_count}`;button.append(count);}
    button.addEventListener('click',()=>context.operate({action:'load',model:item.id}));root.append(button);
  }
  for(const id of ['#compare-robot','#rsi-protocol-robot']){const select=$(id),previous=select.value;select.replaceChildren(...models.map(item=>new Option(item.name,item.id)));select.value=models.some(item=>item.id===previous)?previous:models[0]?.id;}
}
export function initMultiarm(callbacks){
  context=callbacks;
  const label=document.createElement('label');label.className='active-robot-field';label.innerHTML='当前控制臂<select id="active-robot" aria-label="当前控制机械臂"></select><small>其他臂保持各自目标并继续参与物理仿真。Agent 可用 robot_id 在轮间选择控制臂。</small>';
  $('.agent-heading').after(label);label.hidden=true;
  $('#active-robot').addEventListener('change',()=>context.operate({action:'select_robot',robot_id:$('#active-robot').value}));
}
export function renderMultiarm(state,{disabled=false}={}){
  const select=$('#active-robot');if(!select)return;const robots=state.robots||[];
  select.parentElement.hidden=robots.length<2;
  const signature=JSON.stringify(robots.map(robot=>[robot.id,robot.name]));if(signature!==lastRobots){lastRobots=signature;select.replaceChildren(...robots.map(robot=>new Option(`${robot.name} · ${robot.id}`,robot.id)));}
  select.value=state.active_robot||'arm1';
  select.disabled=disabled||!!(state.agent?.active||state.policy?.active||state.comparison?.active||state.research?.active||state.experiment?.active||state.benchmark?.active);
}
