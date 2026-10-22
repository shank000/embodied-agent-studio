import './layout.css';

const STORAGE_KEY='embodied-studio.layout.v1';
const children=group=>[...group.children].filter(node=>node.hasAttribute('data-layout-item'));
const clamp=(value,min,max)=>Math.max(min,Math.min(max,value));

export function initLayout({toast}={}){
  const toggle=document.querySelector('#layout-toggle'),reset=document.querySelector('#layout-reset'),message=document.querySelector('#layout-status');
  if(!toggle||toggle.dataset.ready)return;toggle.dataset.ready='true';
  const groups=new Map(),items=new Map();let editing=false,dragging=null,resizing=null,settings={orders:{},widths:{},heights:{}};
  const tell=text=>{message.textContent=text;};
  for(const group of document.querySelectorAll('[data-layout-group]')){
    const nodes=children(group);if(!nodes.length)continue;
    const anchor=document.createComment('layout end');nodes.at(-1).after(anchor);
    groups.set(group.dataset.layoutGroup,{group,ids:nodes.map(node=>node.dataset.layoutItem),anchor});
    for(const node of nodes)items.set(node.dataset.layoutItem,{node,group});
  }
  try{
    const stored=JSON.parse(localStorage.getItem(STORAGE_KEY)||'null');
    if(stored){if(stored.version!==1||['orders','widths','heights'].some(key=>!stored[key]||typeof stored[key]!=='object'||Array.isArray(stored[key])))throw new Error('配置格式无效');settings={orders:stored.orders,widths:stored.widths,heights:stored.heights};}
  }catch(error){tell('保存的布局无法读取，正在使用默认布局；可用“恢复默认”清除配置。');}
  function save(){
    try{localStorage.setItem(STORAGE_KEY,JSON.stringify({version:1,...settings}));tell('布局已保存在当前浏览器。拖拽只调整同组板块；窄屏也可使用前移、后移与尺寸按钮。');}
    catch(error){tell(`布局已应用，但无法在此浏览器保存：${error.message}`);}
  }
  function weights(group){return Object.fromEntries(children(group).map(node=>{const value=settings.widths[node.dataset.layoutItem];return [node.dataset.layoutItem,typeof value==='number'&&Number.isFinite(value)&&value>0&&value<=10000?value:Number(node.dataset.layoutWeight)||1];}));}
  function applyColumns(group){
    if(!group.hasAttribute('data-layout-columns'))return;
    const nodes=children(group),custom=nodes.some(node=>Object.hasOwn(settings.widths,node.dataset.layoutItem));group.classList.toggle('layout-custom-columns',custom);
    if(custom){const values=weights(group);group.style.setProperty('--layout-columns',nodes.map(node=>`minmax(${Number(node.dataset.layoutMinWidth)||220}px,${values[node.dataset.layoutItem]}fr)`).join(' '));}
    else group.style.removeProperty('--layout-columns');
  }
  function applyHeight(node){
    const height=settings.heights[node.dataset.layoutItem],valid=typeof height==='number'&&Number.isFinite(height)&&height>=Number(node.dataset.layoutMinHeight||150)&&height<=2400;
    node.classList.toggle('layout-fixed-height',valid);if(valid)node.style.setProperty('--layout-height',`${height}px`);else node.style.removeProperty('--layout-height');
  }
  function updateArrows(group){const nodes=children(group);nodes.forEach((node,index)=>{node.querySelector(':scope > .layout-tools [data-layout-previous]').disabled=index===0;node.querySelector(':scope > .layout-tools [data-layout-next]').disabled=index===nodes.length-1;});}
  function rememberOrder(group){
    settings.orders[group.dataset.layoutGroup]=children(group).map(node=>node.dataset.layoutItem);group.classList.add('layout-arranged');
    if(group.hasAttribute('data-layout-columns'))Object.assign(settings.widths,weights(group));applyColumns(group);updateArrows(group);save();
  }
  function move(node,step){const group=node.parentElement,list=children(group),index=list.indexOf(node),other=list[index+step];if(!other)return;if(step<0)other.before(node);else other.after(node);rememberOrder(group);node.querySelector(':scope > .layout-tools .layout-drag-handle').focus();}
  function clearDrop(){document.querySelectorAll('.layout-drop-before,.layout-drop-after').forEach(node=>node.classList.remove('layout-drop-before','layout-drop-after'));}
  function dropTarget(event,group){let target=event.target.closest('[data-layout-item]');while(target&&target.parentElement!==group)target=target.parentElement.closest('[data-layout-item]');return target;}
  function setWidth(node,percent){
    const group=node.parentElement,values=weights(group),id=node.dataset.layoutItem,others=Object.keys(values).filter(key=>key!==id),remaining=others.reduce((total,key)=>total+values[key],0),share=clamp(percent,10,80);
    settings.widths[id]=share;for(const key of others)settings.widths[key]=values[key]/remaining*(100-share);applyColumns(group);
  }
  const dialog=document.createElement('dialog');dialog.id='layout-size-dialog';dialog.setAttribute('aria-labelledby','layout-size-title');
  dialog.innerHTML='<h2 id="layout-size-title">调整板块尺寸</h2><form class="config-form"><p id="layout-size-name" class="config-help"></p><label id="layout-width-field">同组列宽占比 <output id="layout-width-value"></output><input id="layout-width-input" type="range" min="10" max="80" step="1"></label><label>板块高度 / px<input id="layout-height-input" type="number" max="2400" step="1" placeholder="留空使用内容自然高度"></label><p class="config-help">留空高度可恢复自动排版。列宽在宽屏生效；窄屏自动纵向排列。</p><div class="layout-dialog-actions"><button id="layout-size-cancel" type="button" class="secondary">取消</button><button type="submit" class="primary-button">应用尺寸</button></div></form>';
  document.body.append(dialog);let sizedItem=null;
  const widthInput=dialog.querySelector('#layout-width-input'),heightInput=dialog.querySelector('#layout-height-input');
  widthInput.addEventListener('input',()=>{dialog.querySelector('#layout-width-value').textContent=`${widthInput.value}%`;});
  function openSize(node){
    sizedItem=node;const group=node.parentElement,values=weights(group),total=Object.values(values).reduce((sum,value)=>sum+value,0);
    dialog.querySelector('#layout-size-name').textContent=node.dataset.layoutName;dialog.querySelector('#layout-width-field').hidden=!group.hasAttribute('data-layout-columns');
    widthInput.value=Math.round(values[node.dataset.layoutItem]/total*100);dialog.querySelector('#layout-width-value').textContent=`${widthInput.value}%`;heightInput.min=node.dataset.layoutMinHeight||150;heightInput.value=settings.heights[node.dataset.layoutItem]||'';dialog.showModal();
  }
  dialog.querySelector('#layout-size-cancel').addEventListener('click',()=>dialog.close());
  dialog.querySelector('form').addEventListener('submit',event=>{event.preventDefault();if(!sizedItem)return;if(sizedItem.parentElement.hasAttribute('data-layout-columns'))setWidth(sizedItem,Number(widthInput.value));if(heightInput.value)settings.heights[sizedItem.dataset.layoutItem]=Number(heightInput.value);else delete settings.heights[sizedItem.dataset.layoutItem];applyHeight(sizedItem);save();dialog.close();});
  for(const [id,{node,group}] of items){
    const bar=document.createElement('div');bar.className='layout-tools';bar.hidden=true;
    const handle=document.createElement('button');handle.type='button';handle.className='layout-drag-handle';handle.textContent=`⠿ ${node.dataset.layoutName||id}`;handle.title='拖动以调整同组板块顺序';handle.draggable=true;
    const previous=document.createElement('button');previous.type='button';previous.dataset.layoutPrevious='';previous.textContent='前移';previous.setAttribute('aria-label',`${node.dataset.layoutName}前移`);previous.addEventListener('click',()=>move(node,-1));
    const next=document.createElement('button');next.type='button';next.dataset.layoutNext='';next.textContent='后移';next.setAttribute('aria-label',`${node.dataset.layoutName}后移`);next.addEventListener('click',()=>move(node,1));
    const size=document.createElement('button');size.type='button';size.textContent='尺寸';size.setAttribute('aria-label',`调整${node.dataset.layoutName}尺寸`);size.addEventListener('click',()=>openSize(node));bar.append(handle,previous,next,size);node.prepend(bar);
    const grip=document.createElement('button');grip.type='button';grip.className='layout-resize-handle';grip.hidden=true;grip.title='拖动调整尺寸，也可点击打开尺寸设置';grip.setAttribute('aria-label',`拖动调整${node.dataset.layoutName}尺寸`);grip.textContent='◢';node.append(grip);
    handle.addEventListener('dragstart',event=>{if(!editing){event.preventDefault();return;}event.stopPropagation();dragging={node,group};event.dataTransfer.setData('text/plain',id);event.dataTransfer.effectAllowed='move';node.classList.add('layout-dragging');});
    handle.addEventListener('dragend',()=>{node.classList.remove('layout-dragging');dragging=null;clearDrop();});
    grip.addEventListener('pointerdown',event=>{if(!editing||event.button!==0)return;event.preventDefault();event.stopPropagation();const rect=node.getBoundingClientRect();resizing={node,grip,pointerId:event.pointerId,x:event.clientX,y:event.clientY,width:rect.width,height:rect.height,moved:false};grip.setPointerCapture(event.pointerId);});
    grip.addEventListener('pointermove',event=>{if(resizing?.node!==node||resizing.pointerId!==event.pointerId)return;const dx=event.clientX-resizing.x,dy=event.clientY-resizing.y;if(Math.abs(dx)+Math.abs(dy)<4)return;resizing.moved=true;settings.heights[id]=Math.round(clamp(resizing.height+dy,Number(node.dataset.layoutMinHeight)||150,2400));applyHeight(node);if(group.hasAttribute('data-layout-columns')&&window.innerWidth>1100){const available=children(group).reduce((sum,item)=>sum+item.getBoundingClientRect().width,0);setWidth(node,(resizing.width+dx)/available*100);}});
    const finishResize=event=>{if(resizing?.node!==node||resizing.pointerId!==event.pointerId)return;const moved=resizing.moved;resizing=null;if(grip.hasPointerCapture(event.pointerId))grip.releasePointerCapture(event.pointerId);if(moved)save();else if(event.type==='pointerup')openSize(node);};
    grip.addEventListener('pointerup',finishResize);grip.addEventListener('pointercancel',finishResize);grip.addEventListener('keydown',event=>{if(['Enter',' '].includes(event.key)){event.preventDefault();openSize(node);}});
    applyHeight(node);
  }
  for(const [id,{group,ids,anchor}] of groups){
    const saved=settings.orders[id];if(Array.isArray(saved)){const order=[...new Set([...saved.filter(key=>ids.includes(key)),...ids])];for(const key of order)group.insertBefore(items.get(key).node,anchor);group.classList.add('layout-arranged');}
    applyColumns(group);updateArrows(group);
    group.addEventListener('dragover',event=>{if(!editing||dragging?.group!==group)return;const target=dropTarget(event,group);if(!target||target===dragging.node)return;event.preventDefault();event.stopPropagation();const rect=target.getBoundingClientRect(),horizontal=group.hasAttribute('data-layout-columns')&&window.innerWidth>1100,after=horizontal?event.clientX>rect.left+rect.width/2:event.clientY>rect.top+rect.height/2;clearDrop();target.classList.add(after?'layout-drop-after':'layout-drop-before');event.dataTransfer.dropEffect='move';});
    group.addEventListener('drop',event=>{if(dragging?.group!==group)return;const target=dropTarget(event,group);if(!target||target===dragging.node)return;event.preventDefault();event.stopPropagation();if(target.classList.contains('layout-drop-after'))target.after(dragging.node);else target.before(dragging.node);dragging.node.classList.remove('layout-dragging');dragging=null;clearDrop();rememberOrder(group);});
  }
  function setEditing(value){
    if(value)for(const {node} of items.values())node.style.setProperty('--layout-base-padding-top',getComputedStyle(node).paddingTop);
    editing=value;document.body.classList.toggle('layout-editing',editing);toggle.textContent=editing?'完成布局':'编辑布局';toggle.setAttribute('aria-pressed',String(editing));reset.hidden=!editing;
    document.querySelectorAll('.layout-tools,.layout-resize-handle').forEach(element=>element.hidden=!editing);tell(editing?'拖动标题可在同组换序，拖动右下角调整尺寸；触屏可用前移、后移和尺寸按钮。':'布局保存在当前浏览器；业务控件可正常操作。');
  }
  toggle.addEventListener('click',()=>setEditing(!editing));
  reset.addEventListener('click',()=>{
    settings={orders:{},widths:{},heights:{}};for(const {group,ids,anchor} of groups.values()){for(const id of ids)group.insertBefore(items.get(id).node,anchor);group.classList.remove('layout-arranged');applyColumns(group);updateArrows(group);}for(const {node} of items.values())applyHeight(node);
    try{localStorage.removeItem(STORAGE_KEY);tell('已恢复默认顺序与尺寸。');}catch(error){tell(`本页已恢复默认，保存的配置未清除：${error.message}`);}toast?.('已恢复默认布局。');
  });
}
