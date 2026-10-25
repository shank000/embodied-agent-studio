// Remember generated instructions so an environment change never overwrites
// a task the researcher has edited by hand.
export function setTaskDraft(input,text){
  input.value=String(text??'');input.dataset.generatedTask=input.value;
}

export function syncTaskDraft(input,text){
  if(!input.value||input.value===input.dataset.generatedTask)setTaskDraft(input,text);
}
