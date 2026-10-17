// app.js — Vesper v4.1 (seguro, sem módulos ES)
(function(){
"use strict";
const $=id=>document.getElementById(id);
let mode='t2i',sourceImg=null,lastBlob=null,lastKind='image';
let worker=null,ready=false,engine='—',discrete=false;
let toolImg=null,toolState={br:0,ct:0,st:0,tm:0},cropRect=null;
const history=[];
let _prevResultUrl=null;

// ---------- helpers seguros ----------
function safeImg(src){const i=document.createElement('img');i.src=src;i.alt='';i.referrerPolicy='no-referrer';return i;}
function setProgress(p,msg){$('prog').classList.remove('hidden');$('bar').style.width=(p*100).toFixed(1)+'%';if(msg)$('status').textContent=msg;}
function show(el){el.classList.remove('hidden');} function hide(el){el.classList.add('hidden');}
function status(msg){setProgress(0,msg);}

// ---------- worker ----------
worker=new Worker('worker.js');
worker.onmessage=function(e){
  const d=e.data;
  if(d.type==='status') setProgress(d.p,d.msg);
  else if(d.type==='ready'){ready=true;engine=d.engine;$('engine').textContent='motor: '+engine;}
  else if(d.type==='fallback'){ready=false;$('engine').textContent='motor: procedural (offline)';}
  else if(d.type==='image') onGenImage(d);
  else if(d.type==='error'){setProgress(1,'erro: '+d.message);console.error(d.message);}
  else if(d.type==='cancelled') setProgress(0,'cancelado');
  $('go').disabled=false;$('cancel').classList.add('hidden');
};
const useGPU=!!navigator.gpu;
worker.postMessage({type:'init',payload:{useWebGPU:useGPU}});

// ---------- tokenizer CLIP (via transformers.js UMD) ----------
let tokenizerPromise=null;
function getTokenizer(){
  if(tokenizerPromise) return tokenizerPromise;
  tokenizerPromise=(async()=>{
    setProgress(0.02,'carregando tokenizer CLIP…');
    await loadScript('https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/transformers.min.js');
    const pipe=await self.transformers.pipeline('feature-extraction','Xenova/clip-vit-large-patch14',{quantized:true});
    return pipe;
  })();
  return tokenizerPromise;
}
function loadScript(src){return new Promise((res,rej)=>{const s=document.createElement('script');s.src=src;s.onload=res;s.onerror=()=>rej(new Error('falha ao carregar '+src));document.head.appendChild(s);});}
async function tokenize(text){
  const tok=await getTokenizer();
  const r=await tok(text,{padding:true,truncation:true,max_length:77,return_tensor:false});
  return r.input_ids[0];
}

// ---------- tradução PT→EN ----------
let translator=null;
async function maybeTranslate(text){
  if(!/[çãõéêíóúáâêô]|de |um |uma |com |para /.test(text.toLowerCase())) return text;
  if(!translator){
    try{ translator=await self.transformers.pipeline('translation','Xenova/opus-mt-pt-en',{quantized:true}); }
    catch(e){ return text; }
  }
  const r=await translator(text);
  return (r[0]&&r[0].translation_text)||text;
}

// ---------- geração ----------
async function generate(){
  const raw=$('prompt').value.trim();
  if(!raw){status('escreva um prompt');return;}
  const [W,H]=$('fmt').value.split('x').map(Number);
  const seed=parseInt($('seed').value,10)||((Math.random()*2**32)>>>0);
  const style=$('style').value;
  const neg=$('neg').value.trim();
  const full=style?raw+', '+style:raw;

  $('go').disabled=true;$('cancel').classList.remove('hidden');
  hide($('acts'));hide($('result'));hide($('video'));hide($('canvas'));show($('prog'));
  try{
    const en=await maybeTranslate(full);
    const tokens=await tokenize(en);
    const negTokens=neg?await tokenize(neg):null;
    let latent=null;
    if(sourceImg&&mode==='i2i'){
      setProgress(0.15,'codificando imagem…');
      const [lh,lw]=[H>>3,W>>3];
      const tmp=document.createElement('canvas');tmp.width=lw;tmp.height=lh;
      tmp.getContext('2d').drawImage(sourceImg,0,0,lw,lh);
      const d=tmp.getContext('2d').getImageData(0,0,lw,lh).data;
      const ch=await new Promise(r=>{const prev=worker.onmessage;worker.onmessage=ev=>{if(ev.data.type==='latent'){worker.onmessage=prev;r(ev.data);}};
        worker.postMessage({type:'encodeImg',payload:{imageData:d.buffer,W,H,strength:parseInt($('strength').value)/100,seed},transfer:[d.buffer]});});
      latent=new Float32Array(ch.data);
    }
    worker.postMessage({type:'generate',payload:{promptTokens:tokens,negTokens,W,H,seed,latent}});
  }catch(e){setProgress(1,'erro: '+e.message);$('go').disabled=false;}
}

function onGenImage(d){
  const c=$('canvas');c.width=d.width;c.height=d.height;
  const ctx=c.getContext('2d');
  const id=ctx.createImageData(d.width,d.height);
  id.data.set(new Uint8ClampedArray(d.pixels));
  ctx.putImageData(id,0,0);
  show($('canvas'));
  if(mode==='t2v'||mode==='i2v'){makeVideo(c,d.width,d.height);lastKind='video';}
  else{
    c.toBlob(b=>{lastBlob=b;lastKind='image';
      if(_prevResultUrl)URL.revokeObjectURL(_prevResultUrl);
      _prevResultUrl=URL.createObjectURL(b);
      show($('result'));$('result').src=_prevResultUrl;hide($('canvas'));
      pushHistory($('prompt').value.trim(),parseInt($('seed').value,10)||0,b);
    },'image/png');
  }
  setProgress(1,'concluído ✓');show($('acts'));
}

// ---------- vídeo ----------
async function makeVideo(baseCanvas,W,H){
  const FRAMES=parseInt($('vdur').value,10)||24,FPS=12;
  const mime=['video/webm;codecs=vp9','video/webm','video/mp4'].find(m=>MediaRecorder.isTypeSupported(m))||'';
  const stream=baseCanvas.captureStream(FPS);
  const rec=new MediaRecorder(stream,{mimeType:mime,videoBitsPerSecond:4000000});
  const chunks=[];rec.ondataavailable=e=>chunks.push(e.data);
  const done=new Promise(r=>rec.onstop=r);rec.start();
  const ctx=baseCanvas.getContext('2d');
  for(let f=0;f<FRAMES;f++){
    setProgress(0.9+0.1*(f/FRAMES),'frame '+(f+1)+'/'+FRAMES);
    const t=f/(FRAMES-1);
    ctx.save();ctx.translate(W/2+Math.sin(t*Math.PI*2)*6,H/2+Math.cos(t*Math.PI*1.3)*4);
    ctx.scale(1+t*.04,1+t*.04);ctx.drawImage(baseCanvas,-W/2,-H/2);ctx.restore();
    ctx.fillStyle='rgba('+((Math.random()*40)|0)+','+((Math.random()*40)|0)+','+((Math.random()*40)|0)+',.04)';
    ctx.fillRect(0,0,W,H);
    await new Promise(r=>setTimeout(r,1000/FPS));
  }
  rec.stop();await done;
  lastBlob=new Blob(chunks,{type:mime.split(';')[0]});
  show($('video'));$('video').src=URL.createObjectURL(lastBlob);
  baseCanvas.width=0;baseCanvas.height=0;
}

// ---------- upload seguro ----------
const MAX_UPLOAD=20*1024*1024;
const VALID_TYPES=['image/jpeg','image/png','image/webp'];
function wireDrop(el,cb){
  el.onclick=()=>fi.click();
  const fi=document.createElement('input');fi.type='file';fi.accept=VALID_TYPES.join(',');fi.autocomplete='off';
  fi.onchange=e=>validate(e.target.files[0]);
  ['dragover','dragenter'].forEach(ev=>el.addEventListener(ev,e=>{e.preventDefault();el.classList.add('over');}));
  ['dragleave','drop'].forEach(ev=>el.addEventListener(ev,e=>{e.preventDefault();el.classList.remove('over');}));
  el.addEventListener('drop',e=>validate(e.dataTransfer.files[0]));
  function validate(f){
    if(!f)return;
    if(!VALID_TYPES.includes(f.type)){status('formato inválido (JPEG, PNG ou WebP)');return;}
    if(f.size>MAX_UPLOAD){status('imagem muito grande (máx. 20 MB)');return;}
    const url=URL.createObjectURL(f);const img=new Image();
    img.onload=()=>cb(img,url);
    img.onerror=()=>{URL.revokeObjectURL(url);status('falha ao carregar imagem');};
    img.src=url;
  }
}
wireDrop($('drop'),(img,url)=>{sourceImg=img;const p=$('prev');p.textContent='';p.appendChild(safeImg(url));});
wireDrop($('tdrop'),(img)=>{toolImg=img;const p=$('tprev');p.textContent='';p.appendChild(safeImg(img.src));applyTools();});

// ---------- ferramentas ----------
['br','ct','st','tm'].forEach(id=>$(id).addEventListener('input',()=>{
  toolState={br:+$('br').value,ct:+$('ct').value,st:+$('st').value,tm:+$('tm').value};applyTools();
}));
document.querySelectorAll('[data-tool]').forEach(b=>b.onclick=()=>{
  if(!toolImg)return;const t=b.dataset.tool;
  if(t==='reset'){$('br').value=$('ct').value=$('st').value=$('tm').value=0;toolState={br:0,ct:0,st:0,tm:0};applyTools();return;}
  if(t==='crop'){applyCrop();return;}
  const c=document.createElement('canvas');
  if(t==='rot90'){c.width=toolImg.height;c.height=toolImg.width;const x=c.getContext('2d');x.translate(c.width/2,c.height/2);x.rotate(Math.PI/2);x.drawImage(toolImg,-toolImg.width/2,-toolImg.height/2);}
  else if(t==='flipH'){c.width=toolImg.width;c.height=toolImg.height;const x=c.getContext('2d');x.translate(c.width,0);x.scale(-1,1);x.drawImage(toolImg,0,0);}
  else if(t==='invert'){c.width=toolImg.width;c.height=toolImg.height;const x=c.getContext('2d');x.drawImage(toolImg,0,0);const id=x.getImageData(0,0,c.width,c.height);for(let i=0;i<id.data.length;i+=4){id.data[i]=255-id.data[i];id.data[i+1]=255-id.data[i+1];id.data[i+2]=255-id.data[i+2];}x.putImageData(id,0,0);}
  const im=new Image();im.onload=()=>{toolImg=im;applyTools();};im.src=c.toDataURL();
});
$('tgo').onclick=()=>{
  if(!toolImg)return;
  const w=parseInt($('rw').value,10)||toolImg.width,h=parseInt($('rh').value,10)||toolImg.height;
  const c=document.createElement('canvas');c.width=w;c.height=h;c.getContext('2d').drawImage(toolImg,0,0,w,h);
  const im=new Image();im.onload=()=>{toolImg=im;applyTools();};im.src=c.toDataURL();
};
function applyTools(){
  if(!toolImg)return;
  const c=document.createElement('canvas');c.width=toolImg.width;c.height=toolImg.height;
  const ctx=c.getContext('2d');
  ctx.filter='brightness('+(100+toolState.br)+'%) contrast('+(100+toolState.ct)+'%) saturate('+(100+toolState.st)+'%)';
  ctx.drawImage(toolImg,0,0);
  if(toolState.tm){ctx.globalCompositeOperation='overlay';ctx.fillStyle='rgba('+(toolState.tm>0?255:0)+','+(toolState.tm>0?140:200)+','+(toolState.tm>0?60:255)+','+(Math.abs(toolState.tm)/200)+')';ctx.fillRect(0,0,c.width,c.height);}
  show($('canvas'));hide($('result'));hide($('video'));
  $('canvas').width=c.width;$('canvas').height=c.height;$('canvas').getContext('2d').drawImage(c,0,0);
  $('canvas').toBlob(b=>lastBlob=b,'image/png');show($('acts'));
}
function applyCrop(){
  if(!toolImg||!cropRect)return;
  const {x,y,w,h}=cropRect;
  const c=document.createElement('canvas');c.width=w;c.height=h;
  c.getContext('2d').drawImage(toolImg,x,y,w,h,0,0,w,h);
  const im=new Image();im.onload=()=>{toolImg=im;cropRect=null;applyTools();};im.src=c.toDataURL();
}
// recorte por arrasto
(function(){
  const cv=$('canvas');let dragging=false,sx=0,sy=0;
  cv.addEventListener('mousedown',e=>{if(mode!=='tools'||!toolImg)return;dragging=true;sx=e.offsetX;sy=e.offsetY;cropRect=null;});
  cv.addEventListener('mousemove',e=>{
    if(!dragging)return;
    const x=Math.min(sx,e.offsetX),y=Math.min(sy,e.offsetY),w=Math.abs(e.offsetX-sx),h=Math.abs(e.offsetY-sy);
    cropRect={x,y,w,h};
    applyTools();
    const ctx=cv.getContext('2d');
    ctx.fillStyle='rgba(0,0,0,.4)';
    ctx.fillRect(0,0,cv.width,x);ctx.fillRect(0,y+h,cv.width,cv.height-y-h);
    ctx.fillRect(0,y,x,h);ctx.fillRect(x+w,y,cv.width-x-w,h);
    ctx.strokeStyle='#c9a86a';ctx.lineWidth=2;ctx.strokeRect(x,y,w,h);
  });
  cv.addEventListener('mouseup',()=>{dragging=false;});
})();

// ---------- histórico ----------
function pushHistory(prompt,seed,blob){
  history.unshift({prompt,seed,url:URL.createObjectURL(blob),ts:Date.now()});
  if(history.length>12)URL.revokeObjectURL(history.pop().url);
  renderHisto();
}
function renderHisto(){
  const h=$('histo');h.textContent='';
  history.forEach(item=>{
    const img=safeImg(item.url);
    img.title=item.prompt.slice(0,80);
    img.onclick=()=>{$('prompt').value=item.prompt;$('seed').value=String(item.seed);};
    h.appendChild(img);
  });
  $('histoWrap').classList.toggle('hidden',!history.length);
}

// ---------- exemplos ----------
const EX=['candlelit bedroom, soft focus, 85mm','golden hour portrait, warm light','rainy window, city lights bokeh','morning mist, forest path, cinematic','neon noir, wet street, reflections','pastel dream, floating petals'];
EX.forEach(p=>{const b=document.createElement('button');b.textContent=p.slice(0,28)+'…';b.onclick=()=>{$('prompt').value=p;};$('examples').appendChild(b);});
$('dice').onclick=()=>{$('prompt').value=EX[(Math.random()*EX.length)|0];$('seed').value='';};

// ---------- modo discreto ----------
$('discreteBtn').onclick=()=>{
  discrete=!discrete;
  document.body.classList.toggle('discrete',discrete);
  document.title=discrete?'Vesper':'Vesper Studio';
  $('discreteBtn').textContent=discrete?'👁‍🗨':'👁';
};

// ---------- atalhos ----------
document.addEventListener('keydown',e=>{
  if(e.ctrlKey&&e.key==='Enter')generate();
  if(e.ctrlKey&&e.key==='s'){e.preventDefault();$('dl').click();}
  const n=parseInt(e.key,10);
  if(n>=1&&n<=5)document.querySelectorAll('.mode')[n-1].click();
});

// ---------- net badge ----------
function updateNet(){$('netBadge').textContent=navigator.onLine?'🌐':'📦 offline';}
window.addEventListener('online',updateNet);window.addEventListener('offline',updateNet);updateNet();

// ---------- modos ----------
document.querySelectorAll('.mode').forEach(b=>b.onclick=()=>{
  document.querySelectorAll('.mode').forEach(x=>x.classList.remove('active'));
  b.classList.add('active');mode=b.dataset.mode;
  const gen=mode!=='tools';
  $('genPanel').classList.toggle('hidden',!gen);
  $('toolsPanel').classList.toggle('hidden',gen);
  $('imgInputWrap').classList.toggle('hidden',!(mode==='i2i'||mode==='i2v'));
  $('strengthWrap').classList.toggle('hidden',mode!=='i2i');
  $('vidRow').classList.toggle('hidden',!(mode==='t2v'||mode==='i2v'));
  document.querySelectorAll('.genOnly').forEach(x=>x.classList.toggle('hidden',!gen));
});
$('strength').addEventListener('input',e=>$('strVal').textContent=e.target.value);

// ---------- ações ----------
$('go').onclick=generate;
$('cancel').onclick=()=>worker.postMessage({type:'cancel'});
$('again').onclick=()=>{$('seed').value='';generate();};
$('sendTools').onclick=()=>{
  const src=$('result').src||$('canvas').toDataURL();
  const im=new Image();im.onload=()=>{toolImg=im;const p=$('tprev');p.textContent='';p.appendChild(safeImg(im.src));
    document.querySelector('[data-mode="tools"]').click();applyTools();};im.src=src;
};
$('dl').onclick=()=>{
  if(!lastBlob)return;
  const a=document.createElement('a');a.href=URL.createObjectURL(lastBlob);
  a.download='vesper-'+Date.now()+'.'+(lastKind==='video'?((MediaRecorder.isTypeSupported('video/webm')?'webm':'mp4')):'png');
  a.click();
};

// ---------- boot ----------
setProgress(0.01,'inicializando…');
})();
