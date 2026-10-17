// worker.js — Vesper v4.1 (seguro)
const ORT_URL = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/ort.min.js';
let ort = null, sessions = {}, tensorMap = {}, cancelled = false;

const CANDIDATES = [
  { repo:'onnxruntime/sdxl-turbo', base:'https://huggingface.co/onnxruntime/sdxl-turbo/resolve/main',
    files:{ te:'text_encoder/model.onnx', unet:'unet/model.onnx', dec:'vae_decoder/model.onnx', enc:'vae_encoder/model.onnx' } },
  { repo:'tlwu/sdxl-turbo-onnxruntime', base:'https://huggingface.co/tlwu/sdxl-turbo-onnxruntime/resolve/main',
    files:{ te:'text_encoder/model.onnx', unet:'unet/model.onnx', dec:'vae_decoder/model.onnx', enc:'vae_encoder/model.onnx' } }
];

self.onmessage = async (e) => {
  const { type, payload } = e.data;
  try {
    if(type==='init') await init(payload);
    else if(type==='generate') await generate(payload);
    else if(type==='encodeImg') await encodeImage(payload);
    else if(type==='cancel'){ cancelled=true; post({type:'cancelled'}); }
  } catch(err){ post({ type:'error', message: String(err.message).slice(0,200) }); }
};
function post(d){ self.postMessage(d); }

async function fetchT(url, ms=180000){
  const ctrl=new AbortController(); const t=setTimeout(()=>ctrl.abort(),ms);
  try{ const r=await fetch(url,{signal:ctrl.signal,cache:'force-cache'});
    if(!r.ok) throw new Error('HTTP '+r.status); return await r.arrayBuffer();
  } finally{ clearTimeout(t); }
}
async function sha256(buf){
  const h=await crypto.subtle.digest('SHA-256',buf);
  return [...new Uint8Array(h)].map(b=>b.toString(16).padStart(2,'0')).join('');
}

async function init({ useWebGPU }){
  cancelled=false;
  post({type:'status',msg:'carregando runtime…',p:0.02});
  const buf=await fetchT(ORT_URL,60000);
  const blobUrl=URL.createObjectURL(new Blob([buf],{type:'application/javascript'}));
  importScripts(blobUrl); URL.revokeObjectURL(blobUrl);
  ort=self.ort;
  const eps=useWebGPU?['webgpu','wasm']:['wasm'];
  ort.env.wasm.numThreads=navigator.hardwareConcurrency||4;
  ort.env.wasm.wasmSimd=true; ort.env.wasm.wasmThreads=true;

  for(const c of CANDIDATES){
    try{
      sessions={};
      for(const [key,path] of Object.entries(c.files)){
        post({type:'status',msg:'baixando '+key+'…',p:0.1+Object.keys(sessions).length*0.15});
        const mbuf=await fetchT(c.base+'/'+path);
        sessions[key]=await ort.InferenceSession.create(mbuf,{executionProviders:eps,wasm:{wasmSimd:true,wasmThreads:true}});
      }
      tensorMap={
        te_in:sessions.te.inputNames[0], te_out:sessions.te.outputNames[0],
        unet_in:Object.fromEntries(sessions.unet.inputNames.map(n=>[n,n])), unet_out:sessions.unet.outputNames[0],
        dec_in:sessions.dec.inputNames[0], dec_out:sessions.dec.outputNames[0],
        enc_in:sessions.enc.inputNames[0], enc_out:sessions.enc.outputNames[0]
      };
      post({type:'ready',engine:eps[0]}); return;
    }catch(err){ console.warn('candidate failed:',c.repo,err.message); }
  }
  post({type:'fallback',message:'nenhum modelo disponível — motor procedural'});
}

async function runTE(tokens){
  const ids=new Int64Array(tokens.length);
  for(let i=0;i<tokens.length;i++) ids[i]=tokens[i];
  const r=await sessions.te.run({[tensorMap.te_in]:new ort.Tensor('int64',[1,tokens.length],ids.buffer)});
  return r[tensorMap.te_out];
}

async function generate({ promptTokens, negTokens, W, H, seed, latent }){
  cancelled=false;
  const [lh,lw]=[H>>3,W>>3], N=4*lh*lw;
  let lat=latent?new Float32Array(latent):new Float32Array(N);
  if(!latent){ const r=mulberry(seed); for(let i=0;i<N;i++) lat[i]=(r()-.5)*.6; }

  post({type:'status',msg:'codificando…',p:0.2});
  const cond=await runTE(promptTokens);
  const timeIds=new Float32Array([W,H,W,H,0,0]);
  const inputs={
    sample:new ort.Tensor('float32',[1,4,lh,lw],lat),
    timestep:new ort.Tensor('int64',[1],new Int32Array([0])),
    encoder_hidden_states:cond
  };
  for(const n of ['added_cond_kwargs.text_embeds','time_ids','added_time_ids'])
    if(tensorMap.unet_in[n]) inputs[n]=new ort.Tensor('float32',[1,6],timeIds);

  post({type:'status',msg:'difusão…',p:0.45});
  const out=await sessions.unet.run(inputs);
  lat=out[tensorMap.unet_out].data;

  post({type:'status',msg:'decodificando…',p:0.75});
  const dOut=await sessions.dec.run({[tensorMap.dec_in]:new ort.Tensor('float32',[1,4,lh,lw],lat)});
  const img=dOut[tensorMap.dec_out].data;

  const px=new Uint8ClampedArray(W*H*4);
  for(let i=0;i<W*H;i++){
    px[i*4]=Math.max(0,Math.min(255,(img[i]*.5+.5)*255));
    px[i*4+1]=Math.max(0,Math.min(255,(img[lh*lw+i]*.5+.5)*255));
    px[i*4+2]=Math.max(0,Math.min(255,(img[2*lh*lw+i]*.5+.5)*255));
    px[i*4+3]=255;
  }
  post({type:'image',width:W,height:H,pixels:px.buffer,transfer:[px.buffer]});
}

async function encodeImage({ imageData, W, H, strength, seed }){
  const [lh,lw]=[H>>3,W>>3];
  const chw=new Float32Array(3*lh*lw), d=imageData;
  for(let y=0;y<lh;y++)for(let x=0;x<lw;x++){
    const i=(y*lw+x)*4;
    chw[y*lw+x]=d[i]/255*2-1; chw[lh*lw+y*lw+x]=d[i+1]/255*2-1; chw[2*lh*lw+y*lw+x]=d[i+2]/255*2-1;
  }
  const res=await sessions.enc.run({[tensorMap.enc_in]:new ort.Tensor('float32',[1,3,lh,lw],chw)});
  const clean=res[tensorMap.enc_out].data, r=mulberry(seed);
  const noise=new Float32Array(clean.length);
  for(let i=0;i<noise.length;i++) noise[i]=(r()-.5)*2;
  const mixed=new Float32Array(clean.length);
  for(let i=0;i<mixed.length;i++) mixed[i]=clean[i]*(1-strength)+noise[i]*strength*1.2;
  post({type:'latent',data:mixed.buffer,transfer:[mixed.buffer]});
}

function mulberry(a){return()=>{a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}
