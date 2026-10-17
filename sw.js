// sw.js — Vesper v4.1
const CACHE='vesper-v4-'+(self.crypto.randomUUID?crypto.randomUUID().slice(0,8):'a1b2c3d4');
const ASSETS=['./','index.html','app.js','styles.css','worker.js','manifest.json'];
const ALLOWED=['https://cdn.jsdelivr.net','https://huggingface.co'];

self.addEventListener('install',e=>{
  e.waitUntil(caches.open(CACHE).then(c=>c.addAll(ASSETS)).then(()=>self.skipWaiting()));
});
self.addEventListener('activate',e=>{
  e.waitUntil(caches.keys().then(keys=>Promise.all(keys.filter(k=>k!==CACHE).map(k=>caches.delete(k)))).then(()=>self.clients.claim()));
});
self.addEventListener('fetch',e=>{
  if(e.request.method!=='GET')return;
  const url=new URL(e.request.url);
  if(url.origin===self.location.origin){
    e.respondWith(caches.match(e.request).then(c=>c||fetch(e.request)));
    return;
  }
  if(ALLOWED.includes(url.origin)){
    e.respondWith(caches.match(e.request).then(c=>c||fetch(e.request).then(res=>{
      if(res.ok){const cl=res.clone();caches.open(CACHE).then(cc=>cc.put(e.request,cl));}
      return res;
    })));
  }
});
