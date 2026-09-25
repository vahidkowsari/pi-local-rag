/** Offline synthetic follow-up probe. No real network or model inference. Run from any cwd with Node 22 --experimental-strip-types. Leaves temporary evidence stores. */
import { mkdtempSync,writeFileSync,existsSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
const root=fileURLToPath(new URL('../../',import.meta.url)).replace(/\/$/,'');
for(const k of Object.keys(process.env)) if(k.startsWith('PI_RAG_')||k==='VOYAGE_API_KEY') delete process.env[k];
process.env.PI_RAG_DIR=mkdtempSync('/private/tmp/pi-rag-followup-');
process.env.PI_RAG_LEGACY_DIR=join(process.env.PI_RAG_DIR,'legacy');
globalThis.fetch=async()=>{throw Error('network disabled');};
const imp=f=>import(`${root}/${f}.ts`);
const chunk=await imp('chunking');
const cfg=await imp('config');
const dbm=await imp('db');
const idx=await imp('indexing');
const repo=await imp('repository');
const local=await imp('providers/embedding/local');
const ext=await imp('index');
const vec=()=>[1,...Array(383).fill(0)];
local.LocalEmbeddingProvider.prototype.embedDocuments=async texts=>texts.map(vec);
local.LocalEmbeddingProvider.prototype.embedQuery=async()=>vec();
const log=(name,data)=>console.log(JSON.stringify({name,...data}));
const same='Important repeated evidence';
const cc=chunk.chunkBlocks([1,2].map(p=>({text:same,section:null,pageStart:p,pageEnd:p})));
log('same_text_distinct_pages',{chunks:cc});
const dir=process.env.PI_RAG_DIR;
const f=join(dir,'file.md');writeFileSync(f,'original evidence');
await idx.indexFiles([f],{});
const ac=new AbortController();
writeFileSync(f,'replacement evidence');
local.LocalEmbeddingProvider.prototype.embedDocuments=async texts=>{ac.abort();return texts.map(vec);};
let result;try{result=await idx.rebuildWithSwitch([f],{},true,[],ac.signal);}catch(e){result={error:e.name};}
log('abort_during_last_embed',{aborted:ac.signal.aborted,result,published:existsSync(join(dir,'active.json')),contents:dbm.getDbConn().prepare('select chunk_content from chunks').all()});
local.LocalEmbeddingProvider.prototype.embedDocuments=async texts=>texts.map(vec);
writeFileSync(join(dir,'config.json'),'{BROKEN');
const toolmap={};ext.default({on(){},registerCommand(){},registerTool(t){toolmap[t.name]=t;}});
const bad=cfg.loadConfigDetailed();
const q=await toolmap.rag_query.execute('probe',{query:'replacement'});
log('broken_config_query',{issues:bad.issues,output:q});
const added=join(dir,'new.md');writeFileSync(added,'added evidence');
await toolmap.rag_index.execute('probe',{path:added});
log('broken_config_index',{afterStatus:cfg.loadConfigDetailed().fileStatus,trackedPaths:cfg.loadConfig().trackedPaths});
dbm.closeDbConn();

{
const {chunkBlocks,estimateTokens}=await import(root+'/chunking.ts');
const text='a'.repeat(400)+'\n\n'+'b'.repeat(960);
console.log(JSON.stringify({name:'overlap_overflow',max:240,chunks:chunkBlocks([{text,section:null,pageStart:null,pageEnd:null,lineStart:1,lineEnd:3}]).map(c=>({tokens:estimateTokens(c.content),length:c.content.length,lines:[c.lineStart,c.lineEnd]}))}));

}

{
const {defaultConfig}=await import(root+'/config.ts');
const {createEmbeddingProvider}=await import(root+'/providers/embedding/factory.ts');
process.env.VOYAGE_API_KEY='synthetic-offline-key';
const cfg=defaultConfig();cfg.embedding={provider:'voyage',model:'voyage-4-lite',dimensions:1024};cfg.http={timeoutMs:30000,maxRetries:3};
const a=createEmbeddingProvider(cfg);
const b=createEmbeddingProvider({...cfg,http:{timeoutMs:1000,maxRetries:0}});
console.log(JSON.stringify({name:'http_config_cache',sameInstance:a===b,requested:{timeoutMs:1000,maxRetries:0},actual:{timeoutMs:b.timeoutMs,maxRetries:b.maxRetries}}));

}
