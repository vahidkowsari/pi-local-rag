// Synthetic model/HTTP evaluation ONLY, not quality metrics. Run from repository root. Restores the preexisting daily eval report.
import {readFileSync,writeFileSync,existsSync,rmSync} from 'node:fs';
const root=process.cwd();
for(const k of Object.keys(process.env)) if(k.startsWith('PI_RAG_')||['VOYAGE_API_KEY','SKIP_EMBEDDING_TESTS','EVAL_SKIP_CLOUD'].includes(k)) delete process.env[k];
process.env.VOYAGE_API_KEY='synthetic-offline-eval';
process.env.PI_RAG_LEGACY_DIR='/private/tmp/rag-third/no-legacy';
const local=await import(root+'/providers/embedding/local.ts');
const vec=n=>[1,...Array(n-1).fill(0)];
local.LocalEmbeddingProvider.prototype.embedDocuments=async texts=>texts.map(()=>vec(384));
local.LocalEmbeddingProvider.prototype.embedQuery=async()=>vec(384);
let documents=0,queries=0,reranks=0;
globalThis.fetch=async(url,o)=>{
 const b=JSON.parse(o.body);
 if(String(url).endsWith('/rerank')){reranks++;return new Response('synthetic rerank unavailable',{status:400});}
 if(!String(url).endsWith('/embeddings')) throw Error('unexpected request blocked');
 if(b.input_type==='document') documents++; else queries++;
 return new Response(JSON.stringify({data:b.input.map((_,index)=>({index,embedding:vec(b.output_dimension)}))}),{status:200});
};
const reportPath=root+'/eval/runs/eval-'+new Date().toISOString().slice(0,10)+'.json';
const original=existsSync(reportPath)?readFileSync(reportPath):null;
try {
await import(root+'/scripts/eval-retrieval.ts');
const report=JSON.parse(readFileSync(root+'/eval/runs/eval-'+new Date().toISOString().slice(0,10)+'.json','utf8'));
console.log(JSON.stringify({name:'eval_rerank_failure',documents,queries,reranks,runs:report.runs.map(r=>({name:r.name,status:r.status,reason:r.reason,degraded:r.degraded,perQuestionHasDegraded:r.perQuestion?.some(q=>'degraded' in q)}))}));

} finally { if(original) writeFileSync(reportPath,original); else rmSync(reportPath,{force:true}); }
