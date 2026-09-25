/** Review diagnostic for d546804. Run with Node 22 --experimental-transform-types.
 * Prints observed behavior; this is not an acceptance suite or a quality metric.
 * Creates only synthetic temporary stores and deliberately retains them for inspection.
 */
import { mkdtempSync, writeFileSync, readFileSync, existsSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { fileURLToPath } from 'node:url';
const root = fileURLToPath(new URL('../../', import.meta.url));
const base = mkdtempSync(join(tmpdir(), 'pi-rag-review-'));
// Synthetic observations only. No real model inference or network requests.
globalThis.fetch = async () => { throw new Error('Unexpected network request in review probe'); };
for (const key of Object.keys(process.env)) {
  if (key.startsWith('PI_RAG_') || key === 'VOYAGE_API_KEY') delete process.env[key];
}
process.env.PI_RAG_LEGACY_DIR = join(base, 'probe-legacy');
process.env.PI_RAG_DIR = join(base, 'probe-initial');
const imp = f => import(`${root}/${f}.ts`);
const [cfg, dbm, repo, mgr, chunks, parsing, indexing, search, retrieval, context, local, ext] = await Promise.all(
  ['config','db','repository','index-manager','chunking','parsing','indexing','search','retrieval','context','providers/embedding/local','index'].map(imp)
);
let queryCalls = 0;
let embedFail = false;
const vec = dim => [1, ...Array(dim - 1).fill(0)];
local.LocalEmbeddingProvider.prototype.embedDocuments = async function(texts) {
  if (embedFail) throw new Error('injected embedding failure');
  return texts.map(() => vec(this.dimensions));
};
local.LocalEmbeddingProvider.prototype.embedQuery = async function() { queryCalls++; return vec(this.dimensions); };
const log = (name, value) => console.log(JSON.stringify({name, ...value}));
function fresh(config = cfg.defaultConfig()) {
  dbm.closeDbConn(); embedFail = false;
  const dir = mkdtempSync(join(base, 'case-'));
  process.env.PI_RAG_DIR = dir;
  writeFileSync(join(dir, 'config.json'), JSON.stringify(config));
  return {dir, db: dbm.getDbConn(), config};
}
function seed(db, config, path = '/synthetic/paper.pdf', content = 'alpha evidence for review', stamp = true) {
  const r = repo.insertChunk(db, {id:path, filePath:path, content, lineStart:1, lineEnd:2, hash:'old', indexedAt:'2020-01-01T00:00:00Z', tokens:8, pageStart:2, pageEnd:2, section:'Method'});
  repo.insertVector(db, Number(r.lastInsertRowid), vec(config.embedding.dimensions));
  repo.upsertFile(db, path, 'old', 1, '2020-01-01T00:00:00Z', content.length, true);
  if (stamp) mgr.stampFingerprints(db, config);
}
function handlers() {
  const tools = {}, commands = {}, hooks = {};
  ext.default({on:(n,f)=>hooks[n]=f, registerTool:t=>tools[t.name]=t, registerCommand:(n,c)=>commands[n]=c});
  const notes = [];
  const ctx = {ui:{notify:(msg,level)=>notes.push({msg,level}), setStatus:()=>{}, setWidget:()=>{}, theme:{bold:s=>s,fg:(_,s)=>s}}};
  return {tools,commands,hooks,ctx,notes};
}
const huge = chunks.chunkBlocks([{text:'a'.repeat(10000),section:null,pageStart:null,pageEnd:null}]);
log('chunk_limit', {sizes:huge.map(c=>c.content.length), estimatedTokens:huge.map(c=>Math.ceil(c.content.length/4)), configuredMax:240});
{
 const {dir} = fresh(); const f=join(dir,'short.md');
 writeFileSync(f, '# Short\n42\n\n# Long\n'+ 'full evidence '.repeat(8));
 const parsed=await parsing.extractBlocks(f);
 log('markdown_omission',{keptShort:parsed.blocks.some(b=>b.text.includes('42')), sections:parsed.blocks.map(b=>b.section)});
}
{
 const {db,config}=fresh(); seed(db,config,undefined,undefined,false);
 log('legacy_query',{compat:mgr.checkIndexCompatibility(db,config), returned:(await search.hybridSearch('alpha',10,.4,db)).length});
 mgr.stampFingerprints(db,config); const changed={...config, embedding:{...config.embedding,model:'other-model'}};
 writeFileSync(join(process.env.PI_RAG_DIR,'config.json'),JSON.stringify(changed));
 log('mismatch_query',{compat:mgr.checkIndexCompatibility(db,changed), returned:(await search.hybridSearch('alpha',10,.4,db)).length});
 const ac=new AbortController(); ac.abort(); queryCalls=0;
 const hits=await retrieval.retrieve('alpha',{db,signal:ac.signal,config});
 log('cancelled_query',{returned:hits.length,queryCalls});
 const built=context.buildContext(hits);
 const {tools}=handlers(); const result=await tools.rag_query.execute('review',{query:'alpha'});
 log('sources_at_output',{storedPage:hits[0].chunk.pageStart,context:built.text,tool:JSON.parse(result.content[0].text)});
}
{
 const {dir,db,config}=fresh(); seed(db,config,undefined,undefined,false);
 const result=await indexing.rebuildWithSwitch([join(dir,'missing.md')],{},true);
 log('failed_parse_published',{result,manifestPublished:existsSync(join(dir,'active.json')),activeChunks:dbm.getIndexStats().totalChunks});
}
{
 const {dir,db,config}=fresh(); const keep=join(dir,'keep.md'); const gone=join(dir,'gone.md');
 writeFileSync(keep,'alpha updated document '.repeat(5)); seed(db,config,keep); seed(db,config,gone);
 embedFail=true;
 const {commands,ctx,notes}=handlers(); await commands.rag.handler('rebuild --force',ctx);
 log('failed_rebuild_prunes',{remaining:repo.listFilePaths(db),notifications:notes});
}
{
 const config=cfg.defaultConfig(); config.embedding={provider:'voyage',model:'voyage-4-lite',dimensions:1024};
 process.env.VOYAGE_API_KEY='synthetic-review-key';
 const {dir,db}=fresh(config); const f=join(dir,'cloud.md'); writeFileSync(f,'alpha new synthetic cloud document '.repeat(4));
 config.trackedPaths=[f]; writeFileSync(join(dir,'config.json'),JSON.stringify(config)); seed(db,config,f);
 repo.setMetadata(db,repo.MetadataKey.LastBuild,'2020-01-01T00:00:00Z');
 const calls=[];
 globalThis.fetch=async (url,opts)=>{const b=JSON.parse(opts.body);calls.push(b.input_type);return new Response(JSON.stringify({data:b.input.map((_,index)=>({index,embedding:vec(1024)}))}),{status:200});};
 const {hooks}=handlers(); await hooks.before_agent_start({prompt:'alpha'},{});
 log('cloud_refresh_disabled',{cloudAutoRefresh:config.cloudAutoRefresh,calls});
 globalThis.fetch=async()=>{throw new TypeError('synthetic network outage');};
 try {await retrieval.retrieve('alpha',{db,config}); log('query_failure',{threw:false});}catch(e){log('query_failure',{threw:true,message:e.message,ftsHits:repo.searchFts(db,'alpha',10).length});}
 delete process.env.VOYAGE_API_KEY;
}
{
 const {db,config}=fresh(); const other={...config,embedding:{provider:'voyage',model:'voyage-4-lite',dimensions:1024}};
 log('empty_wrong_dimension',{tableDim:repo.detectVectorDimensions(db),requestedDim:1024,compat:mgr.checkIndexCompatibility(db,other)});
 const {dir}=fresh(); const a=mgr.prepareStagingDir(config,dir); writeFileSync(a.dbPath,'old index');
 mgr.publishActiveManifest(dir,{version:1,indexId:'other',relativeDbPath:'indexes/other/rag.db',embeddingFingerprint:'',processingFingerprint:'',createdAt:''});
 mgr.prepareStagingDir(config,dir);
 log('old_index_deleted_on_revisit',{oldDbStillExists:existsSync(a.dbPath)});
}
{
 const {dir}=fresh(); writeFileSync(join(dir,'config.json'),'{BROKEN');
 log('invalid_config_fallback',{provider:cfg.loadConfig().embedding.provider,issues:cfg.validateConfig(cfg.loadConfig())});
}
dbm.closeDbConn();
