export interface RagCommandMeta {
  name: string;
  /** Short label used in autocomplete */
  label: string;
  /** Short description used in autocomplete */
  description: string;
  /** Usage string + longer description shown in /rag help */
  usage: string;
  helpText: string;
}

export const RAG_COMMAND_META: RagCommandMeta[] = [
  {
    name: 'index',
    label: 'index',
    description: 'Index a file or directory',
    usage: '/rag index <path>',
    helpText: 'Index a file or directory (chunks, embeds, stores)'
  },
  {
    name: 'search',
    label: 'search',
    description: 'Search the index',
    usage: '/rag search <query>',
    helpText: 'Hybrid BM25 + vector search over the index'
  },
  {
    name: 'find',
    label: 'find',
    description: 'List indexed files matching a glob',
    usage: '/rag find <glob>',
    helpText: 'List indexed files matching a glob (e.g. *.ts, src/*)'
  },
  {
    name: 'status',
    label: 'status',
    description: 'Show index statistics',
    usage: '/rag status',
    helpText: 'Show index stats and active configuration'
  },
  {
    name: 'rebuild',
    label: 'rebuild',
    description: 'Re-embed tracked files (--force to skip hash check + wipe DB)',
    usage: '/rag rebuild [--force]',
    helpText: 'Re-embed tracked files; --force wipes DB and bypasses hash skip'
  },
  {
    name: 'refresh',
    label: 'refresh',
    description: 'Incremental refresh — new/changed files only',
    usage: '/rag refresh',
    helpText: 'Incremental refresh — only new/changed files (also fires automatically every 24h)'
  },
  {
    name: 'clear',
    label: 'clear',
    description: 'Clear the index',
    usage: '/rag clear',
    helpText: 'Delete all indexed chunks'
  },
  {
    name: 'exclude',
    label: 'exclude',
    description: 'Manage gitignore-style exclude patterns',
    usage: '/rag exclude <pattern>',
    helpText: 'Add a gitignore-style exclude pattern (omit to list; -<pattern> to remove)'
  },
  {
    name: 'ext',
    label: 'ext',
    description: 'Manage indexable file-extension allowlist',
    usage: '/rag ext list|add|remove|reset',
    helpText: 'Manage the indexable file-extension allowlist'
  },
  {
    name: 'on',
    label: 'on',
    description: 'Enable auto-injection',
    usage: '/rag on',
    helpText: 'Enable automatic RAG injection before each agent turn'
  },
  {
    name: 'off',
    label: 'off',
    description: 'Disable auto-injection',
    usage: '/rag off',
    helpText: 'Disable automatic RAG injection'
  },
  {
    name: 'help',
    label: 'help',
    description: 'Show all /rag commands',
    usage: '/rag help',
    helpText: 'Show this help'
  },
];
