import type { SearchResult } from '../engines/hybrid';
import type { Document, Qrels } from '../data/types';

interface Props {
  title: string;
  color: 'amber' | 'purple' | 'emerald';
  description?: string;
  results: SearchResult[];
  latencyMs: number;
  corpus: Document[];
  query: string;
  qrels: Qrels;
  queryId: string | null;
}

const colorMap = {
  amber: {
    border: 'border-amber-500/30',
    bg: 'bg-amber-500/10',
    text: 'text-amber-400',
    badge: 'bg-amber-500/20 text-amber-300',
    relevant: 'border-l-amber-400',
  },
  purple: {
    border: 'border-purple-500/30',
    bg: 'bg-purple-500/10',
    text: 'text-purple-400',
    badge: 'bg-purple-500/20 text-purple-300',
    relevant: 'border-l-purple-400',
  },
  emerald: {
    border: 'border-emerald-500/30',
    bg: 'bg-emerald-500/10',
    text: 'text-emerald-400',
    badge: 'bg-emerald-500/20 text-emerald-300',
    relevant: 'border-l-emerald-400',
  },
};

export function ResultColumn({ title, color, description, results, latencyMs, corpus, query, qrels, queryId }: Props) {
  const c = colorMap[color];

  const relevantDocs = queryId && qrels[queryId]
    ? new Set(Object.keys(qrels[queryId]))
    : new Set<string>();

  return (
    <div className={`rounded-lg border ${c.border} ${c.bg} p-4`}>
      <div className="flex items-center justify-between mb-1">
        <h3 className={`font-semibold ${c.text}`}>{title}</h3>
        <span className={`text-xs px-2 py-0.5 rounded-full ${c.badge}`}>
          {latencyMs.toFixed(1)} ms
        </span>
      </div>
      {description && (
        <p className="text-xs text-gray-500 mb-3">{description}</p>
      )}

      <div className="space-y-2">
        {results.length === 0 ? (
          <p className="text-gray-500 text-sm">No results</p>
        ) : (
          results.map((r, i) => {
            const doc = corpus[r.docIndex];
            if (!doc) return null;
            const isRelevant = relevantDocs.has(doc.id);

            return (
              <div
                key={doc.id}
                className={`bg-gray-900/60 rounded p-3 border-l-2 ${
                  isRelevant ? c.relevant : 'border-l-transparent'
                }`}
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="flex items-center gap-2 min-w-0">
                    <span className="text-xs text-gray-600 shrink-0">#{i + 1}</span>
                    <h4 className="text-sm font-medium text-gray-200 truncate">
                      {doc.title || `Doc ${doc.id}`}
                    </h4>
                  </div>
                  <div className="flex items-center gap-1.5 shrink-0">
                    {isRelevant && (
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-green-500/20 text-green-400">
                        relevant
                      </span>
                    )}
                    <span className="text-[10px] text-gray-600">
                      {r.score.toFixed(4)}
                    </span>
                  </div>
                </div>
                <p className="text-xs text-gray-400 mt-1 line-clamp-3">
                  {highlightSnippet(doc.text, query)}
                </p>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}

function highlightSnippet(text: string, _query: string): string {
  return text.slice(0, 200) + (text.length > 200 ? '...' : '');
}
