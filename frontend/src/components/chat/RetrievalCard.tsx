"use client";

import { ChevronDown, Database } from "lucide-react";

import type { RetrievalResult } from "@/lib/api";

export function RetrievalCard({ results }: { results: RetrievalResult[] }) {
  if (!results.length) {
    return null;
  }

  return (
    <details className="group rounded-lg border border-ocean/20 bg-ocean/5">
      <summary className="flex cursor-pointer list-none items-center gap-2 px-3 py-2 text-xs font-semibold text-ocean">
        <Database size={14} />
        <span>记忆检索</span>
        <span>{results.length} 条片段</span>
        <ChevronDown size={14} className="ml-auto transition group-open:rotate-180" />
      </summary>
      <div className="space-y-2 border-t border-ocean/15 bg-white/72 p-2">
        {results.map((item, index) => (
          <div className="rounded-md border border-[var(--color-line)] bg-white p-3" key={`${item.source}-${index}`}>
            <div className="mb-1 flex items-center justify-between text-xs text-[var(--color-ink-soft)]">
              <span>{item.source}</span>
              <span>{item.score.toFixed(3)}</span>
            </div>
            <p className="max-h-28 overflow-auto text-xs leading-5 text-[var(--color-ink)]">{item.text}</p>
          </div>
        ))}
      </div>
    </details>
  );
}
