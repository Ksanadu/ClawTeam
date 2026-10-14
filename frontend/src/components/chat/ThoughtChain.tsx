"use client";

import { CheckCircle2, ChevronDown, Circle, TerminalSquare } from "lucide-react";

import type { ToolCall } from "@/lib/api";

function toolLabel(tool: string) {
  const labels: Record<string, string> = {
    read_file: "读取文件",
    terminal: "执行命令",
    search_memory: "检索记忆",
    search_knowledge: "检索知识库"
  };
  return labels[tool] ?? tool;
}

function compactOutput(output: string) {
  const clean = output.replace(/\s+/g, " ").trim();
  return clean.length > 96 ? `${clean.slice(0, 96)}...` : clean;
}

function summarizeTools(toolCalls: ToolCall[]) {
  const counts = new Map<string, number>();
  for (const toolCall of toolCalls) {
    counts.set(toolCall.tool, (counts.get(toolCall.tool) ?? 0) + 1);
  }
  return Array.from(counts.entries())
    .map(([tool, count]) => `${toolLabel(tool)}${count > 1 ? ` x${count}` : ""}`)
    .join(" · ");
}

export function ThoughtChain({ toolCalls }: { toolCalls: ToolCall[] }) {
  if (!toolCalls.length) {
    return null;
  }

  return (
    <details className="group rounded-lg border border-slate-200 bg-slate-50/70">
      <summary className="flex cursor-pointer list-none items-center gap-2 px-3 py-2 text-xs text-[var(--color-ink-soft)]">
        <TerminalSquare size={14} className="text-ember" />
        <span className="font-semibold uppercase tracking-[0.16em]">动态工具链</span>
        <span className="rounded-full bg-white px-2 py-0.5">{toolCalls.length} 步</span>
        <span className="hidden min-w-0 truncate sm:inline">{summarizeTools(toolCalls)}</span>
        <ChevronDown size={14} className="ml-auto transition group-open:rotate-180" />
      </summary>

      <div className="space-y-0 border-t border-slate-200 px-3 py-3">
        {toolCalls.map((toolCall, index) => {
          const done = !!toolCall.output;
          const isLast = index === toolCalls.length - 1;

          return (
            <div className="grid grid-cols-[22px_1fr] gap-3" key={`${toolCall.tool}-${index}`}>
              <div className="relative flex justify-center">
                <div
                  className={`mt-1 flex h-5 w-5 items-center justify-center rounded-full border ${
                    done
                      ? "border-emerald-600 bg-emerald-50 text-emerald-700"
                      : "border-ember bg-ember/10 text-ember"
                  }`}
                >
                  {done ? <CheckCircle2 size={14} /> : <Circle size={9} fill="currentColor" />}
                </div>
                {!isLast && (
                  <div
                    className={`absolute top-7 h-[calc(100%-8px)] w-px ${
                      done ? "bg-emerald-200" : "bg-slate-200"
                    }`}
                  />
                )}
              </div>

              <details className="group pb-4" open={!done}>
                <summary className="flex cursor-pointer list-none items-start gap-2 rounded-md px-2 py-1.5 hover:bg-white">
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="text-sm font-semibold text-[var(--color-ink)]">
                        {toolLabel(toolCall.tool)}
                      </span>
                      <span className="mono rounded bg-white px-1.5 py-0.5 text-[11px] text-[var(--color-ink-soft)]">
                        {toolCall.tool}
                      </span>
                      <span
                        className={`rounded-full px-2 py-0.5 text-[11px] ${
                          done ? "bg-emerald-50 text-emerald-700" : "bg-ember/10 text-ember"
                        }`}
                      >
                        {done ? "完成" : "运行中"}
                      </span>
                    </div>
                    {done && toolCall.output && (
                      <p className="mt-1 truncate text-xs text-[var(--color-ink-soft)]">
                        {compactOutput(toolCall.output)}
                      </p>
                    )}
                  </div>
                  <ChevronDown size={14} className="mt-1 shrink-0 text-[var(--color-ink-soft)] transition group-open:rotate-180" />
                </summary>

                <div className="ml-2 mt-2 space-y-2 rounded-md border border-[var(--color-line)] bg-white p-3 text-xs">
                  {toolCall.input && (
                    <div>
                      <div className="mb-1 font-medium text-[var(--color-ink-soft)]">Input</div>
                      <pre className="mono max-h-32 overflow-auto rounded-md bg-slate-50 p-2 whitespace-pre-wrap break-all">
                        {toolCall.input}
                      </pre>
                    </div>
                  )}
                  {toolCall.output && (
                    <div>
                      <div className="mb-1 font-medium text-[var(--color-ink-soft)]">Output</div>
                      <pre className="mono max-h-40 overflow-auto rounded-md bg-slate-50 p-2 whitespace-pre-wrap break-all">
                        {toolCall.output}
                      </pre>
                    </div>
                  )}
                </div>
              </details>
            </div>
          );
        })}
      </div>
    </details>
  );
}
