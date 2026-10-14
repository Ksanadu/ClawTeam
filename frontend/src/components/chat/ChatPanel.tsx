"use client";

import { useEffect, useRef } from "react";
import { Activity, BrainCircuit, FlaskConical, ShieldCheck, Users } from "lucide-react";

import { ChatInput } from "@/components/chat/ChatInput";
import { ChatMessage } from "@/components/chat/ChatMessage";
import { RecommendBubbles } from "@/components/chat/RecommendBubbles";
import { useAppStore } from "@/lib/store";

const modeLabel: Record<string, string> = {
  single: "E1 单 Agent",
  multi_no_memory: "E2 多角色",
  multi_memory: "E3 多角色 + 记忆",
  multi_full: "E4 完整会诊"
};

export function ChatPanel() {
  const { messages, sendMessage, isStreaming, tokenStats, experimentMode } = useAppStore();
  const endRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <section className="flex h-full min-w-0 flex-1 flex-col gap-2">
      <div className="grid gap-2 lg:grid-cols-[1fr_250px]">
        <div className="panel rounded-lg px-4 py-3">
          <div className="flex flex-wrap items-center gap-3">
            <div>
              <p className="text-[11px] font-semibold uppercase tracking-[0.2em] text-[var(--color-ink-soft)]">
                Tumor Board Workspace
              </p>
              <h2 className="mt-0.5 text-lg font-semibold tracking-0">ClawTeam 多专科会诊</h2>
            </div>
            <div className="ml-auto flex flex-wrap gap-2 text-xs">
              <span className="inline-flex items-center gap-1 rounded-full bg-ocean/10 px-2.5 py-1 text-ocean">
                <FlaskConical size={13} />
                {modeLabel[experimentMode] ?? experimentMode}
              </span>
              <span className="inline-flex items-center gap-1 rounded-full bg-emerald-50 px-2.5 py-1 text-emerald-700">
                <ShieldCheck size={13} />
                Guardian
              </span>
            </div>
          </div>
        </div>

        <div className="panel grid grid-cols-3 rounded-lg px-3 py-2 text-center">
          <div>
            <div className="mx-auto mb-0.5 flex h-6 w-6 items-center justify-center rounded-full bg-ocean/10 text-ocean">
              <Users size={13} />
            </div>
            <div className="text-[11px] text-[var(--color-ink-soft)]">专科</div>
            <div className="text-sm font-semibold">4</div>
          </div>
          <div>
            <div className="mx-auto mb-0.5 flex h-6 w-6 items-center justify-center rounded-full bg-amber-100 text-amber-700">
              <BrainCircuit size={13} />
            </div>
            <div className="text-[11px] text-[var(--color-ink-soft)]">Round</div>
            <div className="text-sm font-semibold">{experimentMode === "single" ? "1" : "1-2"}</div>
          </div>
          <div>
            <div className="mx-auto mb-0.5 flex h-6 w-6 items-center justify-center rounded-full bg-slate-100 text-slate-700">
              <Activity size={13} />
            </div>
            <div className="text-[11px] text-[var(--color-ink-soft)]">Tokens</div>
            <div className="text-sm font-semibold">{tokenStats ? tokenStats.total_tokens : "-"}</div>
          </div>
        </div>
      </div>

      <div className="panel flex min-h-0 flex-1 flex-col rounded-lg p-3">
        <div className="flex-1 space-y-3 overflow-y-auto pr-2">
          {!messages.length && (
            <div className="rounded-lg border border-dashed border-[var(--color-line)] bg-white/70 p-5">
              <p className="text-[11px] font-semibold uppercase tracking-[0.2em] text-[var(--color-ink-soft)]">
                Ready
              </p>
              <h3 className="mt-2 text-xl font-semibold tracking-0">输入病例，启动 MDT 会诊</h3>
              <p className="mt-2 max-w-3xl text-sm leading-6 text-[var(--color-ink-soft)]">
                系统会展示复杂度判断、四个肿瘤专科意见、Round 2 修正和最终综合结论。
              </p>
              <div className="mt-4">
                <RecommendBubbles />
              </div>
            </div>
          )}

          {messages.map((message) => (
            <ChatMessage
              content={message.content}
              key={message.id}
              retrievals={message.retrievals}
              role={message.role}
              toolCalls={message.toolCalls}
              roleOpinions={message.roleOpinions}
              routing={message.routing}
              guardianBlocked={message.guardianBlocked}
              progress={message.progress}
              attachments={message.attachments}
            />
          ))}
          <div ref={endRef} />
        </div>
      </div>

      <ChatInput disabled={isStreaming} onSend={sendMessage} />
    </section>
  );
}
