"use client";

import { MessageSquare, PanelLeftClose, Plus, Trash2 } from "lucide-react";

import { useAppStore } from "@/lib/store";

function preview(text: string) {
  return text.length > 72 ? `${text.slice(0, 72)}...` : text;
}

export function Sidebar({ onCollapse }: { onCollapse?: () => void }) {
  const {
    sessions,
    currentSessionId,
    selectSession,
    createNewSession,
    removeSession,
    messages
  } = useAppStore();

  return (
    <aside className="panel flex h-full flex-col rounded-lg p-3">
      <div className="mb-3 flex items-center justify-between">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.22em] text-[var(--color-ink-soft)]">
            Cases
          </p>
          <h2 className="text-base font-semibold tracking-0">会诊队列</h2>
        </div>
        <div className="flex items-center gap-1.5">
          <button
            className="flex h-9 w-9 items-center justify-center rounded-lg bg-ocean/10 text-ocean hover:bg-ocean/15"
            onClick={() => void createNewSession()}
            title="新会诊"
            type="button"
          >
            <Plus size={18} />
          </button>
          {onCollapse && (
            <button
              className="flex h-9 w-9 items-center justify-center rounded-lg border border-[var(--color-line)] bg-white/70 text-[var(--color-ink-soft)] hover:bg-white hover:text-ocean"
              onClick={onCollapse}
              title="收起会诊队列"
              type="button"
            >
              <PanelLeftClose size={18} />
            </button>
          )}
        </div>
      </div>

      <div className="space-y-2 overflow-y-auto pr-1">
        {sessions.map((session) => (
          <div
            className={`rounded-lg border px-3 py-3 transition ${
              session.id === currentSessionId
                ? "border-ocean/30 bg-ocean/10"
                : "border-[var(--color-line)] bg-white/58 hover:bg-white"
            }`}
            key={session.id}
          >
            <button
              className="w-full text-left"
              onClick={() => void selectSession(session.id)}
              type="button"
            >
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <p className="truncate text-sm font-semibold">{session.title}</p>
                  <p className="mt-1 text-xs text-[var(--color-ink-soft)]">
                    {session.message_count} 条消息
                  </p>
                </div>
                <MessageSquare className="mt-0.5 shrink-0 text-[var(--color-ink-soft)]" size={15} />
              </div>
            </button>
            <button
              className="mt-2 flex items-center gap-1.5 text-xs text-ember"
              onClick={() => void removeSession(session.id)}
              type="button"
            >
              <Trash2 size={13} />
              删除
            </button>
          </div>
        ))}
      </div>

      <div className="mt-3 flex min-h-0 flex-1 flex-col rounded-lg border border-[var(--color-line)] bg-white/50 p-3">
        <p className="text-xs font-semibold uppercase tracking-[0.22em] text-[var(--color-ink-soft)]">
          Timeline
        </p>
        <div className="mt-3 space-y-2 overflow-y-auto pr-1">
          {messages.map((message) => (
            <div
              className="rounded-lg border border-[var(--color-line)] bg-white/78 px-3 py-2"
              key={message.id}
            >
              <div className="mb-1 flex items-center justify-between text-[11px] font-medium uppercase tracking-[0.16em] text-[var(--color-ink-soft)]">
                <span>{message.role === "user" ? "case" : "mdt"}</span>
                <span>{message.roleOpinions?.length ?? 0} roles</span>
              </div>
              <p className="text-xs leading-5 text-[var(--color-ink-soft)]">{preview(message.content)}</p>
            </div>
          ))}
        </div>
      </div>
    </aside>
  );
}
