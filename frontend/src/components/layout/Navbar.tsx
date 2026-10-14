"use client";

import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { FileStack, FlaskConical, Plus, SlidersHorizontal, Stethoscope } from "lucide-react";

import { InspectorPanel } from "@/components/editor/InspectorPanel";
import type { ExperimentMode } from "@/lib/api";
import { useAppStore } from "@/lib/store";

const MODE_OPTIONS: { value: ExperimentMode; label: string; helper: string }[] = [
  { value: "single", label: "E1 单 Agent", helper: "基线" },
  { value: "multi_no_memory", label: "E2 多角色", helper: "无记忆" },
  { value: "multi_memory", label: "E3 多角色 + 记忆", helper: "共享上下文" },
  { value: "multi_full", label: "E4 完整会诊", helper: "记忆 + Guardian" }
];

export function Navbar() {
  const [isInspectorOpen, setIsInspectorOpen] = useState(false);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  const {
    createNewSession,
    renameCurrentSession,
    sessions,
    currentSessionId,
    experimentMode,
    setExperimentMode
  } = useAppStore();

  const currentTitle =
    sessions.find((session) => session.id === currentSessionId)?.title ?? "新会诊";

  return (
    <header className="panel flex flex-wrap items-center justify-between gap-3 rounded-lg px-4 py-3">
      <div className="flex min-w-0 items-center gap-3">
        <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-lg bg-ocean/10 text-ocean">
          <Stethoscope size={21} />
        </div>
        <div className="min-w-0">
          <p className="text-xs font-semibold uppercase tracking-[0.22em] text-[var(--color-ink-soft)]">
            ClawTeam MDT
          </p>
          <div className="flex min-w-0 items-center gap-2">
            <h1 className="truncate text-lg font-semibold tracking-0">{currentTitle}</h1>
            <button
              className="rounded-md border border-[var(--color-line)] px-2 py-1 text-xs text-[var(--color-ink-soft)] hover:bg-white"
              onClick={() => {
                const next = window.prompt("重命名当前会诊", currentTitle);
                if (next) {
                  void renameCurrentSession(next);
                }
              }}
              type="button"
            >
              重命名
            </button>
          </div>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <label className="flex items-center gap-2 rounded-lg border border-[var(--color-line)] bg-white/70 px-3 py-2 text-sm">
          <FlaskConical size={15} className="text-ocean" />
          <select
            className="cursor-pointer bg-transparent text-sm outline-none"
            value={experimentMode}
            onChange={(e) => setExperimentMode(e.target.value as ExperimentMode)}
          >
            {MODE_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label} - {opt.helper}
              </option>
            ))}
          </select>
        </label>
        <button
          className="flex items-center gap-2 rounded-lg border border-[var(--color-line)] bg-white/70 px-3 py-2 text-sm hover:bg-white"
          onClick={() => void createNewSession()}
          type="button"
        >
          <Plus size={16} />
          新会诊
        </button>
        <div className="hidden items-center gap-2 rounded-lg bg-ember/10 px-3 py-2 text-sm text-ember md:flex">
          <FileStack size={16} />
          File Memory
        </div>
        <button
          className="flex h-9 w-9 items-center justify-center rounded-lg border border-[var(--color-line)] bg-white/70 text-[var(--color-ink-soft)] hover:bg-white hover:text-ink"
          onClick={() => setIsInspectorOpen(true)}
          title="打开配置面板"
          type="button"
        >
          <SlidersHorizontal size={16} />
        </button>
      </div>

      {isInspectorOpen && mounted && createPortal(
        <div className="fixed inset-0 z-[9999] flex items-center justify-center">
          <div
            className="absolute inset-0 bg-black/25 backdrop-blur-sm"
            onClick={() => setIsInspectorOpen(false)}
          />
          <div className="relative z-10 flex h-[80vh] w-[90vw] max-w-4xl flex-col shadow-2xl">
            <InspectorPanel onClose={() => setIsInspectorOpen(false)} />
          </div>
        </div>,
        document.body
      )}
    </header>
  );
}
