"use client";

import { ChevronRight, Sparkles, Stethoscope } from "lucide-react";

import type { Recommendation } from "@/lib/api";
import { useAppStore } from "@/lib/store";

const TUMOR_COLORS: Record<string, string> = {
  lung: "bg-ocean/10 text-ocean border-ocean/30",
  gastric: "bg-amber-50 text-amber-800 border-amber-200",
  liver: "bg-emerald-50 text-emerald-800 border-emerald-200",
  breast: "bg-rose-50 text-rose-800 border-rose-200",
  colorectal: "bg-cyan-50 text-cyan-800 border-cyan-200",
  esophageal: "bg-violet-50 text-violet-800 border-violet-200",
  pancreatic: "bg-indigo-50 text-indigo-800 border-indigo-200",
  thyroid: "bg-sky-50 text-sky-800 border-sky-200",
  prostate: "bg-blue-50 text-blue-800 border-blue-200",
  ovarian: "bg-pink-50 text-pink-800 border-pink-200",
  cervical: "bg-fuchsia-50 text-fuchsia-800 border-fuchsia-200",
  lymphoma: "bg-purple-50 text-purple-800 border-purple-200",
  head_neck: "bg-orange-50 text-orange-800 border-orange-200",
  renal: "bg-teal-50 text-teal-800 border-teal-200",
  sarcoma: "bg-slate-50 text-slate-800 border-slate-200"
};

function tumorTypeColor(type?: string): string {
  if (!type) return "bg-white text-ink border-line";
  return TUMOR_COLORS[type] || "bg-white text-ink border-line";
}

export function RecommendBubbles() {
  const { recommendations, sendMessage, isStreaming } = useAppStore();

  if (!recommendations.length) {
    return null;
  }

  function handleClick(rec: Recommendation) {
    if (isStreaming) return;
    const message = rec.case_data?.case || rec.text;
    void sendMessage(message);
  }

  return (
    <div className="rounded-lg border border-[var(--color-line)] bg-white/72 p-4">
      <div className="mb-3 flex items-center gap-2 text-sm text-[var(--color-ink-soft)]">
        <Sparkles size={14} className="text-ocean" />
        <span className="font-medium text-[var(--color-ink)]">推荐 Tumor Board 病例</span>
        <span className="ml-auto hidden text-xs md:inline">点击后直接进入会诊</span>
      </div>
      <div className="grid grid-cols-1 gap-2 md:grid-cols-2">
        {recommendations.map((rec) => {
          const colorClass = tumorTypeColor(rec.case_data?.tumor_type);
          const stage = rec.case_data?.stage;
          const biomarker = rec.case_data?.biomarker;

          return (
            <button
              key={rec.id}
              type="button"
              disabled={isStreaming}
              onClick={() => handleClick(rec)}
              className={`group flex flex-col gap-1.5 rounded-lg border px-3 py-2.5 text-left transition hover:shadow-sm disabled:cursor-not-allowed disabled:opacity-50 ${colorClass}`}
            >
              <div className="flex items-start gap-2">
                <Stethoscope size={14} className="mt-0.5 shrink-0" />
                <span className="flex-1 text-sm font-medium leading-snug">
                  {rec.text}
                </span>
                <ChevronRight
                  size={14}
                  className="mt-0.5 opacity-0 transition-opacity group-hover:opacity-100"
                />
              </div>
              {(stage || biomarker) && (
                <div className="ml-5 flex items-center gap-2 text-xs opacity-80">
                  {stage && <span>{stage}</span>}
                  {biomarker && <span>{biomarker}</span>}
                </div>
              )}
              {rec.reason && (
                <div className="ml-5 text-xs opacity-70">
                  {rec.reason}
                </div>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
}
