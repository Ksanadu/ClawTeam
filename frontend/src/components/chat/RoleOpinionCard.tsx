"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Activity, Microscope, Pill, Radiation, Scissors, Stethoscope } from "lucide-react";

import type { RoleOpinion } from "@/lib/api";

const ROLE_META: Record<
  string,
  { icon: JSX.Element; label: string; tone: string; badge: string }
> = {
  pathologist: {
    icon: <Microscope size={16} />,
    label: "病理科",
    tone: "border-blue-200 bg-blue-50/70 text-blue-800",
    badge: "bg-blue-100 text-blue-800"
  },
  surgeon: {
    icon: <Scissors size={16} />,
    label: "肿瘤外科",
    tone: "border-emerald-200 bg-emerald-50/70 text-emerald-800",
    badge: "bg-emerald-100 text-emerald-800"
  },
  medical_oncologist: {
    icon: <Pill size={16} />,
    label: "肿瘤内科",
    tone: "border-amber-200 bg-amber-50/70 text-amber-800",
    badge: "bg-amber-100 text-amber-800"
  },
  radiation_oncologist: {
    icon: <Radiation size={16} />,
    label: "放疗科",
    tone: "border-rose-200 bg-rose-50/70 text-rose-800",
    badge: "bg-rose-100 text-rose-800"
  },
  physician: {
    icon: <Stethoscope size={16} />,
    label: "主治医生",
    tone: "border-cyan-200 bg-cyan-50/70 text-cyan-800",
    badge: "bg-cyan-100 text-cyan-800"
  }
};

function roundLabel(opinion: RoleOpinion) {
  const label = `${opinion.role_label} ${opinion.content}`.toLowerCase();
  if (label.includes("round 2") || label.includes("修正") || label.includes("反对")) {
    return "Round 2";
  }
  return "Round 1";
}

export function RoleOpinionCard({ opinion }: { opinion: RoleOpinion }) {
  const meta = ROLE_META[opinion.role] ?? {
    icon: <Activity size={16} />,
    label: opinion.role_label || opinion.role,
    tone: "border-slate-200 bg-slate-50/80 text-slate-800",
    badge: "bg-slate-100 text-slate-700"
  };

  return (
    <details className={`mb-2 overflow-hidden rounded-lg border ${meta.tone}`} open>
      <summary className="flex cursor-pointer select-none items-center gap-2 px-3 py-2 text-sm font-semibold">
        {meta.icon}
        <span>{meta.label}</span>
        <span className={`ml-auto rounded-full px-2 py-0.5 text-[11px] font-medium ${meta.badge}`}>
          {roundLabel(opinion)}
        </span>
      </summary>
      <div className="border-t border-current/10 bg-white/72 px-3 py-3">
        <div className="markdown text-sm">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{opinion.content}</ReactMarkdown>
        </div>
        {opinion.evidence && opinion.evidence.length > 0 && (
          <div className="mt-3 border-t border-current/10 pt-2 text-xs text-[var(--color-ink-soft)]">
            <div className="mb-1 font-medium text-[var(--color-ink)]">证据来源</div>
            <ul className="list-inside list-disc space-y-1">
              {opinion.evidence.map((src, idx) => (
                <li key={idx}>{src}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </details>
  );
}
