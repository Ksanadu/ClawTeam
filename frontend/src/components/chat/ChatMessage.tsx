"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { CheckCircle2, ClipboardList, FileSearch, GitBranch, Loader2, Route, ShieldAlert, Users } from "lucide-react";

import { RetrievalCard } from "@/components/chat/RetrievalCard";
import { RoleOpinionCard } from "@/components/chat/RoleOpinionCard";
import { ThoughtChain } from "@/components/chat/ThoughtChain";
import type {
  ChatAttachment,
  ProgressEvent,
  RetrievalResult,
  RoleOpinion,
  RoutingInfo,
  ToolCall
} from "@/lib/api";

function roleCountLabel(routing?: RoutingInfo | null) {
  if (!routing || !routing.roles.length) {
    return "等待复杂度判断";
  }
  return `${routing.roles.length} 个专科参与`;
}

function progressFor(progress: ProgressEvent[] | undefined, stage: string) {
  return progress?.find((item) => item.stage === stage);
}

function inferredProgress({
  progress,
  stage,
  fallbackDone,
  fallbackRunning,
  doneLabel,
  runningLabel,
  pendingLabel
}: {
  progress?: ProgressEvent[];
  stage: string;
  fallbackDone?: boolean;
  fallbackRunning?: boolean;
  doneLabel: string;
  runningLabel: string;
  pendingLabel: string;
}) {
  const explicit = progressFor(progress, stage);
  if (explicit) {
    return explicit;
  }
  if (fallbackDone) {
    return { stage, status: "done", label: doneLabel };
  }
  if (fallbackRunning) {
    return { stage, status: "running", label: runningLabel };
  }
  return { stage, status: "pending", label: pendingLabel };
}

function statusText(status?: string) {
  if (status === "done") return "已完成";
  if (status === "running") return "进行中";
  if (status === "pending") return "等待中";
  if (status === "blocked") return "已拦截";
  if (status === "error") return "失败";
  return "等待中";
}

function StepIcon({ status, icon }: { status?: string; icon: JSX.Element }) {
  if (status === "done") {
    return <CheckCircle2 size={14} />;
  }
  if (status === "running") {
    return <Loader2 size={14} className="animate-spin" />;
  }
  return icon;
}

function ProcessStep({
  icon,
  title,
  status,
  children,
  progress,
  last = false
}: {
  icon: JSX.Element;
  title: string;
  status: string;
  children?: React.ReactNode;
  progress?: ProgressEvent;
  last?: boolean;
}) {
  const active = progress?.status === "running";
  const done = progress?.status === "done";
  return (
    <div className="grid grid-cols-[24px_1fr] gap-3">
      <div className="relative flex justify-center">
        <div
          className={`mt-1 flex h-6 w-6 items-center justify-center rounded-full border ${
            done
              ? "border-emerald-600 bg-emerald-50 text-emerald-700"
              : active
                ? "border-ocean bg-ocean/10 text-ocean"
                : "border-slate-300 bg-white text-slate-400"
          }`}
        >
          <StepIcon status={progress?.status} icon={icon} />
        </div>
        {!last && <div className={`absolute top-8 h-[calc(100%-4px)] w-px ${done ? "bg-emerald-200" : "bg-slate-200"}`} />}
      </div>
      <div className="pb-4">
        <div className="flex flex-wrap items-center gap-2">
          <div className="text-sm font-semibold">{title}</div>
          <span
            className={`rounded-full px-2 py-0.5 text-[11px] ${
              done
                ? "bg-emerald-50 text-emerald-700"
                : active
                  ? "bg-ocean/10 text-ocean"
                  : "bg-slate-100 text-[var(--color-ink-soft)]"
            }`}
          >
            {progress?.label || status || statusText(progress?.status)}
          </span>
        </div>
        {children && <div className="mt-2">{children}</div>}
      </div>
    </div>
  );
}

export function ChatMessage({
  role,
  content,
  toolCalls,
  retrievals,
  roleOpinions,
  routing,
  guardianBlocked,
  progress,
  attachments = []
}: {
  role: "user" | "assistant";
  content: string;
  toolCalls: ToolCall[];
  retrievals: RetrievalResult[];
  roleOpinions?: RoleOpinion[];
  routing?: RoutingInfo | null;
  guardianBlocked?: { reason: string; message: string } | null;
  progress?: ProgressEvent[];
  attachments?: ChatAttachment[];
}) {
  const isUser = role === "user";
  const hasRoleOpinions = !!(roleOpinions && roleOpinions.length > 0);
  const hasProgress = !!(progress && progress.length > 0);
  const round1Opinions = roleOpinions?.filter((opinion) => (opinion.round ?? 1) === 1) ?? [];
  const round2Opinions = roleOpinions?.filter((opinion) => opinion.round === 2) ?? [];
  const hasFinal = content.trim() !== "";
  const hasActivity = toolCalls.length > 0 || retrievals.length > 0 || !!guardianBlocked;
  const hasToolContext = toolCalls.length > 0 || retrievals.length > 0;
  const hasMdtWorkflow = hasProgress || !!routing || hasRoleOpinions || !!guardianBlocked;
  const routeProgress = inferredProgress({
    progress,
    stage: "complexity",
    fallbackDone: !!(routing && routing.roles.length),
    fallbackRunning: !routing && !hasRoleOpinions && !hasFinal,
    doneLabel: "复杂度判断完成",
    runningLabel: "正在判断复杂度",
    pendingLabel: "等待复杂度判断"
  });
  const memoryProgress = inferredProgress({
    progress,
    stage: "memory",
    fallbackDone: retrievals.length > 0 || toolCalls.length > 0,
    fallbackRunning: !routing && hasActivity,
    doneLabel: "上下文准备完成",
    runningLabel: "正在准备上下文",
    pendingLabel: "按需调用"
  });
  const round1Progress = inferredProgress({
    progress,
    stage: "round1",
    fallbackDone: round1Opinions.length > 0,
    fallbackRunning: !!routing && round1Opinions.length === 0 && !hasFinal,
    doneLabel: "Round 1 已完成",
    runningLabel: "专家正在生成意见",
    pendingLabel: "等待 Round 1"
  });
  const round2Progress = inferredProgress({
    progress,
    stage: "round2",
    fallbackDone: round2Opinions.length > 0,
    fallbackRunning: round1Opinions.length > 0 && round2Opinions.length === 0 && !hasFinal,
    doneLabel: "Round 2 已完成",
    runningLabel: "专家正在反驳修正",
    pendingLabel: "等待 Round 2"
  });
  const synthesisProgress = inferredProgress({
    progress,
    stage: "synthesis",
    fallbackDone: hasFinal,
    fallbackRunning: hasRoleOpinions && !hasFinal,
    doneLabel: "综合结论已生成",
    runningLabel: "正在生成综合结论",
    pendingLabel: "等待综合结论"
  });

  if (isUser) {
    return (
      <article className="ml-auto max-w-[82%] rounded-lg bg-[var(--color-ink)] px-4 py-3 text-white shadow-sm">
        <div className="mb-1 text-xs font-medium text-white/60">病例输入</div>
        <div className="whitespace-pre-wrap leading-7">{content}</div>
        {attachments.length > 0 ? (
          <div className="mt-3 flex flex-wrap gap-2 border-t border-white/10 pt-2">
            {attachments.map((attachment, index) => (
              <span
                className="inline-flex max-w-[220px] items-center gap-1 rounded-md bg-white/10 px-2 py-1 text-xs text-white/80"
                key={`${attachment.name}-${index}`}
              >
                <FileSearch size={13} />
                <span className="truncate">{attachment.name}</span>
              </span>
            ))}
          </div>
        ) : null}
      </article>
    );
  }

  if (!hasMdtWorkflow) {
    return (
      <article className="mr-auto w-full max-w-[980px] rounded-lg border border-[var(--color-line)] bg-white/88 shadow-sm">
        <div className="flex items-center gap-2 border-b border-[var(--color-line)] px-4 py-3 text-sm font-semibold">
          <ClipboardList size={16} className="text-ocean" />
          {hasToolContext ? "执行记录" : "综合结论"}
        </div>
        <div className="space-y-3 p-4">
          {hasToolContext && (
            <div className="rounded-lg border border-[var(--color-line)] bg-slate-50/80 p-3">
              <div className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.16em] text-ember">
                <FileSearch size={14} />
                动态工具链
              </div>
              <div className="space-y-2">
                <RetrievalCard results={retrievals} />
                <ThoughtChain toolCalls={toolCalls} />
              </div>
            </div>
          )}

          {hasFinal ? (
            <div className="markdown rounded-lg border border-[var(--color-line)] bg-slate-50/80 p-4">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
            </div>
          ) : (
            <div className="rounded-lg border border-dashed border-[var(--color-line)] bg-slate-50 p-4 text-sm text-[var(--color-ink-soft)]">
              正在处理请求...
            </div>
          )}
        </div>
      </article>
    );
  }

  return (
    <article className="mr-auto w-full max-w-[980px] rounded-lg border border-[var(--color-line)] bg-white/88 shadow-sm">
      <div className="flex flex-wrap items-center gap-2 border-b border-[var(--color-line)] px-4 py-3">
        <div className="flex items-center gap-2 text-sm font-semibold">
          <ClipboardList size={16} className="text-ocean" />
          MDT 会诊记录
        </div>
        <div className="ml-auto flex flex-wrap items-center gap-2 text-xs text-[var(--color-ink-soft)]">
          <span className="rounded-full bg-[var(--color-ocean-soft)] px-2.5 py-1 text-ocean">
            {roleCountLabel(routing)}
          </span>
          {routing && routing.reason && (
            <span className="rounded-full bg-slate-100 px-2.5 py-1">
              {routing.reason}
            </span>
          )}
        </div>
      </div>

      <div className="space-y-3 p-4">
        {guardianBlocked && (
          <div className="flex items-start gap-2 rounded-lg border border-ember/40 bg-ember/10 p-3 text-sm">
            <ShieldAlert size={16} className="mt-0.5 text-ember" />
            <div>
              <div className="font-medium text-ember">安全守卫已拦截请求</div>
              <div className="mt-1 text-xs text-[var(--color-ink-soft)]">原因: {guardianBlocked.reason}</div>
              <div className="mt-1 text-[var(--color-ink)]">{guardianBlocked.message}</div>
            </div>
          </div>
        )}

        <section className="rounded-lg border border-ocean/15 bg-ocean/5 p-3">
          <div className="mb-3 flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.16em] text-ocean">
            <GitBranch size={14} />
            会诊流程链
          </div>

          <ProcessStep
            icon={<Route size={14} />}
            title="复杂度与路由"
            status={routing && routing.roles.length ? "已完成" : "等待中"}
            progress={routeProgress}
          >
            {routing && routing.roles.length > 0 ? (
              <div className="rounded-md border border-ocean/20 bg-white px-3 py-2 text-xs text-ocean">
                {routing.roles.join(" / ")}
                {routing.reason && <span className="ml-2 text-[var(--color-ink-soft)]">{routing.reason}</span>}
              </div>
            ) : (
              <div className="text-xs text-[var(--color-ink-soft)]">等待后端返回复杂度判断</div>
            )}
          </ProcessStep>

          {(retrievals.length > 0 || toolCalls.length > 0) && (
            <ProcessStep
              icon={<FileSearch size={14} />}
              title="上下文与工具"
              status={`${retrievals.length} 条检索 / ${toolCalls.length} 次工具`}
              progress={memoryProgress}
            >
              <div className="space-y-2">
                <RetrievalCard results={retrievals} />
                <ThoughtChain toolCalls={toolCalls} />
              </div>
            </ProcessStep>
          )}

          <ProcessStep
            icon={<Users size={14} />}
            title="Round 1 独立意见"
            status={round1Opinions.length ? `${round1Opinions.length} 条意见` : "等待中"}
            progress={round1Progress}
            last={!hasFinal && !progressFor(progress, "round2") && round2Progress.status === "pending"}
          >
            {round1Opinions.length ? (
              <div className="grid gap-2 xl:grid-cols-2">
                {round1Opinions.map((opinion, idx) => (
                  <RoleOpinionCard key={`${opinion.role}-${idx}`} opinion={opinion} />
                ))}
              </div>
            ) : (
              <div className="text-xs text-[var(--color-ink-soft)]">等待四个专科生成独立意见</div>
            )}
          </ProcessStep>

          {(progressFor(progress, "round2") || round2Opinions.length > 0 || round2Progress.status === "running" || round2Progress.status === "done") && (
            <ProcessStep
              icon={<Users size={14} />}
              title="Round 2 反驳修正"
              status={round2Opinions.length ? `${round2Opinions.length} 条修正` : "等待中"}
              progress={round2Progress}
              last={!hasFinal}
            >
              {round2Opinions.length ? (
                <div className="grid gap-2 xl:grid-cols-2">
                  {round2Opinions.map((opinion, idx) => (
                    <RoleOpinionCard key={`${opinion.role}-round2-${idx}`} opinion={opinion} />
                  ))}
                </div>
              ) : (
                <div className="text-xs text-[var(--color-ink-soft)]">等待专家阅读彼此意见后给出同意、反对和修正</div>
              )}
            </ProcessStep>
          )}

          {(hasFinal || synthesisProgress.status === "running") && (
            <ProcessStep
              icon={<ClipboardList size={14} />}
              title="综合结论"
              status="已生成"
              progress={synthesisProgress}
              last
            />
          )}
        </section>

        {hasRoleOpinions && hasFinal && (
          <div className="text-xs font-semibold uppercase tracking-[0.18em] text-[var(--color-ink-soft)]">
            综合结论
          </div>
        )}

        {hasFinal && (!toolCalls.length || !toolCalls.some((tc) => tc.output.trim() === content.trim())) && (
          <div className="markdown rounded-lg border border-[var(--color-line)] bg-slate-50/80 p-4">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
          </div>
        )}

        {!hasFinal && !toolCalls.length && !hasRoleOpinions && !guardianBlocked && (
          <div className="rounded-lg border border-dashed border-[var(--color-line)] bg-slate-50 p-4 text-sm text-[var(--color-ink-soft)]">
            正在组织多专科意见...
          </div>
        )}
      </div>
    </article>
  );
}
