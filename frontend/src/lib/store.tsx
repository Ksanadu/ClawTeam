"use client";

import { createContext, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";

import {
  compressSession,
  createSession,
  deleteSession,
  getRagMode,
  getRecommendations,
  getSessionHistory,
  getSessionTokens,
  listSessions,
  listSkills,
  loadFile,
  renameSession,
  saveFile,
  setRagMode,
  streamChat,
  type ChatAttachment,
  type ExperimentMode,
  type ProgressEvent,
  type Recommendation,
  type RetrievalEvent,
  type RetrievalResult,
  type RoleOpinion,
  type RoutingInfo,
  type RoleFailure,
  type RunStatus,
  type SessionSummary,
  type ToolCall
} from "@/lib/api";

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  toolCalls: ToolCall[];
  retrievals: RetrievalResult[];
  retrievalEvents?: RetrievalEvent[];
  roleOpinions?: RoleOpinion[];
  routing?: RoutingInfo | null;
  guardianBlocked?: { reason: string; message: string } | null;
  progress?: ProgressEvent[];
  attachments?: ChatAttachment[];
  collaborationMetric?: Record<string, unknown> | null;
  runId?: string;
  toolCallId?: string;
  status?: RunStatus;
  failedRoles?: RoleFailure[];
  experimentMode?: ExperimentMode;
};

type TokenStats = {
  system_tokens: number;
  message_tokens: number;
  total_tokens: number;
};

type AppStore = {
  sessions: SessionSummary[];
  currentSessionId: string | null;
  messages: Message[];
  isStreaming: boolean;
  isLoadingSession: boolean;
  error: string | null;
  stopGeneration: () => void;
  ragMode: boolean;
  skills: Array<{ name: string; description: string; path: string }>;
  editableFiles: string[];
  inspectorPath: string;
  inspectorContent: string;
  inspectorDirty: boolean;
  sidebarWidth: number;
  inspectorWidth: number;
  tokenStats: TokenStats | null;
  experimentMode: ExperimentMode;
  recommendations: Recommendation[];
  setExperimentMode: (mode: ExperimentMode) => void;
  refreshRecommendations: () => Promise<void>;
  createNewSession: () => Promise<void>;
  selectSession: (sessionId: string) => Promise<void>;
  sendMessage: (value: string, attachments?: ChatAttachment[]) => Promise<boolean>;
  toggleRagMode: () => Promise<void>;
  renameCurrentSession: (title: string) => Promise<void>;
  removeSession: (sessionId: string) => Promise<void>;
  loadInspectorFile: (path: string) => Promise<void>;
  updateInspectorContent: (value: string) => void;
  saveInspector: () => Promise<void>;
  compressCurrentSession: () => Promise<void>;
  setSidebarWidth: (width: number) => void;
  setInspectorWidth: (width: number) => void;
};

const FIXED_FILES = [
  "workspace/SOUL.md",
  "workspace/IDENTITY.md",
  "workspace/USER.md",
  "workspace/AGENTS.md",
  "memory_module_v1/long_term_memory/MEMORY.md",
  "SKILLS_SNAPSHOT.md"
];

const StoreContext = createContext<AppStore | null>(null);

function makeId() {
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

export function toUiMessages(history: Awaited<ReturnType<typeof getSessionHistory>>["messages"]): Message[] {
  return history.map((message) => ({
    id: message.id ?? makeId(),
    role: message.role,
    content: message.content ?? "",
    toolCalls: message.tool_calls ?? [],
    retrievals: (message.retrievals ?? []).flatMap((item): RetrievalResult[] => {
      if ("type" in item && item.type === "retrieval") return item.results;
      if ("type" in item && item.type === "retrieval_v2") {
        return [{ text: item.context, score: 1, source: "memory_v2 (shared)" }];
      }
      return [item as RetrievalResult];
    }),
    retrievalEvents: (message.retrievals ?? []).filter((item): item is RetrievalEvent => "type" in item),
    roleOpinions: message.role_opinions,
    routing: message.routing,
    guardianBlocked: message.guardian_blocked,
    progress: message.progress,
    attachments: message.attachments,
    collaborationMetric: message.collaboration_metric,
    runId: message.run_id,
    toolCallId: message.tool_call_id,
    status: message.status,
    failedRoles: message.failed_roles,
    experimentMode: message.experiment_mode
  }));
}

export function AppProvider({ children }: { children: ReactNode }) {
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [isLoadingSession, setIsLoadingSession] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const mounted = useRef(true);
  const selection = useRef({ id: null as string | null, revision: 0 });
  const historyRequest = useRef(0);
  const sessionListRequest = useRef(0);
  const loadingSession = useRef(false);
  const activeRun = useRef<{ controller: AbortController; sessionId: string | null } | null>(null);
  const [ragMode, setRagModeState] = useState(false);
  const [skills, setSkills] = useState<Array<{ name: string; description: string; path: string }>>([]);
  const [inspectorPath, setInspectorPath] = useState(
    "memory_module_v1/long_term_memory/MEMORY.md"
  );
  const [inspectorContent, setInspectorContent] = useState("");
  const [inspectorDirty, setInspectorDirty] = useState(false);
  const [sidebarWidth, setSidebarWidth] = useState(308);
  const [inspectorWidth, setInspectorWidth] = useState(360);
  const [tokenStats, setTokenStats] = useState<TokenStats | null>(null);
  const [experimentMode, setExperimentMode] = useState<ExperimentMode>("multi_full");
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);

  const editableFiles = useMemo(
    () => [...FIXED_FILES, ...skills.map((skill) => skill.path)],
    [skills]
  );

  function reportError(cause: unknown) {
    if (mounted.current) setError(cause instanceof Error ? cause.message : String(cause));
  }

  async function refreshSessions() {
    const request = ++sessionListRequest.current;
    const result = await listSessions();
    if (mounted.current && request === sessionListRequest.current) setSessions(result);
    return result;
  }

  async function refreshSkills() {
    setSkills(await listSkills());
  }

  function activateSession(sessionId: string | null) {
    selection.current = { id: sessionId, revision: selection.current.revision + 1 };
    ++historyRequest.current;
    loadingSession.current = false;
    setIsLoadingSession(false);
    setCurrentSessionId(sessionId);
    setMessages([]);
    setTokenStats(null);
    setError(null);
    return selection.current.revision;
  }

  async function refreshSessionDetails(sessionId: string, revision = selection.current.revision) {
    if (!mounted.current || selection.current.id !== sessionId || selection.current.revision !== revision) return;
    const request = ++historyRequest.current;
    loadingSession.current = true;
    setIsLoadingSession(true);
    try {
      const [history, tokens] = await Promise.all([
        getSessionHistory(sessionId), getSessionTokens(sessionId)
      ]);
      if (mounted.current && selection.current.id === sessionId &&
          selection.current.revision === revision && request === historyRequest.current) {
        setMessages(toUiMessages(history.messages));
        setTokenStats(tokens);
      }
    } catch (cause) {
      if (selection.current.id === sessionId && selection.current.revision === revision &&
          request === historyRequest.current) reportError(cause);
    } finally {
      if (mounted.current && request === historyRequest.current) {
        loadingSession.current = false;
        setIsLoadingSession(false);
      }
    }
  }

  async function createNewSession() {
    const revision = selection.current.revision;
    const created = await createSession();
    if (mounted.current && selection.current.revision === revision) activateSession(created.id);
    await refreshSessions();
  }

  async function selectSession(sessionId: string) {
    const revision = activateSession(sessionId);
    await refreshSessionDetails(sessionId, revision);
  }

  function stopGeneration() {
    activeRun.current?.controller.abort();
  }

  async function sendMessage(value: string, attachments: ChatAttachment[] = []) {
    const nextValue = value.trim() || (attachments.length ? "请结合上传的病理图像进行 MDT 会诊。" : "");
    if (!nextValue || activeRun.current || loadingSession.current) return false;

    // Lock synchronously, before even creating a session: React state alone
    // cannot prevent two submissions in the same render/event turn.
    const run = { controller: new AbortController(), sessionId: selection.current.id };
    activeRun.current = run;
    setIsStreaming(true);
    setError(null);
    let revision = selection.current.revision;
    let receivedEvent = false;
    let assistantId: string | null = null;
    let userId: string | null = null;
    let completed = false;
    let terminalError = false;
    const isActiveView = () => mounted.current && selection.current.id === run.sessionId &&
      selection.current.revision === revision;
    const patchAssistant = (updater: (message: Message) => Message) => {
      if (isActiveView()) {
        setMessages((prev) => prev.map((message) => message.id === assistantId ? updater(message) : message));
      }
    };

    try {
      if (!run.sessionId) {
        const created = await createSession();
        if (!mounted.current || revision !== selection.current.revision || run.controller.signal.aborted) return false;
        run.sessionId = created.id;
        revision = activateSession(created.id);
        void refreshSessions().catch(reportError);
      }
      if (run.controller.signal.aborted) return false;
      // Invalidate a pending refresh before appending optimistic messages.
      ++historyRequest.current;
      const userMessage: Message = {
        id: makeId(), role: "user", content: nextValue, toolCalls: [], retrievals: [],
        attachments: attachments.length ? attachments : undefined
      };
      const assistantMessage: Message = {
        id: makeId(), role: "assistant", content: "", toolCalls: [], retrievals: [], experimentMode
      };
      userId = userMessage.id;
      assistantId = assistantMessage.id;
      if (isActiveView()) setMessages((prev) => [...prev, userMessage, assistantMessage]);

      await streamChat(
        { message: nextValue, session_id: run.sessionId, experiment_mode: experimentMode, attachments },
        {
          onEvent(event, data) {
            if (event === "error") terminalError = true;
            if (event !== "error" && event !== "title") receivedEvent = true;
            if (data.run_id) patchAssistant((message) => ({ ...message, runId: String(data.run_id) }));
            if (event === "retrieval") {
              patchAssistant((message) => ({
                ...message,
                retrievals: (data.results as RetrievalResult[]) ?? []
              }));
              return;
            }

            if (event === "retrieval_v2") {
              patchAssistant((message) => ({
                ...message,
                retrievals: [
                  {
                    text: String(data.context ?? ""),
                    score: 1,
                    source: "memory_v2 (shared)"
                  }
                ]
              }));
              return;
            }

            if (event === "role_opinion") {
              const opinion: RoleOpinion = {
                role: String(data.role ?? ""),
                role_label: String(data.role_label ?? ""),
                content: String(data.content ?? ""),
                round: Number(data.round ?? 1),
                evidence: (data.evidence as string[]) ?? [],
                tool_calls: data.tool_calls as ToolCall[] | undefined
              };
              patchAssistant((message) => ({
                ...message,
                roleOpinions: [...(message.roleOpinions ?? []), opinion]
              }));
              return;
            }

            if (event === "routing") {
              patchAssistant((message) => ({
                ...message,
                routing: {
                  roles: (data.roles as string[]) ?? [],
                  reason: String(data.reason ?? "")
                }
              }));
              return;
            }

            if (event === "progress") {
              const progress: ProgressEvent = {
                stage: String(data.stage ?? ""),
                status: String(data.status ?? ""),
                label: String(data.label ?? "")
              };
              patchAssistant((message) => {
                const existing = message.progress ?? [];
                const index = existing.findIndex((item) => item.stage === progress.stage);
                const next =
                  index >= 0
                    ? existing.map((item, idx) => (idx === index ? progress : item))
                    : [...existing, progress];
                return { ...message, progress: next };
              });
              return;
            }

            if (event === "guardian_blocked") {
              patchAssistant((message) => ({
                ...message,
                guardianBlocked: {
                  reason: String(data.reason ?? ""),
                  message: String(data.message ?? "请求被安全守卫拦截")
                }
              }));
              return;
            }

            if (event === "synthesis_token") {
              patchAssistant((message) => ({
                ...message,
                content: `${message.content}${String(data.content ?? "")}`
              }));
              return;
            }

            if (event === "token") {
              patchAssistant((message) => ({
                ...message,
                content: `${message.content}${String(data.content ?? "")}`
              }));
              return;
            }

            if (event === "tool_start") {
              const callId = data.tool_call_id ? String(data.tool_call_id) : undefined;
              patchAssistant((message) => ({
                ...message,
                toolCalls: [
                  ...message.toolCalls.filter((call) => !callId || call.tool_call_id !== callId),
                  { tool_call_id: callId, status: String(data.status ?? "running"), tool: String(data.tool ?? "tool"), input: String(data.input ?? ""), output: "" }
                ]
              }));
              return;
            }

            if (event === "tool_end") {
              const callId = data.tool_call_id ? String(data.tool_call_id) : undefined;
              patchAssistant((message) => {
                // Legacy servers without IDs are sequential. Never attach an
                // unknown ID to the most recent (possibly unrelated) call.
                const index = callId
                  ? message.toolCalls.findIndex((call) => call.tool_call_id === callId)
                  : message.toolCalls.reduce((last, call, i) => !call.tool_call_id && !call.output ? i : last, -1);
                const result: ToolCall = {
                  tool_call_id: callId, status: String(data.status ?? "success"), tool: String(data.tool ?? "tool"),
                  input: String(data.input ?? ""), output: String(data.output ?? "")
                };
                return {
                  ...message,
                  toolCalls: index < 0 ? [...message.toolCalls, result] : message.toolCalls.map((call, i) =>
                    i === index ? { ...call, output: result.output, status: result.status } : call)
                };
              });
              return;
            }

            if (event === "new_response") {
              // Keep tool/result events and the final answer grouped in one assistant card.
              // The backend emits new_response after tool calls; splitting here creates
              // many visually noisy tool-only cards.
              return;
            }

            if (event === "collaboration_metric") {
              patchAssistant((message) => ({ ...message, collaborationMetric: data }));
              return;
            }

            if (event === "role_error") {
              const role = String(data.role ?? "unknown");
              patchAssistant((message) => ({
                ...message,
                status: "partial",
                failedRoles: [...(message.failedRoles ?? []), {
                  role, stage: String(data.stage ?? ""), round: data.round as number | undefined,
                  error: String(data.error ?? data.message ?? "专家意见生成失败")
                }],
                progress: [...(message.progress ?? []), {
                  stage: `role_error_${role}`, status: "error",
                  label: `${role}: ${String(data.error ?? data.message ?? "专家意见生成失败")}`
                }]
              }));
              return;
            }

            if (event === "done") {
              completed = true;
              patchAssistant((message) => ({
                ...message,
                content: String(data.content ?? message.content),
                status: (data.status as RunStatus) ?? message.status ?? "success",
                failedRoles: (data.failed_roles as RoleFailure[]) ?? message.failedRoles,
                runId: data.run_id ? String(data.run_id) : message.runId
              }));
              return;
            }

            if (event === "title") {
              void refreshSessions().catch(reportError);
            }
          }
        },
        run.controller.signal
      );
      return true;
    } catch (cause) {
      completed = false;
      const cancelled = run.controller.signal.aborted;
      if (cancelled || receivedEvent) {
        patchAssistant((message) => ({
          ...message,
          status: cancelled ? "cancelled" : terminalError ? "failed" : "interrupted",
          toolCalls: message.toolCalls.map((call) => call.status === "running"
            ? { ...call, status: cancelled ? "cancelled" : "failed" } : call),
          progress: [...(message.progress ?? []), {
            stage: "stream", status: cancelled ? "blocked" : "error",
            label: cancelled ? "生成已停止" : `生成中断: ${cause instanceof Error ? cause.message : String(cause)}`
          }]
        }));
      } else if (isActiveView()) {
        // HTTP/setup failures were not accepted: keep the draft, remove the
        // optimistic pair, and report a useful error instead of rejecting a
        // fire-and-forget UI callback.
        setMessages((prev) => prev.filter((message) => message.id !== userId && message.id !== assistantId));
      }
      if (!cancelled && isActiveView()) reportError(cause);
      return receivedEvent || cancelled;
    } finally {
      if (activeRun.current === run) {
        activeRun.current = null;
        if (mounted.current) setIsStreaming(false);
      }
      if (mounted.current) {
        void refreshSessions().catch(reportError);
        // A successful backend replay is authoritative, but only for the same
        // selection revision. Never erase partial/cancelled local output.
        if (completed && !run.controller.signal.aborted && isActiveView() && run.sessionId) {
          void refreshSessionDetails(run.sessionId, revision);
        }
      }
    }
  }

  async function toggleRagMode() {
    const next = !ragMode;
    setRagModeState(next);
    try {
      await setRagMode(next);
    } catch (error) {
      setRagModeState(!next);
      throw error;
    }
  }

  async function renameCurrentSession(title: string) {
    if (!currentSessionId || !title.trim()) {
      return;
    }
    await renameSession(currentSessionId, title.trim());
    await refreshSessions();
  }

  async function removeSession(sessionId: string) {
    await deleteSession(sessionId);
    await refreshSessions();
    if (selection.current.id === sessionId) {
      const nextSessions = await listSessions();
      setSessions(nextSessions);
      if (nextSessions.length) {
        await selectSession(nextSessions[0].id);
      } else {
        activateSession(null);
      }
    }
  }

  async function loadInspectorFile(path: string) {
    setInspectorPath(path);
    const file = await loadFile(path);
    setInspectorContent(file.content);
    setInspectorDirty(false);
  }

  function updateInspectorContent(value: string) {
    setInspectorContent(value);
    setInspectorDirty(true);
  }

  async function saveInspector() {
    await saveFile(inspectorPath, inspectorContent);
    setInspectorDirty(false);
    await refreshSkills();
  }

  async function compressCurrentSession() {
    if (!currentSessionId) {
      return;
    }
    await compressSession(currentSessionId);
    await refreshSessionDetails(currentSessionId);
    await refreshSessions();
  }

  async function refreshRecommendations() {
    try {
      const result = await getRecommendations();
      setRecommendations(result.recommendations ?? []);
    } catch (error) {
      console.warn("Failed to load recommendations", error);
      setRecommendations([]);
    }
  }

  useEffect(() => {
    let disposed = false;
    mounted.current = true;
    const initialRevision = selection.current.revision;
    loadingSession.current = true;
    setIsLoadingSession(true);
    void (async () => {
      const [initialSessions, rag, initialSkills] = await Promise.all([
        listSessions(), getRagMode(), listSkills()
      ]);
      if (disposed) return;
      setSessions(initialSessions);
      setRagModeState(rag.enabled);
      setSkills(initialSkills);
      if (selection.current.revision === initialRevision) {
        if (initialSessions.length) {
          await selectSession(initialSessions[0].id);
        } else {
          const created = await createSession();
          if (disposed) return;
          if (selection.current.revision === initialRevision) activateSession(created.id);
          setSessions([created]);
        }
      }
      const file = await loadFile("memory_module_v1/long_term_memory/MEMORY.md");
      if (disposed) return;
      setInspectorPath(file.path);
      setInspectorContent(file.content);
      void refreshRecommendations();
    })().catch((cause) => {
      if (!disposed) {
        loadingSession.current = false;
        setIsLoadingSession(false);
        reportError(cause);
      }
    });
    return () => {
      disposed = true;
      mounted.current = false;
      activeRun.current?.controller.abort();
      // This is a request sequence, not a DOM ref; invalidation is intentional.
      // eslint-disable-next-line react-hooks/exhaustive-deps
      ++historyRequest.current;
    };
    // Initialization runs once; all async commits are guarded against stale selection/unmount.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function guarded<Args extends unknown[]>(action: (...args: Args) => Promise<void>) {
    return async (...args: Args) => {
      try { await action(...args); } catch (cause) { reportError(cause); }
    };
  }

  const value: AppStore = {
    sessions,
    currentSessionId,
    messages,
    isStreaming,
    isLoadingSession,
    error,
    stopGeneration,
    ragMode,
    skills,
    editableFiles,
    inspectorPath,
    inspectorContent,
    inspectorDirty,
    sidebarWidth,
    inspectorWidth,
    tokenStats,
    experimentMode,
    recommendations,
    setExperimentMode,
    refreshRecommendations,
    createNewSession: guarded(createNewSession),
    selectSession: guarded(selectSession),
    sendMessage,
    toggleRagMode: guarded(toggleRagMode),
    renameCurrentSession: guarded(renameCurrentSession),
    removeSession: guarded(removeSession),
    loadInspectorFile: guarded(loadInspectorFile),
    updateInspectorContent,
    saveInspector: guarded(saveInspector),
    compressCurrentSession: guarded(compressCurrentSession),
    setSidebarWidth,
    setInspectorWidth
  };

  return <StoreContext.Provider value={value}>{children}</StoreContext.Provider>;
}

export function useAppStore() {
  const value = useContext(StoreContext);
  if (!value) {
    throw new Error("useAppStore must be used inside AppProvider");
  }
  return value;
}
