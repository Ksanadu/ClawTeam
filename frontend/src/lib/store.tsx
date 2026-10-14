"use client";

import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";

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
  type RetrievalResult,
  type RoleOpinion,
  type RoutingInfo,
  type SessionSummary,
  type ToolCall
} from "@/lib/api";

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  toolCalls: ToolCall[];
  retrievals: RetrievalResult[];
  roleOpinions?: RoleOpinion[];
  routing?: RoutingInfo | null;
  guardianBlocked?: { reason: string; message: string } | null;
  progress?: ProgressEvent[];
  attachments?: ChatAttachment[];
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
  sendMessage: (value: string, attachments?: ChatAttachment[]) => Promise<void>;
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

function toUiMessages(history: Awaited<ReturnType<typeof getSessionHistory>>["messages"]) {
  return history.map((message) => ({
    id: makeId(),
    role: message.role,
    content: message.content ?? "",
    toolCalls: message.tool_calls ?? [],
    retrievals: []
  }));
}

export function AppProvider({ children }: { children: ReactNode }) {
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
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

  async function refreshSessions() {
    setSessions(await listSessions());
  }

  async function refreshSkills() {
    setSkills(await listSkills());
  }

  async function refreshSessionDetails(sessionId: string) {
    const [history, tokens] = await Promise.all([
      getSessionHistory(sessionId),
      getSessionTokens(sessionId)
    ]);
    setMessages(toUiMessages(history.messages));
    setTokenStats(tokens);
  }

  async function createNewSession() {
    const created = await createSession();
    await refreshSessions();
    setCurrentSessionId(created.id);
    setMessages([]);
    setTokenStats(null);
  }

  async function selectSession(sessionId: string) {
    setCurrentSessionId(sessionId);
    await refreshSessionDetails(sessionId);
  }

  async function ensureSession() {
    if (currentSessionId) {
      return currentSessionId;
    }

    const created = await createSession();
    setCurrentSessionId(created.id);
    await refreshSessions();
    return created.id;
  }

  async function sendMessage(value: string, attachments: ChatAttachment[] = []) {
    const nextValue =
      value.trim() || (attachments.length ? "请结合上传的病理图像进行 MDT 会诊。" : "");
    if (!nextValue || isStreaming) {
      return;
    }

    const sessionId = await ensureSession();
    const userMessage: Message = {
      id: makeId(),
      role: "user",
      content: nextValue,
      toolCalls: [],
      retrievals: [],
      attachments: attachments.length ? attachments : undefined
    };
    const assistantMessage: Message = {
      id: makeId(),
      role: "assistant",
      content: "",
      toolCalls: [],
      retrievals: []
    };

    setMessages((prev) => [...prev, userMessage, assistantMessage]);
    setIsStreaming(true);

    let activeAssistantId = assistantMessage.id;

    const patchAssistant = (updater: (message: Message) => Message) => {
      setMessages((prev) =>
        prev.map((message) => (message.id === activeAssistantId ? updater(message) : message))
      );
    };

    try {
      await streamChat(
        {
          message: nextValue,
          session_id: sessionId,
          experiment_mode: experimentMode,
          attachments
        },
        {
          onEvent(event, data) {
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
                evidence: (data.evidence as string[]) ?? []
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
              patchAssistant((message) => ({
                ...message,
                toolCalls: [
                  ...message.toolCalls,
                  {
                    tool: String(data.tool ?? "tool"),
                    input: String(data.input ?? ""),
                    output: ""
                  }
                ]
              }));
              return;
            }

            if (event === "tool_end") {
              patchAssistant((message) => {
                const updatedToolCalls = message.toolCalls.map((toolCall, index, list) =>
                  index === list.length - 1
                    ? { ...toolCall, output: String(data.output ?? "") }
                    : toolCall
                );
                
                // 去除可能从内容里混进来的工具输出文本（这里是个简单的清空/重置策略，按需可根据特殊标记清洗）
                // 如果需要确保 content 彻底为空，可以视业务修改
                return {
                  ...message,
                  toolCalls: updatedToolCalls
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

            if (event === "done") {
              const finalContent = String(data.content ?? "");
              patchAssistant((message) =>
                message.content
                  ? message
                  : {
                      ...message,
                      content: finalContent
                    }
              );
              return;
            }

            if (event === "title") {
              void refreshSessions();
              return;
            }

            if (event === "error") {
              patchAssistant((message) => ({
                ...message,
                content:
                  message.content || `请求失败: ${String(data.error ?? "unknown error")}`
              }));
            }
          }
        }
      );
    } finally {
      setIsStreaming(false);
      await refreshSessions();
      await refreshSessionDetails(sessionId);
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
    if (currentSessionId === sessionId) {
      const nextSessions = await listSessions();
      setSessions(nextSessions);
      if (nextSessions.length) {
        setCurrentSessionId(nextSessions[0].id);
        await refreshSessionDetails(nextSessions[0].id);
      } else {
        setCurrentSessionId(null);
        setMessages([]);
        setTokenStats(null);
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
    void (async () => {
      const [initialSessions, rag, initialSkills] = await Promise.all([
        listSessions(),
        getRagMode(),
        listSkills()
      ]);

      setSessions(initialSessions);
      setRagModeState(rag.enabled);
      setSkills(initialSkills);

      if (initialSessions.length) {
        setCurrentSessionId(initialSessions[0].id);
        await refreshSessionDetails(initialSessions[0].id);
      } else {
        const created = await createSession();
        setCurrentSessionId(created.id);
        setSessions([created]);
      }

      const file = await loadFile("memory_module_v1/long_term_memory/MEMORY.md");
      setInspectorPath(file.path);
      setInspectorContent(file.content);

      void refreshRecommendations();
    })();
  }, []);

  const value: AppStore = {
    sessions,
    currentSessionId,
    messages,
    isStreaming,
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
    createNewSession,
    selectSession,
    sendMessage,
    toggleRagMode,
    renameCurrentSession,
    removeSession,
    loadInspectorFile,
    updateInspectorContent,
    saveInspector,
    compressCurrentSession,
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
