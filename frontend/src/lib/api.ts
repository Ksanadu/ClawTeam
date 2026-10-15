export type ToolCall = {
  tool_call_id?: string;
  status?: string;
  tool: string;
  input: string;
  output: string;
};

export type ChatAttachment = {
  type: "image";
  name: string;
  content_type: string;
  url: string;
  size?: number;
};

export type RetrievalResult = {
  text: string;
  score: number;
  source: string;
};

export type RoleOpinion = {
  role: string;
  role_label: string;
  content: string;
  round?: number;
  evidence?: string[];
  tool_calls?: ToolCall[];
};

export type RoutingInfo = {
  roles: string[];
  reason: string;
};

export type ProgressEvent = {
  stage: string;
  status: "pending" | "running" | "done" | "error" | "blocked" | string;
  label: string;
};

export type TumorBoardCase = {
  id: string;
  tumor_type: string;
  subtype?: string;
  stage?: string;
  biomarker?: string;
  title: string;
  case: string;
  tags?: string[];
  complexity?: string;
};

export type Recommendation = {
  id: string;
  text: string;
  reason?: string;
  case_data?: TumorBoardCase | null;
  score?: number;
};

export type ExperimentMode =
  | "single"
  | "multi_no_memory"
  | "multi_memory"
  | "multi_full";

export type SessionSummary = {
  id: string;
  title: string;
  created_at: number;
  updated_at: number;
  message_count: number;
};

export type RunStatus = "running" | "success" | "partial" | "failed" | "blocked" | "interrupted" | "cancelled";
export type RoleFailure = string | { role: string; stage?: string; round?: number; error?: string };
export type RetrievalEvent = { type: "retrieval"; query?: string; results: RetrievalResult[] } |
  { type: "retrieval_v2"; context: string; [key: string]: unknown };

export type HistoryMessage = {
  id?: string;
  role: "user" | "assistant";
  content: string;
  tool_calls?: ToolCall[];
  tool_call_id?: string;
  role_opinions?: RoleOpinion[];
  routing?: RoutingInfo | null;
  retrievals?: Array<RetrievalResult | RetrievalEvent>;
  progress?: ProgressEvent[];
  attachments?: ChatAttachment[];
  guardian_blocked?: { reason: string; message: string } | null;
  collaboration_metric?: Record<string, unknown> | null;
  run_id?: string;
  status?: RunStatus;
  failed_roles?: RoleFailure[];
  experiment_mode?: ExperimentMode;
};

export type SessionHistory = {
  id: string;
  title: string;
  created_at: number;
  updated_at: number;
  compressed_context?: string;
  messages: HistoryMessage[];
};

export type StreamHandlers = {
  onEvent: (event: string, data: Record<string, unknown>) => void;
};

export function getApiBase() {
  // A relative base (e.g. /api) supports a same-origin reverse proxy. Next inlines
  // NEXT_PUBLIC_* values at build time; an explicitly empty base also means /api.
  const configured = process.env.NEXT_PUBLIC_API_BASE_URL;
  if (configured !== undefined) {
    return (configured.trim() || "/api").replace(/\/+$/, "");
  }
  if (typeof window === "undefined") {
    return "http://127.0.0.1:8002/api";
  }
  const { protocol, hostname, port } = window.location;
  const isLocal = ["localhost", "127.0.0.1", "[::1]"].includes(hostname);
  if (protocol === "http:" && (isLocal || port === "7788" || port === "3000")) {
    return `${protocol}//${hostname}:8002/api`;
  }
  // Never downgrade an HTTPS page to the legacy HTTP development backend.
  return "/api";
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${getApiBase()}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {})
    }
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed: ${response.status}`);
  }

  return (await response.json()) as T;
}

export async function listSessions() {
  return request<SessionSummary[]>("/sessions");
}

export async function createSession(title = "新会诊") {
  return request<SessionSummary>("/sessions", {
    method: "POST",
    body: JSON.stringify({ title })
  });
}

export async function renameSession(sessionId: string, title: string) {
  return request<SessionSummary>(`/sessions/${sessionId}`, {
    method: "PUT",
    body: JSON.stringify({ title })
  });
}

export async function deleteSession(sessionId: string) {
  return request<{ ok: boolean }>(`/sessions/${sessionId}`, {
    method: "DELETE"
  });
}

export async function getSessionHistory(sessionId: string) {
  return request<SessionHistory>(`/sessions/${sessionId}/history`);
}

export async function getSessionTokens(sessionId: string) {
  return request<{
    system_tokens: number;
    message_tokens: number;
    total_tokens: number;
  }>(`/tokens/session/${sessionId}`);
}

export async function listSkills() {
  return request<Array<{ name: string; description: string; path: string }>>("/skills");
}

export async function loadFile(path: string) {
  return request<{ path: string; content: string }>(
    `/files?path=${encodeURIComponent(path)}`
  );
}

export async function saveFile(path: string, content: string) {
  return request<{ ok: boolean; path: string }>("/files", {
    method: "POST",
    body: JSON.stringify({ path, content })
  });
}

export async function getRagMode() {
  return request<{ enabled: boolean }>("/config/rag-mode");
}

export async function setRagMode(enabled: boolean) {
  return request<{ enabled: boolean }>("/config/rag-mode", {
    method: "PUT",
    body: JSON.stringify({ enabled })
  });
}

export async function compressSession(sessionId: string) {
  return request<{ archived_count: number; remaining_count: number }>(
    `/sessions/${sessionId}/compress`,
    { method: "POST" }
  );
}

export async function getRecommendations() {
  return request<{ recommendations: Recommendation[] }>("/recommend");
}

export async function streamChat(
  payload: {
    message: string;
    session_id: string;
    experiment_mode?: ExperimentMode;
    attachments?: ChatAttachment[];
  },
  handlers: StreamHandlers,
  signal?: AbortSignal
) {
  const response = await fetch(`${getApiBase()}/chat`, {
    method: "POST",
    signal,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      ...payload,
      stream: true,
      experiment_mode: payload.experiment_mode ?? "single"
    })
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Chat request failed: ${response.status}`);
  }
  if (!response.body) {
    throw new Error("Chat response has no readable stream");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8", { fatal: true });
  let line = "";
  let skipLF = false;
  let event = "message";
  let dataLines: string[] = [];
  let receivedDone = false;
  const throwIfAborted = () => {
    if (signal?.aborted) {
      throw new DOMException("Generation cancelled", "AbortError");
    }
  };
  const cancelReader = () => { void reader.cancel().catch(() => {}); };

  const consumeLine = () => {
    if (!line) {
      if (dataLines.length) {
        const eventName = event;
        let data: unknown;
        try {
          data = JSON.parse(dataLines.join("\n"));
        } catch {
          throw new Error(`Invalid JSON in SSE ${eventName} event`);
        }
        if (!data || typeof data !== "object" || Array.isArray(data)) {
          throw new Error(`Invalid payload in SSE ${eventName} event`);
        }
        handlers.onEvent(eventName, data as Record<string, unknown>);
        if (eventName === "error") {
          const error = data as Record<string, unknown>;
          throw new Error(String(error.error ?? error.message ?? "Chat generation failed"));
        }
        if (eventName === "done") receivedDone = true;
      }
      event = "message";
      dataLines = [];
    } else if (!line.startsWith(":")) {
      const colon = line.indexOf(":");
      const field = colon < 0 ? line : line.slice(0, colon);
      let value = colon < 0 ? "" : line.slice(colon + 1);
      if (value.startsWith(" ")) value = value.slice(1);
      if (field === "event") event = value || "message";
      if (field === "data") dataLines.push(value);
    }
    line = "";
  };
  const consume = (text: string) => {
    // Handle CRLF even when CR and LF arrive in different byte chunks. A CR or
    // LF ends a line; only a blank line dispatches an event (never EOF itself).
    for (const char of text) {
      if (skipLF && char === "\n") { skipLF = false; continue; }
      skipLF = false;
      if (char === "\r" || char === "\n") {
        consumeLine();
        skipLF = char === "\r";
      } else {
        line += char;
      }
    }
  };

  signal?.addEventListener("abort", cancelReader, { once: true });
  try {
    throwIfAborted();
    while (true) {
      const { value, done } = await reader.read();
      throwIfAborted();
      consume(decoder.decode(value, { stream: !done }));
      if (done) break;
    }
    if (!receivedDone) {
      throw new Error("Chat stream interrupted before done (truncated response)");
    }
    // Keep reading through EOF: the server may send title after done.
  } finally {
    signal?.removeEventListener("abort", cancelReader);
    try {
      await reader.cancel();
    } catch {
      // A failed/aborted transport may already have closed the reader.
    } finally {
      reader.releaseLock();
    }
  }
}
