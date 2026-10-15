import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { HistoryMessage, SessionHistory } from "@/lib/api";
import { AppProvider, toUiMessages, useAppStore } from "@/lib/store";

const encoder = new TextEncoder();
const last = <T,>(items: T[]) => items[items.length - 1];
const session = (id: string) => ({ id, title: id, created_at: 1, updated_at: 1, message_count: 1 });
const json = (value: unknown) => new Response(JSON.stringify(value), { headers: { "Content-Type": "application/json" } });
const history = (id: string, messages?: HistoryMessage[]): SessionHistory => ({
  ...session(id), messages: messages ?? [{ id: `message-${id}`, role: "assistant", content: `history ${id}` }]
});
function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => { resolve = done; });
  return { promise, resolve };
}
function liveStream() {
  let writer!: ReadableStreamDefaultController<Uint8Array>;
  const cancel = vi.fn();
  const body = new ReadableStream<Uint8Array>({ start(controller) { writer = controller; }, cancel });
  return {
    body, cancel,
    response: new Response(body),
    event(event: string, data: Record<string, unknown>) {
      writer.enqueue(encoder.encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`));
    },
    close: () => writer.close()
  };
}
let histories: Map<string, () => Response | Promise<Response>>;
let chat: () => Response | Promise<Response>;
let mockFetch: ReturnType<typeof vi.fn>;

beforeEach(() => {
  histories = new Map();
  chat = () => json({});
  mockFetch = vi.fn(async (input: string, init?: RequestInit) => {
    const path = new URL(input, "http://localhost").pathname.replace(/^\/api/, "");
    if (path === "/sessions") return json(init?.method === "POST" ? session("new") : [session("a"), session("b")]);
    if (path === "/config/rag-mode") return json({ enabled: false });
    if (path === "/skills") return json([]);
    if (path === "/files") return json({ path: "MEMORY.md", content: "memory" });
    if (path === "/recommend") return json({ recommendations: [] });
    if (path.startsWith("/tokens/session/")) return json({ total_tokens: path.endsWith("a") ? 10 : 20, message_tokens: 1, system_tokens: 1 });
    const match = path.match(/^\/sessions\/([^/]+)\/history$/);
    if (match) return histories.get(match[1])?.() ?? json(history(match[1]));
    if (path === "/chat") return chat();
    throw new Error(`Unexpected request: ${input}`);
  });
  vi.stubGlobal("fetch", mockFetch);
});
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

function renderStore() {
  return renderHook(() => useAppStore(), { wrapper: ({ children }: { children: ReactNode }) => <AppProvider>{children}</AppProvider> });
}
async function ready(result: ReturnType<typeof renderStore>["result"]) {
  await waitFor(() => {
    expect(result.current.currentSessionId).toBe("a");
    expect(result.current.messages[0]?.content).toBe("history a");
    expect(result.current.isLoadingSession).toBe(false);
  });
}

describe("production store session isolation", () => {
  it("rejects out-of-order history across rapid A/B/A switching, including same-session stale loads", async () => {
    const { result } = renderStore();
    await ready(result);
    const staleA = deferred<Response>();
    const staleB = deferred<Response>();
    const latestA = deferred<Response>();
    histories.set("a", () => staleA.promise);
    let first!: Promise<void>;
    act(() => { first = result.current.selectSession("a"); });
    histories.set("b", () => staleB.promise);
    let second!: Promise<void>;
    act(() => { second = result.current.selectSession("b"); });
    histories.set("a", () => latestA.promise);
    let third!: Promise<void>;
    act(() => { third = result.current.selectSession("a"); });
    await act(async () => {
      latestA.resolve(json(history("a", [{ id: "latest", role: "assistant", content: "latest A" }])));
      await third;
    });
    await act(async () => {
      staleB.resolve(json(history("b"))); staleA.resolve(json(history("a")));
      await Promise.all([first, second]);
    });
    expect(result.current.currentSessionId).toBe("a");
    expect(result.current.messages.map((message) => message.content)).toEqual(["latest A"]);
    expect(result.current.tokenStats?.total_tokens).toBe(10);
  });

  it("does not let an old stream finish or its history refresh overwrite another selected session", async () => {
    const { result } = renderStore();
    await ready(result);
    const stream = liveStream();
    chat = () => stream.response;
    let running!: Promise<boolean>;
    act(() => { running = result.current.sendMessage("question a"); });
    await waitFor(() => expect(mockFetch.mock.calls.filter(([url]) => String(url).endsWith("/chat"))).toHaveLength(1));
    act(() => stream.event("token", { content: "answer a" }));
    await act(async () => { await result.current.selectSession("b"); });
    await act(async () => {
      stream.event("done", { content: "answer a", status: "success" });
      stream.event("title", { title: "renamed a" }); stream.close();
      await running;
    });
    expect(result.current.currentSessionId).toBe("b");
    expect(result.current.messages.map((message) => message.content)).toEqual(["history b"]);
    expect(result.current.tokenStats?.total_tokens).toBe(20);
    expect(mockFetch.mock.calls.filter(([url]) => String(url).endsWith("/sessions/a/history"))).toHaveLength(1);
    expect(result.current.isStreaming).toBe(false);
  });

  it("uses a synchronous send lock and cancellation retains partial output without authoritative replay", async () => {
    const { result } = renderStore();
    await ready(result);
    const stream = liveStream(); chat = () => stream.response;
    let running!: Promise<boolean>;
    let duplicate!: Promise<boolean>;
    act(() => {
      running = result.current.sendMessage("first");
      duplicate = result.current.sendMessage("duplicate");
    });
    await expect(duplicate).resolves.toBe(false);
    await waitFor(() => expect(result.current.isStreaming).toBe(true));
    act(() => {
      stream.event("tool_start", { tool_call_id: "pending", tool: "search", input: "query" });
      stream.event("token", { content: "partial result" });
    });
    await waitFor(() => expect(last(result.current.messages)?.content).toBe("partial result"));
    await act(async () => { result.current.stopGeneration(); await running; });
    expect(stream.cancel).toHaveBeenCalledOnce();
    expect(stream.body.locked).toBe(false);
    expect(last(result.current.messages)).toMatchObject({ content: "partial result", status: "cancelled" });
    expect(last(result.current.messages)?.toolCalls[0]).toMatchObject({ tool_call_id: "pending", status: "cancelled" });
    expect(result.current.isStreaming).toBe(false);
    expect(mockFetch.mock.calls.filter(([url]) => String(url).endsWith("/chat"))).toHaveLength(1);
    expect(mockFetch.mock.calls.filter(([url]) => String(url).endsWith("/sessions/a/history"))).toHaveLength(1);
  });

  it("matches overlapping tool outputs by tool_call_id and catches truncated EOF without erasing evidence", async () => {
    const { result } = renderStore(); await ready(result);
    const stream = liveStream(); chat = () => stream.response;
    let running!: Promise<boolean>;
    act(() => { running = result.current.sendMessage("tools"); });
    await waitFor(() => expect(result.current.isStreaming).toBe(true));
    await act(async () => {
      stream.event("tool_start", { tool_call_id: "one", tool: "search", input: "first" });
      stream.event("tool_start", { tool_call_id: "two", tool: "search", input: "second" });
      stream.event("tool_end", { tool_call_id: "one", output: "first output" });
      stream.event("tool_end", { tool_call_id: "two", output: "second output", status: "failed" });
      stream.event("tool_end", { tool_call_id: "unknown", tool: "third", output: "own result" });
      stream.event("tool_start", { tool_call_id: "pending", tool: "fourth", input: "pending" });
      stream.event("token", { content: "partial" }); stream.close(); await running;
    });
    const assistant = last(result.current.messages)!;
    expect(assistant.toolCalls.map((call) => [call.tool_call_id, call.output, call.status])).toEqual([
      ["one", "first output", "success"], ["two", "second output", "failed"],
      ["unknown", "own result", "success"], ["pending", "", "failed"]
    ]);
    expect(assistant.status).toBe("interrupted");
    expect(assistant.content).toBe("partial");
    expect(result.current.error).toMatch(/before done/);
    expect(result.current.isStreaming).toBe(false);
  });

  it("catches HTTP and terminal stream failures, removing only immediately rejected optimistic messages", async () => {
    const { result } = renderStore(); await ready(result);
    chat = () => new Response("offline", { status: 503 });
    let accepted = true;
    await act(async () => { accepted = await result.current.sendMessage("retain draft"); });
    expect(accepted).toBe(false);
    expect(result.current.messages).toHaveLength(1);
    expect(result.current.error).toBe("offline");
    const stream = liveStream(); chat = () => stream.response;
    let running!: Promise<boolean>;
    act(() => { running = result.current.sendMessage("fail later"); });
    await act(async () => {
      stream.event("token", { content: "evidence" });
      stream.event("error", { error: "terminal error" });
      await running;
    });
    expect(last(result.current.messages)).toMatchObject({ content: "evidence", status: "failed" });
    expect(result.current.error).toBe("terminal error");
  });

  it("catches initialization errors instead of leaving an unhandled rejection", async () => {
    mockFetch.mockRejectedValue(new Error("initial network failed"));
    const { result } = renderStore();
    await waitFor(() => expect(result.current.error).toBe("initial network failed"));
    expect(result.current.isLoadingSession).toBe(false);
  });

  it("replays persisted metadata, stable IDs, partial status and tool IDs through the real store", async () => {
    const message: HistoryMessage = {
      id: "stable", role: "assistant", content: "persisted",
      role_opinions: [{ role: "surgeon", role_label: "外科", content: "opinion", round: 2, evidence: ["ref"] }],
      routing: { roles: ["surgeon"], reason: "routing reason" },
      retrievals: [
        { type: "retrieval", query: "case", results: [{ text: "evidence", score: 0.8, source: "memory" }] },
        { type: "retrieval_v2", query: "case", context: "shared context" }
      ],
      progress: [{ stage: "synthesis", status: "done", label: "complete" }],
      attachments: [{ type: "image", name: "pathology.png", content_type: "image/png", url: "data:image/png;base64,AA==" }],
      guardian_blocked: { reason: "safety", message: "blocked" },
      collaboration_metric: { agreement: 0.8 }, run_id: "run-1", tool_call_id: "message-call",
      tool_calls: [{ tool_call_id: "call-1", tool: "search", input: "query", output: "result" }],
      status: "partial", failed_roles: [{ role: "oncologist", stage: "round2", round: 2, error: "timed out" }], experiment_mode: "multi_full"
    };
    histories.set("a", () => json(history("a", [message])));
    const { result } = renderStore();
    await waitFor(() => expect(result.current.messages[0]?.id).toBe("stable"));
    expect(result.current.messages[0]).toEqual({
      id: message.id, role: message.role, content: message.content,
      toolCalls: message.tool_calls,
      retrievals: [{ text: "evidence", score: 0.8, source: "memory" }, { text: "shared context", score: 1, source: "memory_v2 (shared)" }],
      retrievalEvents: message.retrievals, roleOpinions: message.role_opinions,
      routing: message.routing, guardianBlocked: message.guardian_blocked, progress: message.progress,
      attachments: message.attachments, collaborationMetric: message.collaboration_metric, runId: message.run_id,
      toolCallId: message.tool_call_id, status: message.status, failedRoles: message.failed_roles, experimentMode: message.experiment_mode
    });
    expect(toUiMessages([message])[0].id).toBe("stable");
  });
});
