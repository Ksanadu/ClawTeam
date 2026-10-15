import { afterEach, describe, expect, it, vi } from "vitest";

import { getApiBase, streamChat } from "@/lib/api";

const payload = { message: "test", session_id: "a" };
const encoder = new TextEncoder();

function respond(bytes: Uint8Array[], close = true) {
  let controller!: ReadableStreamDefaultController<Uint8Array>;
  const cancel = vi.fn();
  const body = new ReadableStream<Uint8Array>({
    start(value) {
      controller = value;
      for (const bytesChunk of bytes) value.enqueue(bytesChunk);
      if (close) value.close();
    },
    cancel
  });
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(body)));
  return { body, cancel, controller };
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});

describe("production streamChat parser", () => {
  it.each(["\n", "\r\n", "\r"])("handles %j framing at every byte split with UTF-8, multiline data and trailing title", async (newline) => {
    const wire = [
      ": keepalive", "id: 1", "event: token", 'data: {"content":', 'data: "会诊🩺"}', "",
      "event: done", 'data: {"content":"会诊🩺", "status":"success"}', "",
      "event: title", 'data: {"title":"病历"}', "", ""
    ].join(newline);
    const bytes = encoder.encode(wire);
    for (let split = 0; split <= bytes.length; split++) {
      const { body } = respond([bytes.slice(0, split), bytes.slice(split)]);
      const events: Array<[string, Record<string, unknown>]> = [];
      await streamChat(payload, { onEvent: (event, data) => events.push([event, data]) });
      expect(events.map(([event]) => event)).toEqual(["token", "done", "title"]);
      expect(events[0][1].content).toBe("会诊🩺");
      expect(body.locked).toBe(false);
    }
    respond(Array.from(bytes, (byte) => new Uint8Array([byte])));
    await expect(streamChat(payload, { onEvent: vi.fn() })).resolves.toBeUndefined();
  });

  it.each([
    'event: token\ndata: {"content":"partial"}\n\n',
    'event: done\ndata: {}\n',
    'event: done\ndata: {}'
  ])("rejects truncated EOF, including an undispatched done", async (wire) => {
    const { body } = respond([encoder.encode(wire)]);
    await expect(streamChat(payload, { onEvent: vi.fn() })).rejects.toThrow(/before done/);
    expect(body.locked).toBe(false);
  });

  it.each([
    ['event: token\ndata: broken\n\n', /Invalid JSON/],
    ['event: token\ndata: []\n\n', /Invalid payload/],
    ['event: error\ndata: {"error":"backend unavailable"}\n\n', /backend unavailable/]
  ])("rejects malformed or terminal error events and releases the reader", async (wire, error) => {
    const { body, cancel } = respond([encoder.encode(String(wire))], false);
    await expect(streamChat(payload, { onEvent: vi.fn() })).rejects.toThrow(error);
    expect(cancel).toHaveBeenCalledOnce();
    expect(body.locked).toBe(false);
  });

  it("treats role_error as recoverable and failed/partial done as protocol completion", async () => {
    respond([encoder.encode('event: role_error\ndata: {"role":"surgeon"}\n\nevent: done\ndata: {"status":"partial"}\n\n')]);
    const onEvent = vi.fn();
    await streamChat(payload, { onEvent });
    expect(onEvent.mock.calls.map(([event]) => event)).toEqual(["role_error", "done"]);
  });

  it("cancels a pending read on AbortSignal, passes signal to fetch and releases its lock", async () => {
    const { body, cancel } = respond([], false);
    const controller = new AbortController();
    const running = streamChat(payload, { onEvent: vi.fn() }, controller.signal);
    await Promise.resolve();
    controller.abort();
    await expect(running).rejects.toMatchObject({ name: "AbortError" });
    expect(cancel).toHaveBeenCalledOnce();
    expect(body.locked).toBe(false);
    expect(fetch).toHaveBeenCalledWith(expect.any(String), expect.objectContaining({ signal: controller.signal }));
  });

  it("cleans up if a handler throws and rejects invalid UTF-8", async () => {
    let response = respond([encoder.encode('event: token\ndata: {}\n\n')], false);
    await expect(streamChat(payload, { onEvent: () => { throw new Error("handler failed"); } })).rejects.toThrow("handler failed");
    expect(response.body.locked).toBe(false);
    response = respond([new Uint8Array([0xff])]);
    await expect(streamChat(payload, { onEvent: vi.fn() })).rejects.toThrow();
    expect(response.body.locked).toBe(false);
  });

  it("reports HTTP detail and missing stream errors", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("bad request", { status: 400 })));
    await expect(streamChat(payload, { onEvent: vi.fn() })).rejects.toThrow("bad request");
    vi.mocked(fetch).mockResolvedValue(new Response(null));
    await expect(streamChat(payload, { onEvent: vi.fn() })).rejects.toThrow(/no readable stream/);
  });
});

describe("API base", () => {
  it("preserves local default and accepts explicit absolute/relative bases", () => {
    vi.stubEnv("NEXT_PUBLIC_API_BASE_URL", undefined);
    expect(getApiBase()).toBe("http://localhost:8002/api");
    vi.stubEnv("NEXT_PUBLIC_API_BASE_URL", "https://api.example.test/api/");
    expect(getApiBase()).toBe("https://api.example.test/api");
    vi.stubEnv("NEXT_PUBLIC_API_BASE_URL", "/api/");
    expect(getApiBase()).toBe("/api");
    vi.stubEnv("NEXT_PUBLIC_API_BASE_URL", "");
    expect(getApiBase()).toBe("/api");
  });

  it("defaults HTTPS and deployed hosts to a same-origin reverse proxy", () => {
    vi.stubEnv("NEXT_PUBLIC_API_BASE_URL", undefined);
    vi.stubGlobal("window", { location: { protocol: "https:", hostname: "example.test", port: "" } });
    expect(getApiBase()).toBe("/api");
    vi.stubGlobal("window", { location: { protocol: "http:", hostname: "example.test", port: "" } });
    expect(getApiBase()).toBe("/api");
  });
});
