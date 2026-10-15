import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ChatInput } from "@/components/chat/ChatInput";

afterEach(cleanup);

describe("ChatInput send/stop lifecycle", () => {
  it("retains text and attachments on immediate failure and consumes the callback rejection", async () => {
    const onSend = vi.fn().mockRejectedValue(new Error("network offline"));
    const { container } = render(<ChatInput disabled={false} onSend={onSend} />);
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "my exact draft " } });
    const file = new File(["image"], "pathology.png", { type: "image/png" });
    fireEvent.change(container.querySelector('input[type="file"]')!, { target: { files: [file] } });
    await waitFor(() => expect(screen.getByText("pathology.png")).toBeTruthy());
    fireEvent.click(screen.getByRole("button", { name: "发送" }));
    await waitFor(() => expect(screen.getByText("network offline")).toBeTruthy());
    expect((screen.getByRole("textbox") as HTMLTextAreaElement).value).toBe("my exact draft ");
    expect(screen.getByText("pathology.png")).toBeTruthy();
    expect(onSend).toHaveBeenCalledWith("my exact draft", [expect.objectContaining({ name: "pathology.png" })]);
  });

  it("retains a draft when the store reports an unaccepted send", async () => {
    const onSend = vi.fn().mockResolvedValue(false);
    render(<ChatInput disabled={false} onSend={onSend} />);
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "retry me" } });
    fireEvent.click(screen.getByRole("button", { name: "发送" }));
    await waitFor(() => expect(onSend).toHaveBeenCalledOnce());
    expect((screen.getByRole("textbox") as HTMLTextAreaElement).value).toBe("retry me");
  });

  it("blocks Ctrl/Cmd+Enter while disabled and exposes a working stop button", () => {
    const onSend = vi.fn(); const onStop = vi.fn();
    render(<ChatInput disabled isStreaming onSend={onSend} onStop={onStop} />);
    const input = screen.getByRole("textbox");
    fireEvent.change(input, { target: { value: "draft" } });
    fireEvent.keyDown(input, { key: "Enter", ctrlKey: true });
    fireEvent.keyDown(input, { key: "Enter", metaKey: true });
    expect(onSend).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "停止生成" }));
    expect(onStop).toHaveBeenCalledOnce();
  });

  it("locks duplicate shortcuts synchronously and does not clear edits made during generation", async () => {
    let resolve!: (accepted: boolean) => void;
    const onSend = vi.fn(() => new Promise<boolean>((done) => { resolve = done; }));
    render(<ChatInput disabled={false} onSend={onSend} />);
    const input = screen.getByRole("textbox");
    fireEvent.change(input, { target: { value: "first draft" } });
    fireEvent.keyDown(input, { key: "Enter", ctrlKey: true });
    fireEvent.keyDown(input, { key: "Enter", ctrlKey: true });
    expect(onSend).toHaveBeenCalledOnce();
    fireEvent.change(input, { target: { value: "next draft" } });
    await act(async () => resolve(true));
    expect((input as HTMLTextAreaElement).value).toBe("next draft");
  });
});
