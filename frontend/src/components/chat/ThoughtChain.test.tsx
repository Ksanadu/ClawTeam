import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { ThoughtChain } from "@/components/chat/ThoughtChain";

afterEach(cleanup);

describe("persisted tool status", () => {
  it.each([
    ["cancelled", "", "已停止"],
    ["failed", "error output", "失败"],
    ["success", "", "完成"],
    ["running", "", "运行中"]
  ])("renders %s independently of output presence", (status, output, label) => {
    render(<ThoughtChain toolCalls={[{
      tool_call_id: "stable-call", tool: "read_file", input: "file.md", output, status
    }]} />);
    expect(screen.getByText(label)).toBeTruthy();
    if (status !== "running") expect(screen.queryByText("运行中")).toBeNull();
  });
});
