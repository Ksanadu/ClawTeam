import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { ChatMessage } from "@/components/chat/ChatMessage";

afterEach(cleanup);

describe("run history progress replay", () => {
  it("shows the most recent progress event for a stage", () => {
    render(<ChatMessage role="assistant" content="final answer" toolCalls={[]} retrievals={[]}
      progress={[
        { stage: "round1", status: "pending", label: "old pending stage" },
        { stage: "round1", status: "running", label: "old running stage" },
        { stage: "round1", status: "done", label: "latest completed stage" }
      ]} />);
    expect(screen.getByText("latest completed stage")).toBeTruthy();
    expect(screen.queryByText("old pending stage")).toBeNull();
    expect(screen.queryByText("old running stage")).toBeNull();
  });
});
