"use client";

import { PanelLeftOpen } from "lucide-react";
import { useState } from "react";

import { ChatPanel } from "@/components/chat/ChatPanel";
import { Navbar } from "@/components/layout/Navbar";
import { ResizeHandle } from "@/components/layout/ResizeHandle";
import { Sidebar } from "@/components/layout/Sidebar";
import { AppProvider, useAppStore } from "@/lib/store";

function Workspace() {
  const { sidebarWidth, setSidebarWidth } = useAppStore();
  const [sidebarOpen, setSidebarOpen] = useState(true);

  return (
    <main className="flex h-screen flex-col p-3 md:p-4">
      <div className="mx-auto flex min-h-0 w-full max-w-[1840px] flex-1 flex-col gap-3">
        <Navbar />
        <div className="flex min-h-0 flex-1 gap-0">
          <div
            style={{ width: sidebarOpen ? sidebarWidth : 44, flexShrink: 0 }}
            className="hidden h-full md:block"
          >
            {sidebarOpen ? (
              <Sidebar onCollapse={() => setSidebarOpen(false)} />
            ) : (
              <button
                className="panel flex h-full w-11 items-start justify-center rounded-lg pt-3 text-[var(--color-ink-soft)] hover:text-ocean"
                onClick={() => setSidebarOpen(true)}
                title="展开会诊队列"
                type="button"
              >
                <PanelLeftOpen size={18} />
              </button>
            )}
          </div>
          {sidebarOpen && (
            <div className="hidden md:block">
              <ResizeHandle onResize={(delta) => setSidebarWidth(Math.max(260, sidebarWidth + delta))} />
            </div>
          )}
          <div className="h-full min-w-0 flex-1">
            <ChatPanel />
          </div>
        </div>
      </div>
    </main>
  );
}

export default function Page() {
  return (
    <AppProvider>
      <Workspace />
    </AppProvider>
  );
}
