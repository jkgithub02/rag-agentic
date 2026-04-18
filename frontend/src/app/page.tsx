"use client";

import { useState } from "react";

import { Tabs } from "@/components/tabs";
import { ChatTab } from "@/features/chat/chat-tab";
import { KnowledgeBaseTab } from "@/features/knowledge/knowledge-base-tab";
import { UploadTab } from "@/features/upload/upload-tab";

type AppTab = "chat" | "upload" | "knowledge";

export default function Home() {
  const [activeTab, setActiveTab] = useState<AppTab>("chat");
  const [refreshKey, setRefreshKey] = useState(0);

  return (
    <div className="relative min-h-screen overflow-hidden px-4 py-10 sm:px-8">
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_15%_20%,_#fef3c7_0,_transparent_45%),radial-gradient(circle_at_85%_8%,_#dbeafe_0,_transparent_50%),radial-gradient(circle_at_80%_82%,_#dcfce7_0,_transparent_45%)]" />

      <main className="relative mx-auto w-full max-w-5xl space-y-6">
        <header className="space-y-2">
          <p className="text-xs font-semibold uppercase tracking-[0.22em] text-[var(--ink-muted)]">
            Agentic RAG Console
          </p>
          <h1 className="text-3xl font-semibold tracking-tight text-[var(--ink)] sm:text-4xl">
            Search, cite, and manage your knowledge base
          </h1>
        </header>

        <Tabs
          active={activeTab}
          onChange={(next) => setActiveTab(next as AppTab)}
          items={[
            { key: "chat", label: "Chat" },
            { key: "upload", label: "Upload" },
            { key: "knowledge", label: "Knowledge Base" },
          ]}
        />

        <section className="rounded-3xl border border-[var(--line)] bg-white/80 p-5 shadow-[0_20px_55px_-32px_rgba(15,23,42,0.5)] backdrop-blur-sm sm:p-6">
          {activeTab === "chat" ? <ChatTab defaultThreadId="" /> : null}
          {activeTab === "upload" ? (
            <UploadTab onKnowledgeChanged={() => setRefreshKey((value) => value + 1)} />
          ) : null}
          {activeTab === "knowledge" ? <KnowledgeBaseTab refreshKey={refreshKey} /> : null}
        </section>
      </main>
    </div>
  );
}
