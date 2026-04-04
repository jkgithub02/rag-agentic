"use client";

import { type FormEvent, useEffect, useState } from "react";

import { askQuestion, checkBackendHealth, getApiBaseUrl } from "@/lib/api-client";

interface ChatTabProps {
    defaultThreadId: string;
}

interface ChatMessage {
    id: string;
    role: "user" | "assistant";
    content: string;
    citations?: string[];
    traceId?: string;
    safeFail?: boolean;
}

export function ChatTab({ defaultThreadId }: ChatTabProps) {
    const [query, setQuery] = useState("");
    const [threadId, setThreadId] = useState(defaultThreadId);
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [isBackendConnected, setIsBackendConnected] = useState<boolean | null>(null);

    useEffect(() => {
        if (!threadId.trim()) {
            setThreadId(`session-${crypto.randomUUID().slice(0, 8)}`);
        }
    }, [threadId]);

    useEffect(() => {
        let active = true;
        void checkBackendHealth().then((isHealthy) => {
            if (active) {
                setIsBackendConnected(isHealthy);
            }
        });

        return () => {
            active = false;
        };
    }, []);

    const rotateThread = () => {
        setThreadId(`session-${crypto.randomUUID().slice(0, 8)}`);
        setMessages([]);
        setError(null);
    };

    const onSubmit = async (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();
        const trimmed = query.trim();
        if (!trimmed) {
            return;
        }

        const userMessage: ChatMessage = {
            id: crypto.randomUUID(),
            role: "user",
            content: trimmed,
        };
        setMessages((prev) => [...prev, userMessage]);
        setQuery("");
        setIsLoading(true);
        setError(null);

        try {
            const response = await askQuestion({
                query: trimmed,
                threadId: threadId.trim() || undefined,
            });
            const assistantMessage: ChatMessage = {
                id: crypto.randomUUID(),
                role: "assistant",
                content: response.answer,
                citations: response.citations,
                traceId: response.trace_id,
                safeFail: response.safe_fail,
            };
            setMessages((prev) => [...prev, assistantMessage]);
        } catch (submitError) {
            setError(submitError instanceof Error ? submitError.message : "Request failed");
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <section className="space-y-4">
            <header className="flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-[var(--line)] bg-white px-4 py-3">
                <div>
                    <p className="text-xs font-semibold uppercase tracking-wide text-[var(--ink-muted)]">
                        Session
                    </p>
                    <p className="text-sm font-semibold text-[var(--ink)]">{threadId}</p>
                </div>
                <div className="flex items-center gap-2">
                    <span
                        className={`rounded-full px-3 py-1 text-xs font-semibold ${isBackendConnected === true
                                ? "bg-emerald-100 text-emerald-800"
                                : isBackendConnected === false
                                    ? "bg-rose-100 text-rose-800"
                                    : "bg-slate-100 text-slate-700"
                            }`}
                    >
                        {isBackendConnected === true
                            ? "Backend connected"
                            : isBackendConnected === false
                                ? "Backend offline"
                                : "Checking backend"}
                    </span>
                    <button
                        type="button"
                        onClick={rotateThread}
                        className="rounded-xl border border-[var(--line)] bg-white px-3 py-2 text-xs font-semibold text-[var(--ink)]"
                    >
                        New Chat
                    </button>
                </div>
            </header>

            <div className="max-h-[420px] space-y-3 overflow-y-auto rounded-2xl border border-[var(--line)] bg-[var(--paper)] p-4">
                {messages.length === 0 ? (
                    <p className="text-sm text-[var(--ink-muted)]">
                        Start a conversation by typing a question below.
                    </p>
                ) : null}

                {messages.map((message) => (
                    <article
                        key={message.id}
                        className={`max-w-[90%] rounded-2xl px-4 py-3 text-sm shadow-sm ${message.role === "user"
                                ? "ml-auto bg-[var(--accent)] text-white"
                                : "bg-white text-[var(--ink)]"
                            }`}
                    >
                        <p className="mb-2 text-xs font-semibold uppercase tracking-wide opacity-70">
                            {message.role === "user" ? "You" : "Assistant"}
                        </p>
                        <p className="whitespace-pre-wrap leading-6">{message.content}</p>

                        {message.role === "assistant" && message.citations && message.citations.length > 0 ? (
                            <ul className="mt-3 flex flex-wrap gap-2">
                                {message.citations.map((citation) => (
                                    <li
                                        key={`${message.id}-${citation}`}
                                        className="rounded-full border border-[var(--line)] bg-[var(--paper)] px-2 py-1 text-[11px]"
                                    >
                                        {citation}
                                    </li>
                                ))}
                            </ul>
                        ) : null}

                        {message.role === "assistant" && message.traceId ? (
                            <p className="mt-2 text-[11px] text-[var(--ink-muted)]">
                                Trace ID: {message.traceId}
                                {message.safeFail ? " | Safe-fail" : ""}
                            </p>
                        ) : null}
                    </article>
                ))}

                {isLoading ? (
                    <div className="max-w-[90%] rounded-2xl bg-white px-4 py-3 text-sm text-[var(--ink-muted)] shadow-sm">
                        Assistant is thinking...
                    </div>
                ) : null}
            </div>

            <form className="space-y-3" onSubmit={onSubmit}>
                <label className="block text-sm font-semibold text-[var(--ink)]">Message</label>
                <textarea
                    value={query}
                    onChange={(event) => setQuery(event.target.value)}
                    placeholder="Ask about uploaded documents..."
                    className="h-24 w-full rounded-2xl border border-[var(--line)] bg-white px-4 py-3 text-sm text-[var(--ink)] shadow-sm outline-none focus:border-[var(--accent)]"
                />

                <details className="rounded-xl border border-[var(--line)] bg-white px-3 py-2 text-sm text-[var(--ink)]">
                    <summary className="cursor-pointer font-semibold">Advanced session settings</summary>
                    <div className="mt-2 space-y-2">
                        <label className="block text-xs font-semibold uppercase tracking-wide text-[var(--ink-muted)]">
                            Thread ID override
                        </label>
                        <input
                            value={threadId}
                            onChange={(event) => setThreadId(event.target.value)}
                            className="w-full rounded-xl border border-[var(--line)] bg-white px-3 py-2 text-sm outline-none focus:border-[var(--accent)]"
                        />
                        <p className="text-xs text-[var(--ink-muted)]">
                            API base: {getApiBaseUrl()}
                        </p>
                    </div>
                </details>

                <button
                    type="submit"
                    disabled={isLoading}
                    className="rounded-xl bg-[var(--accent)] px-4 py-2 text-sm font-semibold text-white transition hover:brightness-110 disabled:opacity-60"
                >
                    {isLoading ? "Sending..." : "Send"}
                </button>
            </form>

            {error ? <p className="rounded-xl bg-red-50 p-3 text-sm text-red-700">{error}</p> : null}
        </section>
    );
}
