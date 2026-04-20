"use client";

import { type FormEvent, type KeyboardEvent, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { askQuestionStream, checkBackendHealth, getTrace } from "@/lib/api-client";
import { type PipelineStageInfo, type PipelineTrace } from "@/lib/types";

const CHAT_HISTORY_KEY = "agentic_rag_chat_history_v1";

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
    stages?: PipelineStageInfo[];
}

interface ChatHistorySession {
    threadId: string;
    updatedAt: string;
    messages: ChatMessage[];
}

function NewChatIcon() {
    return (
        <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M12 5v14" />
            <path d="M5 12h14" />
        </svg>
    );
}

function SendIcon() {
    return (
        <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M22 2 11 13" />
            <path d="m22 2-7 20-4-9-9-4Z" />
        </svg>
    );
}

function TrashIcon() {
    return (
        <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M3 6h18" />
            <path d="M8 6V4h8v2" />
            <path d="M19 6v14H5V6" />
            <path d="M10 11v6" />
            <path d="M14 11v6" />
        </svg>
    );
}

function ExportIcon() {
    return (
        <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M12 3v12" />
            <path d="m8 11 4 4 4-4" />
            <path d="M5 21h14" />
        </svg>
    );
}

const STAGE_LABELS: Record<string, string> = {
    rewrite_query: "Analyzing query",
    detect_query_type: "Classifying query",
    prepare_decomposition: "Decomposing query",
    agent_think: "Planning next step",
    retrieve: "Searching documents",
    agent_act: "Executing action",
    agent_reflect: "Evaluating evidence",
    validate: "Validating results",
    generate: "Generating answer",
    verify_grounding: "Verifying grounding",
    tool_web_search: "Searching the web",
    tool_fetch_chunks_by_ids: "Fetching evidence",
    compress_context: "Compressing context",
    answer: "Finalizing",
};

function PipelineProgressIndicator({ stages }: { stages: PipelineStageInfo[] }) {
    const completed = stages.filter((s) => s.status === "completed");
    const active = stages.find((s) => s.status === "active");

    if (completed.length === 0 && !active) {
        return <div className="text-sm text-[var(--ink-muted)]">Thinking...</div>;
    }

    return (
        <div className="space-y-1">
            {completed.map((s, i) => (
                <div key={i} className="flex items-center gap-2 text-xs text-[var(--ink-muted)]">
                    <span className="text-emerald-500">&#10003;</span>
                    <span>{STAGE_LABELS[s.stage] || s.stage}</span>
                </div>
            ))}
            {active && (
                <div className="flex items-center gap-2 text-xs text-[var(--ink)]">
                    <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-[var(--accent)]" />
                    <span className="font-medium">{STAGE_LABELS[active.stage] || active.stage}</span>
                </div>
            )}
        </div>
    );
}

function PipelineTracePanel({ trace }: { trace: PipelineTrace }) {
    const rewriteEvent = trace.events.find((e) => e.stage === "rewrite_query");
    const decompEvent = trace.events.find((e) => e.stage === "prepare_decomposition");
    const agentThinks = trace.events.filter((e) => e.stage === "agent_think");
    const retrieves = trace.events.filter((e) => e.stage === "retrieve");
    const validates = trace.events.filter((e) => e.stage === "validate");
    const generateEvent = trace.events.find((e) => e.stage === "generate");
    const groundingEvent = trace.events.find((e) => e.stage === "verify_grounding");
    const queryWasRewritten = trace.original_query !== trace.rewritten_query;

    const totalChunks = retrieves.reduce((sum, e) => {
        const hits = e.payload.hits;
        return sum + (Array.isArray(hits) ? hits.length : 0);
    }, 0);

    return (
        <details className="mt-3 rounded-xl border border-[var(--line)] bg-[var(--paper)]">
            <summary className="cursor-pointer px-3 py-2 text-[11px] font-semibold text-[var(--ink-muted)] hover:text-[var(--ink)]">
                Pipeline details ({trace.events.length} stages)
            </summary>
            <div className="space-y-3 px-3 pb-3 text-[11px] text-[var(--ink-muted)]">
                {/* Query Understanding */}
                {queryWasRewritten && (
                    <div>
                        <p className="font-semibold text-[var(--ink)]">Query Understanding</p>
                        <p><span className="font-medium">Original:</span> {trace.original_query}</p>
                        <p><span className="font-medium">Rewritten:</span> {trace.rewritten_query}</p>
                        {rewriteEvent?.payload.rewrite_source != null && (
                            <p><span className="font-medium">Source:</span> {String(rewriteEvent.payload.rewrite_source)}</p>
                        )}
                    </div>
                )}

                {/* Decomposition */}
                {decompEvent && decompEvent.payload.applied === true && (
                    <div>
                        <p className="font-semibold text-[var(--ink)]">Query Decomposition</p>
                        <p>{String(decompEvent.payload.query_count || 0)} sub-queries created</p>
                    </div>
                )}

                {/* Agent Reasoning */}
                {agentThinks.length > 0 && (
                    <div>
                        <p className="font-semibold text-[var(--ink)]">Agent Reasoning ({agentThinks.length} iteration{agentThinks.length !== 1 ? "s" : ""})</p>
                        {agentThinks.map((t, i) => (
                            <div key={i} className="ml-2 mt-1 border-l-2 border-[var(--line)] pl-2">
                                <p className="font-medium">Step {i + 1}: {String(t.payload.recommended_action || "thinking")}</p>
                                {t.payload.reasoning != null && (
                                    <p className="italic">{String(t.payload.reasoning).slice(0, 150)}</p>
                                )}
                            </div>
                        ))}
                    </div>
                )}

                {/* Retrieval */}
                {totalChunks > 0 && (
                    <div>
                        <p className="font-semibold text-[var(--ink)]">Evidence Retrieved</p>
                        <p>{totalChunks} chunk{totalChunks !== 1 ? "s" : ""} across {retrieves.length} retrieval{retrieves.length !== 1 ? "s" : ""}</p>
                    </div>
                )}

                {/* Validation */}
                {validates.length > 0 && (
                    <div>
                        <p className="font-semibold text-[var(--ink)]">Validation</p>
                        <p>Status: {String(validates[0].payload.status)} (confidence: {String(validates[0].payload.confidence)})</p>
                    </div>
                )}

                {/* Generation */}
                {generateEvent && (
                    <div>
                        <p className="font-semibold text-[var(--ink)]">Answer Generation</p>
                        <p>Safe fail: {String(generateEvent.payload.safe_fail)}</p>
                    </div>
                )}

                {/* Grounding */}
                {groundingEvent && (
                    <div>
                        <p className="font-semibold text-[var(--ink)]">Grounding Verification</p>
                        <p>Status: <span className={
                            groundingEvent.payload.status === "supported" ? "text-emerald-600 font-medium" :
                            groundingEvent.payload.status === "partial" ? "text-amber-600 font-medium" :
                            "text-rose-600 font-medium"
                        }>{String(groundingEvent.payload.status)}</span></p>
                        {groundingEvent.payload.reason != null && (
                            <p className="italic">{String(groundingEvent.payload.reason)}</p>
                        )}
                    </div>
                )}
            </div>
        </details>
    );
}

function readHistory(): ChatHistorySession[] {
    try {
        const raw = localStorage.getItem(CHAT_HISTORY_KEY);
        if (!raw) {
            return [];
        }
        const parsed = JSON.parse(raw) as ChatHistorySession[];
        if (!Array.isArray(parsed)) {
            return [];
        }
        return parsed.filter((item) => {
            if (typeof item.threadId !== "string") {
                return false;
            }
            if (!Array.isArray(item.messages)) {
                return false;
            }
            return item.messages.some(
                (message) =>
                    message &&
                    typeof message.id === "string" &&
                    (message.role === "user" || message.role === "assistant") &&
                    typeof message.content === "string",
            );
        });
    } catch {
        return [];
    }
}

function writeHistory(sessions: ChatHistorySession[]) {
    localStorage.setItem(CHAT_HISTORY_KEY, JSON.stringify(sessions));
}

function generateMessageId(): string {
    if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
        return crypto.randomUUID();
    }
    return `msg-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

function generateThreadId(): string {
    if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
        return `session-${crypto.randomUUID().slice(0, 8)}`;
    }
    return `session-${Math.random().toString(36).slice(2, 10)}`;
}

export function ChatTab({ defaultThreadId }: ChatTabProps) {
    const [query, setQuery] = useState("");
    const [threadId, setThreadId] = useState(defaultThreadId);
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [historySessions, setHistorySessions] = useState<ChatHistorySession[]>([]);
    const [historySelection, setHistorySelection] = useState("");
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [isBackendConnected, setIsBackendConnected] = useState<boolean | null>(null);
    const [tracesById, setTracesById] = useState<Record<string, PipelineTrace>>({});
    const messageContainerRef = useRef<HTMLDivElement | null>(null);
    const requestedTraceIdsRef = useRef<Set<string>>(new Set());

    useEffect(() => {
        if (!threadId.trim()) {
            setThreadId(generateThreadId());
        }
    }, [threadId]);

    useEffect(() => {
        if (!messageContainerRef.current) {
            return;
        }

        messageContainerRef.current.scrollTo({
            top: messageContainerRef.current.scrollHeight,
            behavior: "smooth",
        });
    }, [messages, isLoading]);

    useEffect(() => {
        const traceIds = messages
            .filter((message) => message.role === "assistant" && typeof message.traceId === "string")
            .map((message) => message.traceId)
            .filter((value): value is string => typeof value === "string" && value.length > 0);

        const missingIds = traceIds.filter((traceId) => {
            if (tracesById[traceId]) {
                return false;
            }
            if (requestedTraceIdsRef.current.has(traceId)) {
                return false;
            }
            return true;
        });

        if (missingIds.length === 0) {
            return;
        }

        for (const traceId of missingIds) {
            requestedTraceIdsRef.current.add(traceId);
        }

        void Promise.all(
            missingIds.map(async (traceId) => {
                try {
                    const trace = await getTrace(traceId);
                    setTracesById((prev) => ({ ...prev, [traceId]: trace }));
                } catch {
                    // Keep chat experience resilient even when trace fetch fails.
                }
            }),
        );
    }, [messages, tracesById]);

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

    useEffect(() => {
        const sessions = readHistory().sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));
        setHistorySessions(sessions);
    }, []);

    useEffect(() => {
        if (!threadId.trim()) {
            return;
        }
        if (messages.length === 0) {
            return;
        }

        setHistorySessions((prev) => {
            const current: ChatHistorySession = {
                threadId,
                updatedAt: new Date().toISOString(),
                messages,
            };

            const merged = [current, ...prev.filter((session) => session.threadId !== threadId)].slice(0, 20);
            writeHistory(merged);
            return merged;
        });
    }, [threadId, messages]);

    const rotateThread = () => {
        setThreadId(generateThreadId());
        setMessages([]);
        setHistorySelection("");
        setError(null);
    };

    const loadSelectedHistory = (selectedThreadId: string) => {
        if (!selectedThreadId) {
            return;
        }
        const selected = historySessions.find((session) => session.threadId === selectedThreadId);
        if (!selected) {
            return;
        }
        setThreadId(selected.threadId);
        setMessages(selected.messages.map((message) => ({ ...message })));
        setError(null);
    };

    const onHistorySelectionChange = (selectedThreadId: string) => {
        setHistorySelection(selectedThreadId);
        loadSelectedHistory(selectedThreadId);
    };

    const clearHistory = () => {
        localStorage.removeItem(CHAT_HISTORY_KEY);
        setHistorySessions([]);
        setHistorySelection("");
    };

    const exportCurrentSession = () => {
        const payload = {
            threadId,
            exportedAt: new Date().toISOString(),
            messages,
        };
        const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const anchor = document.createElement("a");
        anchor.href = url;
        anchor.download = `chat-${threadId || "session"}.json`;
        anchor.click();
        URL.revokeObjectURL(url);
    };

    const submitQuery = async () => {
        const trimmed = query.trim();
        if (!trimmed) {
            return;
        }

        const userMessage: ChatMessage = {
            id: generateMessageId(),
            role: "user",
            content: trimmed,
        };
        setMessages((prev) => [...prev, userMessage]);
        setQuery("");
        setIsLoading(true);
        setError(null);

        try {
            const assistantMessageId = generateMessageId();
            setMessages((prev) => [
                ...prev,
                {
                    id: assistantMessageId,
                    role: "assistant",
                    content: "",
                    citations: [],
                },
            ]);

            await askQuestionStream({
                query: trimmed,
                threadId: threadId.trim() || undefined,
            }, (event) => {
                setMessages((prev) => {
                    const next = [...prev];
                    const index = next.findIndex((message) => message.id === assistantMessageId);
                    if (index === -1) {
                        return prev;
                    }

                    const target = { ...next[index] };
                    if (event.type === "delta") {
                        target.content = `${target.content}${event.text}`;
                    } else if (event.type === "stage") {
                        const prevStages = (target.stages || []).map((s) =>
                            s.status === "active" ? { ...s, status: "completed" as const } : s,
                        );
                        target.stages = [
                            ...prevStages,
                            {
                                stage: event.stage,
                                detail: event as unknown as Record<string, unknown>,
                                status: "active",
                            },
                        ];
                    } else if (event.type === "thinking") {
                        // Keep the progress indicator visible while waiting for first token.
                    } else if (event.type === "done") {
                        // Mark any remaining active stage as completed
                        if (target.stages) {
                            target.stages = target.stages.map((s) =>
                                s.status === "active" ? { ...s, status: "completed" as const } : s,
                            );
                        }
                        target.citations = event.citations;
                        target.traceId = event.trace_id;
                        target.safeFail = event.safe_fail;
                    } else if (event.type === "error") {
                        target.content = target.content || event.message;
                        target.safeFail = true;
                    }

                    next[index] = target;
                    return next;
                });
            });
        } catch (submitError) {
            setError(submitError instanceof Error ? submitError.message : "Request failed");
        } finally {
            setIsLoading(false);
        }
    };

    const onSubmit = async (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();
        await submitQuery();
    };

    const onMessageKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            void submitQuery();
        }
    };

    return (
        <section className="space-y-4">
            <header className="flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-[var(--line)] bg-white px-4 py-3">
                <div>
                    <p className="text-xs font-semibold uppercase tracking-wide text-[var(--ink-muted)]">
                        Session
                    </p>
                    <p data-testid="chat-thread-id" className="text-sm font-semibold text-[var(--ink)]">{threadId}</p>
                </div>
                <div className="flex items-center gap-2">
                    <span
                        data-testid="chat-backend-status"
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
                        className="inline-flex items-center gap-1 rounded-xl border border-[var(--line)] bg-white px-3 py-2 text-xs font-semibold text-[var(--ink)]"
                    >
                        <NewChatIcon />
                        New Chat
                    </button>
                    <button
                        type="button"
                        onClick={exportCurrentSession}
                        disabled={messages.length === 0}
                        className="inline-flex items-center gap-1 rounded-xl border border-[var(--line)] bg-white px-3 py-2 text-xs font-semibold text-[var(--ink)] disabled:opacity-60"
                    >
                        <ExportIcon />
                        Export
                    </button>
                </div>
            </header>

            <div className="rounded-2xl border border-[var(--line)] bg-white p-3">
                <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-[var(--ink-muted)]">
                    Chat History
                </p>
                <div className="flex flex-wrap items-center gap-2">
                    <select
                        value={historySelection}
                        onChange={(event) => onHistorySelectionChange(event.target.value)}
                        className="min-w-56 flex-1 rounded-xl border border-[var(--line)] bg-white px-3 py-2 text-sm text-[var(--ink)]"
                    >
                        <option value="">Select previous session</option>
                        {historySessions.map((session) => (
                            <option key={session.threadId} value={session.threadId}>
                                {session.threadId} ({new Date(session.updatedAt).toLocaleString()})
                            </option>
                        ))}
                    </select>

                    <button
                        type="button"
                        onClick={clearHistory}
                        disabled={historySessions.length === 0}
                        className="inline-flex items-center gap-1 rounded-xl border border-rose-300 bg-rose-50 px-3 py-2 text-xs font-semibold text-rose-700 disabled:opacity-60"
                    >
                        <TrashIcon />
                        Clear History
                    </button>
                </div>
            </div>

            <div
                ref={messageContainerRef}
                className="max-h-[900px] space-y-3 overflow-y-auto rounded-2xl border border-[var(--line)] bg-[var(--paper)] p-4 flex flex-col"
            >
                {messages.length === 0 ? (
                    <div className="flex-1 flex items-center justify-center">
                        <div className="text-center">
                            <h2 className="text-5xl font-bold text-[var(--ink)] mb-4">
                                Hello there, how can I help?
                            </h2>
                            <p className="text-lg text-[var(--ink-muted)]">
                                Ask me anything about your uploaded documents
                            </p>
                        </div>
                    </div>
                ) : null}

                {messages.map((message) => (
                    message.role === "assistant" && !message.content.trim() ? null : (
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
                            {message.role === "assistant" ? (
                                <div className="prose prose-sm max-w-none leading-6 prose-p:my-2 prose-li:my-1">
                                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
                                </div>
                            ) : (
                                <p className="whitespace-pre-wrap leading-6">{message.content}</p>
                            )}

                            {message.role === "assistant" && message.citations && message.citations.length > 0 ? (
                                <ul className="mt-3 flex flex-wrap gap-2">
                                    {message.citations.map((citation) => {
                                        // Check if citation is a web URL
                                        const isWebUrl = citation.startsWith("http://") || citation.startsWith("https://");
                                        if (isWebUrl) {
                                            // Extract URL and chunk_id from "url#chunk_id"
                                            const [url, chunkId] = citation.split("#");
                                            try {
                                                const domain = new URL(url).hostname.replace("www.", "");
                                                const displayText = chunkId ? `${domain} (${chunkId})` : domain;
                                                return (
                                                    <li key={`${message.id}-${citation}`}>
                                                        <a
                                                            href={url}
                                                            target="_blank"
                                                            rel="noopener noreferrer"
                                                            className="inline-block rounded-full border border-blue-300 bg-blue-50 px-2 py-1 text-[11px] text-blue-700 hover:bg-blue-100 hover:border-blue-400"
                                                            title={url}
                                                        >
                                                            🌐 {displayText}
                                                        </a>
                                                    </li>
                                                );
                                            } catch {
                                                // Fallback if URL parsing fails
                                                return (
                                                    <li
                                                        key={`${message.id}-${citation}`}
                                                        className="rounded-full border border-[var(--line)] bg-[var(--paper)] px-2 py-1 text-[11px]"
                                                    >
                                                        {citation}
                                                    </li>
                                                );
                                            }
                                        }
                                        // Local document citation
                                        return (
                                            <li
                                                key={`${message.id}-${citation}`}
                                                className="rounded-full border border-[var(--line)] bg-[var(--paper)] px-2 py-1 text-[11px]"
                                            >
                                                📄 {citation}
                                            </li>
                                        );
                                    })}
                                </ul>
                            ) : null}

                            {message.role === "assistant" && message.traceId && tracesById[message.traceId] ? (
                                <PipelineTracePanel trace={tracesById[message.traceId]} />
                            ) : null}
                        </article>
                    )
                ))}

                {isLoading ? (
                    <div className="max-w-[90%] rounded-2xl bg-white px-4 py-3 text-sm text-[var(--ink-muted)] shadow-sm">
                        <PipelineProgressIndicator stages={messages[messages.length - 1]?.stages || []} />
                    </div>
                ) : null}
            </div>

            <form className="space-y-3" onSubmit={onSubmit}>
                <label className="block text-sm font-semibold text-[var(--ink)]">Message</label>
                <textarea
                    value={query}
                    onChange={(event) => setQuery(event.target.value)}
                    onKeyDown={onMessageKeyDown}
                    placeholder="Ask about uploaded documents..."
                    className="h-24 w-full rounded-2xl border border-[var(--line)] bg-white px-4 py-3 text-sm text-[var(--ink)] shadow-sm outline-none focus:border-[var(--accent)]"
                />

                <button
                    type="submit"
                    disabled={isLoading}
                    className="inline-flex items-center gap-1 rounded-xl bg-[var(--accent)] px-4 py-2 text-sm font-semibold text-white transition hover:brightness-110 disabled:opacity-60"
                >
                    <SendIcon />
                    {isLoading ? "Sending..." : "Send"}
                </button>
            </form>

            {error ? <p className="rounded-xl bg-red-50 p-3 text-sm text-red-700">{error}</p> : null}
        </section>
    );
}
