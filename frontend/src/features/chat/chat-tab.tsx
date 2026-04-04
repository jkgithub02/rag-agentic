"use client";

import { type FormEvent, useState } from "react";

import { askQuestion } from "@/lib/api-client";
import type { AskResponse } from "@/lib/types";

interface ChatTabProps {
    defaultThreadId: string;
}

export function ChatTab({ defaultThreadId }: ChatTabProps) {
    const [query, setQuery] = useState("");
    const [threadId, setThreadId] = useState(defaultThreadId);
    const [result, setResult] = useState<AskResponse | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const onSubmit = async (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();
        const trimmed = query.trim();
        if (!trimmed) {
            return;
        }

        setIsLoading(true);
        setError(null);

        try {
            const response = await askQuestion({
                query: trimmed,
                threadId: threadId.trim() || undefined,
            });
            setResult(response);
        } catch (submitError) {
            setError(submitError instanceof Error ? submitError.message : "Request failed");
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <section className="space-y-4">
            <form className="space-y-3" onSubmit={onSubmit}>
                <label className="block text-sm font-semibold text-[var(--ink)]">Question</label>
                <textarea
                    value={query}
                    onChange={(event) => setQuery(event.target.value)}
                    placeholder="Ask about uploaded documents..."
                    className="h-28 w-full rounded-2xl border border-[var(--line)] bg-white px-4 py-3 text-sm text-[var(--ink)] shadow-sm outline-none focus:border-[var(--accent)]"
                />

                <label className="block text-sm font-semibold text-[var(--ink)]">
                    Session Thread ID (optional)
                </label>
                <input
                    value={threadId}
                    onChange={(event) => setThreadId(event.target.value)}
                    className="w-full rounded-xl border border-[var(--line)] bg-white px-3 py-2 text-sm outline-none focus:border-[var(--accent)]"
                />

                <button
                    type="submit"
                    disabled={isLoading}
                    className="rounded-xl bg-[var(--accent)] px-4 py-2 text-sm font-semibold text-white transition hover:brightness-110 disabled:opacity-60"
                >
                    {isLoading ? "Running..." : "Run Query"}
                </button>
            </form>

            {error ? <p className="rounded-xl bg-red-50 p-3 text-sm text-red-700">{error}</p> : null}

            {result ? (
                <article className="space-y-3 rounded-2xl border border-[var(--line)] bg-white p-4 shadow-sm">
                    <header className="flex items-center justify-between gap-4">
                        <h3 className="text-sm font-semibold text-[var(--ink)]">Response</h3>
                        <span
                            className={`rounded-full px-3 py-1 text-xs font-semibold ${result.safe_fail
                                    ? "bg-amber-100 text-amber-800"
                                    : "bg-emerald-100 text-emerald-800"
                                }`}
                        >
                            {result.safe_fail ? "Safe-fail" : "Supported"}
                        </span>
                    </header>

                    <p className="text-sm leading-6 text-[var(--ink)]">{result.answer}</p>

                    <div>
                        <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-[var(--ink-muted)]">
                            Citations
                        </p>
                        {result.citations.length === 0 ? (
                            <p className="text-sm text-[var(--ink-muted)]">No citations returned.</p>
                        ) : (
                            <ul className="flex flex-wrap gap-2">
                                {result.citations.map((citation) => (
                                    <li
                                        key={citation}
                                        className="rounded-full border border-[var(--line)] bg-[var(--paper)] px-3 py-1 text-xs text-[var(--ink)]"
                                    >
                                        {citation}
                                    </li>
                                ))}
                            </ul>
                        )}
                    </div>

                    <p className="text-xs text-[var(--ink-muted)]">Trace ID: {result.trace_id}</p>
                </article>
            ) : null}
        </section>
    );
}
