"use client";

import { useEffect, useMemo, useState } from "react";

import { clearDocuments, deleteDocument, listDocuments } from "@/lib/api-client";
import type { DocumentRecord } from "@/lib/types";

interface KnowledgeBaseTabProps {
    refreshKey: number;
}

function formatBytes(size: number): string {
    if (size < 1024) {
        return `${size} B`;
    }
    if (size < 1024 * 1024) {
        return `${(size / 1024).toFixed(1)} KB`;
    }
    return `${(size / (1024 * 1024)).toFixed(1)} MB`;
}

export function KnowledgeBaseTab({ refreshKey }: KnowledgeBaseTabProps) {
    const [records, setRecords] = useState<DocumentRecord[]>([]);
    const [query, setQuery] = useState("");
    const [error, setError] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(false);

    const load = async () => {
        setIsLoading(true);
        setError(null);
        try {
            setRecords(await listDocuments());
        } catch (loadError) {
            setError(loadError instanceof Error ? loadError.message : "Failed to load documents");
        } finally {
            setIsLoading(false);
        }
    };

    useEffect(() => {
        void load();
        // refreshKey is explicit external trigger for reload.
    }, [refreshKey]);

    const filtered = useMemo(() => {
        const needle = query.trim().toLowerCase();
        if (!needle) {
            return records;
        }
        return records.filter((record) => record.filename.toLowerCase().includes(needle));
    }, [records, query]);

    const onDeleteOne = async (filename: string) => {
        try {
            await deleteDocument(filename);
            await load();
        } catch (deleteError) {
            setError(deleteError instanceof Error ? deleteError.message : "Delete failed");
        }
    };

    const onClearAll = async () => {
        try {
            await clearDocuments();
            await load();
        } catch (clearError) {
            setError(clearError instanceof Error ? clearError.message : "Clear-all failed");
        }
    };

    return (
        <section className="space-y-4">
            <div className="flex flex-wrap items-center gap-2">
                <input
                    value={query}
                    onChange={(event) => setQuery(event.target.value)}
                    placeholder="Filter by filename"
                    className="min-w-52 flex-1 rounded-xl border border-[var(--line)] bg-white px-3 py-2 text-sm"
                />
                <button
                    type="button"
                    onClick={() => void load()}
                    className="rounded-xl border border-[var(--line)] bg-white px-3 py-2 text-sm font-semibold text-[var(--ink)]"
                >
                    Refresh
                </button>
                <button
                    type="button"
                    onClick={() => void onClearAll()}
                    className="rounded-xl bg-rose-600 px-3 py-2 text-sm font-semibold text-white"
                >
                    Clear All
                </button>
            </div>

            {error ? <p className="rounded-xl bg-red-50 p-3 text-sm text-red-700">{error}</p> : null}

            {isLoading ? <p className="text-sm text-[var(--ink-muted)]">Loading documents...</p> : null}

            {!isLoading && filtered.length === 0 ? (
                <p className="text-sm text-[var(--ink-muted)]">No documents found.</p>
            ) : null}

            <ul className="space-y-2">
                {filtered.map((record) => (
                    <li
                        key={record.filename}
                        className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-[var(--line)] bg-white px-3 py-3"
                    >
                        <div>
                            <p className="text-sm font-semibold text-[var(--ink)]">{record.filename}</p>
                            <p className="text-xs text-[var(--ink-muted)]">
                                {formatBytes(record.size_bytes)} | {record.chunks_indexed} chunk(s)
                            </p>
                        </div>
                        <button
                            type="button"
                            onClick={() => void onDeleteOne(record.filename)}
                            className="rounded-lg border border-rose-300 bg-rose-50 px-3 py-1 text-xs font-semibold text-rose-700"
                        >
                            Delete
                        </button>
                    </li>
                ))}
            </ul>
        </section>
    );
}
