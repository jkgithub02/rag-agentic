"use client";

import { type FormEvent, useState } from "react";

import { uploadDocument } from "@/lib/api-client";
import type { ConflictPolicy, UploadResponse } from "@/lib/types";

interface UploadTabProps {
    onKnowledgeChanged: () => void;
}

export function UploadTab({ onKnowledgeChanged }: UploadTabProps) {
    const [file, setFile] = useState<File | null>(null);
    const [pendingConflict, setPendingConflict] = useState<File | null>(null);
    const [conflictPayload, setConflictPayload] = useState<UploadResponse | null>(null);
    const [message, setMessage] = useState<string | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(false);

    const executeUpload = async (policy: ConflictPolicy, fileToUpload: File) => {
        setIsLoading(true);
        setError(null);
        setMessage(null);

        try {
            const response = await uploadDocument({ file: fileToUpload, conflictPolicy: policy });
            if (response.status === "conflict") {
                setPendingConflict(fileToUpload);
                setConflictPayload(response);
                setMessage(response.message);
                return;
            }

            setPendingConflict(null);
            setConflictPayload(null);
            setMessage(
                `Stored ${response.stored_filename} with ${response.chunks_added ?? 0} indexed chunk(s).`,
            );
            onKnowledgeChanged();
        } catch (uploadError) {
            setError(uploadError instanceof Error ? uploadError.message : "Upload failed");
        } finally {
            setIsLoading(false);
        }
    };

    const onSubmit = async (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();
        if (!file) {
            setError("Please choose a file first.");
            return;
        }

        await executeUpload("ask", file);
    };

    return (
        <section className="space-y-4">
            <form className="space-y-3" onSubmit={onSubmit}>
                <label className="block text-sm font-semibold text-[var(--ink)]">Upload file</label>
                <input
                    type="file"
                    accept=".pdf,.txt,.md"
                    onChange={(event) => setFile(event.target.files?.[0] ?? null)}
                    className="w-full rounded-xl border border-[var(--line)] bg-white px-3 py-2 text-sm"
                />
                <button
                    type="submit"
                    disabled={isLoading}
                    className="rounded-xl bg-[var(--accent)] px-4 py-2 text-sm font-semibold text-white transition hover:brightness-110 disabled:opacity-60"
                >
                    {isLoading ? "Uploading..." : "Upload"}
                </button>
            </form>

            {message ? <p className="rounded-xl bg-sky-50 p-3 text-sm text-sky-800">{message}</p> : null}
            {error ? <p className="rounded-xl bg-red-50 p-3 text-sm text-red-700">{error}</p> : null}

            {pendingConflict && conflictPayload ? (
                <article className="space-y-3 rounded-2xl border border-amber-200 bg-amber-50 p-4">
                    <p className="text-sm font-semibold text-amber-900">Filename conflict detected</p>
                    <p className="text-sm text-amber-800">
                        Existing: {conflictPayload.existing_filename} | Suggested: {conflictPayload.suggested_filename}
                    </p>
                    <div className="flex flex-wrap gap-2">
                        <button
                            type="button"
                            onClick={() => void executeUpload("replace", pendingConflict)}
                            className="rounded-xl bg-amber-600 px-3 py-2 text-xs font-semibold text-white"
                        >
                            Replace existing
                        </button>
                        <button
                            type="button"
                            onClick={() => void executeUpload("keep_both", pendingConflict)}
                            className="rounded-xl border border-amber-400 bg-white px-3 py-2 text-xs font-semibold text-amber-900"
                        >
                            Keep both
                        </button>
                    </div>
                </article>
            ) : null}
        </section>
    );
}
