import {
    type AskResponse,
    type ConflictPolicy,
    type DocumentRecord,
    type PipelineTrace,
    type UploadResponse,
} from "@/lib/types";

const API_BASE =
    process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ?? "http://127.0.0.1:8000";

async function parseError(response: Response): Promise<string> {
    try {
        const payload = await response.json();
        if (typeof payload.detail === "string") {
            return payload.detail;
        }
    } catch {
        // ignore parse errors and fall back to status text.
    }

    return response.statusText || `Request failed with status ${response.status}`;
}

export async function askQuestion(input: {
    query: string;
    threadId?: string;
}): Promise<AskResponse> {
    const response = await fetch(`${API_BASE}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            query: input.query,
            thread_id: input.threadId?.trim() || undefined,
        }),
    });

    if (!response.ok) {
        throw new Error(await parseError(response));
    }

    return (await response.json()) as AskResponse;
}

export async function uploadDocument(input: {
    file: File;
    conflictPolicy: ConflictPolicy;
}): Promise<UploadResponse> {
    const formData = new FormData();
    formData.set("file", input.file);
    formData.set("conflict_policy", input.conflictPolicy);

    const response = await fetch(`${API_BASE}/upload`, {
        method: "POST",
        body: formData,
    });

    if (!response.ok) {
        throw new Error(await parseError(response));
    }

    return (await response.json()) as UploadResponse;
}

export async function listDocuments(): Promise<DocumentRecord[]> {
    const response = await fetch(`${API_BASE}/documents`, { method: "GET" });

    if (!response.ok) {
        throw new Error(await parseError(response));
    }

    const payload = (await response.json()) as { documents: DocumentRecord[] };
    return payload.documents;
}

export async function deleteDocument(sourceName: string): Promise<void> {
    const response = await fetch(`${API_BASE}/documents/${encodeURIComponent(sourceName)}`, {
        method: "DELETE",
    });

    if (!response.ok) {
        throw new Error(await parseError(response));
    }
}

export async function clearDocuments(): Promise<number> {
    const response = await fetch(`${API_BASE}/documents`, { method: "DELETE" });

    if (!response.ok) {
        throw new Error(await parseError(response));
    }

    const payload = (await response.json()) as { deleted_count: number };
    return payload.deleted_count;
}

export async function listTraces(limit = 20): Promise<PipelineTrace[]> {
    const response = await fetch(`${API_BASE}/traces?limit=${limit}`, { method: "GET" });

    if (!response.ok) {
        throw new Error(await parseError(response));
    }

    return (await response.json()) as PipelineTrace[];
}
