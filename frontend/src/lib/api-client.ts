import {
    type AskResponse,
    type AskStreamEvent,
    type ConflictPolicy,
    type DocumentRecord,
    type PipelineTrace,
    type UploadResponse,
} from "@/lib/types";

const API_BASE =
    process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ?? "http://127.0.0.1:8000";

export function getApiBaseUrl(): string {
    return API_BASE;
}

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

export async function askQuestionStream(
    input: { query: string; threadId?: string },
    onEvent: (event: AskStreamEvent) => void,
): Promise<void> {
    const response = await fetch(`${API_BASE}/ask/stream`, {
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

    if (!response.body) {
        throw new Error("Streaming response body is unavailable");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    const emitChunk = (chunk: string) => {
        const lines = chunk.split("\n");
        let eventType = "message";
        const dataLines: string[] = [];

        for (const line of lines) {
            if (line.startsWith("event:")) {
                eventType = line.slice(6).trim();
            } else if (line.startsWith("data:")) {
                dataLines.push(line.slice(5).trim());
            }
        }

        const data = dataLines.join("\n");
        if (!data) {
            return;
        }

        try {
            const payload = JSON.parse(data) as Record<string, unknown>;
            switch (eventType) {
                case "start":
                    onEvent({ type: "start", trace_id: String(payload.trace_id ?? "") || undefined });
                    break;
                case "delta":
                    onEvent({ type: "delta", text: String(payload.text ?? "") });
                    break;
                case "thinking":
                    onEvent({ type: "thinking", status: String(payload.status ?? "running") });
                    break;
                case "done":
                    onEvent({
                        type: "done",
                        citations: Array.isArray(payload.citations)
                            ? payload.citations.map((value) => String(value))
                            : [],
                        safe_fail: Boolean(payload.safe_fail),
                        trace_id: String(payload.trace_id ?? ""),
                    });
                    break;
                case "error":
                    onEvent({ type: "error", message: String(payload.message ?? "Streaming failed") });
                    break;
                default:
                    break;
            }
        } catch {
            onEvent({ type: "error", message: "Failed to parse streaming event" });
        }
    };

    while (true) {
        const { done, value } = await reader.read();
        if (done) {
            break;
        }

        buffer += decoder.decode(value, { stream: true });
        let separatorIndex = buffer.indexOf("\n\n");
        while (separatorIndex !== -1) {
            const chunk = buffer.slice(0, separatorIndex).trim();
            buffer = buffer.slice(separatorIndex + 2);
            if (chunk) {
                emitChunk(chunk);
            }
            separatorIndex = buffer.indexOf("\n\n");
        }
    }
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

export async function getTrace(traceId: string): Promise<PipelineTrace> {
    const response = await fetch(`${API_BASE}/trace/${encodeURIComponent(traceId)}`, { method: "GET" });

    if (!response.ok) {
        throw new Error(await parseError(response));
    }

    return (await response.json()) as PipelineTrace;
}

export async function checkBackendHealth(): Promise<boolean> {
    try {
        const response = await fetch(`${API_BASE}/health`, { method: "GET" });
        if (!response.ok) {
            return false;
        }

        const payload = (await response.json()) as { status?: string };
        return payload.status === "ok";
    } catch {
        return false;
    }
}
