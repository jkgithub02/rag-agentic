export type ConflictPolicy = "ask" | "replace" | "keep_both";

export type UploadStatus = "success" | "conflict";

export interface UploadResponse {
    status: UploadStatus;
    message: string;
    original_filename: string;
    stored_filename: string | null;
    chunks_added: number | null;
    existing_filename: string | null;
    suggested_filename: string | null;
    conflict_options: ConflictPolicy[];
}

export interface AskResponse {
    answer: string;
    citations: string[];
    safe_fail: boolean;
    trace_id: string;
}

export type AskStreamEvent =
    | { type: "start"; trace_id?: string }
    | { type: "thinking"; status: string }
    | { type: "delta"; text: string }
    | { type: "done"; citations: string[]; safe_fail: boolean; trace_id: string }
    | { type: "error"; message: string };

export interface DocumentRecord {
    filename: string;
    size_bytes: number;
    chunks_indexed: number;
}

export interface PipelineTraceEvent {
    stage: string;
    payload: Record<string, unknown>;
    timestamp: string;
}

export interface PipelineTrace {
    trace_id: string;
    original_query: string;
    rewritten_query: string;
    retry_triggered: boolean;
    retry_reason: string | null;
    final_grounding_status: "supported" | "partial" | "unsupported";
    events: PipelineTraceEvent[];
}
