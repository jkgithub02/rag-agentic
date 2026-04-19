# Agentic RAG Frontend

Next.js frontend for the Agentic RAG system.

## Features

- Chat UI with synchronous and streaming answer support
- Document upload with conflict policy handling (`ask`, `replace`, `keep_both`)
- Knowledge base view to list and delete indexed documents
- Trace browsing support through backend observability endpoints

## Requirements

- Node.js 20+
- Running backend API (default: `http://127.0.0.1:8000`)

## Environment

Create `frontend/.env.local`:

```bash
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
```

If omitted, the app defaults to `http://127.0.0.1:8000`.

## Run Locally

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`.

## Build

```bash
npm run build
npm run start
```

## Main UI Modules

- `src/features/chat/chat-tab.tsx`: query input, chat timeline, citation display
- `src/features/upload/upload-tab.tsx`: file upload and conflict resolution UX
- `src/features/knowledge/knowledge-base-tab.tsx`: indexed document list/delete
- `src/lib/api-client.ts`: backend API client (`/ask`, `/ask/stream`, `/upload`, `/documents`, `/trace`, `/traces`)

## Backend Dependency Notes

- Backend CORS defaults already allow `http://127.0.0.1:3000` and `http://localhost:3000`
- Streaming uses SSE-style chunk parsing from `POST /ask/stream`
- Trace panel functions rely on `GET /traces` and `GET /trace/{trace_id}`
