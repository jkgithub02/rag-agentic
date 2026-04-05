import { expect, test } from "@playwright/test";

const API_BASE_URL = process.env.E2E_API_BASE_URL ?? "http://127.0.0.1:8000";

test.describe("Agentic RAG end-to-end flow", () => {
    test("backend capabilities work end-to-end", async ({ request }) => {
        const fileName = `e2e-${Date.now()}.md`;
        const marker = `E2E_MARKER_${Date.now()}`;

        const health = await request.get(`${API_BASE_URL}/health`);
        expect(health.ok()).toBeTruthy();

        const upload = await request.post(`${API_BASE_URL}/upload`, {
            multipart: {
                conflict_policy: "ask",
                file: {
                    name: fileName,
                    mimeType: "text/markdown",
                    buffer: Buffer.from(`# ${fileName}\n\n${marker} lives in this document.`),
                },
            },
        });
        expect(upload.ok()).toBeTruthy();

        const docsResponse = await request.get(`${API_BASE_URL}/documents`);
        expect(docsResponse.ok()).toBeTruthy();
        const docsPayload = (await docsResponse.json()) as { documents: Array<{ filename: string }> };
        expect(docsPayload.documents.some((doc) => doc.filename === fileName)).toBeTruthy();

        const tracesResponse = await request.get(`${API_BASE_URL}/traces?limit=5`);
        expect(tracesResponse.ok()).toBeTruthy();
        const traces = (await tracesResponse.json()) as unknown[];
        expect(Array.isArray(traces)).toBeTruthy();

        const deleteResponse = await request.delete(`${API_BASE_URL}/documents/${encodeURIComponent(fileName)}`);
        expect(deleteResponse.ok()).toBeTruthy();
    });

    test("frontend can reach backend from browser runtime", async ({ page }) => {
        await page.goto("/");
        await expect(page.getByRole("heading", { name: "Search, cite, and manage your knowledge base" })).toBeVisible();

        const browserHealth = await page.evaluate(async (apiBase) => {
            const response = await fetch(`${apiBase}/health`);
            if (!response.ok) {
                return false;
            }
            const payload = (await response.json()) as { status?: string };
            return payload.status === "ok";
        }, API_BASE_URL);

        expect(browserHealth).toBeTruthy();

        const corsPreflight = await page.request.fetch(`${API_BASE_URL}/ask`, {
            method: "OPTIONS",
            headers: {
                Origin: "http://127.0.0.1:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        });
        expect(corsPreflight.ok()).toBeTruthy();
    });
});
