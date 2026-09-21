export class HttpError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly retryable: boolean,
  ) {
    super(message);
    this.name = "HttpError";
  }
}

function redact(s: string): string {
  return s.replace(/Bearer\s+\S+/gi, "Bearer [redacted]");
}

function retryDelayMs(res: Response | undefined, attempt: number): number {
  const header = res?.headers.get("retry-after");
  if (header) {
    const seconds = Number(header);
    if (Number.isFinite(seconds) && seconds >= 0) return Math.min(seconds * 1000, 30_000);
    const when = Date.parse(header);
    if (!Number.isNaN(when)) return Math.max(0, Math.min(when - Date.now(), 30_000));
  }
  return Math.min(1000 * 2 ** attempt, 8_000);
}

function throwIfAborted(signal?: AbortSignal) {
  if (!signal?.aborted) return;
  const err = new Error("Request cancelled");
  err.name = "AbortError";
  throw err;
}

export interface PostJsonOptions {
  apiKey: string;
  timeoutMs: number;
  maxRetries: number;
  signal?: AbortSignal;
}

export async function postJson<T>(url: string, body: unknown, opts: PostJsonOptions): Promise<T> {
  let lastErr: unknown;
  for (let attempt = 0; attempt <= opts.maxRetries; attempt++) {
    throwIfAborted(opts.signal);
    const ac = new AbortController();
    const timer = setTimeout(() => ac.abort(), opts.timeoutMs);
    const onAbort = () => ac.abort();
    opts.signal?.addEventListener("abort", onAbort, { once: true });
    try {
      const res = await fetch(url, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${opts.apiKey}`,
        },
        body: JSON.stringify(body),
        signal: ac.signal,
      });
      if (res.ok) return await res.json() as T;
      const text = redact(await res.text().catch(() => ""));
      const retryable = res.status === 429 || (res.status >= 500 && res.status <= 599);
      const err = new HttpError(`HTTP ${res.status}: ${text.slice(0, 200)}`, res.status, retryable);
      if (!retryable || attempt === opts.maxRetries) throw err;
      lastErr = err;
      await new Promise(r => setTimeout(r, retryDelayMs(res, attempt)));
    } catch (err) {
      if ((err as Error).name === "AbortError") {
        if (opts.signal?.aborted) throw err;
        lastErr = new HttpError(`HTTP timeout after ${opts.timeoutMs}ms`, 0, true);
        if (attempt === opts.maxRetries) throw lastErr;
        await new Promise(r => setTimeout(r, retryDelayMs(undefined, attempt)));
        continue;
      }
      if (err instanceof HttpError) {
        if (!err.retryable || attempt === opts.maxRetries) throw err;
        lastErr = err;
        await new Promise(r => setTimeout(r, retryDelayMs(undefined, attempt)));
        continue;
      }
      throw err;
    } finally {
      clearTimeout(timer);
      opts.signal?.removeEventListener("abort", onAbort);
    }
  }
  throw lastErr instanceof Error ? lastErr : new Error(String(lastErr));
}
