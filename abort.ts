/** Abort / deadline helpers shared by retrieval, indexing, and HTTP. */

export function isAbortError(err: unknown): boolean {
  if (!err || typeof err !== "object") return false;
  const name = (err as { name?: string }).name;
  if (name === "AbortError") return true;
  const msg = (err as { message?: string }).message ?? "";
  return /aborted|cancelled|canceled/i.test(msg);
}

export function throwIfAborted(signal?: AbortSignal, message = "Operation cancelled"): void {
  if (!signal?.aborted) return;
  const err = new Error(message);
  err.name = "AbortError";
  throw err;
}

export function abortError(message = "Operation cancelled"): Error {
  const err = new Error(message);
  err.name = "AbortError";
  return err;
}

/** Merge an optional parent signal with a wall-clock deadline. */
export function withDeadline(parent: AbortSignal | undefined, deadlineMs: number): AbortSignal {
  const timeout = AbortSignal.timeout(deadlineMs);
  if (!parent) return timeout;
  return AbortSignal.any([parent, timeout]);
}

export function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  if (ms <= 0) return Promise.resolve();
  throwIfAborted(signal);
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    const onAbort = () => {
      clearTimeout(timer);
      reject(abortError("Request cancelled"));
    };
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}
