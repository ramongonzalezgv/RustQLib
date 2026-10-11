// Promise-based RPC to the pricing worker.
import type { Op, Reply } from "./worker";
import type {
  Capability,
  EngineKind,
  ModelKind,
  PathSample,
  PricingRequest,
  Quote,
} from "./types";

export interface Timed<T> {
  result: T;
  ms: number;
}

export class CancelledError extends Error {
  constructor() {
    super("cancelled");
  }
}

export class Pricer {
  private worker: Worker;
  private nextId = 0;
  private pending = new Map<
    number,
    { resolve: (v: Timed<unknown>) => void; reject: (e: Error) => void }
  >();

  constructor() {
    this.worker = this.spawn();
  }

  /** Abandons all in-flight work (e.g. a long MC run whose inputs changed). */
  cancelAll(): void {
    if (this.pending.size === 0) return;
    this.worker.terminate();
    for (const entry of this.pending.values()) entry.reject(new CancelledError());
    this.pending.clear();
    this.worker = this.spawn();
  }

  private spawn(): Worker {
    const worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
    worker.onmessage = (event: MessageEvent<Reply>) => {
      const reply = event.data;
      const entry = this.pending.get(reply.id);
      if (!entry) return;
      this.pending.delete(reply.id);
      if (reply.ok) entry.resolve({ result: reply.result, ms: reply.ms });
      else entry.reject(new Error(reply.error));
    };
    worker.onerror = (event) => {
      const error = new Error(event.message || "pricing worker failed to start");
      for (const entry of this.pending.values()) entry.reject(error);
      this.pending.clear();
    };
    return worker;
  }

  private call<T>(op: Op): Promise<Timed<T>> {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve: resolve as (v: Timed<unknown>) => void, reject });
      this.worker.postMessage({ id, ...op });
    });
  }

  capabilities() {
    return this.call<Capability[]>({ op: "capabilities" });
  }
  price(request: PricingRequest, engine: EngineKind) {
    return this.call<Quote>({ op: "price", request, engine });
  }
  ttm(request: PricingRequest) {
    return this.call<number>({ op: "ttm", request });
  }
  priceCurve(request: PricingRequest, engine: EngineKind, xs: number[]) {
    return this.call<(number | null)[]>({ op: "priceCurve", request, engine, xs });
  }
  payoffCurve(request: PricingRequest, xs: number[]) {
    return this.call<number[]>({ op: "payoffCurve", request, xs });
  }
  smileEngine(model: ModelKind) {
    return this.call<EngineKind | null>({ op: "smileEngine", model });
  }
  smile(request: PricingRequest, strikes: number[]) {
    return this.call<(number | null)[]>({ op: "smile", request, strikes });
  }
  mcPaths(request: PricingRequest, nDisplay: number) {
    return this.call<PathSample>({ op: "mcPaths", request, nDisplay });
  }
}
