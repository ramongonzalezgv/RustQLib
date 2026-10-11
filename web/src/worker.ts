// Runs the WASM pricer off the main thread so long Monte Carlo runs never
// block the UI.
import init, * as ql from "./wasm/quantlib_wasm.js";

export type Op =
  | { op: "capabilities" }
  | { op: "price"; request: unknown; engine: string }
  | { op: "ttm"; request: unknown }
  | { op: "priceCurve"; request: unknown; engine: string; xs: number[] }
  | { op: "payoffCurve"; request: unknown; xs: number[] }
  | { op: "smileEngine"; model: string }
  | { op: "smile"; request: unknown; strikes: number[] }
  | { op: "mcPaths"; request: unknown; nDisplay: number };

export type Reply =
  | { id: number; ok: true; result: unknown; ms: number }
  | { id: number; ok: false; error: string };

const ready = init();

function run(msg: Op): unknown {
  switch (msg.op) {
    case "capabilities":
      return ql.capabilities();
    case "price":
      return ql.price(msg.request, msg.engine);
    case "ttm":
      return ql.ttm(msg.request);
    case "priceCurve":
      return ql.priceCurve(msg.request, msg.engine, Float64Array.from(msg.xs));
    case "payoffCurve":
      return Array.from(ql.payoffCurve(msg.request, Float64Array.from(msg.xs)));
    case "smileEngine":
      return ql.smileEngine(msg.model);
    case "smile":
      return ql.smile(msg.request, Float64Array.from(msg.strikes));
    case "mcPaths":
      return ql.mcPaths(msg.request, msg.nDisplay);
  }
}

self.onmessage = async (event: MessageEvent<{ id: number } & Op>) => {
  const { id } = event.data;
  try {
    await ready;
    const start = performance.now();
    const result = run(event.data);
    const reply: Reply = { id, ok: true, result, ms: performance.now() - start };
    self.postMessage(reply);
  } catch (e) {
    const reply: Reply = { id, ok: false, error: e instanceof Error ? e.message : String(e) };
    self.postMessage(reply);
  }
};
