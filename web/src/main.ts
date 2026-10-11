import "./style.css";
import { CancelledError, Pricer } from "./client";
import { echarts, fmt, pathsOption, payoffOption, smileOption } from "./charts";
import {
  ENGINES,
  MODELS,
  PRODUCTS,
  decodeState,
  defaultModel,
  defaultRequest,
  encodeState,
  modelMeta,
} from "./state";
import type { Capability, EngineKind, ModelKind, PricingRequest, ProductKind, Quote } from "./types";

type Tab = "payoff" | "paths" | "smile";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

let req: PricingRequest =
  decodeState(new URLSearchParams(location.search).get("s") ?? "") ?? defaultRequest();
let caps: Capability[] = [];
let tab: Tab = "payoff";
/** Incremented on every input change; stale async results are dropped. */
let runId = 0;
let results = new Map<EngineKind, { quote?: Quote; ms?: number; error?: string }>();

const pricer = new Pricer();
const chart = echarts.init($("chart"));
new ResizeObserver(() => chart.resize()).observe($("chart"));

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function enginesFor(product: ProductKind, model: ModelKind): EngineKind[] {
  return caps.find((c) => c.product === product && c.model === model)?.engines ?? [];
}

function supported(): EngineKind[] {
  return enginesFor(req.product.kind, req.model.kind);
}

/** First supported engine in table order; it is the comparison baseline. */
function referenceEngine(): EngineKind | undefined {
  return ENGINES.map((e) => e.kind).find((e) => supported().includes(e));
}

function engineLabel(kind: EngineKind): string {
  return ENGINES.find((e) => e.kind === kind)!.label;
}

function isForwardModel(): boolean {
  return modelMeta(req.model.kind).forward;
}

function underlyingSymbol(): string {
  return isForwardModel() ? "F" : "S";
}

function escapeHtml(s: string): string {
  return s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]!);
}

function getPath(path: string): unknown {
  return path.split(".").reduce<any>((o, k) => o?.[k], req);
}

function setPath(path: string, value: unknown): void {
  const keys = path.split(".");
  const last = keys.pop()!;
  const target = keys.reduce<any>((o, k) => o[k], req);
  target[last] = value;
}

function toast(message: string): void {
  const el = $("toast");
  el.textContent = message;
  el.hidden = false;
  window.setTimeout(() => (el.hidden = true), 1800);
}

// ---------------------------------------------------------------------------
// Inputs
// ---------------------------------------------------------------------------

function numberField(path: string, label: string, opts: { step?: number; min?: number; max?: number; symbol?: string; help?: string } = {}): string {
  const value = getPath(path) as number;
  const attrs = [
    `type="number"`,
    `id="f-${path}"`,
    `data-path="${path}"`,
    `value="${value}"`,
    `step="${opts.step ?? "any"}"`,
    opts.min !== undefined ? `min="${opts.min}"` : "",
    opts.max !== undefined ? `max="${opts.max}"` : "",
    `inputmode="decimal"`,
  ].join(" ");
  const sym = opts.symbol ? `<span class="sym">${opts.symbol}</span>` : "";
  const title = opts.help ? ` title="${escapeHtml(opts.help)}"` : "";
  return `<div class="field"${title}><label for="f-${path}"><span>${label}</span>${sym}</label><input ${attrs} /></div>`;
}

function dateField(path: string, label: string): string {
  return `<div class="field"><label for="f-${path}">${label}</label><input type="date" id="f-${path}" data-path="${path}" value="${getPath(path)}" /></div>`;
}

function segmented(path: string, options: { value: string; label: string }[], ariaLabel: string): string {
  const current = getPath(path);
  const buttons = options
    .map((o) => `<button type="button" data-set="${path}" data-value="${o.value}" aria-pressed="${o.value === current}">${o.label}</button>`)
    .join("");
  return `<div class="segmented" role="group" aria-label="${ariaLabel}">${buttons}</div>`;
}

function renderInputs(): void {
  const meta = modelMeta(req.model.kind);
  const p = req.product;

  const modelOptions = MODELS.map((m) => {
    const available = enginesFor(p.kind, m.kind).length > 0;
    const suffix = available ? "" : " — n/a for this product";
    return `<option value="${m.kind}" ${m.kind === req.model.kind ? "selected" : ""}>${m.label}${suffix}</option>`;
  }).join("");

  const productExtras =
    p.kind === "digital"
      ? `<div class="grid-2">
          <div class="field"><label for="f-digital">Pays</label>
            <select id="f-digital" data-path="product.digitalPayoff">
              <option value="cash" ${p.digitalPayoff === "cash" ? "selected" : ""}>Cash</option>
              <option value="asset" ${p.digitalPayoff === "asset" ? "selected" : ""}>Asset</option>
            </select></div>
          ${p.digitalPayoff === "cash" ? numberField("product.payout", "Payout", { step: 1, min: 0 }) : "<div></div>"}
        </div>`
      : p.kind === "asian"
        ? `<div class="field"><label for="f-avg">Averaging</label>
            <select id="f-avg" data-path="product.averaging">
              <option value="arithmetic" ${p.averaging === "arithmetic" ? "selected" : ""}>Arithmetic</option>
              <option value="geometric" ${p.averaging === "geometric" ? "selected" : ""}>Geometric</option>
            </select></div>`
        : "";

  const params = meta.params
    .map((pm) => numberField(`model.${pm.key}`, pm.label, { step: pm.step, min: pm.min, max: pm.max, symbol: pm.symbol, help: pm.help }))
    .join("");

  $("inputs").innerHTML = `
    <div class="section">
      <h3 class="section-title">Product</h3>
      ${segmented("product.kind", PRODUCTS.map((x) => ({ value: x.kind, label: x.label })), "Product")}
      ${segmented("product.optionType", [{ value: "call", label: "Call" }, { value: "put", label: "Put" }], "Option type")}
      ${numberField("product.strike", "Strike", { step: 1, min: 0, symbol: "K" })}
      <div class="grid-2">
        ${dateField("product.valuationDate", "Valuation")}
        ${dateField("product.expiryDate", "Expiry")}
      </div>
      ${productExtras}
    </div>
    <div class="section">
      <h3 class="section-title">Market</h3>
      <div class="grid-2">
        ${numberField("market.underlying", isForwardModel() ? "Forward" : "Spot", { step: 1, min: 0, symbol: underlyingSymbol() })}
        ${numberField("market.r", "Rate", { step: 0.005, symbol: "r", help: "Continuously compounded risk-free rate" })}
      </div>
      ${isForwardModel() ? "" : `<div class="grid-2">${numberField("market.q", "Dividend yield", { step: 0.005, symbol: "q", help: "Continuous dividend yield" })}<div></div></div>`}
    </div>
    <div class="section">
      <h3 class="section-title">Model</h3>
      <div class="field"><label for="f-model">Model</label><select id="f-model" data-model>${modelOptions}</select></div>
      <div class="grid-2">${params}</div>
    </div>
    <div class="section">
      <details class="settings">
        <summary><h3 class="section-title">Engine settings</h3></summary>
        <div class="grid-2">
          ${numberField("settings.mcPaths", "MC paths", { step: 1000, min: 8 })}
          ${numberField("settings.mcSteps", "MC steps", { step: 1, min: 1 })}
        </div>
        <div class="grid-2">
          ${numberField("settings.mcSeed", "MC seed", { step: 1, min: 0 })}
          ${numberField("settings.binomialSteps", "Tree steps", { step: 50, min: 1 })}
        </div>
        <div class="grid-2">
          <div class="field"><label for="f-fft">FFT points</label>
            <select id="f-fft" data-path="settings.fftN" data-number>
              ${[1024, 2048, 4096, 8192, 16384].map((n) => `<option value="${n}" ${n === req.settings.fftN ? "selected" : ""}>${n}</option>`).join("")}
            </select></div>
          <div></div>
        </div>
      </details>
    </div>`;
}

function onInput(event: Event): void {
  const el = event.target as HTMLInputElement | HTMLSelectElement;

  if (el.matches("[data-model]")) {
    req.model = defaultModel(el.value as ModelKind);
    renderInputs();
    schedule(0);
    return;
  }

  const path = el.dataset.path;
  if (!path) return;

  if (el instanceof HTMLInputElement && el.type === "number") {
    const value = el.valueAsNumber;
    const invalid = !Number.isFinite(value) || (el.min !== "" && value < Number(el.min)) || (el.max !== "" && value > Number(el.max));
    el.setAttribute("aria-invalid", String(invalid));
    if (invalid) return;
    setPath(path, value);
  } else if (el.dataset.number !== undefined) {
    setPath(path, Number(el.value));
  } else {
    setPath(path, el.value);
  }

  // Fields whose value changes which other fields are shown.
  if (path === "product.digitalPayoff") renderInputs();
  schedule();
}

function onClick(event: Event): void {
  const button = (event.target as HTMLElement).closest<HTMLButtonElement>("button[data-set]");
  if (!button) return;
  setPath(button.dataset.set!, button.dataset.value);
  renderInputs();
  schedule(0);
}

$("inputs").addEventListener("input", onInput);
$("inputs").addEventListener("change", (e) => {
  if ((e.target as HTMLElement).matches("select")) onInput(e);
});
$("inputs").addEventListener("click", onClick);

// ---------------------------------------------------------------------------
// Engine table & hero
// ---------------------------------------------------------------------------

function renderNotice(): void {
  const notices: string[] = [];
  if (supported().length === 0) {
    const product = PRODUCTS.find((x) => x.kind === req.product.kind)!.label;
    const models = MODELS.filter((m) => enginesFor(req.product.kind, m.kind).length > 0).map((m) => m.label);
    notices.push(`No RustQLib engine prices ${product} options under ${modelMeta(req.model.kind).label}. Available models: ${models.join(", ")}.`);
  }
  if (req.model.kind === "sabr") {
    notices.push(
      `SABR uses Hagan's implied-vol formulas, which have known errors in this version (<a href="https://github.com/ramongonzalezgv/RustQLib/issues/1" target="_blank" rel="noopener">#1</a>). Treat SABR prices as indicative.`,
    );
  }
  const el = $("notice");
  el.hidden = notices.length === 0;
  el.innerHTML = notices.join("<br />");
}

function renderRows(): void {
  const ref = referenceEngine();
  const refQuote = ref ? results.get(ref)?.quote : undefined;

  $("engine-rows").innerHTML = ENGINES.map((e) => {
    const name = `<span class="engine-name">${e.label}${e.kind === ref ? '<span class="badge">ref</span>' : ""}</span><span class="engine-detail">${e.detail}</span>`;
    if (!supported().includes(e.kind)) {
      return `<tr class="unsupported"><td>${name}</td><td class="num">—</td><td class="num">n/a</td><td class="num col-time"></td></tr>`;
    }
    const r = results.get(e.kind);
    if (!r) {
      return `<tr><td>${name}</td><td class="num"><span class="spinner sm"></span></td><td class="num"></td><td class="num col-time"></td></tr>`;
    }
    if (r.error) {
      return `<tr><td>${name}</td><td colspan="3" class="err">${escapeHtml(r.error)}</td></tr>`;
    }
    const q = r.quote!;
    const pm = q.stderr !== null ? ` <span class="pm">± ${fmt(3 * q.stderr, 3)}</span>` : "";
    let diff = "";
    if (e.kind === ref) {
      diff = '<span class="muted">—</span>';
    } else if (refQuote) {
      const d = q.price - refQuote.price;
      if (q.stderr !== null && q.stderr > 0) {
        const z = Math.abs(d) / q.stderr;
        const cls = z < 3 ? "diff-ok" : "diff-warn";
        diff = `<span class="${cls}" title="${z.toFixed(2)} standard errors">${d >= 0 ? "+" : ""}${fmt(d, 4)} (${z.toFixed(1)}σ)</span>`;
      } else {
        diff = `${d >= 0 ? "+" : ""}${fmt(d, 6)}`;
      }
    }
    const time = r.ms !== undefined ? `${r.ms < 10 ? r.ms.toFixed(2) : r.ms.toFixed(0)} ms` : "";
    return `<tr><td>${name}</td><td class="num">${fmt(q.price, 6)}${pm}</td><td class="num">${diff}</td><td class="num muted col-time">${time}</td></tr>`;
  }).join("");

  const mc = supported().includes("monteCarlo");
  $("engine-caption").textContent = mc
    ? `MC: ${req.settings.mcPaths.toLocaleString()} paths, ± is 3 standard errors`
    : "";
}

function renderHero(ttm: number | null): void {
  const ref = referenceEngine();
  const quote = ref ? results.get(ref)?.quote : undefined;
  $("hero-label").textContent = ref ? `Price · ${engineLabel(ref)}` : "Price";
  $("hero-price").textContent = quote ? fmt(quote.price, 6) : "—";

  const { underlying, r, q } = req.market;
  const facts: [string, string][] = [];
  if (ttm !== null) {
    const forward = isForwardModel() ? underlying : underlying * Math.exp((r - q) * ttm);
    facts.push(["Maturity", `${ttm.toFixed(4)} y`], ["Forward", fmt(forward, 4)], ["Discount", fmt(Math.exp(-r * ttm), 6)]);
  }
  facts.push(["Moneyness K/F", ttm !== null ? fmt(req.product.strike / (isForwardModel() ? underlying : underlying * Math.exp((r - q) * ttm)), 4) : "—"]);
  $("hero-facts").innerHTML = facts.map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join("");
}

// ---------------------------------------------------------------------------
// Charts
// ---------------------------------------------------------------------------

function chartMessage(message: string | null): void {
  const el = $("chart-empty");
  el.hidden = message === null;
  el.textContent = message ?? "";
}

function chartBusy(busy: boolean): void {
  $("chart-busy").hidden = !busy;
}

function linspace(a: number, b: number, n: number): number[] {
  return Array.from({ length: n }, (_, i) => a + ((b - a) * i) / (n - 1));
}

async function renderChart(id: number, ttm: number | null): Promise<void> {
  chartBusy(true);
  chartMessage(null);
  $("chart-caption").textContent = "";
  try {
    if (tab === "payoff") await renderPayoff(id);
    else if (tab === "paths") await renderPaths(id);
    else await renderSmile(id, ttm);
  } catch (e) {
    if (e instanceof CancelledError || id !== runId) return;
    chart.clear();
    chartMessage(e instanceof Error ? e.message : String(e));
  } finally {
    if (id === runId) chartBusy(false);
  }
}

async function renderPayoff(id: number): Promise<void> {
  const { strike } = req.product;
  const u = req.market.underlying;
  const ref = referenceEngine();
  const mcOnly = ref === "monteCarlo";
  const xs = linspace(0.4 * Math.min(strike, u), 1.6 * Math.max(strike, u), mcOnly ? 41 : 121);

  const payoff = (await pricer.payoffCurve(req, xs)).result;
  let value: (number | null)[] | null = null;
  if (ref) {
    // Monte Carlo per grid point is expensive: use a lighter configuration.
    const curveReq: PricingRequest = mcOnly
      ? { ...req, settings: { ...req.settings, mcPaths: 4000, mcSteps: Math.min(req.settings.mcSteps, 50) } }
      : req;
    value = (await pricer.priceCurve(curveReq, ref, xs)).result;
  }
  if (id !== runId) return;

  chart.setOption(
    payoffOption({
      xs,
      payoff,
      value,
      valueLabel: ref ? `Value today (${engineLabel(ref)})` : "",
      strike,
      underlying: u,
      underlyingLabel: `${underlyingSymbol()} (underlying)`,
    }),
    true,
  );
  const captions = [
    mcOnly
      ? "Value today from Monte Carlo with 4,000 paths per point"
      : ref
        ? `Value today from the ${engineLabel(ref)} engine`
        : "",
    req.product.kind === "asian" ? "payoff plotted against the realised average" : "",
  ];
  $("chart-caption").textContent = captions.filter(Boolean).join(" · ");
}

async function renderPaths(id: number): Promise<void> {
  if (req.model.kind !== "blackScholes" && req.model.kind !== "heston") {
    chart.clear();
    chartMessage("Path simulation is available for the Black–Scholes and Heston models.");
    return;
  }
  const nDisplay = 120;
  const sample = (await pricer.mcPaths(req, nDisplay)).result;
  if (id !== runId) return;
  chart.setOption(pathsOption({ sample, strike: req.product.strike }), true);
  const simulated = Math.min(Math.max(req.settings.mcPaths, nDisplay), 10_000);
  const steps = Math.min(req.settings.mcSteps, 252);
  $("chart-caption").textContent = `${sample.paths.length} of ${simulated.toLocaleString()} simulated paths · ${steps} steps · bands and histogram use all paths`;
}

async function renderSmile(id: number, ttm: number | null): Promise<void> {
  const engine = (await pricer.smileEngine(req.model.kind)).result;
  if (!engine || ttm === null) {
    chart.clear();
    chartMessage("This model has no European engine to build a smile from.");
    return;
  }
  const { underlying, r, q } = req.market;
  const forward = isForwardModel() ? underlying : underlying * Math.exp((r - q) * ttm);
  const strikes = linspace(0.6 * forward, 1.6 * forward, 81);
  const vols = (await pricer.smile(req, strikes)).result;
  if (id !== runId) return;
  chart.setOption(
    smileOption({ strikes, vols, forward, strike: req.product.strike, label: `${modelMeta(req.model.kind).label} implied vol` }),
    true,
  );
  $("chart-caption").textContent = `Black implied vol of out-of-the-money European options · ${engineLabel(engine)} engine`;
}

// ---------------------------------------------------------------------------
// Run loop
// ---------------------------------------------------------------------------

let timer: number | undefined;
let lastTtm: number | null = null;

function schedule(delay = 250): void {
  window.clearTimeout(timer);
  timer = window.setTimeout(run, delay);
}

function updateUrl(): void {
  const url = new URL(location.href);
  url.searchParams.set("s", encodeState(req));
  history.replaceState(null, "", url);
}

async function run(): Promise<void> {
  const id = ++runId;
  pricer.cancelAll();
  updateUrl();
  results = new Map();
  renderNotice();
  renderRows();

  try {
    lastTtm = (await pricer.ttm(req)).result;
  } catch (e) {
    if (e instanceof CancelledError || id !== runId) return;
    lastTtm = null;
    for (const engine of supported()) results.set(engine, { error: e instanceof Error ? e.message : String(e) });
    renderRows();
    renderHero(null);
    chart.clear();
    chartMessage(e instanceof Error ? e.message : String(e));
    return;
  }
  renderHero(lastTtm);

  // Fast engines first, then the chart, then Monte Carlo (the slow one).
  const engines = supported();
  const fast = engines.filter((e) => e !== "monteCarlo");
  const slow = engines.filter((e) => e === "monteCarlo");

  const priceOne = async (engine: EngineKind) => {
    try {
      const { result, ms } = await pricer.price(req, engine);
      if (id !== runId) return;
      results.set(engine, { quote: result, ms });
    } catch (e) {
      if (e instanceof CancelledError || id !== runId) return;
      results.set(engine, { error: e instanceof Error ? e.message : String(e) });
    }
    renderRows();
    renderHero(lastTtm);
  };

  for (const engine of fast) await priceOne(engine);
  if (id !== runId) return;
  const chartDone = renderChart(id, lastTtm);
  for (const engine of slow) await priceOne(engine);
  await chartDone;
}

// ---------------------------------------------------------------------------
// Tabs, share, theme
// ---------------------------------------------------------------------------

$("tabs").addEventListener("click", (event) => {
  const button = (event.target as HTMLElement).closest<HTMLButtonElement>("button[data-tab]");
  if (!button || button.dataset.tab === tab) return;
  tab = button.dataset.tab as Tab;
  for (const b of $("tabs").querySelectorAll("button")) b.setAttribute("aria-selected", String(b === button));
  void renderChart(runId, lastTtm);
});

$("share").addEventListener("click", async () => {
  updateUrl();
  try {
    await navigator.clipboard.writeText(location.href);
    toast("Link copied");
  } catch {
    toast("Copy the address bar to share");
  }
});

const darkQuery = window.matchMedia("(prefers-color-scheme: dark)");
darkQuery.addEventListener("change", () => {
  void renderChart(runId, lastTtm);
});

/** Explicit choice overrides the OS setting; remembered per browser. */
$("theme").addEventListener("click", () => {
  const root = document.documentElement;
  const isDark = root.dataset.theme ? root.dataset.theme === "dark" : darkQuery.matches;
  const next = isDark ? "light" : "dark";
  root.dataset.theme = next;
  try {
    localStorage.setItem("theme", next);
  } catch {
    // Storage unavailable (private mode): the choice lasts for this page only.
  }
  void renderChart(runId, lastTtm);
});

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

async function boot(): Promise<void> {
  try {
    caps = (await pricer.capabilities()).result;
  } catch (e) {
    $("notice").hidden = false;
    $("notice").textContent = `Failed to load the WebAssembly pricer: ${e instanceof Error ? e.message : String(e)}`;
    return;
  }
  renderInputs();
  void run();
}

void boot();
