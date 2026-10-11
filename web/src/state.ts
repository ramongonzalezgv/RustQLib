// Form metadata, defaults and shareable-URL encoding.
import type { EngineKind, ModelKind, ModelSpec, PricingRequest, ProductKind } from "./types";

export interface ParamMeta {
  key: string;
  label: string;
  /** Symbol shown next to the label, e.g. "κ". */
  symbol: string;
  value: number;
  step: number;
  min?: number;
  max?: number;
  help: string;
}

export interface ModelMeta {
  kind: ModelKind;
  label: string;
  /** Forward-quoted models take F instead of S and ignore q. */
  forward: boolean;
  params: ParamMeta[];
}

export const MODELS: ModelMeta[] = [
  {
    kind: "blackScholes",
    label: "Black–Scholes",
    forward: false,
    params: [
      { key: "sigma", label: "Volatility", symbol: "σ", value: 0.2, step: 0.01, min: 0.001, help: "Annualised lognormal volatility" },
    ],
  },
  {
    kind: "black76",
    label: "Black-76",
    forward: true,
    params: [
      { key: "sigma", label: "Volatility", symbol: "σ", value: 0.2, step: 0.01, min: 0.001, help: "Annualised lognormal volatility of the forward" },
    ],
  },
  {
    kind: "heston",
    label: "Heston",
    forward: false,
    params: [
      { key: "kappa", label: "Mean reversion", symbol: "κ", value: 2, step: 0.1, min: 0, help: "Speed at which variance reverts to θ" },
      { key: "theta", label: "Long-run variance", symbol: "θ", value: 0.04, step: 0.005, min: 0, help: "Long-run variance (0.04 = 20% vol)" },
      { key: "sigma", label: "Vol of vol", symbol: "ξ", value: 0.3, step: 0.05, min: 0, help: "Volatility of the variance process" },
      { key: "rho", label: "Correlation", symbol: "ρ", value: -0.7, step: 0.05, min: -0.999, max: 0.999, help: "Spot/variance correlation (skew)" },
      { key: "v0", label: "Initial variance", symbol: "v₀", value: 0.04, step: 0.005, min: 0, help: "Variance today" },
    ],
  },
  {
    kind: "bates",
    label: "Bates",
    forward: false,
    params: [
      { key: "kappa", label: "Mean reversion", symbol: "κ", value: 2, step: 0.1, min: 0, help: "Speed at which variance reverts to θ" },
      { key: "theta", label: "Long-run variance", symbol: "θ", value: 0.04, step: 0.005, min: 0, help: "Long-run variance" },
      { key: "sigma", label: "Vol of vol", symbol: "ξ", value: 0.3, step: 0.05, min: 0, help: "Volatility of the variance process" },
      { key: "rho", label: "Correlation", symbol: "ρ", value: -0.7, step: 0.05, min: -0.999, max: 0.999, help: "Spot/variance correlation" },
      { key: "v0", label: "Initial variance", symbol: "v₀", value: 0.04, step: 0.005, min: 0, help: "Variance today" },
      { key: "lambda", label: "Jump intensity", symbol: "λ", value: 0.5, step: 0.1, min: 0, help: "Expected jumps per year" },
      { key: "muJ", label: "Mean jump", symbol: "μⱼ", value: -0.1, step: 0.01, help: "Mean of log jump size" },
      { key: "sigmaJ", label: "Jump vol", symbol: "σⱼ", value: 0.15, step: 0.01, min: 0, help: "Std. dev. of log jump size" },
    ],
  },
  {
    kind: "kou",
    label: "Kou",
    forward: false,
    params: [
      { key: "sigma", label: "Diffusion vol", symbol: "σ", value: 0.2, step: 0.01, min: 0.001, help: "Volatility of the diffusion part" },
      { key: "lambda", label: "Jump intensity", symbol: "λ", value: 1, step: 0.1, min: 0, help: "Expected jumps per year" },
      { key: "pUp", label: "Up-jump prob.", symbol: "p", value: 0.4, step: 0.05, min: 0, max: 1, help: "Probability a jump is upward" },
      { key: "eta1", label: "Up-jump rate", symbol: "η₁", value: 10, step: 0.5, min: 1.001, help: "Exponential rate of up jumps (> 1)" },
      { key: "eta2", label: "Down-jump rate", symbol: "η₂", value: 5, step: 0.5, min: 0.001, help: "Exponential rate of down jumps" },
    ],
  },
  {
    kind: "sabr",
    label: "SABR",
    forward: true,
    params: [
      { key: "alpha", label: "Alpha", symbol: "α", value: 2, step: 0.1, min: 0.0001, help: "Initial volatility level" },
      { key: "beta", label: "Beta", symbol: "β", value: 0.5, step: 0.1, min: 0, max: 1, help: "CEV exponent: 1 lognormal, 0 normal (then α is a normal vol)" },
      { key: "rho", label: "Correlation", symbol: "ρ", value: -0.3, step: 0.05, min: -0.999, max: 0.999, help: "Forward/vol correlation" },
      { key: "nu", label: "Vol of vol", symbol: "ν", value: 0.4, step: 0.05, min: 0, help: "Volatility of volatility" },
    ],
  },
];

export const PRODUCTS: { kind: ProductKind; label: string }[] = [
  { kind: "european", label: "European" },
  { kind: "american", label: "American" },
  { kind: "digital", label: "Digital" },
  { kind: "asian", label: "Asian" },
];

export const ENGINES: { kind: EngineKind; label: string; detail: string }[] = [
  { kind: "analytical", label: "Analytical", detail: "Closed form" },
  { kind: "fft", label: "FFT", detail: "Carr–Madan / Lewis" },
  { kind: "binomial", label: "Binomial", detail: "CRR tree" },
  { kind: "monteCarlo", label: "Monte Carlo", detail: "8 seeded batches" },
];

export function modelMeta(kind: ModelKind): ModelMeta {
  return MODELS.find((m) => m.kind === kind)!;
}

export function defaultModel(kind: ModelKind): ModelSpec {
  const spec: Record<string, number | string> = { kind };
  for (const p of modelMeta(kind).params) spec[p.key] = p.value;
  return spec as unknown as ModelSpec;
}

function isoDate(d: Date): string {
  return d.toISOString().slice(0, 10);
}

export function defaultRequest(): PricingRequest {
  const today = new Date();
  const expiry = new Date(today);
  expiry.setFullYear(today.getFullYear() + 1);
  return {
    product: {
      kind: "european",
      optionType: "call",
      strike: 100,
      valuationDate: isoDate(today),
      expiryDate: isoDate(expiry),
      digitalPayoff: "cash",
      payout: 1,
      averaging: "arithmetic",
    },
    model: defaultModel("blackScholes"),
    market: { underlying: 100, r: 0.05, q: 0 },
    settings: { binomialSteps: 500, mcPaths: 50_000, mcSteps: 252, mcSeed: 42, fftN: 4096 },
  };
}

// --- Shareable URLs -------------------------------------------------------
// The whole request is stored as base64url JSON in `?s=`. Unknown or invalid
// fields fall back to defaults so old links keep working.

export function encodeState(req: PricingRequest): string {
  const bytes = new TextEncoder().encode(JSON.stringify(req));
  let bin = "";
  for (const b of bytes) bin += String.fromCharCode(b);
  return btoa(bin).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

export function decodeState(encoded: string): PricingRequest | null {
  try {
    const b64 = encoded.replace(/-/g, "+").replace(/_/g, "/");
    const bin = atob(b64);
    const bytes = Uint8Array.from(bin, (c) => c.charCodeAt(0));
    const parsed = JSON.parse(new TextDecoder().decode(bytes)) as Partial<PricingRequest>;
    const base = defaultRequest();
    const kind = parsed.model?.kind;
    if (!kind || !MODELS.some((m) => m.kind === kind)) return null;
    return {
      product: { ...base.product, ...parsed.product },
      model: { ...defaultModel(kind), ...parsed.model } as ModelSpec,
      market: { ...base.market, ...parsed.market },
      settings: { ...base.settings, ...parsed.settings },
    };
  } catch {
    return null;
  }
}
