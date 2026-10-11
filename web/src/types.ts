// Mirrors the serde types in quantlib_wasm/src/api.rs.

export type ProductKind = "european" | "american" | "digital" | "asian";
export type EngineKind = "analytical" | "fft" | "binomial" | "monteCarlo";
export type ModelKind = "blackScholes" | "black76" | "heston" | "bates" | "kou" | "sabr";
export type CallPut = "call" | "put";

export type ModelSpec =
  | { kind: "blackScholes"; sigma: number }
  | { kind: "black76"; sigma: number }
  | { kind: "heston"; kappa: number; theta: number; sigma: number; rho: number; v0: number }
  | {
      kind: "bates";
      kappa: number;
      theta: number;
      sigma: number;
      rho: number;
      v0: number;
      lambda: number;
      muJ: number;
      sigmaJ: number;
    }
  | { kind: "kou"; sigma: number; lambda: number; pUp: number; eta1: number; eta2: number }
  | { kind: "sabr"; alpha: number; beta: number; rho: number; nu: number };

export interface ProductSpec {
  kind: ProductKind;
  optionType: CallPut;
  strike: number;
  valuationDate: string;
  expiryDate: string;
  digitalPayoff: "cash" | "asset";
  payout: number;
  averaging: "arithmetic" | "geometric";
}

export interface MarketSpec {
  underlying: number;
  r: number;
  q: number;
}

export interface EngineSettings {
  binomialSteps: number;
  mcPaths: number;
  mcSteps: number;
  mcSeed: number;
  fftN: number;
}

export interface PricingRequest {
  product: ProductSpec;
  model: ModelSpec;
  market: MarketSpec;
  settings: EngineSettings;
}

export interface Quote {
  price: number;
  stderr: number | null;
}

export interface Capability {
  product: ProductKind;
  model: ModelKind;
  engines: EngineKind[];
}

export interface PathSample {
  times: number[];
  paths: number[][];
  quantiles: number[][];
  terminal: { edges: number[]; density: number[] };
  modelDensity: number[] | null;
}
