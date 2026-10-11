// Browser smoke test for the built playground (run `npm run build` first).
//   npm run smoke               # assertions only
//   SCREENSHOTS=1 npm run smoke # also writes e2e/out/*.png
import { mkdirSync } from "node:fs";
import { preview } from "vite";
import { chromium } from "playwright";

const SHOTS = process.env.SCREENSHOTS === "1";
if (SHOTS) mkdirSync("e2e/out", { recursive: true });

/** Same encoding as `encodeState` in src/state.ts. */
const encode = (req) => Buffer.from(JSON.stringify(req)).toString("base64url");

// Fixed dates so the expected price does not depend on today's date.
const base = {
  product: {
    kind: "european", optionType: "call", strike: 100,
    valuationDate: "2026-01-01", expiryDate: "2027-01-01",
    digitalPayoff: "cash", payout: 1, averaging: "arithmetic",
  },
  model: { kind: "blackScholes", sigma: 0.2 },
  market: { underlying: 100, r: 0.05, q: 0 },
  settings: { binomialSteps: 500, mcPaths: 50000, mcSteps: 252, mcSeed: 42, fftN: 4096 },
};
const withModel = (model) => ({ ...base, model });
const withProduct = (product) => ({ ...base, product: { ...base.product, ...product } });

const scenarios = [
  { name: "bs-payoff", req: base, tab: "payoff", expect: { analytical: "10.450584" } },
  { name: "bs-paths", req: base, tab: "paths" },
  { name: "bs-smile", req: base, tab: "smile" },
  { name: "heston-smile-dark", req: withModel({ kind: "heston", kappa: 2, theta: 0.04, sigma: 0.3, rho: -0.7, v0: 0.04 }), tab: "smile", scheme: "dark" },
  { name: "heston-paths", req: withModel({ kind: "heston", kappa: 2, theta: 0.04, sigma: 0.3, rho: -0.7, v0: 0.04 }), tab: "paths" },
  { name: "bates", req: withModel({ kind: "bates", kappa: 2, theta: 0.04, sigma: 0.3, rho: -0.7, v0: 0.04, lambda: 0.5, muJ: -0.1, sigmaJ: 0.15 }), tab: "smile" },
  { name: "kou", req: withModel({ kind: "kou", sigma: 0.2, lambda: 1, pUp: 0.4, eta1: 10, eta2: 5 }), tab: "smile" },
  { name: "sabr", req: withModel({ kind: "sabr", alpha: 2, beta: 0.5, rho: -0.3, nu: 0.4 }), tab: "smile" },
  { name: "sabr-normal", req: withModel({ kind: "sabr", alpha: 20, beta: 0, rho: -0.3, nu: 0.4 }), tab: "smile" },
  { name: "sabr-lognormal", req: withModel({ kind: "sabr", alpha: 0.2, beta: 1, rho: -0.3, nu: 0.4 }), tab: "smile" },
  { name: "black76", req: withModel({ kind: "black76", sigma: 0.2 }), tab: "payoff" },
  { name: "american-put", req: withProduct({ kind: "american", optionType: "put" }), tab: "payoff" },
  { name: "digital", req: withProduct({ kind: "digital" }), tab: "payoff" },
  { name: "asian", req: withProduct({ kind: "asian" }), tab: "payoff" },
  { name: "american-heston-unsupported", req: { ...withProduct({ kind: "american" }), model: { kind: "heston", kappa: 2, theta: 0.04, sigma: 0.3, rho: -0.7, v0: 0.04 } }, tab: "payoff", notice: true, noEngines: true },
  { name: "mobile-paths", req: base, tab: "paths", width: 390, height: 900 },
];

const server = await preview({ preview: { port: 4173, strictPort: false }, logLevel: "warn" });
const url = server.resolvedUrls.local[0];
const browser = await chromium.launch();
const failures = [];

for (const s of scenarios) {
  const page = await browser.newPage({ viewport: { width: s.width ?? 1360, height: s.height ?? 1100 }, colorScheme: s.scheme ?? "light" });
  const errors = [];
  page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
  page.on("pageerror", (e) => errors.push(e.message));
  const fail = (msg) => failures.push(`${s.name}: ${msg}`);

  try {
    await page.goto(`${url}?s=${encode(s.req)}`);
    await page.waitForSelector("#engine-rows tr", { timeout: 15000 });
    if (s.tab !== "payoff") await page.click(`button[data-tab="${s.tab}"]`);
    await page.waitForFunction(
      () => !document.querySelector("#engine-rows .spinner") && document.querySelector("#chart-busy").hidden,
      null,
      { timeout: 60000 },
    );
    await page.waitForTimeout(300);

    const rows = await page.$$eval("#engine-rows tr", (trs) => trs.map((t) => t.innerText.replace(/\s+/g, " ").trim()));
    const priced = rows.filter((r) => /\d+\.\d{6}/.test(r));
    const rowErrors = await page.$$eval("#engine-rows .err", (els) => els.map((e) => e.innerText));
    const notice = await page.$eval("#notice", (n) => (n.hidden ? "" : n.innerText));
    const empty = await page.$eval("#chart-empty", (n) => (n.hidden ? "" : n.innerText));

    if (errors.length) fail(`console errors: ${errors.join(" | ")}`);
    if (rowErrors.length) fail(`engine errors: ${rowErrors.join(" | ")}`);
    if (s.noEngines ? priced.length !== 0 : priced.length === 0) fail(`unexpected priced rows: ${JSON.stringify(rows)}`);
    if (Boolean(s.notice) !== Boolean(notice)) fail(`notice mismatch: "${notice}"`);
    if (!s.noEngines && empty) fail(`chart not drawn: "${empty}"`);
    if (s.expect?.analytical && !rows.some((r) => r.startsWith("Analytical") && r.includes(s.expect.analytical))) {
      fail(`expected analytical ${s.expect.analytical}, rows=${JSON.stringify(rows)}`);
    }
    console.log(`${failures.some((f) => f.startsWith(s.name)) ? "FAIL" : "ok  "} ${s.name}`);
    if (SHOTS) await page.screenshot({ path: `e2e/out/${s.name}.png`, fullPage: true });
  } catch (e) {
    fail(String(e));
    console.log(`FAIL ${s.name}`);
  }
  await page.close();
}

// Theme toggle: flips data-theme and the page background.
{
  const page = await browser.newPage({ colorScheme: "light" });
  await page.goto(`${url}?s=${encode(base)}`);
  await page.waitForSelector("#engine-rows tr");
  const bg = () => page.evaluate(() => getComputedStyle(document.body).backgroundColor);
  const before = await bg();
  await page.click("#theme");
  const theme = await page.evaluate(() => document.documentElement.dataset.theme);
  const after = await bg();
  if (theme !== "dark" || before === after) failures.push(`theme-toggle: theme=${theme}, bg ${before} -> ${after}`);
  await page.reload();
  await page.waitForSelector("#engine-rows tr");
  const persisted = await page.evaluate(() => document.documentElement.dataset.theme);
  if (persisted !== "dark") failures.push(`theme-toggle: not persisted (${persisted})`);
  if (SHOTS) await page.screenshot({ path: "e2e/out/theme-toggle-dark.png", fullPage: true });
  console.log(`${failures.some((f) => f.startsWith("theme-toggle")) ? "FAIL" : "ok  "} theme-toggle`);
  await page.close();
}

await browser.close();
server.httpServer.close();
if (failures.length) {
  console.error(`\n${failures.length} failure(s):\n${failures.join("\n")}`);
  process.exit(1);
}
console.log(`\nall ${scenarios.length} scenarios passed`);
