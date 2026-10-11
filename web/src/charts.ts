// ECharts option builders. Colours come from the CSS tokens so the charts
// follow light/dark mode.
import * as echarts from "echarts/core";
import { CustomChart, LineChart } from "echarts/charts";
import {
  DataZoomComponent,
  GridComponent,
  LegendComponent,
  MarkLineComponent,
  TooltipComponent,
} from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";
import type { EChartsCoreOption } from "echarts/core";
import type { PathSample } from "./types";

echarts.use([
  LineChart,
  CustomChart,
  GridComponent,
  TooltipComponent,
  LegendComponent,
  MarkLineComponent,
  DataZoomComponent,
  CanvasRenderer,
]);

export { echarts };

interface Palette {
  text: string;
  muted: string;
  border: string;
  surface: string;
  accent: string;
  c2: string;
  c3: string;
  c4: string;
  mono: string;
  font: string;
}

function palette(): Palette {
  const css = getComputedStyle(document.documentElement);
  const v = (name: string) => css.getPropertyValue(name).trim();
  return {
    text: v("--text"),
    muted: v("--muted"),
    border: v("--border"),
    surface: v("--surface"),
    accent: v("--accent"),
    c2: v("--c2"),
    c3: v("--c3"),
    c4: v("--c4"),
    mono: v("--mono"),
    font: v("--font"),
  };
}

/** `#rrggbb` token to `rgba()`; ECharts parses colours itself and does not
 * understand `color-mix()`. */
function withAlpha(color: string, alpha: number): string {
  const m = /^#([0-9a-f]{6})$/i.exec(color);
  if (!m) return color;
  const n = parseInt(m[1], 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`;
}

export function fmt(x: number, digits = 4): string {
  if (!Number.isFinite(x)) return "—";
  const a = Math.abs(x);
  if (a !== 0 && (a < 1e-3 || a >= 1e7)) return x.toExponential(digits - 1);
  return x.toFixed(digits);
}

function base(p: Palette) {
  const axis = {
    axisLine: { lineStyle: { color: p.border } },
    axisTick: { lineStyle: { color: p.border } },
    axisLabel: {
      color: p.muted,
      fontFamily: p.mono,
      fontSize: 11,
      // Data-driven bounds (min/max) would otherwise print every digit.
      formatter: (v: number) => String(Number(v.toFixed(2))),
    },
    splitLine: { lineStyle: { color: p.border, type: "dashed" as const } },
    nameTextStyle: { color: p.muted, fontFamily: p.font },
  };
  const tooltip = {
    trigger: "axis" as const,
    backgroundColor: p.surface,
    borderColor: p.border,
    textStyle: { color: p.text, fontFamily: p.font, fontSize: 12 },
    axisPointer: { lineStyle: { color: p.muted } },
  };
  const legend = {
    // One scrollable line, so narrow screens never wrap it over the plot.
    type: "scroll" as const,
    top: 0,
    right: 0,
    left: 40,
    pageIconColor: p.muted,
    pageTextStyle: { color: p.muted },
    textStyle: { color: p.muted, fontFamily: p.font },
    icon: "roundRect",
    itemWidth: 14,
    itemHeight: 4,
  };
  return { axis, tooltip, legend };
}

function vline(x: number, label: string, color: string) {
  return {
    xAxis: x,
    label: { formatter: label, color, fontSize: 11 },
    lineStyle: { color, type: "dashed" as const, width: 1 },
  };
}

/** Strike marker plus a second marker (spot or forward); merged into one
 * "K = S" line when they would overlap. */
function markers(strike: number, other: number, otherLabel: string, range: number, p: Palette) {
  if (Math.abs(strike - other) < 0.03 * range) {
    return [vline(strike, `K = ${otherLabel}`, p.c3)];
  }
  return [vline(strike, "K", p.c3), vline(other, otherLabel, p.accent)];
}

// --- Payoff --------------------------------------------------------------

export function payoffOption(args: {
  xs: number[];
  payoff: number[];
  value: (number | null)[] | null;
  valueLabel: string;
  strike: number;
  underlying: number;
  underlyingLabel: string;
}): EChartsCoreOption {
  const p = palette();
  const { axis, tooltip, legend } = base(p);
  const pts = (ys: (number | null)[]) => args.xs.map((x, i) => [x, ys[i]]);
  return {
    animationDuration: 300,
    grid: { left: 68, right: 18, top: 36, bottom: 44 },
    tooltip: {
      ...tooltip,
      valueFormatter: (v: unknown) => (typeof v === "number" ? fmt(v) : "—"),
    },
    legend,
    xAxis: { type: "value", name: args.underlyingLabel, nameLocation: "middle", nameGap: 28, min: "dataMin", max: "dataMax", ...axis, splitLine: { show: false } },
    yAxis: { type: "value", name: "Value", nameLocation: "middle", nameGap: 48, ...axis },
    series: [
      {
        name: "Payoff at expiry",
        type: "line",
        data: pts(args.payoff),
        showSymbol: false,
        lineStyle: { color: p.muted, width: 1.5, type: "dashed" },
        itemStyle: { color: p.muted },
        markLine: {
          silent: true,
          symbol: "none",
          data: markers(
            args.strike,
            args.underlying,
            args.underlyingLabel.split(" ")[0],
            args.xs[args.xs.length - 1] - args.xs[0],
            p,
          ),
        },
      },
      ...(args.value
        ? [
            {
              name: args.valueLabel,
              type: "line",
              data: pts(args.value),
              showSymbol: false,
              connectNulls: false,
              lineStyle: { color: p.accent, width: 2.5 },
              itemStyle: { color: p.accent },
              areaStyle: {
                color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                  { offset: 0, color: withAlpha(p.accent, 0.18) },
                  { offset: 1, color: withAlpha(p.accent, 0) },
                ]),
              },
            },
          ]
        : []),
    ],
  };
}

// --- Monte Carlo paths ---------------------------------------------------

export function pathsOption(args: { sample: PathSample; strike: number }): EChartsCoreOption {
  const p = palette();
  const { axis, tooltip, legend } = base(p);
  const { times, paths, quantiles, terminal, modelDensity } = args.sample;
  const [q05, q25, q50, q75, q95] = quantiles;
  const lo = Math.min(terminal.edges[0], ...q05);
  const hi = Math.max(terminal.edges[terminal.edges.length - 1], ...q95);
  const pad = (hi - lo) * 0.04;
  const yRange = { min: lo - pad, max: hi + pad };
  const t = (ys: number[]) => times.map((x, i) => [x, ys[i]]);
  const diff = (a: number[], b: number[]) => a.map((x, i) => x - b[i]);

  const pathSeries = paths.map((path, i) => ({
    name: "Sample paths",
    type: "line",
    data: t(path),
    showSymbol: false,
    silent: true,
    z: 1,
    lineStyle: { width: 0.7, color: i % 2 ? p.c2 : p.accent, opacity: 0.22 },
    itemStyle: { color: p.accent },
    tooltip: { show: false },
    emphasis: { disabled: true },
  }));

  const band = (name: string, lower: number[], upper: number[], stack: string, alpha: number) => [
    {
      name: `${stack}-base`,
      type: "line",
      data: t(lower),
      stack,
      showSymbol: false,
      silent: true,
      lineStyle: { opacity: 0 },
      tooltip: { show: false },
      z: 2,
    },
    {
      name,
      type: "line",
      data: t(diff(upper, lower)),
      stack,
      showSymbol: false,
      silent: true,
      lineStyle: { opacity: 0 },
      areaStyle: { color: withAlpha(p.c3, alpha) },
      itemStyle: { color: withAlpha(p.c3, 0.6) },
      tooltip: { show: false },
      z: 2,
    },
  ];

  const centres = terminal.edges.slice(0, -1).map((e, i) => (e + terminal.edges[i + 1]) / 2);
  const maxDensity = Math.max(...terminal.density, ...(modelDensity ?? [0]));

  return {
    animation: false,
    grid: [
      { left: 68, right: "28%", top: 36, bottom: 44 },
      { left: "75%", right: 18, top: 36, bottom: 44 },
    ],
    legend: {
      ...legend,
      data: ["Sample paths", "5–95%", "25–75%", "Median", "Terminal histogram", ...(modelDensity ? ["Model density"] : [])],
    },
    tooltip: {
      ...tooltip,
      valueFormatter: (v: unknown) => (typeof v === "number" ? fmt(v) : "—"),
    },
    xAxis: [
      { type: "value", gridIndex: 0, name: "Time (years)", nameLocation: "middle", nameGap: 28, min: 0, max: times[times.length - 1], ...axis, splitLine: { show: false } },
      { type: "value", gridIndex: 1, name: "Density", nameLocation: "middle", nameGap: 28, min: 0, max: maxDensity * 1.05, ...axis, axisLabel: { show: false }, splitLine: { show: false } },
    ],
    yAxis: [
      { type: "value", gridIndex: 0, name: "Underlying", nameLocation: "middle", nameGap: 48, ...yRange, ...axis, axisLabel: { ...axis.axisLabel, showMinLabel: false, showMaxLabel: false } },
      { type: "value", gridIndex: 1, ...yRange, ...axis, axisLabel: { show: false } },
    ],
    series: [
      ...pathSeries,
      ...band("5–95%", q05, q95, "outer", 0.14),
      ...band("25–75%", q25, q75, "inner", 0.22),
      {
        name: "Median",
        type: "line",
        data: t(q50),
        showSymbol: false,
        z: 3,
        lineStyle: { color: p.c3, width: 2 },
        itemStyle: { color: p.c3 },
        markLine: {
          silent: true,
          symbol: "none",
          data: [{ yAxis: args.strike, label: { formatter: "K", color: p.c4 }, lineStyle: { color: p.c4, type: "dashed" } }],
        },
      },
      {
        name: "Terminal histogram",
        type: "custom",
        xAxisIndex: 1,
        yAxisIndex: 1,
        itemStyle: { color: withAlpha(p.accent, 0.45) },
        renderItem: (_params: unknown, api: any) => {
          const i = api.value(2) as number;
          const start = api.coord([0, terminal.edges[i]]);
          const end = api.coord([terminal.density[i], terminal.edges[i + 1]]);
          return {
            type: "rect",
            shape: { x: start[0], y: end[1], width: end[0] - start[0], height: start[1] - end[1] },
            style: api.style(),
          };
        },
        encode: { x: 0, y: 1 },
        data: terminal.density.map((d, i) => [d, centres[i], i]),
        tooltip: { show: false },
      },
      ...(modelDensity
        ? [
            {
              name: "Model density",
              type: "line",
              xAxisIndex: 1,
              yAxisIndex: 1,
              data: modelDensity.map((d, i) => [d, centres[i]]),
              showSymbol: false,
              smooth: true,
              lineStyle: { color: p.c4, width: 2 },
              itemStyle: { color: p.c4 },
              tooltip: { show: false },
            },
          ]
        : []),
    ],
  };
}

// --- Implied volatility smile -------------------------------------------

export function smileOption(args: {
  strikes: number[];
  vols: (number | null)[];
  forward: number;
  strike: number;
  label: string;
}): EChartsCoreOption {
  const p = palette();
  const { axis, tooltip, legend } = base(p);
  const pct = args.vols.filter((v): v is number => v !== null).map((v) => v * 100);
  const [lo, hi] = [Math.min(...pct), Math.max(...pct)];
  // Flat smiles (Black–Scholes) would otherwise zoom into rounding noise.
  const flat = pct.length > 0 && hi - lo < 0.5;
  const yBounds = flat ? { min: Math.floor(lo - 2), max: Math.ceil(hi + 2) } : {};
  return {
    animationDuration: 300,
    grid: { left: 68, right: 18, top: 36, bottom: 44 },
    legend,
    tooltip: {
      ...tooltip,
      valueFormatter: (v: unknown) => (typeof v === "number" ? `${v.toFixed(2)}%` : "—"),
    },
    xAxis: { type: "value", name: "Strike", nameLocation: "middle", nameGap: 28, min: "dataMin", max: "dataMax", ...axis, splitLine: { show: false } },
    yAxis: {
      type: "value",
      name: "Implied vol (%)",
      nameLocation: "middle",
      nameGap: 44,
      scale: true,
      ...yBounds,
      ...axis,
      axisLabel: { ...axis.axisLabel, formatter: (v: number) => v.toFixed(1) },
    },
    series: [
      {
        name: args.label,
        type: "line",
        data: args.strikes.map((k, i) => [k, args.vols[i] === null ? null : args.vols[i]! * 100]),
        showSymbol: false,
        connectNulls: false,
        smooth: false,
        lineStyle: { color: p.accent, width: 2.5 },
        itemStyle: { color: p.accent },
        markLine: {
          silent: true,
          symbol: "none",
          data: markers(args.strike, args.forward, "F", args.strikes[args.strikes.length - 1] - args.strikes[0], p),
        },
      },
    ],
  };
}
