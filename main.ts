// GTA VI trophy rarity forecast: a refresh-aware forecaster.
//
// What the practice teaches: Exophase refreshes a game's counts on its own
// schedule, not continuously. On launch night the page sat unchanged for
// hours (638 players tracked, every rate flat), then one refresh jumped to
// 3,868 players and every rate but the 100% one FELL: the newly tracked
// players hold far fewer trophies. The totals pin the size of the drop: the
// refreshed page shows 4,276 trophies earned over 3,868 players, so the 54
// non-100% trophies sum to 10.55 rate points, down from 47.78 -- a factor of
// about 0.22. Persistence (the house baseline) scores ~1.51 on the practice
// because its narrow band sits on the stale rates and the refresh takes every
// trophy outside it.
//
// This forecaster therefore has two regimes, chosen per trophy by how much
// the history actually moved:
//
//  * FLAT history (no real movement -- the page is a stale snapshot): the
//    median is the last rate times a dilution factor D = 1 / (1 + S / 1.25),
//    where S is the hours from the first read to the target (the snapshot's
//    age). For the practice S is ~4.7 h, so D is ~0.21, matching the observed
//    refresh. The spread is a flat 0.6 logit units, wide enough to cover the
//    unknown per-trophy dilution.
//  * MOVING history (the final's two days of reads): two curve fits are
//    blended in logit space. (a) A saturating exponential in linear time,
//    R(t) = Rb + (C - Rb) * (1 - exp(-k t)), fitted by a grid search over k
//    with C by least squares -- the shape a growing, diluting player base
//    produces (new cohorts join, existing players saturate). (b) The house
//    curve's log-time linear fit over the last 24 h, anchored at the last
//    level with its slope shrunk by its own standard error and its move
//    capped. Their disagreement widens the spread, and the blend leans on the
//    trend only as far as the history's movement warrants.
//
// A trophy already at 100% is forecast at 100% with a hair of lower tail:
// every tracked player holds it within minutes of starting, so it does not
// dilute.
//
// Node 22, standard library only, no network, deterministic: the only
// inputs are the case files, and every computation is a pure function of
// them, so two runs write identical bytes.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];

/** Hours of snapshot staleness that halve the rate in the flat regime. */
const TAU_DIL = 1.25;
/** Floor on the dilution factor for a stale snapshot. */
const DIL_FLOOR = 0.08;
/** Dilution factor when the snapshot is more than a day stale: the
 *  1/(1+S/TAU_DIL) form is calibrated on a launch-night refresh a few hours
 *  old; over days the old cohort's own earning offsets the dilution, so the
 *  factor is not allowed to fall this far. */
const DIL_LONG = 0.2;
/** Logit spread in the flat regime. */
const SPREAD_FLAT = 0.6;
/** Random-walk spread, in logit units per square-root day of horizon. */
const SIGMA_RW = 0.35;
/** Logit movement scale at which the trend earns half the blend weight. */
const W_MOVE = 0.15;
/** Cap on the log-linear extrapolation move, in logit units. */
const MAX_MOVE = 1.2;
/** Weight of the saturating model in the trend blend (rest: log-linear).
 *  Leans saturating: a growing, diluting player base saturates in linear
 *  time by the cohort argument, while the log-linear shape is kept as a hedge. */
const W_SAT = 0.6;
/** Grid of saturation rates (per day) tried by the saturating fit. */
const K_GRID: number[] = [];
for (let i = 0; i < 28; i++) K_GRID.push(0.05 * Math.pow(1.35, i));

interface Trophy {
  id: number;
  name: string;
  tier: string;
}

interface HistoryRead {
  at: string;
  players_tracked: number;
  rates: Array<number | null>;
}

interface Case {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  game: { release_date: string };
  trophies: Trophy[];
  history: HistoryRead[];
}

const clamp = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const logit = (pct: number): number => {
  const p = clamp(pct);
  return Math.log(p / (100 - p));
};
const invLogit = (l: number): number => 100 / (1 + Math.exp(-l));
/** x rounded to 4 decimals. */
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;
const parseTs = (s: string): number => new Date(s).getTime();

/** Distinct (days-since-release, rate) points for one trophy, in time order.
 *  Consecutive duplicate rates carry no new information (the page had not
 *  refreshed), so only the first read showing each value is kept. */
function distinctPoints(c: Case, idx: number, releaseMs: number): Array<[number, number]> {
  const pts: Array<[number, number]> = [];
  for (const h of c.history) {
    const r = h.rates[idx];
    if (r === null || r === undefined) continue;
    if (pts.length > 0 && Math.abs(pts[pts.length - 1][1] - r) < 1e-12) continue;
    const days = (parseTs(h.at) - releaseMs) / 86_400_000;
    pts.push([days, r]);
  }
  return pts;
}

/** Saturating fit R(t) = rb + (C - rb) * (1 - exp(-k t)) over the distinct
 *  points: k by grid search, C by least squares given k. Returns the median
 *  rate at the target (days) and the fit's root-mean-square error. */
function fitSaturating(pts: Array<[number, number]>, tTgtDays: number): { med: number; rmse: number } | null {
  const rb = pts[0][1];
  let best: { sse: number; c: number; k: number } | null = null;
  for (const k of K_GRID) {
    let num = 0;
    let den = 0;
    for (const [td, r] of pts) {
      const g = 1 - Math.exp(-k * Math.max(0, td));
      num += (r - rb) * g;
      den += g * g;
    }
    if (den < 1e-12) continue;
    let cc = rb + num / den;
    if (cc < rb) cc = rb;
    let sse = 0;
    for (const [td, r] of pts) {
      const pred = rb + (cc - rb) * (1 - Math.exp(-k * Math.max(0, td)));
      sse += (r - pred) * (r - pred);
    }
    if (best === null || sse < best.sse) best = { sse, c: cc, k };
  }
  if (best === null) return null;
  const rmse = Math.sqrt(best.sse / Math.max(1, pts.length - 2));
  const med = rb + (best.c - rb) * (1 - Math.exp(-best.k * Math.max(0, tTgtDays)));
  return { med, rmse };
}

/** Log-time linear fit over the last 24 h of distinct points (the house
 *  curve's shape): returns the extrapolated logit median at the target, the
 *  residual spread, and the slope's standard error times the extrapolation
 *  distance. Anchored at the last point, slope shrunk by its own error. */
function fitLogLinear(
  pts: Array<[number, number, number]>,
  atMs: number,
  tTgtDays: number,
): { medLogit: number; sRes: number; seDx: number } | null {
  const pts24: Array<[number, number]> = [];
  for (const [atMsRead, days, r] of pts) {
    if (atMs - atMsRead > 86_400_000) continue;
    const x = Math.log1p(Math.max(0, days * 24));
    const y = logit(r);
    if (pts24.length > 0 && Math.abs(pts24[pts24.length - 1][1] - y) < 1e-12) continue;
    pts24.push([x, y]);
  }
  if (pts24.length < 3) return null;
  const n = pts24.length;
  const mx = pts24.reduce((s, p) => s + p[0], 0) / n;
  const my = pts24.reduce((s, p) => s + p[1], 0) / n;
  let sxx = 0;
  let sxy = 0;
  for (const [x, y] of pts24) {
    sxx += (x - mx) * (x - mx);
    sxy += (x - mx) * (y - my);
  }
  const b = sxx > 1e-12 ? sxy / sxx : 0;
  const a = my - b * mx;
  let s2 = 0;
  for (const [x, y] of pts24) s2 += (y - (a + b * x)) * (y - (a + b * x));
  s2 /= Math.max(1, n - 2);
  const seB = sxx > 1e-12 ? Math.sqrt(s2 / sxx) : 0;
  const xTgt = Math.log1p(Math.max(0, tTgtDays * 24));
  const [xLast, yLast] = pts24[pts24.length - 1];
  const dx = xTgt - xLast;
  const k = b * b + seB * seB > 0 ? (b * b) / (b * b + seB * seB) : 0;
  const move = Math.max(-MAX_MOVE, Math.min(MAX_MOVE, k * b * dx));
  return { medLogit: yLast + move, sRes: Math.sqrt(Math.max(s2, 0)), seDx: seB * dx };
}

/** Five quantiles for one trophy. */
function forecastTrophy(
  c: Case,
  idx: number,
  releaseMs: number,
  targetMs: number,
  horizonDays: number,
  dilution: number,
): number[] {
  const lastRaw = c.history[c.history.length - 1].rates[idx];
  const last = lastRaw === null || lastRaw === undefined ? 0.01 : lastRaw;
  // A trophy every tracked player holds stays held: it is earned within
  // minutes of starting, so new players dilute it least of all.
  if (last >= 99.99) return [99.9, 99.99, 100, 100, 100];

  const lFlat = logit(last * dilution);
  const pts = distinctPoints(c, idx, releaseMs);
  if (pts.length < 3) {
    return Z.map((z) => round4(invLogit(lFlat + z * SPREAD_FLAT)));
  }
  const ys = pts.map((p) => logit(p[1]));
  const movement = Math.max(...ys) - Math.min(...ys);
  if (movement < 0.05) {
    return Z.map((z) => round4(invLogit(lFlat + z * SPREAD_FLAT)));
  }

  const tTgtDays = (targetMs - releaseMs) / 86_400_000;
  const atMs = parseTs(c.at);
  const sat = fitSaturating(pts, tTgtDays);
  const ptsDt: Array<[number, number, number]> = [];
  for (const h of c.history) {
    const r = h.rates[idx];
    if (r === null || r === undefined) continue;
    const days = (parseTs(h.at) - releaseMs) / 86_400_000;
    if (ptsDt.length > 0 && Math.abs(ptsDt[ptsDt.length - 1][2] - r) < 1e-12) continue;
    ptsDt.push([parseTs(h.at), days, r]);
  }
  const loglin = fitLogLinear(ptsDt, atMs, tTgtDays);

  let med: number;
  let s: number;
  const rw = SIGMA_RW * Math.sqrt(horizonDays);
  if (sat === null && loglin === null) {
    med = lFlat;
    s = SPREAD_FLAT;
  } else if (loglin === null) {
    med = logit(sat!.med);
    s = Math.max(0.35, Math.sqrt(0.15 * 0.15 + rw * rw));
  } else if (sat === null) {
    med = loglin.medLogit;
    s = Math.max(0.35, Math.sqrt(Math.max(loglin.sRes, 0.15) ** 2 + loglin.seDx ** 2 + rw * rw));
  } else {
    const lSat = logit(sat.med);
    med = W_SAT * lSat + (1 - W_SAT) * loglin.medLogit;
    const dis = Math.abs(lSat - loglin.medLogit);
    s = Math.max(
      0.35,
      Math.sqrt(Math.max(loglin.sRes, 0.15) ** 2 + loglin.seDx ** 2 + rw * rw + (0.5 * dis) ** 2),
    );
  }
  // Blend the trend toward the dilution prior when the history barely moved.
  const w = (movement * movement) / (movement * movement + W_MOVE * W_MOVE);
  med = w * med + (1 - w) * lFlat;
  if (s < 0.35) s = 0.35;
  return Z.map((z) => round4(invLogit(med + z * s)));
}

function forecastCase(c: Case): { trophies: Array<{ id: number; q: number[] }> } {
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${c.id} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  if (c.history.length === 0) throw new Error(`${c.id} has no history`);
  const releaseDate = c.game.release_date.includes('T')
    ? c.game.release_date
    : `${c.game.release_date}T00:00:00Z`;
  const releaseMs = parseTs(releaseDate);
  const targetMs = parseTs(c.target);
  const horizonDays = c.horizon_minutes / 1440;
  // Staleness: hours from the first read to the target. The snapshot the
  // history shows can be that old when the page has not refreshed.
  const stalenessH = (targetMs - parseTs(c.history[0].at)) / 3_600_000;
  const dilution =
    stalenessH > 24 ? DIL_LONG : Math.max(DIL_FLOOR, 1 / (1 + stalenessH / TAU_DIL));
  const trophies = c.trophies.map((t, i) => ({
    id: t.id,
    q: forecastTrophy(c, i, releaseMs, targetMs, horizonDays, dilution),
  }));
  return { trophies };
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify(forecastCase(c))}\n`);
}
