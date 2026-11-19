// GTA VI trophy rarity forecast: a refresh-aware forecaster, v3.
//
// DIAGNOSIS (tuning round 2, 19 Nov 2026). v2 scored 0.0967 on the server
// practice (rank 55/3736, inside the top 100) and its flat regime is
// calibrated: the practice history is a stale snapshot, the trend regime
// never engages there, so v3 keeps the flat regime EXACTLY as v2 (same
// rate-dependent dilution, same 0.30 spread) and its practice numbers are
// unchanged by construction (local: POW-g1.0 0.074, uniform 0.107).
// The final is decided by the TREND regime, and v2's trend regime has two
// measured faults on the local synthetic-final harness (five invented
// finals: saturating-rate, log-linear-rate, and three cohort truths with
// player growth; v2 mean loss 0.214):
//   * Its spread is ~0.78 logit units on finals: the SIGMA_RW * sqrt(days)
//     random-walk term charges for the whole horizon's movement even though
//     the trend already projects that movement. Re-spreading v2's own
//     medians at s = 0.25-0.35 halves the loss on every synthetic final.
//   * Its median is a fixed 0.6/0.4 saturating/log-linear blend. The
//     saturating shape wins only when the truth saturates (median-only loss
//     0.063 vs 0.398 on the saturating final); the log-linear shape wins on
//     every other family (0.030-0.105 vs 0.31-0.66). A fixed blend is wrong
//     in both directions, and in-sample fit cannot tell them apart: the
//     log-linear fit hugs a saturating curve almost exactly over the
//     observed 2-day window (in-sample logit RMSE 0.006 vs 0.024) and only
//     overshoots at day 7. A backtest on the last 30% of history fails the
//     same way: it rewards trend-following over too short a horizon.
// The signal that DOES separate them is the CURVATURE of the observed
// logit-rate curve against log-time: a saturating process bends down
// (quadratic coefficient median -0.59, all 55 trophies negative on the
// saturating final), while log-linear and cohort truths are straight or
// bend up (+0.001 to +0.039, no trophy below -0.05). v3 therefore weights
// the saturating model per trophy by that curvature, through a sigmoid
// centred at -0.2: a clearly decelerating trophy gets wS = 0.75, a
// straight-or-accelerating one wS = 0.10. A third shape was tried and
// DROPPED: a hazard/count model (dE/dt = lam * (P - E), earners over
// players tracked) extrapolates worst on every family (median-only loss
// 0.23-1.48) because it cannot represent a saturating rate while players
// keep joining; it is not in this file.
//   * The spread drops the sqrt(days) term: s = sqrt(wmean(rmse^2) +
//     slope-error^2 + (0.5 * model disagreement)^2 + (0.10 * dx)^2 +
//     0.15^2), floored at 0.30 and capped at 0.60, where dx is the log-time
//     extrapolation distance. The horizon still widens the band, but
//     through the distance the fit actually extrapolates, not the raw days.
// Verified with the real program (node, p00+p01, run twice byte-identical,
// plus the five synthetic finals): practice POW-g1.0 0.074 / uniform 0.107
// (identical to v2), synthetic-final mean loss 0.085 vs v2's 0.214, worst
// final 0.096 vs v2's 0.243, coverage 1.00 on all five (noiseless truths;
// the floor and the disagreement term keep the band honest on the real
// final, whose refresh timing and player growth no synthetic family
// captures).
//
// The rest of the forecaster is v2 unchanged: two regimes chosen per trophy
// by how much the history moved. A flat history means the page is a stale
// snapshot, so the median is the last rate times the dilution factor (now
// rate-dependent) and the spread is SPREAD_FLAT. A moving history blends a
// saturating-exponential fit in linear time with the house curve's log-time
// linear fit, anchored at the last level, slope shrunk by its own error,
// move capped; their disagreement widens the spread, and the blend leans on
// the trend only as far as the history's movement warrants. A trophy already
// at 100% is forecast at 100% with a hair of lower tail.
//
// Node 22, standard library only, no network, deterministic: the only
// inputs are the case files, and every computation is a pure function of
// them, so two runs write identical bytes. Rates are percents (0-100);
// GAMMA_DIL is calibrated in those units (r = 1.0 is the pivot).

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];

/** Hours of snapshot staleness that halve the rate in the flat regime. */
const TAU_DIL = 1.4;
/** Floor on the dilution factor for a stale snapshot. */
const DIL_FLOOR = 0.08;
/** Dilution factor when the snapshot is more than a day stale: the
 *  1/(1+S/TAU_DIL) form is calibrated on a launch-night refresh a few hours
 *  old; over days the old cohort's own earning offsets the dilution, so the
 *  factor is not allowed to fall this far. */
const DIL_LONG = 0.2;
/** Rate-dependence of the dilution factor (v2): the median rate in the
 *  flat regime is D * r^(1 + GAMMA_DIL), so a trophy's dilution factor is
 *  D * r^GAMMA_DIL -- easy trophies dilute less. Calibrated in percent
 *  units; the launch-night anchors put the true exponent near 1.0 and 0.7
 *  keeps most of the gain while staying ahead of v1 if it is lower. */
const GAMMA_DIL = 0.7;
/** Logit spread in the flat regime (v2: 0.30, was 0.6 -- v1's coverage
 *  0.982 showed the old band far too wide for the calibrated scatter). */
const SPREAD_FLAT = 0.3;
/** Logit movement scale at which the trend earns half the blend weight. */
const W_MOVE = 0.15;
/** Cap on the log-linear extrapolation move, in logit units. */
const MAX_MOVE = 1.2;
/** Grid of saturation rates (per day) tried by the saturating fit. */
const K_GRID: number[] = [];
for (let i = 0; i < 28; i++) K_GRID.push(0.05 * Math.pow(1.35, i));
/** Curvature-adaptive blend (v3): the saturating model's weight per trophy
 *  is W_S_LO + (W_S_HI - W_S_LO) * sigmoid((CURV_C0 - curv) / CURV_K),
 *  where curv is the quadratic coefficient of logit(rate) against
 *  log1p(hours since release) over the distinct points. Saturating truths
 *  show curv ~ -0.6, log-linear and cohort truths ~ 0.0 to +0.04, so the
 *  sigmoid is centred at -0.2 with a 0.05 width: clearly decelerating
 *  trophies lean saturating, straight ones lean log-linear. */
const W_S_LO = 0.1;
const W_S_HI = 0.75;
const CURV_C0 = -0.2;
const CURV_K = 0.05;
/** Distinct points needed before the curvature switch is trusted; fewer
 *  and the weight stays at W_S_LO (a 3-point quadratic is exact and its
 *  coefficient is noise). */
const MIN_CURV_PTS = 5;
/** Trend-regime spread (v3): s = sqrt(wmean(rmse^2) + seDx^2 +
 *  (0.5 * disagreement)^2 + (S_HOR * dx)^2 + SPREAD_BASE^2), floored and
 *  capped. The old SIGMA_RW * sqrt(days) term is gone: it charged ~0.75
 *  logit units for movement the trend already projects. */
const SPREAD_BASE = 0.15;
const S_HOR = 0.1;
const SPREAD_FLOOR = 0.3;
const SPREAD_CAP = 0.6;

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
/** Logistic sigmoid, guarded against overflow. */
const sigmoid = (x: number): number => {
  const t = Math.max(-30, Math.min(30, x));
  return 1 / (1 + Math.exp(-t));
};

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
 *  rate at the target (days) and the fit's root-mean-square error in logit
 *  space over the distinct points. */
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
  const predAt = (td: number): number =>
    rb + (best!.c - rb) * (1 - Math.exp(-best!.k * Math.max(0, td)));
  let sseLogit = 0;
  for (const [td, r] of pts) sseLogit += (logit(r) - logit(predAt(td))) ** 2;
  return { med: predAt(tTgtDays), rmse: Math.sqrt(sseLogit / pts.length) };
}

/** Log-time linear fit over the last 24 h of distinct points (the house
 *  curve's shape): returns the extrapolated logit median at the target, the
 *  residual spread in logit space, the slope's standard error times the
 *  extrapolation distance, and that distance itself. Anchored at the last
 *  point, slope shrunk by its own error. */
function fitLogLinear(
  pts: Array<[number, number, number]>,
  atMs: number,
  tTgtDays: number,
): { medLogit: number; rmse: number; seDx: number; dx: number } | null {
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
  let sse = 0;
  for (const [x, y] of pts24) sse += (y - (a + b * x)) * (y - (a + b * x));
  const seB = sxx > 1e-12 ? Math.sqrt(sse / Math.max(1, n - 2) / sxx) : 0;
  const xTgt = Math.log1p(Math.max(0, tTgtDays * 24));
  const [xLast, yLast] = pts24[pts24.length - 1];
  const dx = xTgt - xLast;
  const k = b * b + seB * seB > 0 ? (b * b) / (b * b + seB * seB) : 0;
  const move = Math.max(-MAX_MOVE, Math.min(MAX_MOVE, k * b * dx));
  return { medLogit: yLast + move, rmse: Math.sqrt(sse / n), seDx: seB * dx, dx };
}

/** Quadratic coefficient of logit(rate) against log1p(hours since release)
 *  over the distinct points: negative when the trophy's growth is
 *  decelerating (saturating), near zero when it is log-linear. Solved from
 *  the centred normal equations by Gaussian elimination with partial
 *  pivoting; returns 0 when there are too few points or the system is
 *  singular. */
function fitCurvature(pts: Array<[number, number]>): number {
  if (pts.length < MIN_CURV_PTS) return 0;
  const xs: number[] = [];
  const ys: number[] = [];
  for (const [td, r] of pts) {
    xs.push(Math.log1p(Math.max(0, td * 24)));
    ys.push(logit(r));
  }
  const n = xs.length;
  const mx = xs.reduce((s, x) => s + x, 0) / n;
  const my = ys.reduce((s, y) => s + y, 0) / n;
  // Normal equations for y - my = c0 + c1 (x - mx) + c2 (x - mx)^2.
  const mat: number[][] = [];
  const vec: number[] = [];
  for (let a = 0; a < 3; a++) {
    const row: number[] = [];
    for (let b = 0; b < 3; b++) {
      let s = 0;
      for (const x of xs) s += Math.pow(x - mx, a + b);
      row.push(s);
    }
    let t = 0;
    for (let i = 0; i < n; i++) t += (ys[i] - my) * Math.pow(xs[i] - mx, a);
    mat.push(row);
    vec.push(t);
  }
  const aug: number[][] = mat.map((row, i) => [...row, vec[i]]);
  for (let col = 0; col < 3; col++) {
    let piv = col;
    for (let r = col + 1; r < 3; r++) {
      if (Math.abs(aug[r][col]) > Math.abs(aug[piv][col])) piv = r;
    }
    if (Math.abs(aug[piv][col]) < 1e-12) return 0;
    if (piv !== col) {
      const tmp = aug[col];
      aug[col] = aug[piv];
      aug[piv] = tmp;
    }
    for (let r = col + 1; r < 3; r++) {
      const f = aug[r][col] / aug[col][col];
      for (let cc = col; cc < 4; cc++) aug[r][cc] -= f * aug[col][cc];
    }
  }
  const coef: number[] = [0, 0, 0];
  for (let r = 2; r >= 0; r--) {
    let s = aug[r][3];
    for (let cc = r + 1; cc < 3; cc++) s -= aug[r][cc] * coef[cc];
    coef[r] = s / aug[r][r];
  }
  return Number.isFinite(coef[2]) ? coef[2] : 0;
}

/** Five quantiles for one trophy. */
function forecastTrophy(
  c: Case,
  idx: number,
  releaseMs: number,
  targetMs: number,
  dilution: number,
): number[] {
  const lastRaw = c.history[c.history.length - 1].rates[idx];
  const last = lastRaw === null || lastRaw === undefined ? 0.01 : lastRaw;
  // A trophy every tracked player holds stays held: it is earned within
  // minutes of starting, so new players dilute it least of all.
  if (last >= 99.99) return [99.9, 99.99, 100, 100, 100];

  // Flat-regime median (v2): the dilution factor itself grows with the
  // rate -- D * r^GAMMA_DIL -- so the median is D * r^(1 + GAMMA_DIL).
  // Rates are percents; r = 1.0 is the pivot of the power law.
  const safeLast = Math.max(last, 0.01);
  const lFlat = logit(dilution * Math.pow(safeLast, 1 + GAMMA_DIL));
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

  if (sat === null && loglin === null) {
    return Z.map((z) => round4(invLogit(lFlat + z * SPREAD_FLAT)));
  }

  // Curvature-adaptive blend (v3): decelerating trophies lean on the
  // saturating shape, straight-or-accelerating ones on the log-linear one.
  // A missing shape takes no weight.
  const curv = fitCurvature(pts);
  const wSatRaw =
    W_S_LO + (W_S_HI - W_S_LO) * sigmoid((CURV_C0 - curv) / CURV_K);
  const wSat = sat === null ? 0 : loglin === null ? 1 : wSatRaw;
  const wLog = 1 - wSat;
  const lSat = sat === null ? 0 : logit(sat.med);
  const lLog = loglin === null ? 0 : loglin.medLogit;
  const medTrend = wSat * lSat + wLog * lLog;
  const dis = sat === null || loglin === null ? 0 : Math.abs(lSat - lLog);
  const rmseW = Math.sqrt(
    (sat === null ? 0 : wSat * sat.rmse * sat.rmse) +
      (loglin === null ? 0 : wLog * loglin.rmse * loglin.rmse),
  );
  const seW = loglin === null ? 0 : Math.sqrt(wLog) * loglin.seDx;
  const dx = loglin === null ? 0 : loglin.dx;
  // Spread (v3): fit residuals, slope error, model disagreement and the
  // log-time extrapolation distance -- no raw sqrt(days) term.
  const s = Math.min(
    SPREAD_CAP,
    Math.max(
      SPREAD_FLOOR,
      Math.sqrt(
        rmseW * rmseW +
          seW * seW +
          (0.5 * dis) * (0.5 * dis) +
          (S_HOR * dx) * (S_HOR * dx) +
          SPREAD_BASE * SPREAD_BASE,
      ),
    ),
  );
  // Blend the trend toward the dilution prior when the history barely moved.
  const w = (movement * movement) / (movement * movement + W_MOVE * W_MOVE);
  const med = w * medTrend + (1 - w) * lFlat;
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
  // Staleness: hours from the first read to the target. The snapshot the
  // history shows can be that old when the page has not refreshed.
  const stalenessH = (targetMs - parseTs(c.history[0].at)) / 3_600_000;
  const dilution =
    stalenessH > 24 ? DIL_LONG : Math.max(DIL_FLOOR, 1 / (1 + stalenessH / TAU_DIL));
  const trophies = c.trophies.map((t, i) => ({
    id: t.id,
    q: forecastTrophy(c, i, releaseMs, targetMs, dilution),
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
