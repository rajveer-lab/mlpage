#!/usr/bin/env node
/*
 * Checks the JavaScript port (docs/engine.js) against the Python package.
 *
 *     node tools/web_parity/check.js [path/to/cases.json]
 *
 * Replays tools/web_parity/cases.json (written by export.py), prints a
 * compact summary and exits with status 1 on any failure:
 *
 *   - any fault count, learned-policy decision count, next_use, matmul
 *     trace, MT19937 number, frame count or replayed generator trace that
 *     differs from Python;
 *   - a feature that differs from compute_features by more than 1e-9, or a
 *     model probability that differs from predict_proba by more than 1e-9;
 *   - the sampled learned policy (k = 16 < frames) more than 3% away from
 *     Python on a case where Python takes at least 500 faults. Its sampler
 *     uses a different PRNG, so this is the only comparison that is not
 *     exact. With k >= frames it scores every slot and must match exactly.
 */
"use strict";

const fs = require("fs");
const path = require("path");

const ROOT = path.resolve(__dirname, "..", "..");
const MLPR = require(path.join(ROOT, "docs", "engine.js"));
const doc = JSON.parse(fs.readFileSync(process.argv[2] || path.join(__dirname, "cases.json"), "utf8"));

const FEATURE_TOL = 1e-9;
const PROB_TOL = 1e-9;
const APPROX_TOL = 0.03;
const APPROX_MIN_FAULTS = 500;
const WEIGHT_TOL = 1e-9;

const failures = [];
function fail(message) {
  failures.push(message);
}

function pad(text, width) {
  text = String(text);
  return text.length >= width ? text : text + " ".repeat(width - text.length);
}

function lpad(text, width) {
  text = String(text);
  return text.length >= width ? text : " ".repeat(width - text.length) + text;
}

function pct(x) {
  return (100 * x).toFixed(2) + "%";
}

// ------------------------------------------------------------- policies

const started = Date.now();
const groups = ["textbook", "random", "workload"];
const tally = {}; // config id -> group -> {ok, total}
const approxRows = [];
let maxWeightDiff = 0;
let statsChecked = 0;

doc.configs.forEach(function (c) {
  tally[c.id] = {};
  groups.forEach(function (g) { tally[c.id][g] = { ok: 0, total: 0 }; });
});

doc.cases.forEach(function (kase) {
  const trace = kase.trace;
  kase.runs.forEach(function (run) {
    const frames = run.frames;
    doc.configs.forEach(function (config) {
      const opts = Object.assign({}, config.opts, { trace: trace });
      if (opts.k === "frames") opts.k = frames;
      if (config.policy === "Learned") opts.model = doc.model;
      const policy = MLPR.makePolicy(config.policy, frames, opts);
      const result = MLPR.simulate(trace, policy, { timed: false });
      const py = run.faults[config.id];
      const where = kase.name + " frames=" + frames + " " + config.id;
      const sampled = config.approx && policy.k < frames;
      const cell = tally[config.id][kase.group];
      cell.total++;
      if (sampled) {
        const rel = py ? (result.faults - py) / py : 0;
        const others = [py].concat(run.learned16_other_seeds || []);
        const spread = (Math.max.apply(null, others) - Math.min.apply(null, others)) / py;
        const judged = py >= APPROX_MIN_FAULTS;
        const ok = !judged || Math.abs(rel) <= APPROX_TOL;
        approxRows.push({ where: kase.name + " f=" + frames, py: py, js: result.faults, rel: rel, spread: spread, judged: judged, ok: ok });
        if (ok) cell.ok++;
        else fail(where + ": " + result.faults + " faults vs Python " + py + " (" + pct(rel) + ", limit 3%)");
        return;
      }
      let ok = result.faults === py;
      if (!ok) fail(where + ": " + result.faults + " faults vs Python " + py);
      if (config.policy === "Learned") {
        const want = run.learned_stats[config.id];
        const got = [policy.decisions, policy.fallbacks, policy.guarded];
        statsChecked++;
        if (got.join() !== want.join()) {
          ok = false;
          fail(where + ": decisions/fallbacks/guarded " + got.join("/") + " vs Python " + want.join("/"));
        }
        if (result.extra.fallbackRate !== (want[0] ? want[1] / want[0] : 0) || result.extra.guardedRate !== (want[0] ? want[2] / want[0] : 0)) {
          ok = false;
          fail(where + ": fallback/guarded rates differ");
        }
      }
      if (config.policy === "LeCaR") {
        const diff = Math.abs(policy.wLru - run.lecar_w_lru) / Math.max(1e-300, Math.abs(run.lecar_w_lru));
        maxWeightDiff = Math.max(maxWeightDiff, diff);
        if (!(diff <= WEIGHT_TOL)) {
          ok = false;
          fail(where + ": LeCaR weight " + policy.wLru + " vs Python " + run.lecar_w_lru);
        }
      }
      if (ok) cell.ok++;
    });
  });
});

console.log("Policies: fault counts against Python (runs matching / runs)");
console.log(pad("config", 28) + groups.map(function (g) { return lpad(g, 10); }).join("") + lpad("total", 10) + "  result");
doc.configs.forEach(function (config) {
  let ok = 0, total = 0;
  const cells = groups.map(function (g) {
    const c = tally[config.id][g];
    ok += c.ok;
    total += c.total;
    return lpad(c.ok + "/" + c.total, 10);
  });
  let verdict = ok === total ? "exact" : "MISMATCH";
  if (config.approx) {
    const judged = approxRows.filter(function (r) { return r.judged; }).length;
    verdict = (ok === total ? "ok" : "FAIL") + ": exact on " + (total - approxRows.length) + " runs with k >= frames; " + approxRows.length +
      " sampled runs, the " + judged + " with >= 500 faults within 3%";
  }
  console.log(pad(config.id, 28) + cells.join("") + lpad(ok + "/" + total, 10) + "  " + verdict);
});
console.log("Learned decision counts (decisions, fallbacks, guarded) compared on " + statsChecked + " exact runs; LeCaR weight max rel diff " + maxWeightDiff.toExponential(1));

if (approxRows.length) {
  console.log("\nLearned k=16 with k < frames (sampled; JS PRNG differs from NumPy's):");
  console.log(pad("case", 30) + lpad("Python", 8) + lpad("JS", 8) + lpad("diff", 9) + lpad("Python seeds 0-4 spread", 25));
  approxRows.forEach(function (r) {
    console.log(pad(r.where, 30) + lpad(r.py, 8) + lpad(r.js, 8) + lpad(pct(r.rel), 9) + lpad(pct(r.spread), 25) + (r.judged ? "" : "  (< 500 faults, not judged)") + (r.ok ? "" : "  FAIL"));
  });
  const judged = approxRows.filter(function (r) { return r.judged; });
  const worst = judged.reduce(function (m, r) { return Math.max(m, Math.abs(r.rel)); }, 0);
  console.log("max |diff| over " + judged.length + " judged runs: " + pct(worst));
}

// ------------------------------------------------------- features, models

let maxFeature = 0;
let maxProb = 0;
let maxProbJsFeatures = 0;
let treeMismatch = 0;
let rows = 0;
doc.features.forEach(function (group) {
  const x = MLPR.computeFeatures(group.t, group.raw, group.frames);
  x.forEach(function (row, i) {
    rows++;
    row.forEach(function (v, j) {
      const diff = Math.abs(v - group.x[i][j]);
      if (!(diff <= maxFeature)) maxFeature = Number.isNaN(diff) ? Infinity : diff;
    });
  });
  const p = MLPR.scoreModel(doc.model, group.x);
  const pJs = MLPR.scoreModel(doc.model, x);
  p.forEach(function (v, i) {
    maxProb = Math.max(maxProb, Math.abs(v - group.p_logistic[i]));
    maxProbJsFeatures = Math.max(maxProbJsFeatures, Math.abs(pJs[i] - group.p_logistic[i]));
  });
  MLPR.scoreModel(doc.tree_model, group.x).forEach(function (v, i) {
    if (v !== group.p_tree[i]) treeMismatch++;
  });
});
if (JSON.stringify(MLPR.FEATURE_NAMES) !== JSON.stringify(doc.feature_names)) fail("FEATURE_NAMES differ from Python");
if (!(maxFeature <= FEATURE_TOL)) fail("features: max abs diff " + maxFeature + " > " + FEATURE_TOL);
if (!(maxProb <= PROB_TOL && maxProbJsFeatures <= PROB_TOL)) fail("logistic probabilities: max abs diff " + Math.max(maxProb, maxProbJsFeatures));
if (treeMismatch) fail("tree model: " + treeMismatch + " probabilities differ");
console.log("\nFeatures: " + rows + " rows, max abs diff " + maxFeature.toExponential(1) +
  "; logistic probabilities max abs diff " + maxProb.toExponential(1) + " (on JS features " + maxProbJsFeatures.toExponential(1) + ")" +
  "; tree probabilities " + (treeMismatch ? treeMismatch + " differ" : "identical"));

// ------------------------------------------------------- deterministic bits

function sameArray(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

const notes = [];
doc.matmul.forEach(function (m) {
  const js = MLPR.traces.matmul({ n: m.n, pageBytes: m.page_bytes });
  const ok = sameArray(js, m.trace);
  if (!ok) fail("matmul_trace(" + m.n + ", " + m.page_bytes + ") differs (" + js.length + " vs " + m.trace.length + " references)");
  notes.push("matmul_trace(" + m.n + ", " + m.page_bytes + "): " + m.trace.length + " references " + (ok ? "equal" : "DIFFER"));
});

if (!sameArray(MLPR.nextUse(doc.next_use.trace), doc.next_use.next)) fail("next_use differs");
notes.push("next_use: " + (sameArray(MLPR.nextUse(doc.next_use.trace), doc.next_use.next) ? "equal" : "DIFFERS"));

let mtOk = 0, mtTotal = 0;
doc.mt19937.forEach(function (m) {
  const r = new MLPR.PyRandom(m.seed);
  m.random.forEach(function (v, i) {
    mtTotal++;
    const got = r.random();
    if (got === v) mtOk++;
    else fail("PyRandom(" + m.seed + ") draw " + i + ": " + got + " vs Python " + v);
  });
});
notes.push("random.Random: " + mtOk + "/" + mtTotal + " draws equal over " + doc.mt19937.length + " seeds");

let framesOk = 0;
doc.frames_for.forEach(function (f) {
  const trace = [];
  for (let i = 0; i < f.distinct; i++) trace.push(i);
  const got = MLPR.framesFor(trace, f.fraction);
  if (got === f.frames) framesOk++;
  else fail("framesFor(" + f.distinct + " pages, " + f.fraction + "): " + got + " vs Python " + f.frames);
});
doc.cases.filter(function (k) { return k.group === "workload"; }).forEach(function (k) {
  const got = MLPR.memorySizes(k.trace, [0.02, 0.10]).map(function (s) { return s.frames; });
  const want = k.runs.map(function (r) { return r.frames; });
  if (got.join() !== want.join()) fail("memorySizes(" + k.name + "): " + got + " vs Python " + want);
});
notes.push("framesFor: " + framesOk + "/" + doc.frames_for.length + " equal (round half to even); memorySizes equal on the workload cases");

let metricsOk = 0;
doc.metrics.forEach(function (m) {
  const red = MLPR.faultReduction(m.f_lru, m.f);
  const gap = MLPR.gapClosed(m.f_lru, m.f, m.f_opt);
  const gapOk = m.gap_closed === null ? Number.isNaN(gap) : gap === m.gap_closed;
  if (red === m.reduction && gapOk) metricsOk++;
  else fail("metrics(" + [m.f_lru, m.f, m.f_opt] + "): " + red + ", " + gap + " vs Python " + m.reduction + ", " + m.gap_closed);
});
notes.push("faultReduction/gapClosed: " + metricsOk + "/" + doc.metrics.length + " equal");

// --------------------------------------------------- generators (replayed)

/** Feeds a generator the NumPy draws Python's generator made, checking each call. */
class ReplayRng {
  constructor(name, log) {
    this.name = name;
    this.log = log;
    this.pos = 0;
  }

  _next(method, args, n) {
    if (this.pos >= this.log.length) throw new Error(this.name + ": more draws than Python made (" + method + ")");
    const entry = this.log[this.pos++];
    const where = this.name + " draw " + this.pos + " (" + method + ")";
    if (entry[0] !== method) throw new Error(where + ": Python called " + entry[0]);
    const want = entry[1];
    Object.keys(args).forEach(function (key) {
      if (key === "p") {
        const p = args.p;
        if (p.length !== want.p.length) throw new Error(where + ": p has " + p.length + " entries, Python " + want.p.length);
        for (let i = 0; i < p.length; i++) {
          if (Math.abs(p[i] - want.p[i]) > 1e-12 * want.p[i]) throw new Error(where + ": p[" + i + "] " + p[i] + " vs Python " + want.p[i]);
        }
      } else if (args[key] !== want[key] && !(args[key] === undefined && want[key] === null)) {
        throw new Error(where + ": " + key + " " + args[key] + " vs Python " + want[key]);
      }
    });
    const values = entry[2];
    if (n === undefined) {
      if (Array.isArray(values)) throw new Error(where + ": Python drew an array");
      return values;
    }
    if (!Array.isArray(values) || values.length !== n) throw new Error(where + ": size " + n + " vs Python " + (Array.isArray(values) ? values.length : "scalar"));
    return values.slice();
  }

  random(n) { return this._next("random", { size: n }, n); }
  integers(lo, hi, n) { return this._next("integers", { low: lo, high: hi, size: n }, n); }
  permutation(n) { return this._next("permutation", { n: n }, n); }
  choice(n, size, p) { return this._next("choice", { n: n, size: size, p: Array.from(p) }, size); }
  choiceNoReplace(n, size) { return this._next("choiceNoReplace", { n: n, size: size }, size); }
}

const genNotes = [];
doc.generators.forEach(function (g) {
  const rng = new ReplayRng(g.name, g.draws);
  try {
    const trace = MLPR.traces[g.name](Object.assign({}, g.options, { rng: rng }));
    if (rng.pos !== g.draws.length) throw new Error(g.name + ": used " + rng.pos + " of Python's " + g.draws.length + " draws");
    if (!sameArray(trace, g.trace)) {
      let i = 0;
      while (i < trace.length && trace[i] === g.trace[i]) i++;
      throw new Error(g.name + ": trace differs at reference " + i + " (" + trace.length + " vs " + g.trace.length + " references)");
    }
    genNotes.push(g.name + " ok");
  } catch (err) {
    fail("generator " + err.message);
    genNotes.push(g.name + " FAIL");
  }
});
notes.push("generators fed Python's draws: " + genNotes.join(", "));

// -------------------------------------------------------------- workloads

const wlNotes = [];
const pyWorkloads = doc.workloads;
if (MLPR.WORKLOADS.map(function (w) { return w.name; }).join() !== pyWorkloads.map(function (w) { return w.name; }).join()) {
  fail("WORKLOADS names differ from the Python test split");
}
MLPR.WORKLOADS.forEach(function (w, i) {
  const py = pyWorkloads[i];
  if (!py) return;
  if (w.kind !== py.kind || w.tests !== py.tests) fail("workload " + w.name + ": kind/tests differ from Python");
  const trace = w.build({ truncate: false });
  const distinct = new Set(trace).size;
  // Lengths are fixed for synthetic traces and for matmul and mergesort (no data-dependent collapsing).
  if (["bfs"].indexOf(w.name) < 0 && trace.length !== py.accesses) fail("workload " + w.name + ": " + trace.length + " references vs Python " + py.accesses);
  if (["matmul", "mergesort"].indexOf(w.name) >= 0 && distinct !== py.distinct_pages) fail("workload " + w.name + ": " + distinct + " distinct pages vs Python " + py.distinct_pages);
  const short = w.build();
  if (short.length !== Math.min(100000, trace.length)) fail("workload " + w.name + ": default build has " + short.length + " references");
  wlNotes.push(w.name + " " + trace.length + "/" + distinct + " (Py " + py.accesses + "/" + py.distinct_pages + ")");
});
notes.push("workloads, references/distinct pages: " + wlNotes.join(", "));

// ------------------------- end to end: the committed results for matmul

// matmul has no randomness, so the full test trace is Python's exactly and
// the fault counts in results/results.csv (via docs/data.js) must come out
// again: exactly, except the sampled learned policy when frames > k.
const dataPath = path.join(ROOT, "docs", "data.js");
if (fs.existsSync(dataPath)) {
  require(dataPath);
  const bundle = MLPR.data();
  const rows = bundle.results.filter(function (r) { return r.workload === "matmul"; });
  const trace = MLPR.WORKLOADS.filter(function (w) { return w.name === "matmul"; })[0].build({ truncate: false });
  let exact = 0, close = 0, worst = 0;
  rows.forEach(function (r) {
    if (r.accesses !== trace.length) fail("results.csv matmul has " + r.accesses + " references, JS " + trace.length);
    const policy = MLPR.makePolicy(r.policy, r.frames, { trace: trace, model: bundle.model });
    const faults = MLPR.simulate(trace, policy, { timed: false }).faults;
    const where = "results.csv matmul frames=" + r.frames + " " + r.policy;
    if (r.policy === "Learned" && policy.k < r.frames) {
      const rel = (faults - r.faults) / r.faults;
      worst = Math.max(worst, Math.abs(rel));
      if (r.faults >= APPROX_MIN_FAULTS && Math.abs(rel) > APPROX_TOL) fail(where + ": " + faults + " vs " + r.faults + " (" + pct(rel) + ")");
      else close++;
    } else if (faults === r.faults) {
      exact++;
    } else {
      fail(where + ": " + faults + " faults vs " + r.faults);
    }
  });
  notes.push("results.csv, matmul (" + trace.length + " references): " + exact + " rows reproduced exactly, " + close +
    " sampled Learned rows within " + pct(worst));
} else {
  notes.push("results.csv check skipped: no docs/data.js (run python3 tools/build_web.py)");
}

// --------------------------------------------- traceFrames and simulate

let framesChecked = 0;
[[MLPR.TEXTBOOK, 3], [doc.cases[1].trace.slice(0, 600), 8]].forEach(function (pair) {
  const trace = pair[0], frames = pair[1];
  MLPR.POLICY_NAMES.forEach(function (name) {
    const opts = { trace: trace, model: doc.model };
    const steps = MLPR.traceFrames(trace, MLPR.makePolicy(name, frames, opts));
    const shadow = MLPR.makePolicy(name, frames, opts);
    let prev = new Array(frames).fill(null);
    steps.forEach(function (s, t) {
      const hit = shadow.access(trace[t], t);
      const resident = shadow.residentPages().slice().sort(function (a, b) { return a - b; });
      const shown = s.frames.filter(function (p) { return p !== null; }).sort(function (a, b) { return a - b; });
      const changed = s.frames.map(function (p, i) { return p !== prev[i] ? i : -1; }).filter(function (i) { return i >= 0; });
      const okSlot = hit ? changed.length === 0 : changed.length === 1 && s.frames[changed[0]] === trace[t] &&
        (s.victim === null ? prev[changed[0]] === null : prev[changed[0]] === s.victim);
      if (s.page !== trace[t] || s.hit !== hit || resident.join() !== shown.join() || !okSlot) {
        fail("traceFrames " + name + " frames=" + frames + " step " + t + " inconsistent");
      }
      prev = s.frames;
    });
    framesChecked++;
  });
});
notes.push("traceFrames: consistent with residentPages for " + framesChecked + " policy runs");

const timedCase = doc.cases[doc.cases.length - 1];
MLPR.POLICY_NAMES.forEach(function (name) {
  const frames = timedCase.runs[0].frames;
  const opts = { trace: timedCase.trace, model: doc.model, seed: 0 };
  const timedPolicy = MLPR.makePolicy(name, frames, opts);
  const r = MLPR.simulate(timedCase.trace, timedPolicy);
  const plainPolicy = MLPR.makePolicy(name, frames, opts);
  const want = MLPR.simulate(timedCase.trace, plainPolicy, { timed: false }).faults;
  const sorted = function (p) { return p.residentPages().slice().sort(function (a, b) { return a - b; }).join(); };
  if (r.faults !== want || !(r.usPerFault >= 0) || !Number.isFinite(r.usPerFault) || !(r.usPerHit >= 0)) fail("timed simulate " + name + " inconsistent");
  // The timing copies must leave the caller's policy exactly as one pass over the trace left it.
  if (sorted(timedPolicy) !== sorted(plainPolicy) || timedPolicy.decisions !== plainPolicy.decisions) fail("timed simulate " + name + " changed the policy's state");
});
notes.push("timed simulate: same faults as untimed, finite costs, caller's policy left as after one pass");

console.log("\n" + notes.join("\n"));
console.log("\ncases.json from Python at " + doc.source + "; checked in " + ((Date.now() - started) / 1000).toFixed(1) + " s");
if (failures.length) {
  console.log("\nFAILED (" + failures.length + "):");
  failures.slice(0, 40).forEach(function (f) { console.log("  " + f); });
  if (failures.length > 40) console.log("  ... and " + (failures.length - 40) + " more");
  process.exit(1);
}
console.log("\nALL CHECKS PASSED");
