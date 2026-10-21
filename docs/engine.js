/*
 * MLPR engine: the page replacement simulator of the mlpr Python package,
 * ported to plain JavaScript (ES2017, no dependencies, no build step) so it
 * runs in a phone browser.
 *
 * The Python sources under mlpr/ are the ground truth. Every policy here
 * follows them line by line, including tie-breaks, so fault counts match
 * Python exactly. Two things cannot match bit for bit and are documented
 * where they occur:
 *
 *   - the Learned policy with k < frames samples slots with this file's own
 *     PRNG instead of NumPy's PCG64, so it matches Python statistically
 *     (with k >= frames every slot is scored and it matches exactly);
 *   - random workloads use this file's PRNG instead of NumPy's, so they
 *     follow the Python formulas exactly but draw different numbers.
 *
 * LeCaR draws from a port of CPython's Mersenne Twister, seeded exactly like
 * random.Random(seed), so its fault counts match Python exactly.
 *
 * Model scores agree with NumPy to within an ulp or two (NumPy's BLAS sums
 * the dot product in its own order, and log2/exp may round differently);
 * that could only change a decision on an exact tie or a score within
 * 1e-16 of the threshold, which the parity cases never hit.
 *
 * tools/web_parity/check.js replays cases exported from Python
 * (tools/web_parity/export.py) and fails on any mismatch.
 */
(function (root, factory) { if (typeof module === "object" && module.exports) module.exports = factory(root); else root.MLPR = factory(root); })(typeof self !== "undefined" ? self : this, function (root) {
  "use strict";

  const VERSION = "1.0.0";

  // ================================================================ helpers

  function firstKey(collection) {
    return collection.keys().next().value;
  }

  /** Remove and return the oldest entry of a Set or Map (OrderedDict.popitem(last=False)). */
  function popFirst(collection, what) {
    if (collection.size === 0) throw new Error("pop from an empty " + what);
    const key = firstKey(collection);
    collection.delete(key);
    return key;
  }

  function checkFrames(frames) {
    if (typeof frames !== "number" || !Number.isInteger(frames) || frames < 1) {
      throw new Error("frames must be at least 1, got " + frames);
    }
  }

  function given(value) {
    return value !== undefined && value !== null;
  }

  function range(n) {
    const out = new Array(n);
    for (let i = 0; i < n; i++) out[i] = i;
    return out;
  }

  /** Python's round(): halves go to the even neighbour (banker's rounding). */
  function roundHalfEven(x) {
    const floor = Math.floor(x);
    const diff = x - floor;
    if (diff > 0.5) return floor + 1;
    if (diff < 0.5) return floor;
    return floor % 2 === 0 ? floor : floor + 1;
  }

  const now = (function () {
    const perf = (root && root.performance) || (typeof performance !== "undefined" ? performance : null);
    if (perf && typeof perf.now === "function") return function () { return perf.now(); };
    return function () { return Date.now(); };
  })();

  /** Lexicographic "<" on numeric tuples, like Python's tuple comparison. */
  function tupleLess(a, b) {
    for (let i = 0; i < a.length; i++) {
      if (a[i] !== b[i]) return a[i] < b[i];
    }
    return false;
  }

  /** A binary min-heap of numeric tuples (the subset of heapq the policies use). */
  class TupleHeap {
    constructor(items) {
      this.items = items || [];
      for (let i = (this.items.length >> 1) - 1; i >= 0; i--) this._down(i, this.items[i]);
    }

    get size() {
      return this.items.length;
    }

    peek() {
      return this.items[0];
    }

    push(item) {
      const a = this.items;
      let i = a.length;
      a.push(item);
      while (i > 0) {
        const parent = (i - 1) >> 1;
        if (!tupleLess(item, a[parent])) break;
        a[i] = a[parent];
        i = parent;
      }
      a[i] = item;
    }

    pop() {
      const a = this.items;
      if (a.length === 0) throw new Error("pop from an empty heap");
      const top = a[0];
      const last = a.pop();
      if (a.length > 0) this._down(0, last);
      return top;
    }

    _down(i, item) {
      const a = this.items;
      const n = a.length;
      for (;;) {
        let child = 2 * i + 1;
        if (child >= n) break;
        if (child + 1 < n && tupleLess(a[child + 1], a[child])) child++;
        if (!tupleLess(a[child], item)) break;
        a[i] = a[child];
        i = child;
      }
      a[i] = item;
    }
  }

  // ======================================================= random numbers

  /**
   * CPython's random.Random: MT19937 seeded like random.seed(int).
   *
   * The seed's absolute value is split into 32-bit words, least significant
   * first (seed 0 gives the key [0]), and fed to init_by_array. random()
   * combines two outputs into a 53-bit float exactly like genrand_res53, so
   * the sequence equals Python's random.Random(seed).random().
   */
  class PyRandom {
    constructor(seed) {
      this.mt = new Uint32Array(624);
      this.mti = 625;
      this.seed(given(seed) ? seed : 0);
    }

    seed(seed) {
      if (typeof seed !== "number" || !Number.isSafeInteger(seed)) {
        throw new Error("seed must be an integer between -(2^53 - 1) and 2^53 - 1, got " + seed);
      }
      let n = Math.abs(seed);
      const key = [];
      do {
        key.push(n % 4294967296);
        n = Math.floor(n / 4294967296);
      } while (n > 0);
      this._initByArray(key);
    }

    _initGenrand(s) {
      const mt = this.mt;
      mt[0] = s >>> 0;
      for (let i = 1; i < 624; i++) {
        const prev = mt[i - 1] ^ (mt[i - 1] >>> 30);
        mt[i] = (Math.imul(1812433253, prev) + i) >>> 0;
      }
      this.mti = 624;
    }

    _initByArray(key) {
      const mt = this.mt;
      const N = 624;
      this._initGenrand(19650218);
      let i = 1;
      let j = 0;
      for (let k = Math.max(N, key.length); k > 0; k--) {
        const prev = mt[i - 1] ^ (mt[i - 1] >>> 30);
        mt[i] = ((mt[i] ^ Math.imul(prev, 1664525)) + key[j] + j) >>> 0;
        i++;
        j++;
        if (i >= N) { mt[0] = mt[N - 1]; i = 1; }
        if (j >= key.length) j = 0;
      }
      for (let k = N - 1; k > 0; k--) {
        const prev = mt[i - 1] ^ (mt[i - 1] >>> 30);
        mt[i] = ((mt[i] ^ Math.imul(prev, 1566083941)) - i) >>> 0;
        i++;
        if (i >= N) { mt[0] = mt[N - 1]; i = 1; }
      }
      mt[0] = 0x80000000;
      this.mti = N;
    }

    /** The next 32-bit output (genrand_uint32). */
    uint32() {
      const mt = this.mt;
      if (this.mti >= 624) {
        let kk = 0;
        for (; kk < 624 - 397; kk++) {
          const y = (mt[kk] & 0x80000000) | (mt[kk + 1] & 0x7fffffff);
          mt[kk] = mt[kk + 397] ^ (y >>> 1) ^ (y & 1 ? 0x9908b0df : 0);
        }
        for (; kk < 623; kk++) {
          const y = (mt[kk] & 0x80000000) | (mt[kk + 1] & 0x7fffffff);
          mt[kk] = mt[kk - 227] ^ (y >>> 1) ^ (y & 1 ? 0x9908b0df : 0);
        }
        const y = (mt[623] & 0x80000000) | (mt[0] & 0x7fffffff);
        mt[623] = mt[396] ^ (y >>> 1) ^ (y & 1 ? 0x9908b0df : 0);
        this.mti = 0;
      }
      let y = mt[this.mti++];
      y ^= y >>> 11;
      y ^= (y << 7) & 0x9d2c5680;
      y ^= (y << 15) & 0xefc60000;
      y ^= y >>> 18;
      return y >>> 0;
    }

    /** A float in [0, 1) with 53 random bits, identical to Python's random(). */
    random() {
      const a = this.uint32() >>> 5;
      const b = this.uint32() >>> 6;
      return (a * 67108864 + b) / 9007199254740992;
    }
  }

  /**
   * The PRNG for workloads and the Learned policy's slot sampler:
   * xoshiro128** (Blackman and Vigna), its four 32-bit state words filled
   * by splitmix32 from the seed. NumPy's PCG64 is not ported, so draws differ
   * from Python's; the methods mirror the numpy.random.Generator calls the
   * Python code makes:
   *
   *   random(n)              uniform floats in [0, 1), 53 bits each
   *   integers(lo, hi, n)    uniform integers in [lo, hi)
   *   permutation(n)         a shuffled 0..n-1 (Fisher-Yates)
   *   choice(n, size, p)     indices drawn with probabilities p, with
   *                          replacement, by NumPy's method: cumulative sum
   *                          normalised by its last value, then a right-side
   *                          binary search for each uniform draw
   *   choiceNoReplace(n, size)  a random subset in random order
   *
   * With n omitted, random() and integers() return a single number.
   * Integers are floor(u * span) with a 53-bit u; the bias is below 2^-30.
   */
  class Rng {
    constructor(seed) {
      seed = given(seed) ? seed : 0;
      if (typeof seed !== "number" || !Number.isFinite(seed)) throw new Error("seed must be a number, got " + seed);
      seed = Math.trunc(seed);
      let a = (seed % 4294967296) >>> 0;
      const high = Math.floor(Math.abs(seed) / 4294967296) >>> 0;
      a = (a ^ Math.imul(high, 0x85ebca6b) ^ (seed < 0 ? 0x5bd1e995 : 0)) >>> 0;
      const s = new Uint32Array(4);
      for (let i = 0; i < 4; i++) {
        // splitmix32
        a = (a + 0x9e3779b9) | 0;
        let z = a ^ (a >>> 16);
        z = Math.imul(z, 0x21f0aaad);
        z ^= z >>> 15;
        z = Math.imul(z, 0x735a2d97);
        z ^= z >>> 15;
        s[i] = z >>> 0;
      }
      if ((s[0] | s[1] | s[2] | s[3]) === 0) s[0] = 1;
      this.s = s;
    }

    /** The next 32-bit output of xoshiro128**. */
    uint32() {
      const s = this.s;
      const m = Math.imul(s[1], 5);
      const result = Math.imul((m << 7) | (m >>> 25), 9) >>> 0;
      const t = s[1] << 9;
      s[2] ^= s[0];
      s[3] ^= s[1];
      s[1] ^= s[2];
      s[0] ^= s[3];
      s[2] ^= t;
      s[3] = (s[3] << 11) | (s[3] >>> 21);
      return result;
    }

    _float() {
      const a = this.uint32() >>> 5;
      const b = this.uint32() >>> 6;
      return (a * 67108864 + b) / 9007199254740992;
    }

    random(n) {
      if (n === undefined) return this._float();
      const out = new Float64Array(n);
      for (let i = 0; i < n; i++) out[i] = this._float();
      return out;
    }

    integers(lo, hi, n) {
      const span = hi - lo;
      if (!(span > 0)) throw new Error("integers: need lo < hi, got " + lo + ", " + hi);
      if (n === undefined) return lo + Math.floor(this._float() * span);
      const out = new Array(n);
      for (let i = 0; i < n; i++) out[i] = lo + Math.floor(this._float() * span);
      return out;
    }

    /** A uniform slot index in [0, n): the hot path of the Learned sampler. */
    below(n) {
      return Math.floor(this._float() * n);
    }

    permutation(n) {
      const out = range(n);
      for (let i = n - 1; i > 0; i--) {
        const j = Math.floor(this._float() * (i + 1));
        const tmp = out[i];
        out[i] = out[j];
        out[j] = tmp;
      }
      return out;
    }

    choice(n, size, p) {
      if (!p) return this.integers(0, n, size);
      if (p.length !== n) throw new Error("choice: p must have " + n + " entries");
      const cdf = new Float64Array(n);
      let acc = 0;
      for (let i = 0; i < n; i++) {
        acc += p[i];
        cdf[i] = acc;
      }
      const last = cdf[n - 1];
      for (let i = 0; i < n; i++) cdf[i] /= last;
      const out = new Array(size);
      for (let s = 0; s < size; s++) {
        const u = this._float();
        let lo = 0;
        let hi = n;
        while (lo < hi) {
          // first index with cdf > u (searchsorted side="right")
          const mid = (lo + hi) >>> 1;
          if (cdf[mid] <= u) lo = mid + 1;
          else hi = mid;
        }
        out[s] = lo;
      }
      return out;
    }

    choiceNoReplace(n, size) {
      if (size > n) throw new Error("choice: cannot take a larger sample than the population without replacement");
      const pool = range(n);
      for (let i = 0; i < size; i++) {
        const j = i + Math.floor(this._float() * (n - i));
        const tmp = pool[i];
        pool[i] = pool[j];
        pool[j] = tmp;
      }
      return pool.slice(0, size);
    }
  }

  // ================================================ classical policies

  /** First-in, first-out: evicts the page that has been in memory the longest. */
  class FIFO {
    constructor(frames, opts) {
      checkFrames(frames);
      this.name = "FIFO";
      this.frames = frames;
      this.options = opts || {};
      this._queue = new Array(frames); // ring buffer, oldest at _head
      this._head = 0;
      this._resident = new Set();
    }

    access(page, t) {
      const resident = this._resident;
      if (resident.has(page)) return true;
      const n = resident.size;
      if (n >= this.frames) {
        // The queue is full: the oldest page leaves and the new one takes its place at the tail.
        resident.delete(this._queue[this._head]);
        this._queue[this._head] = page;
        this._head = (this._head + 1) % this.frames;
      } else {
        this._queue[(this._head + n) % this.frames] = page;
      }
      resident.add(page);
      return false;
    }

    residentPages() {
      return Array.from(this._resident);
    }
  }

  /** Least recently used: evicts the page whose last reference is oldest. */
  class LRU {
    constructor(frames, opts) {
      checkFrames(frames);
      this.name = "LRU";
      this.frames = frames;
      this.options = opts || {};
      this._pages = new Set(); // insertion order = recency order, like OrderedDict
    }

    access(page, t) {
      const pages = this._pages;
      if (pages.has(page)) {
        pages.delete(page);
        pages.add(page);
        return true;
      }
      if (pages.size >= this.frames) pages.delete(firstKey(pages));
      pages.add(page);
      return false;
    }

    residentPages() {
      return Array.from(this._pages);
    }
  }

  /**
   * Least frequently used: evicts the resident page with the fewest
   * references since it was loaded. Ties go to the least recently used page.
   * A heap with lazy deletion keeps each decision at O(log n); stale heap
   * entries are dropped when popped and the heap is rebuilt when it grows
   * well past the number of frames.
   */
  class LFU {
    constructor(frames, opts) {
      checkFrames(frames);
      this.name = "LFU";
      this.frames = frames;
      this.options = opts || {};
      this._key = new Map(); // page -> [count, last access]
      this._heap = new TupleHeap(); // [count, last access, page]
    }

    access(page, t) {
      const key = this._key.get(page);
      const hit = key !== undefined;
      let count = 1;
      if (hit) {
        count = key[0] + 1;
      } else if (this._key.size >= this.frames) {
        this._evict();
      }
      this._key.set(page, [count, t]);
      this._heap.push([count, t, page]);
      if (this._heap.size > 4 * this.frames + 64) this._compact();
      return hit;
    }

    _evict() {
      const keys = this._key;
      for (;;) {
        const entry = this._heap.pop();
        const key = keys.get(entry[2]);
        if (key !== undefined && key[0] === entry[0] && key[1] === entry[1]) {
          keys.delete(entry[2]);
          return;
        }
      }
    }

    _compact() {
      const items = [];
      this._key.forEach(function (key, page) { items.push([key[0], key[1], page]); });
      this._heap = new TupleHeap(items);
    }

    residentPages() {
      return Array.from(this._key.keys());
    }
  }

  /**
   * CLOCK (second chance), the LRU approximation used by real kernels.
   * Frames sit on a circle with one reference bit each. A hit sets the
   * page's bit. On a fault the hand sweeps forward, clearing set bits, and
   * evicts the first page whose bit is already clear.
   */
  class CLOCK {
    constructor(frames, opts) {
      checkFrames(frames);
      this.name = "CLOCK";
      this.frames = frames;
      this.options = opts || {};
      this._pages = new Array(frames).fill(null);
      this._ref = new Uint8Array(frames);
      this._slot = new Map(); // page -> slot
      this._hand = 0;
    }

    access(page, t) {
      let slot = this._slot.get(page);
      if (slot !== undefined) {
        this._ref[slot] = 1;
        return true;
      }
      if (this._slot.size < this.frames) {
        slot = this._slot.size;
      } else {
        const ref = this._ref;
        const n = this.frames;
        while (ref[this._hand]) {
          ref[this._hand] = 0;
          this._hand = (this._hand + 1) % n;
        }
        slot = this._hand;
        this._slot.delete(this._pages[slot]);
        this._hand = (slot + 1) % n;
      }
      this._pages[slot] = page;
      this._ref[slot] = 1;
      this._slot.set(page, slot);
      return false;
    }

    residentPages() {
      return Array.from(this._slot.keys());
    }
  }

  /**
   * Adaptive Replacement Cache (Megiddo and Modha, FAST 2003).
   * T1 holds pages referenced once recently and T2 pages referenced at least
   * twice. B1 and B2 are ghost lists that remember pages recently evicted
   * from T1 and T2. A miss that hits B1 means T1 was too small, so the target
   * size p of T1 grows; a miss that hits B2 shrinks it. Every list is kept
   * in LRU order (oldest first). p is a float, as in the Python code.
   */
  class ARC {
    constructor(frames, opts) {
      checkFrames(frames);
      this.name = "ARC";
      this.frames = frames;
      this.options = opts || {};
      this.p = 0.0;
      this.t1 = new Set();
      this.t2 = new Set();
      this.b1 = new Set();
      this.b2 = new Set();
    }

    access(page, t) {
      const c = this.frames;
      const t1 = this.t1, t2 = this.t2, b1 = this.b1, b2 = this.b2;

      if (t1.has(page)) {
        t1.delete(page);
        t2.add(page);
        return true;
      }
      if (t2.has(page)) {
        t2.delete(page);
        t2.add(page);
        return true;
      }

      if (b1.has(page)) {
        this.p = Math.min(c, this.p + Math.max(b2.size / b1.size, 1.0));
        this._replace(false);
        b1.delete(page);
        t2.add(page);
        return false;
      }
      if (b2.has(page)) {
        this.p = Math.max(0.0, this.p - Math.max(b1.size / b2.size, 1.0));
        this._replace(true);
        b2.delete(page);
        t2.add(page);
        return false;
      }

      // Brand-new page.
      const l1 = t1.size + b1.size;
      if (l1 === c) {
        if (t1.size < c) {
          popFirst(b1, "B1");
          this._replace(false);
        } else {
          popFirst(t1, "T1");
        }
      } else {
        const total = l1 + t2.size + b2.size;
        if (total >= c) {
          if (total === 2 * c) popFirst(b2, "B2");
          this._replace(false);
        }
      }
      t1.add(page);
      return false;
    }

    _replace(inB2) {
      const t1 = this.t1;
      const takeT1 = t1.size > 0 && (t1.size > this.p || (inB2 && t1.size === this.p));
      if (takeT1 || this.t2.size === 0) {
        this.b1.add(popFirst(t1, "T1"));
      } else {
        this.b2.add(popFirst(this.t2, "T2"));
      }
    }

    residentPages() {
      return Array.from(this.t1).concat(Array.from(this.t2));
    }
  }

  /**
   * LeCaR (Vietri et al., HotStorage 2018): online learning over LRU and LFU.
   * Picks LRU's or LFU's victim at random according to two learned weights.
   * Each evicted page goes into the history of the expert that chose it. If a
   * page comes back while it is still in an expert's history, that expert
   * made a mistake (regret), so the other expert's weight is increased by an
   * amount that decays with the time the page spent in history.
   *
   * The coin flips come from PyRandom(seed), so fault counts equal Python's.
   */
  class LeCaR {
    constructor(frames, opts) {
      checkFrames(frames);
      opts = opts || {};
      this.name = "LeCaR";
      this.frames = frames;
      this.options = opts;
      this.learningRate = given(opts.learningRate) ? Number(opts.learningRate) : 0.45;
      this.discount = Math.pow(0.005, 1.0 / frames);
      this.wLru = 0.5;
      this.wLfu = 0.5;
      this._rng = new PyRandom(given(opts.seed) ? opts.seed : 0);
      this._lru = new Set();
      this._freq = new Map();
      this._last = new Map();
      this._heap = new TupleHeap(); // [freq, last access, page]
      // page -> [eviction time, frequency at eviction]
      this._histLru = new Map();
      this._histLfu = new Map();
    }

    access(page, t) {
      const known = this._freq.get(page);
      if (known !== undefined) {
        this._lru.delete(page);
        this._lru.add(page);
        this._touch(page, known + 1, t);
        return true;
      }

      let freq = 1;
      let lruErred = true;
      let entry = this._histLru.get(page);
      if (entry !== undefined) {
        this._histLru.delete(page);
      } else {
        lruErred = false;
        entry = this._histLfu.get(page);
        if (entry !== undefined) this._histLfu.delete(page);
      }
      if (entry !== undefined) {
        const boost = Math.exp(this.learningRate * Math.pow(this.discount, t - entry[0]));
        if (lruErred) this.wLfu *= boost;
        else this.wLru *= boost;
        const total = this.wLru + this.wLfu;
        this.wLru /= total;
        this.wLfu /= total;
        freq = entry[1] + 1;
      }

      if (this._freq.size >= this.frames) this._evict(t);
      this._lru.add(page);
      this._touch(page, freq, t);
      return false;
    }

    _touch(page, freq, t) {
      this._freq.set(page, freq);
      this._last.set(page, t);
      this._heap.push([freq, t, page]);
      if (this._heap.size > 4 * this.frames + 64) {
        const items = [];
        const last = this._last;
        this._freq.forEach(function (f, p) { items.push([f, last.get(p), p]); });
        this._heap = new TupleHeap(items);
      }
    }

    _lfuVictim() {
      const heap = this._heap;
      for (;;) {
        const top = heap.peek();
        if (this._freq.get(top[2]) === top[0] && this._last.get(top[2]) === top[1]) return top[2];
        heap.pop();
      }
    }

    _evict(t) {
      const lruVictim = firstKey(this._lru);
      const lfuVictim = this._lfuVictim();
      let victim;
      let history;
      if (lruVictim === lfuVictim) {
        victim = lruVictim;
        history = null;
      } else if (this._rng.random() < this.wLru) {
        victim = lruVictim;
        history = this._histLru;
      } else {
        victim = lfuVictim;
        history = this._histLfu;
      }
      if (history !== null) {
        history.set(victim, [t, this._freq.get(victim)]);
        if (history.size > this.frames) popFirst(history, "history");
      }
      this._lru.delete(victim);
      this._freq.delete(victim);
      this._last.delete(victim);
    }

    residentPages() {
      return Array.from(this._freq.keys());
    }
  }

  /**
   * For every position t, the index of the next reference to trace[t].
   * Pages that are never referenced again get trace.length.
   */
  function nextUseTyped(trace) {
    const n = trace.length;
    const result = n < 2147483648 ? new Int32Array(n) : new Float64Array(n);
    const seen = new Map();
    for (let t = n - 1; t >= 0; t--) {
      const page = trace[t];
      const next = seen.get(page);
      result[t] = next === undefined ? n : next;
      seen.set(page, t);
    }
    return result;
  }

  function nextUse(trace) {
    return Array.from(nextUseTyped(trace));
  }

  /**
   * Belady's optimal replacement (OPT / MIN): evicts the page whose next
   * reference is furthest in the future (ties, which only happen between
   * pages never used again, go to the smallest page number, as Python's
   * heap of (-next use, page) does). It needs the whole trace up front.
   * A reference past the end of the trace (only simulate's timing replay
   * makes one) counts as never used again.
   */
  class OPT {
    constructor(frames, opts) {
      checkFrames(frames);
      opts = opts || {};
      if (!given(opts.trace)) throw new Error("OPT needs the whole trace: it looks into the future (pass opts.trace)");
      this.name = "OPT";
      this.frames = frames;
      this.options = opts;
      this._next = nextUseTyped(opts.trace);
      this._due = new Map(); // resident page -> time of its next use
      this._heap = new TupleHeap(); // [-next use, page], lazy deletion
    }

    access(page, t) {
      const due = this._due;
      const hit = due.has(page);
      if (!hit && due.size >= this.frames) this._evict();
      const next = t < this._next.length ? this._next[t] : this._next.length;
      due.set(page, next);
      this._heap.push([-next, page]);
      if (this._heap.size > 4 * this.frames + 64) this._compact();
      return hit;
    }

    _evict() {
      const due = this._due;
      for (;;) {
        const entry = this._heap.pop();
        if (due.get(entry[1]) === -entry[0]) {
          due.delete(entry[1]);
          return;
        }
      }
    }

    _compact() {
      const items = [];
      this._due.forEach(function (next, page) { items.push([-next, page]); });
      this._heap = new TupleHeap(items);
    }

    residentPages() {
      return Array.from(this._due.keys());
    }
  }

  // ===================================================== features, models

  /*
   * Every feature is known at fault time without any future knowledge:
   *
   *   age                 time since the page was last referenced
   *   count               number of references seen so far
   *   gap1, gap2, gap3    the last three gaps between consecutive references
   *   mean_gap            average gap between references
   *   age_over_mean_gap   how "overdue" the page is relative to its usual gap
   *
   * Time quantities are divided by the number of frames so a single model
   * works across memory sizes. Every feature is capped at CAP and
   * log-scaled. A gap not observed yet is censored: the first missing gap
   * (and the mean gap of a page seen once) is filled in with the age, older
   * missing gaps with the newest known one.
   */
  const FEATURE_NAMES = ["age", "count", "gap1", "gap2", "gap3", "mean_gap", "age_over_mean_gap"];
  const RAW_FIELDS = ["last", "count", "gap1", "gap2", "gap3", "gap_sum"];
  const LAST = 0, COUNT = 1, GAP1 = 2, GAP2 = 3, GAP3 = 4, GAP_SUM = 5;
  const UNKNOWN = -1;
  const CAP = 1024.0;

  /** Write the 7 features of one raw record into out, in compute_features' order of operations. */
  function featuresInto(out, t, rec, frames) {
    const age = t - rec[LAST];
    const count = rec[COUNT];
    const gap1 = rec[GAP1] >= 0 ? rec[GAP1] : age;
    const gap2 = rec[GAP2] >= 0 ? rec[GAP2] : gap1;
    const gap3 = rec[GAP3] >= 0 ? rec[GAP3] : gap2;
    const gapsSeen = count - 1;
    const meanGap = gapsSeen > 0 ? rec[GAP_SUM] / Math.max(gapsSeen, 1) : age;
    const ratio = age / Math.max(meanGap, 1.0);
    out[0] = Math.log2(Math.min(age / frames, CAP) + 1.0);
    out[1] = Math.log2(Math.min(count, CAP) + 1.0);
    out[2] = Math.log2(Math.min(gap1 / frames, CAP) + 1.0);
    out[3] = Math.log2(Math.min(gap2 / frames, CAP) + 1.0);
    out[4] = Math.log2(Math.min(gap3 / frames, CAP) + 1.0);
    out[5] = Math.log2(Math.min(meanGap / frames, CAP) + 1.0);
    out[6] = Math.log2(Math.min(ratio, CAP) + 1.0);
    return out;
  }

  /**
   * featuresInto for the learned policy's hot loop. Once a page's last three
   * gaps are known, features 1-5 and its mean gap depend only on its record,
   * not on t, so they are cached in the record (entries 6-12) until its next
   * reference changes the count. They come from the same expressions, so the
   * features are bit for bit those of featuresInto; only 2 of 7 logarithms
   * are left per scored page.
   */
  function featuresCached(out, t, rec, frames) {
    const count = rec[COUNT];
    if (!(count > 1 && rec[GAP1] >= 0 && rec[GAP2] >= 0 && rec[GAP3] >= 0)) return featuresInto(out, t, rec, frames);
    const age = t - rec[LAST];
    if (rec[6] !== count) {
      featuresInto(out, t, rec, frames);
      rec[6] = count;
      rec[7] = rec[GAP_SUM] / Math.max(count - 1, 1);
      for (let j = 1; j <= 5; j++) rec[7 + j] = out[j];
      return out;
    }
    out[0] = Math.log2(Math.min(age / frames, CAP) + 1.0);
    out[1] = rec[8];
    out[2] = rec[9];
    out[3] = rec[10];
    out[4] = rec[11];
    out[5] = rec[12];
    out[6] = Math.log2(Math.min(age / Math.max(rec[7], 1.0), CAP) + 1.0);
    return out;
  }

  /**
   * Feature matrix with one row per page, like mlpr.features.compute_features.
   * rawRows has one row per page: [last, count, gap1, gap2, gap3, gapSum],
   * with -1 for a gap not seen yet.
   */
  function computeFeatures(t, rawRows, frames) {
    return rawRows.map(function (row) { return Array.from(featuresInto(new Float64Array(7), t, row, frames)); });
  }

  /**
   * Metadata of a page referenced for the first time at t: the six
   * RAW_FIELDS, then room for the feature cache of featuresCached (its count
   * entry 0 marks the cache empty).
   */
  function newRecord(t) {
    return [t, 1, UNKNOWN, UNKNOWN, UNKNOWN, 0, 0, 0, 0, 0, 0, 0, 0];
  }

  /** Account for a new reference at t. */
  function updateRecord(record, t) {
    const gap = t - record[LAST];
    record[GAP3] = record[GAP2];
    record[GAP2] = record[GAP1];
    record[GAP1] = gap;
    record[GAP_SUM] += gap;
    record[COUNT] += 1;
    record[LAST] = t;
  }

  const compiledModels = typeof WeakMap === "function" ? new WeakMap() : null;

  /**
   * Turn a model.json object into a scorer with score(x) -> probability that
   * the page will NOT be needed soon, x being the 7 features of one page.
   *   logistic: sigmoid(x[features] . weights + bias), z clipped to [-35, 35]
   *   tree:     walk the nodes from the root, "<=" goes left, feature -1 is a leaf
   */
  function compileModel(model) {
    if (!model || typeof model !== "object") throw new Error("a model must be a model.json object");
    if (compiledModels && compiledModels.has(model)) return compiledModels.get(model);
    const kind = model.kind || "logistic";
    const features = Int32Array.from(given(model.features) ? model.features : range(FEATURE_NAMES.length));
    let score;
    if (kind === "logistic") {
      const weights = Float64Array.from(model.weights || []);
      const bias = Number(model.bias || 0);
      if (weights.length !== features.length) throw new Error("logistic model: " + weights.length + " weights for " + features.length + " features");
      const m = features.length;
      if (m === 7 && features.every(function (f, j) { return f === j; })) {
        // The usual model: all seven features in order, the same sum unrolled.
        const w0 = weights[0], w1 = weights[1], w2 = weights[2], w3 = weights[3], w4 = weights[4], w5 = weights[5], w6 = weights[6];
        score = function (x) {
          let z = x[0] * w0 + x[1] * w1 + x[2] * w2 + x[3] * w3 + x[4] * w4 + x[5] * w5 + x[6] * w6 + bias;
          z = Math.min(35.0, Math.max(-35.0, z));
          return 1.0 / (1.0 + Math.exp(-z));
        };
      } else {
        score = function (x) {
          let z = 0.0;
          for (let j = 0; j < m; j++) z += x[features[j]] * weights[j];
          z += bias;
          z = Math.min(35.0, Math.max(-35.0, z));
          return 1.0 / (1.0 + Math.exp(-z));
        };
      }
    } else if (kind === "tree") {
      const nodes = model.nodes;
      const feature = Int32Array.from(nodes.feature);
      const threshold = Float64Array.from(nodes.threshold);
      const left = Int32Array.from(nodes.left);
      const right = Int32Array.from(nodes.right);
      const value = Float64Array.from(nodes.value);
      score = function (x) {
        let node = 0;
        while (feature[node] >= 0) node = x[features[feature[node]]] <= threshold[node] ? left[node] : right[node];
        return value[node];
      };
    } else {
      throw new Error("unknown model kind " + JSON.stringify(kind));
    }
    const compiled = { kind: kind, meta: model.meta || {}, score: score };
    if (compiledModels) compiledModels.set(model, compiled);
    return compiled;
  }

  /** Probabilities for feature rows, like model.predict_proba(X). */
  function scoreModel(model, featureRows) {
    const scorer = compileModel(model);
    return featureRows.map(function (row) { return scorer.score(row); });
  }

  /** The data bundle written by tools/build_web.py (docs/data.js), or null. */
  function data() {
    if (root && root.MLPR_DATA) return root.MLPR_DATA;
    if (typeof globalThis !== "undefined" && globalThis.MLPR_DATA) return globalThis.MLPR_DATA;
    return null;
  }

  // ================================================== the learned policy

  /**
   * Metadata for resident pages, stored per frame slot. Slots fill in order
   * while memory has free frames; afterwards a newly loaded page always
   * takes the slot of the page it replaced. Metadata of recently evicted
   * pages is kept in a bounded ghost table, so a page that comes back keeps
   * its reference count and gap history.
   */
  class MetadataTable {
    constructor(frames, ghostSize) {
      this.frames = frames;
      this.ghostSize = given(ghostSize) ? ghostSize : frames;
      this.pages = new Array(frames).fill(null);
      this.slotOf = new Map();
      this.records = [];
      for (let i = 0; i < frames; i++) this.records.push(newRecord(0));
      this.ghost = new Map();
    }

    freeSlot() {
      const used = this.slotOf.size;
      return used < this.frames ? used : null;
    }

    last(slot) {
      return this.records[slot][LAST];
    }

    /** Record a hit on a resident page. */
    touch(page, t) {
      updateRecord(this.records[this.slotOf.get(page)], t);
    }

    /** Place a faulting page in slot, restoring its history if it is a ghost. */
    load(page, t, slot) {
      let record = this.ghost.get(page);
      if (record === undefined) {
        record = newRecord(t);
      } else {
        this.ghost.delete(page);
        updateRecord(record, t);
      }
      this.pages[slot] = page;
      this.slotOf.set(page, slot);
      this.records[slot] = record;
    }

    /** Remove the page in slot and remember its metadata in the ghost table. */
    evict(slot) {
      const page = this.pages[slot];
      this.slotOf.delete(page);
      if (this.ghostSize > 0) {
        this.ghost.set(page, this.records[slot]);
        if (this.ghost.size > this.ghostSize) popFirst(this.ghost, "ghost table");
      }
      return page;
    }

    /** Feature rows for the pages in slots. */
    features(t, slots) {
      const records = this.records;
      const frames = this.frames;
      return slots.map(function (s) { return Array.from(featuresInto(new Float64Array(7), t, records[s], frames)); });
    }
  }

  /**
   * Draws k frame slots uniformly at random, with replacement, which keeps
   * each draw O(k); a repeated slot just means fewer distinct candidates
   * that time. With k >= frames every slot is a candidate, in slot order.
   * Python draws with NumPy's PCG64; this uses Rng (xoshiro128**), so the
   * draws differ while their distribution is the same.
   */
  class SlotSampler {
    constructor(frames, k, seed) {
      this.frames = frames;
      this.k = k;
      this.everySlot = k >= frames ? range(frames) : null;
      this._rng = new Rng(given(seed) ? seed : 0);
      this._row = new Array(k);
    }

    draw() {
      if (this.everySlot !== null) return this.everySlot;
      const row = this._row;
      for (let i = 0; i < this.k; i++) row[i] = this._rng.below(this.frames);
      return row;
    }
  }

  /**
   * Switches the learned policy to plain LRU while it is losing to LRU.
   * A shadow LRU directory (page numbers only) counts the faults LRU would
   * have taken on the same references. At the end of every window the
   * learned policy's faults are compared with the shadow's. If they exceed
   * them by more than margin, eviction switches to exact LRU for backoff
   * windows, then the model is tried again; backoff doubles after every
   * losing window and halves after every winning one.
   */
  class RegretGuard {
    constructor(frames, window, margin, maxBackoff) {
      this.frames = frames;
      this.window = given(window) ? window : Math.max(1000, 8 * frames);
      this.margin = given(margin) ? margin : 0.02;
      this.maxBackoff = given(maxBackoff) ? maxBackoff : 64;
      this.shadow = new Set();
      this.useLru = false;
      this.backoff = 1;
      this._lruWindowsLeft = 0;
      this._seen = 0;
      this._faults = 0;
      this._shadowFaults = 0;
      this.windows = 0;
      this.lruWindows = 0;
    }

    observe(page, faulted) {
      const shadow = this.shadow;
      if (shadow.has(page)) {
        shadow.delete(page);
        shadow.add(page);
      } else {
        this._shadowFaults += 1;
        shadow.add(page);
        if (shadow.size > this.frames) shadow.delete(firstKey(shadow));
      }
      if (faulted) this._faults += 1;
      this._seen += 1;
      if (this._seen >= this.window) this._endWindow();
    }

    _endWindow() {
      this.windows += 1;
      if (this.useLru) {
        this.lruWindows += 1;
        this._lruWindowsLeft -= 1;
        if (this._lruWindowsLeft <= 0) this.useLru = false;
      } else if (this._faults > this._shadowFaults * (1.0 + this.margin)) {
        this.useLru = true;
        this._lruWindowsLeft = this.backoff;
        this.backoff = Math.min(2 * this.backoff, this.maxBackoff);
      } else {
        this.backoff = Math.max(1, Math.floor(this.backoff / 2));
      }
      this._seen = 0;
      this._faults = 0;
      this._shadowFaults = 0;
    }
  }

  /**
   * Sampled eviction scored by a small model, with two LRU safety nets.
   * On every fault with memory full, k resident pages are sampled and the
   * model estimates for each the probability that it will not be needed
   * soon. The page with the highest score is evicted (the first one on a
   * tie, like numpy.argmax). If even the best score is below threshold the
   * model is treated as unsure and the least recently used sampled page is
   * evicted instead (the first sampled slot with the smallest last access,
   * like Python's min); and with guard on, a RegretGuard hands eviction to
   * exact LRU while the model is doing worse than LRU would.
   */
  class Learned {
    constructor(frames, opts) {
      checkFrames(frames);
      opts = opts || {};
      let model = opts.model;
      if (!given(model)) {
        const bundle = data();
        model = bundle && bundle.model;
      }
      if (!given(model)) {
        throw new Error("the Learned policy needs a trained model: load data.js (MLPR_DATA.model) or pass opts.model");
      }
      this.name = "Learned";
      this.frames = frames;
      this.options = opts;
      this.model = model;
      this._scorer = compileModel(model);
      const meta = model.meta || {};
      this.k = Math.trunc(Number(given(opts.k) ? opts.k : (given(meta.k) ? meta.k : 16)));
      this.threshold = Number(given(opts.threshold) ? opts.threshold : (given(meta.threshold) ? meta.threshold : 0.5));
      const useGhost = Boolean(given(opts.ghost) ? opts.ghost : (given(meta.ghost) ? meta.ghost : true));
      const useGuard = Boolean(given(opts.guard) ? opts.guard : (given(meta.guard) ? meta.guard : true));
      if (!(this.k >= 1)) throw new Error("k must be at least 1, got " + opts.k);
      this.table = new MetadataTable(frames, useGhost ? frames : 0);
      this.sampler = new SlotSampler(frames, this.k, given(opts.seed) ? opts.seed : 0);
      this.guard = useGuard ? new RegretGuard(frames) : null;
      // Exact recency order of resident pages, only needed when the guard hands over to LRU.
      this._recency = useGuard ? new Set() : null;
      this._x = new Float64Array(7);
      this.decisions = 0;
      this.fallbacks = 0;
      this.guarded = 0;
    }

    access(page, t) {
      const table = this.table;
      const recency = this._recency;
      const hit = table.slotOf.has(page);
      if (hit) {
        table.touch(page, t);
        if (recency !== null) {
          recency.delete(page);
          recency.add(page);
        }
      } else {
        let slot = table.freeSlot();
        if (slot === null) {
          slot = this._chooseSlot(t);
          const victim = table.evict(slot);
          if (recency !== null) recency.delete(victim);
        }
        table.load(page, t, slot);
        if (recency !== null) recency.add(page);
      }
      if (this.guard !== null) this.guard.observe(page, !hit);
      return hit;
    }

    _chooseSlot(t) {
      this.decisions += 1;
      if (this.guard !== null && this.guard.useLru) {
        this.guarded += 1;
        return this.table.slotOf.get(firstKey(this._recency));
      }
      // choose_victim: score the sampled slots, evict the best unless the model is unsure.
      const slots = this.sampler.draw();
      const records = this.table.records;
      const frames = this.frames;
      const x = this._x;
      const score = this._scorer.score;
      let best = 0;
      let bestScore = -Infinity;
      for (let i = 0; i < slots.length; i++) {
        const s = score(featuresCached(x, t, records[slots[i]], frames));
        if (s > bestScore) {
          bestScore = s;
          best = i;
        }
      }
      if (bestScore >= this.threshold) return slots[best];
      this.fallbacks += 1;
      let victim = slots[0];
      let oldest = records[victim][LAST];
      for (let i = 1; i < slots.length; i++) {
        const last = records[slots[i]][LAST];
        if (last < oldest) {
          oldest = last;
          victim = slots[i];
        }
      }
      return victim;
    }

    /** Share of evictions where the model was unsure and LRU among the sample decided. */
    get fallbackRate() {
      return this.decisions ? this.fallbacks / this.decisions : 0.0;
    }

    /** Share of evictions made by exact LRU because the guard had taken over. */
    get guardedRate() {
      return this.decisions ? this.guarded / this.decisions : 0.0;
    }

    residentPages() {
      return Array.from(this.table.slotOf.keys());
    }
  }

  // ============================================================ factory

  const POLICY_CLASSES = { FIFO: FIFO, LRU: LRU, LFU: LFU, CLOCK: CLOCK, ARC: ARC, LeCaR: LeCaR, Learned: Learned, OPT: OPT };
  const POLICY_NAMES = ["FIFO", "LRU", "LFU", "CLOCK", "ARC", "LeCaR", "Learned", "OPT"];

  function resolveName(name) {
    const wanted = String(name).trim().toLowerCase();
    for (let i = 0; i < POLICY_NAMES.length; i++) {
      if (POLICY_NAMES[i].toLowerCase() === wanted) return POLICY_NAMES[i];
    }
    throw new Error("unknown policy " + JSON.stringify(name) + "; choose from " + POLICY_NAMES.join(", "));
  }

  /**
   * Build a policy by name (case-insensitive).
   * opts: trace (required for OPT), model, k, threshold, ghost, guard and
   * seed (Learned; model defaults to MLPR_DATA.model and the others to the
   * model's meta), seed and learningRate (LeCaR).
   */
  function makePolicy(name, frames, opts) {
    return new POLICY_CLASSES[resolveName(name)](frames, opts || {});
  }

  // ========================================================= simulation

  // How much time the timing in simulate measures at least, and its limits.
  const TIMING_MS = 10;
  const CHUNK_MS = 2;
  const TIMING_MAX_REFS = 4000000;
  const TIMING_MAX_RUNS = 1000;

  /** Whether fresh copies of policy can be made (it is one of this file's policies). */
  function canCopy(policy) {
    return Boolean(policy.options) && POLICY_CLASSES[policy.name] === policy.constructor;
  }

  /** A fresh policy of the same kind and options as policy, with the given frames. */
  function freshCopy(policy, frames) {
    return new policy.constructor(frames, policy.options);
  }

  /** Mean of the middle 60% of the values (the fastest and slowest fifth dropped). */
  function trimmedMean(values) {
    const sorted = values.slice().sort(function (a, b) { return a - b; });
    const cut = Math.floor(sorted.length / 5);
    const kept = sorted.slice(cut, sorted.length - cut);
    return kept.reduce(function (a, b) { return a + b; }, 0) / kept.length;
  }

  /**
   * Run time of a short trace: fresh copies, built beforehand, run in batches
   * of about 20000 references per timer reading until TIMING_MS have been
   * measured; total and runs are what was measured already.
   */
  function shortRunMs(trace, policy, total, runs) {
    const n = trace.length;
    const batch = Math.max(1, Math.min(TIMING_MAX_RUNS, Math.ceil(20000 / Math.max(n, 1))));
    while (total < TIMING_MS && runs * n < TIMING_MAX_REFS && runs < TIMING_MAX_RUNS) {
      const copies = [];
      for (let b = 0; b < batch; b++) copies.push(freshCopy(policy, policy.frames));
      const start = now();
      for (let b = 0; b < batch; b++) {
        const copy = copies[b];
        for (let t = 0; t < n; t++) copy.access(trace[t], t);
      }
      total += now() - start;
      runs += batch;
    }
    return total / runs;
  }

  /**
   * Timing estimates for simulate, from fresh copies of policy (same kind,
   * frames and options, so the same decisions as a fresh policy):
   *
   * runMs: a copy repeats the run, timed, as a second sample. The two agree
   * unless a garbage collection or JIT pause hit one, so their mean is
   * taken, or the smaller one if they differ by more than 25%. A run shorter
   * than TIMING_MS is repeated in batches instead and averaged.
   *
   * usPerHit: the copy now holds exactly the policy's final state. The run's
   * hits whose page is still resident are replayed on it, each at its
   * original time, cycling: a hit never evicts, so every replayed reference
   * is a hit and the resident set stays the same, while the popularity skew
   * of the real hits (and, for OPT, their real next-use times) is kept. If
   * no hit is on a page that is still resident, the resident pages are
   * replayed round-robin at times after the end of the trace. The replay is
   * timed in chunks of at least CHUNK_MS until TIMING_MS have passed, and
   * the cost per hit is the trimmed mean over the chunks.
   */
  function estimateCosts(trace, policy, firstMs, wantHits) {
    const n = trace.length;
    const copy = freshCopy(policy, policy.frames);
    const hit = new Uint8Array(n);
    const start = now();
    for (let t = 0; t < n; t++) {
      if (copy.access(trace[t], t)) hit[t] = 1;
    }
    const copyMs = now() - start;
    let runMs = Math.max(firstMs, copyMs) <= 1.25 * Math.min(firstMs, copyMs) ? (firstMs + copyMs) / 2 : Math.min(firstMs, copyMs);
    if (firstMs + copyMs < TIMING_MS) runMs = shortRunMs(trace, policy, firstMs + copyMs, 2);
    if (!wantHits) return { runMs: runMs, usPerHit: 0 };

    const resident = copy.residentPages();
    const residentSet = new Set(resident);
    let pages = [];
    let times = [];
    for (let t = 0; t < n; t++) {
      if (hit[t] && residentSet.has(trace[t])) {
        pages.push(trace[t]);
        times.push(t);
      }
    }
    const roundRobin = pages.length === 0;
    if (roundRobin) {
      pages = resident;
      times = resident.map(function (_, i) { return n + i; });
    }
    const m = pages.length;
    if (m === 0) return { runMs: runMs, usPerHit: 0 };
    const rates = [];
    let chunk = 2048;
    let spent = 0;
    let refs = 0;
    let i = 0;
    let shift = 0;
    while ((spent < TIMING_MS || rates.length < 5) && refs < TIMING_MAX_REFS) {
      const begin = now();
      for (let c = 0; c < chunk; c++) {
        copy.access(pages[i], times[i] + shift);
        if (++i === m) {
          i = 0;
          if (roundRobin) shift += m; // keep round-robin times moving forward
        }
      }
      const ms = now() - begin;
      spent += ms;
      refs += chunk;
      if (ms >= CHUNK_MS) rates.push((ms * 1000) / chunk);
      else chunk = Math.min(TIMING_MAX_REFS / 4, Math.ceil(chunk * (ms > 0 ? (1.5 * CHUNK_MS) / ms : 8)));
    }
    return { runMs: runMs, usPerHit: rates.length ? trimmedMean(rates) : (spent * 1000) / refs };
  }

  /**
   * Replay trace through policy and count faults.
   *
   * Timing (timed: true, the default). Browsers clamp performance.now() to
   * 5-100 us (1 ms in some), far coarser than one access, so accesses are
   * not timed one by one as in the Python simulator. Instead the whole run
   * is timed, a fresh copy of the policy repeats it as a second sample and
   * then replays the run's own hits to measure the cost of one hit, and
   *     usPerFault = (runMs - hits x usPerHit) / faults, floored at 0
   * (see estimateCosts). Both costs include the loop overhead, as Python's
   * per-access timer did. The policy passed in (which should be fresh) sees
   * the trace once; the copies are separate objects. Timing costs about one
   * more pass and 10 ms per call. For an object not built by makePolicy (no
   * copies possible) usPerHit is 0 and usPerFault is the whole run divided
   * by the faults, an upper bound.
   * With timed: false only the faults are counted.
   */
  function simulate(trace, policy, opts) {
    const timed = !opts || opts.timed === undefined ? true : Boolean(opts.timed);
    const n = trace.length;
    let faults = 0;
    const start = timed ? now() : 0;
    for (let t = 0; t < n; t++) {
      if (!policy.access(trace[t], t)) faults++;
    }
    let runMs = timed ? now() - start : 0;
    const extra = {};
    if ("fallbackRate" in policy) extra.fallbackRate = policy.fallbackRate;
    if ("guardedRate" in policy) extra.guardedRate = policy.guardedRate;
    const hits = n - faults;
    let usPerHit = 0;
    let usPerFault = 0;
    if (timed && n > 0) {
      if (canCopy(policy)) {
        const est = estimateCosts(trace, policy, runMs, hits > 0 && faults > 0);
        runMs = est.runMs;
        usPerHit = est.usPerHit;
      }
      usPerFault = faults ? Math.max(0, runMs * 1000 - hits * usPerHit) / faults : 0;
    }
    const faultRate = n ? faults / n : 0.0;
    return {
      policy: policy.name,
      frames: policy.frames,
      accesses: n,
      faults: faults,
      hits: hits,
      faultRate: faultRate,
      hitRatio: 1.0 - faultRate,
      usPerFault: usPerFault,
      usPerHit: usPerHit,
      runMs: runMs,
      extra: extra,
    };
  }

  /**
   * Step-by-step frame table for small examples, as drawn in OS lab sheets:
   * one entry per reference {page, hit, frames, victim}, where frames lists
   * the page in each frame slot (null when free). A newly loaded page takes
   * the slot of the page it evicted, else the first free slot. The victim is
   * found from the resident set before and after the access, so this works
   * for every policy without knowing its internals.
   */
  function traceFrames(trace, policy) {
    const slots = new Array(policy.frames).fill(null);
    const slotOf = new Map();
    policy.residentPages().forEach(function (page, i) {
      slots[i] = page;
      slotOf.set(page, i);
    });
    const steps = [];
    for (let t = 0; t < trace.length; t++) {
      const page = trace[t];
      const hit = policy.access(page, t);
      let victim = null;
      if (!hit) {
        const resident = new Set(policy.residentPages());
        for (const entry of slotOf) {
          if (!resident.has(entry[0])) {
            victim = entry[0];
            break;
          }
        }
        let slot;
        if (victim !== null) {
          slot = slotOf.get(victim);
          slotOf.delete(victim);
        } else {
          slot = slots.indexOf(null);
          if (slot < 0) throw new Error("traceFrames: " + policy.name + " loaded a page without evicting one");
        }
        slots[slot] = page;
        slotOf.set(page, slot);
      }
      steps.push({ page: page, hit: hit, frames: slots.slice(), victim: victim });
    }
    return steps;
  }

  // ============================================================ metrics

  /** Fault reduction vs LRU, in percent: (F_LRU - F_policy) / F_LRU x 100. */
  function faultReduction(fLru, f) {
    return fLru ? (100.0 * (fLru - f)) / fLru : 0.0;
  }

  /**
   * Share of the LRU-to-OPT gap closed, in percent:
   * (F_LRU - F_policy) / (F_LRU - F_OPT) x 100. NaN when LRU is already
   * optimal on the trace (there is no gap to close).
   */
  function gapClosed(fLru, f, fOpt) {
    const gap = fLru - fOpt;
    return gap > 0 ? (100.0 * (fLru - f)) / gap : NaN;
  }

  // ============================================================= traces

  // The classic reference string from OS textbooks. With 3 frames:
  // FIFO gives 15 faults, LRU 12 and OPT 9.
  const TEXTBOOK = [7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2, 1, 2, 0, 1, 7, 0, 1];

  function zipfProbs(n, alpha) {
    const p = new Float64Array(n);
    let sum = 0;
    for (let i = 0; i < n; i++) {
      p[i] = Math.pow(i + 1, -alpha);
      sum += p[i];
    }
    for (let i = 0; i < n; i++) p[i] /= sum;
    return p;
  }

  function rngFrom(o) {
    return o.rng || new Rng(given(o.seed) ? o.seed : 0);
  }

  function opt(o, key, fallback) {
    return given(o[key]) ? o[key] : fallback;
  }

  /**
   * A loop over more pages than fit in memory: LRU's worst case.
   * noise is the fraction of references replaced by a random page from the loop.
   */
  function loopTrace(o) {
    o = o || {};
    const nPages = opt(o, "nPages", 2000), length = opt(o, "length", 100000), noise = opt(o, "noise", 0.0);
    const rng = rngFrom(o);
    const trace = new Array(length);
    for (let i = 0; i < length; i++) trace[i] = i % nPages;
    if (noise > 0) {
      const u = rng.random(length);
      const hit = [];
      for (let i = 0; i < length; i++) if (u[i] < noise) hit.push(i);
      const pages = rng.integers(0, nPages, hit.length);
      for (let i = 0; i < hit.length; i++) trace[hit[i]] = pages[i];
    }
    return trace;
  }

  /** Independent references with skewed (Zipf) popularity: frequency matters more than recency. */
  function zipfTrace(o) {
    o = o || {};
    const nPages = opt(o, "nPages", 5000), length = opt(o, "length", 100000), alpha = opt(o, "alpha", 1.0);
    const rng = rngFrom(o);
    const pages = rng.permutation(nPages);
    const idx = rng.choice(nPages, length, zipfProbs(nPages, alpha));
    return idx.map(function (i) { return pages[i]; });
  }

  /** The working set moves: each phase draws (Zipf) from its own random subset of pages. */
  function phaseTrace(o) {
    o = o || {};
    const nPages = opt(o, "nPages", 4000), length = opt(o, "length", 100000);
    const nPhases = opt(o, "nPhases", 6), workingSet = opt(o, "workingSet", 500), alpha = opt(o, "alpha", 0.7);
    const rng = rngFrom(o);
    const probs = zipfProbs(workingSet, alpha);
    const out = [];
    const perPhase = Math.floor(length / nPhases);
    for (let phase = 0; phase < nPhases; phase++) {
      const n = phase < nPhases - 1 ? perPhase : length - perPhase * (nPhases - 1);
      const ws = rng.choiceNoReplace(nPages, workingSet);
      const idx = rng.choice(workingSet, n, probs);
      for (let i = 0; i < n; i++) out.push(ws[idx[i]]);
    }
    return out;
  }

  /**
   * A hot set with long sequential scans mixed in; each scan flushes the hot
   * set under LRU. Pages 0..hotPages-1 are the hot set; scans read scanLen
   * consecutive pages of a large region, starting at a random offset and
   * wrapping around its end.
   */
  function scanHotTrace(o) {
    o = o || {};
    const hotPages = opt(o, "hotPages", 500), scanPages = opt(o, "scanPages", 6000), length = opt(o, "length", 100000);
    const hotBurst = opt(o, "hotBurst", 2000), scanLen = opt(o, "scanLen", 1000), alpha = opt(o, "alpha", 0.8);
    const rng = rngFrom(o);
    const probs = zipfProbs(hotPages, alpha);
    const out = [];
    let size = 0;
    while (size < length) {
      const hot = rng.choice(hotPages, hotBurst, probs);
      const start = rng.integers(0, scanPages);
      for (let i = 0; i < hot.length; i++) out.push(hot[i]);
      for (let i = 0; i < scanLen; i++) out.push(hotPages + ((start + i) % scanPages));
      size += hotBurst + scanLen;
    }
    out.length = Math.min(out.length, length);
    return out;
  }

  /** Two programs sharing memory: one loops over an array, the other has Zipf popularity. */
  function mixedTrace(o) {
    o = o || {};
    const loopPages = opt(o, "loopPages", 1200), zipfPages = opt(o, "zipfPages", 3000), length = opt(o, "length", 100000);
    const alpha = opt(o, "alpha", 0.9), loopShare = opt(o, "loopShare", 0.5);
    const rng = rngFrom(o);
    const u = rng.random(length);
    const trace = new Array(length);
    let nLoop = 0;
    for (let i = 0; i < length; i++) {
      if (u[i] < loopShare) trace[i] = nLoop++ % loopPages;
    }
    const zipfIds = rng.permutation(zipfPages);
    const idx = rng.choice(zipfPages, length - nLoop, zipfProbs(zipfPages, alpha));
    let j = 0;
    for (let i = 0; i < length; i++) {
      if (!(u[i] < loopShare)) trace[i] = zipfIds[idx[j++]] + loopPages;
    }
    return trace;
  }

  /** Base address of each array when placed back to back on page boundaries. */
  function layout(sizes, pageBytes) {
    const bases = [];
    let addr = 0;
    for (let i = 0; i < sizes.length; i++) {
      bases.push(addr);
      addr += Math.ceil(sizes[i] / pageBytes) * pageBytes;
    }
    return bases;
  }

  /** Turns addresses into page numbers, collapsing consecutive references to the same page. */
  class PageWriter {
    constructor(pageBytes) {
      this.pageBytes = pageBytes;
      this.pages = [];
      this._last = NaN;
    }

    add(addr) {
      const page = Math.floor(addr / this.pageBytes);
      if (page !== this._last) {
        this.pages.push(page);
        this._last = page;
      }
    }
  }

  /**
   * Naive i-j-k multiplication of two n x n row-major double matrices,
   * C = A x B. A is read along rows (good locality), B down columns (a new
   * page almost every step) and C written once per element.
   */
  function matmulTrace(o) {
    o = typeof o === "number" ? { n: o } : o || {};
    const n = opt(o, "n", 40), pageBytes = opt(o, "pageBytes", 64);
    const elem = 8;
    const bases = layout([n * n * elem, n * n * elem, n * n * elem], pageBytes);
    const aBase = bases[0], bBase = bases[1], cBase = bases[2];
    const w = new PageWriter(pageBytes);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        for (let k = 0; k < n; k++) {
          w.add(aBase + (i * n + k) * elem);
          w.add(bBase + (k * n + j) * elem);
        }
        w.add(cBase + (i * n + j) * elem);
      }
    }
    return w.pages;
  }

  /** Bottom-up merge sort of n random 4-byte integers, ping-ponging between two buffers. */
  function mergesortTrace(o) {
    o = o || {};
    const n = opt(o, "n", 8192), pageBytes = opt(o, "pageBytes", 64);
    const rng = rngFrom(o);
    const elem = 4;
    const bases = layout([n * elem, n * elem], pageBytes);
    let srcBase = bases[0], dstBase = bases[1];
    let src = rng.permutation(n);
    let dst = new Array(n).fill(0);
    const w = new PageWriter(pageBytes);
    let width = 1;
    while (width < n) {
      for (let lo = 0; lo < n; lo += 2 * width) {
        const mid = Math.min(lo + width, n), hi = Math.min(lo + 2 * width, n);
        let i = lo, j = mid;
        for (let k = lo; k < hi; k++) {
          if (i < mid && (j >= hi || src[i] <= src[j])) {
            w.add(srcBase + i * elem);
            dst[k] = src[i];
            i++;
          } else {
            w.add(srcBase + j * elem);
            dst[k] = src[j];
            j++;
          }
          w.add(dstBase + k * elem);
        }
      }
      const tmp = src; src = dst; dst = tmp;
      const tmpBase = srcBase; srcBase = dstBase; dstBase = tmpBase;
      width *= 2;
    }
    return w.pages;
  }

  /**
   * Breadth-first search over a random graph in CSR form, repeated from
   * several sources. localShare of each node's edges go to nearby node ids
   * (within 20, wrapping around), the rest to random nodes. The visited
   * array is cleared before every run.
   */
  function bfsTrace(o) {
    o = o || {};
    const n = opt(o, "nNodes", 6000), degree = opt(o, "degree", 5), runs = opt(o, "runs", 2);
    const localShare = opt(o, "localShare", 0.5), pageBytes = opt(o, "pageBytes", 64);
    const rng = rngFrom(o);
    const m = n * degree;
    const local = rng.random(m);
    const nearOffset = rng.integers(-20, 21, m);
    const far = rng.integers(0, n, m);
    const adj = new Array(m);
    for (let u = 0; u < n; u++) {
      for (let d = 0; d < degree; d++) {
        const e = u * degree + d;
        adj[e] = local[e] < localShare ? (((u + nearOffset[e]) % n) + n) % n : far[e];
      }
    }
    const bases = layout([(n + 1) * 4, m * 4, n, n * 4], pageBytes);
    const offBase = bases[0], adjBase = bases[1], visBase = bases[2], queueBase = bases[3];
    const w = new PageWriter(pageBytes);
    for (let r = 0; r < runs; r++) {
      for (let a = visBase; a < visBase + n; a += pageBytes) w.add(a); // clear the visited array
      const visited = new Uint8Array(n);
      const source = rng.integers(0, n);
      visited[source] = 1;
      const queue = [source];
      w.add(visBase + source);
      w.add(queueBase);
      let head = 0;
      while (head < queue.length) {
        const u = queue[head];
        w.add(queueBase + 4 * head);
        w.add(offBase + 4 * u);
        w.add(offBase + 4 * (u + 1));
        head++;
        for (let e = u * degree; e < (u + 1) * degree; e++) {
          const v = adj[e];
          w.add(adjBase + 4 * e);
          w.add(visBase + v);
          if (!visited[v]) {
            visited[v] = 1;
            w.add(queueBase + 4 * queue.length);
            queue.push(v);
          }
        }
      }
    }
    return w.pages;
  }

  // ========================================================== workloads

  const MEMORY_FRACTIONS = [0.01, 0.02, 0.05, 0.10];
  const MIN_FRAMES = 8;
  const TEST_SEED = 300;

  function workload(name, kind, tests, params, make) {
    return {
      name: name,
      kind: kind,
      tests: tests,
      params: params,
      /**
       * build({length = 100000, seed = 300, truncate = true}) -> page numbers.
       * length sets the length of synthetic traces; program traces are cut
       * to length when longer (pass truncate: false for the full trace, as
       * in the Python test split). seed is the split's base seed: each family
       * adds its own offset (+0 .. +6) as mlpr/workloads.py does.
       */
      build: function (o) {
        o = o || {};
        const length = given(o.length) ? o.length : 100000;
        const seed = given(o.seed) ? o.seed : TEST_SEED;
        if (!Number.isInteger(length) || length < 1) throw new Error("length must be a positive integer, got " + o.length);
        const trace = make(length, seed);
        if (kind === "program" && o.truncate !== false && trace.length > length) trace.length = length;
        return trace;
      },
    };
  }

  // The test split of mlpr/workloads.py: same families and parameters.
  const WORKLOADS = [
    workload("loop", "synthetic", "loop larger than memory (LRU's worst case)", "2000 pages, 2% noise",
      function (length, s) { return loopTrace({ nPages: 2000, length: length, noise: 0.02, seed: s }); }),
    workload("zipf", "synthetic", "skewed popularity (frequency beats recency)", "5000 pages, alpha 1.0",
      function (length, s) { return zipfTrace({ nPages: 5000, length: length, alpha: 1.0, seed: s + 1 }); }),
    workload("phase", "synthetic", "working set shifts over time", "4000 pages, 6 phases, working set 500, alpha 0.7",
      function (length, s) { return phaseTrace({ nPages: 4000, length: length, nPhases: 6, workingSet: 500, alpha: 0.7, seed: s + 2 }); }),
    workload("scan_hot", "synthetic", "one-time scans flushing a hot set", "hot set 500, scan region 6000, bursts of 2000, scans of 1000, alpha 0.8",
      function (length, s) { return scanHotTrace({ hotPages: 500, scanPages: 6000, length: length, hotBurst: 2000, scanLen: 1000, alpha: 0.8, seed: s + 3 }); }),
    workload("mixed", "synthetic", "a loop and a Zipf program sharing memory", "loop of 1200 pages + Zipf over 3000 pages (alpha 0.9), half each",
      function (length, s) { return mixedTrace({ loopPages: 1200, zipfPages: 3000, length: length, alpha: 0.9, loopShare: 0.5, seed: s + 4 }); }),
    workload("matmul", "program", "naive matrix multiply", "40 x 40 doubles, 64-byte pages",
      function () { return matmulTrace({ n: 40, pageBytes: 64 }); }),
    workload("mergesort", "program", "bottom-up merge sort", "8192 integers, 64-byte pages",
      function (length, s) { return mergesortTrace({ n: 8192, pageBytes: 64, seed: s + 5 }); }),
    workload("bfs", "program", "breadth-first search on a graph", "6000 nodes, degree 5, 2 runs, half the edges local, 64-byte pages",
      function (length, s) { return bfsTrace({ nNodes: 6000, degree: 5, runs: 2, localShare: 0.5, pageBytes: 64, seed: s + 6 }); }),
  ];

  function distinctCount(trace) {
    return new Set(trace).size;
  }

  /** Memory size as a fraction of the trace's distinct pages, with a floor. */
  function framesFor(trace, fraction, minimum) {
    minimum = given(minimum) ? minimum : MIN_FRAMES;
    return Math.max(minimum, roundHalfEven(fraction * distinctCount(trace)));
  }

  /** {fraction, frames} pairs, dropping fractions that collapse onto the same frame count. */
  function memorySizes(trace, fractions) {
    fractions = fractions || MEMORY_FRACTIONS;
    const distinct = distinctCount(trace);
    const seen = new Set();
    const sizes = [];
    for (let i = 0; i < fractions.length; i++) {
      const frames = Math.max(MIN_FRAMES, roundHalfEven(fractions[i] * distinct));
      if (!seen.has(frames)) {
        seen.add(frames);
        sizes.push({ fraction: fractions[i], frames: frames });
      }
    }
    return sizes;
  }

  return {
    VERSION: VERSION,
    POLICY_NAMES: POLICY_NAMES,
    TEXTBOOK: TEXTBOOK,
    FEATURE_NAMES: FEATURE_NAMES,
    RAW_FIELDS: RAW_FIELDS,
    CAP: CAP,
    MEMORY_FRACTIONS: MEMORY_FRACTIONS,
    MIN_FRAMES: MIN_FRAMES,
    WORKLOADS: WORKLOADS,
    makePolicy: makePolicy,
    resolveName: resolveName,
    simulate: simulate,
    traceFrames: traceFrames,
    computeFeatures: computeFeatures,
    scoreModel: scoreModel,
    nextUse: nextUse,
    framesFor: framesFor,
    memorySizes: memorySizes,
    faultReduction: faultReduction,
    gapClosed: gapClosed,
    data: data,
    traces: {
      loop: loopTrace,
      zipf: zipfTrace,
      phase: phaseTrace,
      scanHot: scanHotTrace,
      mixed: mixedTrace,
      matmul: matmulTrace,
      mergesort: mergesortTrace,
      bfs: bfsTrace,
    },
    policies: POLICY_CLASSES,
    PyRandom: PyRandom,
    Rng: Rng,
    MetadataTable: MetadataTable,
    SlotSampler: SlotSampler,
    RegretGuard: RegretGuard,
  };
});
