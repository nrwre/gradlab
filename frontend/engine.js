// In-browser fallback engine so the static site works with NO backend.
// Mirrors the Python nanograd optimisers (Experiment 1) and provides a light
// classifier for offline mode. When API_BASE is set, app.js calls the real
// Python engine instead and these are unused.
(function (global) {
  // ---- surfaces ----
  function quadratic(kappa) {
    return {
      value: (x) => 0.5 * (x[0] * x[0] + kappa * x[1] * x[1]),
      grad: (x) => [x[0], kappa * x[1]],
    };
  }
  function rosenbrock(a = 1, b = 100) {
    return {
      value: (x) => Math.pow(a - x[0], 2) + b * Math.pow(x[1] - x[0] * x[0], 2),
      grad: (x) => [
        -2 * (a - x[0]) - 4 * b * x[0] * (x[1] - x[0] * x[0]),
        2 * b * (x[1] - x[0] * x[0]),
      ],
    };
  }

  // ---- optimisers ----
  function optimise(surface, opt, lr, steps, start) {
    let x = start.slice();
    let v = [0, 0], m = [0, 0], vv = [0, 0], t = 0;
    const b1 = 0.9, b2 = 0.999, eps = 1e-8, beta = 0.9;
    const traj = [x.slice()];
    for (let i = 0; i < steps; i++) {
      let g;
      if (opt === "nesterov") {
        const look = [x[0] + beta * v[0], x[1] + beta * v[1]];
        g = surface.grad(look);
      } else g = surface.grad(x);
      if (opt === "sgd") {
        x = [x[0] - lr * g[0], x[1] - lr * g[1]];
      } else if (opt === "momentum" || opt === "nesterov") {
        v = [beta * v[0] - lr * g[0], beta * v[1] - lr * g[1]];
        x = [x[0] + v[0], x[1] + v[1]];
      } else { // adam
        t++;
        m = [b1 * m[0] + (1 - b1) * g[0], b1 * m[1] + (1 - b1) * g[1]];
        vv = [b2 * vv[0] + (1 - b2) * g[0] * g[0], b2 * vv[1] + (1 - b2) * g[1] * g[1]];
        const mh = [m[0] / (1 - b1 ** t), m[1] / (1 - b1 ** t)];
        const vh = [vv[0] / (1 - b2 ** t), vv[1] / (1 - b2 ** t)];
        x = [x[0] - lr * mh[0] / (Math.sqrt(vh[0]) + eps),
             x[1] - lr * mh[1] / (Math.sqrt(vh[1]) + eps)];
      }
      if (!isFinite(x[0]) || Math.abs(x[0]) > 1e6) break;
      traj.push(x.slice());
    }
    return traj;
  }

  // ---- toy datasets (match Python seeds loosely) ----
  function randn(rng) { // Box-Muller
    let u = 0, v = 0; while (u === 0) u = rng(); while (v === 0) v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  function mulberry(seed) { return function () {
    seed |= 0; seed = seed + 0x6D2B79F5 | 0;
    let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }

  function dataset(name, noise, n = 220, seed = 0) {
    const rng = mulberry(seed), X = [], y = [], h = n / 2;
    for (let i = 0; i < h; i++) {
      const t = Math.PI * i / h;
      if (name === "moons") {
        X.push([Math.cos(t) + noise * randn(rng), Math.sin(t) + noise * randn(rng)]); y.push(0);
        X.push([1 - Math.cos(t) + noise * randn(rng), 0.5 - Math.sin(t) + noise * randn(rng)]); y.push(1);
      } else if (name === "circles") {
        const a = 2 * Math.PI * i / h;
        X.push([Math.cos(a) + noise * randn(rng), Math.sin(a) + noise * randn(rng)]); y.push(0);
        X.push([0.4 * Math.cos(a) + noise * randn(rng), 0.4 * Math.sin(a) + noise * randn(rng)]); y.push(1);
      } else { // blobs
        X.push([-2 + 0.8 * randn(rng), -2 + 0.8 * randn(rng)]); y.push(0);
        X.push([2 + 0.8 * randn(rng), 2 + 0.8 * randn(rng)]); y.push(1);
      }
    }
    return { X, y };
  }

  // ---- offline classifier: RBF-weighted kNN (smooth boundary, no training loop) ----
  function classifyGrid(X, y, model) {
    const gridN = 50;
    let minx = Infinity, maxx = -Infinity, miny = Infinity, maxy = -Infinity;
    X.forEach(p => { minx = Math.min(minx, p[0]); maxx = Math.max(maxx, p[0]);
                     miny = Math.min(miny, p[1]); maxy = Math.max(maxy, p[1]); });
    minx -= 1; maxx += 1; miny -= 1; maxy += 1;
    const gx = [], gy = [];
    for (let i = 0; i < gridN; i++) { gx.push(minx + (maxx - minx) * i / (gridN - 1));
                                      gy.push(miny + (maxy - miny) * i / (gridN - 1)); }
    const gamma = model === "svm-linear" ? 0.0 : 1.5;
    const z = [];
    let correct = 0;
    const pred = (px, py) => {
      let s0 = 0, s1 = 0;
      for (let k = 0; k < X.length; k++) {
        const d2 = (px - X[k][0]) ** 2 + (py - X[k][1]) ** 2;
        const w = gamma > 0 ? Math.exp(-gamma * d2) : 1 / (d2 + 1e-6);
        if (y[k] === 0) s0 += w; else s1 += w;
      }
      return s1 > s0 ? 1 : 0;
    };
    for (let j = 0; j < gridN; j++) { const row = [];
      for (let i = 0; i < gridN; i++) row.push(pred(gx[i], gy[j])); z.push(row); }
    for (let k = 0; k < X.length; k++) if (pred(X[k][0], X[k][1]) === y[k]) correct++;
    return { grid_x: gx, grid_y: gy, grid_z: z, points: X, labels: y,
             accuracy: correct / X.length };
  }

  global.GradLabEngine = { quadratic, rosenbrock, optimise, dataset, classifyGrid };
})(window);
