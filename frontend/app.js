// UI wiring: reads controls, runs either the local engine (offline) or the
// deployed Python backend, and renders with Plotly.
(function () {
  const API = (window.GRADLAB_CONFIG && window.GRADLAB_CONFIG.API_BASE) || "";
  const E = window.GradLabEngine;
  const $ = (id) => document.getElementById(id);
  const online = API !== "";

  const layout = (title) => ({
    title: { text: title, font: { color: "#e8edf7", size: 14 } },
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "#0c1426",
    font: { color: "#94a3c4" }, margin: { l: 40, r: 16, t: 36, b: 36 },
    showlegend: false,
  });
  const CONF = { displayModeBar: false, responsive: true };

  async function api(path, body) {
    const r = await fetch(API.replace(/\/$/, "") + path, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r.ok) throw new Error("backend " + r.status);
    return r.json();
  }

  // ---------- tabs ----------
  document.querySelectorAll(".tab").forEach(t => t.onclick = () => {
    document.querySelectorAll(".tab").forEach(x => x.classList.remove("active"));
    t.classList.add("active");
    $("view-optimize").style.display = t.dataset.tab === "optimize" ? "" : "none";
    $("view-classify").style.display = t.dataset.tab === "classify" ? "" : "none";
  });
  const modeNote = online ? "Using the live Python engine." : "Running in-browser (offline demo). Set API_BASE in config.js to use the Python engine.";
  $("o-mode").textContent = modeNote;
  $("c-mode").textContent = modeNote;

  // ---------- optimiser ----------
  const lrMap = (v) => +(v / 100).toFixed(2);   // slider 1..60 -> 0.01..0.60
  $("o-kappa").oninput = e => $("o-kappa-v").textContent = e.target.value;
  $("o-lr").oninput = e => $("o-lr-v").textContent = lrMap(e.target.value);
  $("o-steps").oninput = e => $("o-steps-v").textContent = e.target.value;
  $("o-surface").onchange = e =>
    $("o-kappa-row").style.display = e.target.value === "quadratic" ? "" : "none";

  async function runOptimize() {
    const surface = $("o-surface").value;
    const kappa = +$("o-kappa").value;
    const opt = $("o-opt").value;
    const lr = lrMap(+$("o-lr").value);
    const steps = +$("o-steps").value;
    const start = surface === "quadratic" ? [5, 5] : [-1.5, 2];

    let traj, losses;
    if (online) {
      const r = await api("/optimize", { surface, kappa, optimizer: opt, lr, steps, start });
      traj = r.trajectory; losses = r.losses;
    } else {
      const s = surface === "quadratic" ? E.quadratic(kappa) : E.rosenbrock();
      traj = E.optimise(s, opt, lr, steps, start);
      losses = traj.map(s2 => (surface === "quadratic" ? E.quadratic(kappa) : E.rosenbrock()).value(s2));
    }

    // contour
    const s = surface === "quadratic" ? E.quadratic(kappa) : E.rosenbrock();
    const N = 60, span = surface === "quadratic" ? 6 : 2.2;
    const cx = [], cy = [], cz = [];
    for (let i = 0; i < N; i++) { cx.push(-span + 2 * span * i / (N - 1)); cy.push(-span + 2 * span * i / (N - 1)); }
    for (let j = 0; j < N; j++) { const row = [];
      for (let i = 0; i < N; i++) row.push(Math.log1p(s.value([cx[i], cy[j]]))); cz.push(row); }

    Plotly.react("o-plot", [
      { type: "contour", x: cx, y: cy, z: cz, showscale: false,
        colorscale: "Blues", contours: { coloring: "fill" }, opacity: 0.85 },
      { type: "scatter", mode: "lines+markers", x: traj.map(p => p[0]), y: traj.map(p => p[1]),
        line: { color: "#f5a623", width: 2 }, marker: { size: 5, color: "#f5a623" } },
      { type: "scatter", mode: "markers", x: [traj[traj.length - 1][0]], y: [traj[traj.length - 1][1]],
        marker: { size: 12, color: "#3ddc84", symbol: "star" } },
    ], layout(`${opt} on ${surface}`), CONF);

    $("o-metrics").innerHTML =
      `<span class="metric">start loss <b>${losses[0].toFixed(3)}</b></span>` +
      `<span class="metric">final loss <b>${losses[losses.length - 1].toExponential(2)}</b></span>` +
      `<span class="metric">steps <b>${traj.length - 1}</b></span>`;

    explain($("o-tutor"), {
      kind: "optimize", optimizer: opt, surface, kappa,
      start_loss: losses[0], final_loss: losses[losses.length - 1], steps: traj.length - 1,
    });
  }
  $("o-run").onclick = () => runOptimize().catch(err => alert(err.message));

  // ---------- classifier ----------
  $("c-noise").oninput = e => $("c-noise-v").textContent = (e.target.value / 100).toFixed(2);

  async function runClassify() {
    const dataset = $("c-data").value;
    const modelSel = $("c-model").value;
    const noise = +$("c-noise").value / 100;
    let r;
    if (online) {
      const model = modelSel === "svm-linear" ? "svm" : modelSel;
      const kernel = modelSel === "svm-linear" ? "linear" : "rbf";
      r = await api("/classify", { dataset, model, kernel, noise });
    } else {
      const d = E.dataset(dataset, noise);
      r = E.classifyGrid(d.X, d.y, modelSel);
    }
    const pts0 = r.points.filter((_, i) => r.labels[i] === 0);
    const pts1 = r.points.filter((_, i) => r.labels[i] === 1);
    Plotly.react("c-plot", [
      { type: "contour", x: r.grid_x, y: r.grid_y, z: r.grid_z, showscale: false,
        colorscale: [[0, "#16335e"], [1, "#5a3a13"]], contours: { coloring: "fill" }, opacity: 0.7 },
      { type: "scatter", mode: "markers", x: pts0.map(p => p[0]), y: pts0.map(p => p[1]),
        marker: { color: "#5b9cff", size: 7, line: { color: "#0c1426", width: 1 } } },
      { type: "scatter", mode: "markers", x: pts1.map(p => p[0]), y: pts1.map(p => p[1]),
        marker: { color: "#f5a623", size: 7, line: { color: "#0c1426", width: 1 } } },
    ], layout(`${modelSel} on ${dataset}`), CONF);

    let m = `<span class="metric">accuracy <b>${(r.accuracy * 100).toFixed(1)}%</b></span>`;
    if (r.oob_score != null) m += `<span class="metric">OOB <b>${(r.oob_score * 100).toFixed(1)}%</b></span>`;
    $("c-metrics").innerHTML = m;

    explain($("c-tutor"), {
      kind: "classify", model: modelSel, dataset, accuracy: r.accuracy,
      kernel: modelSel === "svm-linear" ? "linear" : "rbf",
    });
  }
  $("c-run").onclick = () => runClassify().catch(err => alert(err.message));

  // ---------- AI tutor ----------
  async function explain(el, run) {
    el.style.display = "";
    el.innerHTML = `<div class="who">AI tutor</div><span class="muted">thinking…</span>`;
    let text;
    try {
      if (online) { text = (await api("/explain", { run })).text; }
      else { text = localExplain(run); }
    } catch (e) { text = localExplain(run); }
    el.innerHTML = `<div class="who">AI tutor</div>${text}`;
  }

  function localExplain(run) {
    if (run.kind === "optimize") {
      let t = `Using <b>${run.optimizer}</b>, the loss fell from ${(+run.start_loss).toPrecision(3)} to ${(+run.final_loss).toExponential(2)} in ${run.steps} steps. `;
      if (run.surface === "quadratic")
        t += run.kappa >= 30
          ? `With a high condition number (κ=${run.kappa}) the bowl is stretched, so plain gradient descent zig-zags; momentum and Adam smooth that out and reach the minimum faster.`
          : `The condition number κ=${run.kappa} is mild, so every method converges quickly and they look similar.`;
      else t += `Rosenbrock's curved valley is hard: notice how the path snakes along the floor of the banana toward (1,1).`;
      return t;
    }
    const linear = run.kernel === "linear";
    return `<b>${run.model}</b> reached ${(run.accuracy * 100).toFixed(1)}% on the ${run.dataset} data. ` +
      (linear ? "A straight line is its only option, so curved data costs accuracy."
              : "A non-linear boundary lets it bend around the classes — watch the coloured region curve.");
  }

  // initial render
  runOptimize();
  runClassify();
})();
