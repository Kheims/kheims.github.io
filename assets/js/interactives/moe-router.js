(function () {
  const root = document.querySelector("[data-moe-router]");
  if (!root) return;

  const canvas = root.querySelector("canvas");
  const ctx = canvas.getContext("2d");
  const controls = {
    tokens: root.querySelector('[data-control="tokens"]'),
    experts: root.querySelector('[data-control="experts"]'),
    topK: root.querySelector('[data-control="topK"]'),
    capacity: root.querySelector('[data-control="capacity"]'),
    balance: root.querySelector('[data-control="balance"]')
  };
  const outputs = {
    tokens: root.querySelector('[data-output="tokens"]'),
    experts: root.querySelector('[data-output="experts"]'),
    capacity: root.querySelector('[data-output="capacity"]'),
    balance: root.querySelector('[data-output="balance"]')
  };
  const stats = {
    routed: root.querySelector('[data-stat="routed"]'),
    dropped: root.querySelector('[data-stat="dropped"]'),
    capacity: root.querySelector('[data-stat="capacity"]'),
    imbalance: root.querySelector('[data-stat="imbalance"]')
  };
  const toggle = root.querySelector('[data-action="toggle"]');
  const reroll = root.querySelector('[data-action="reroll"]');

  const palette = {
    ink: "#1a1a1a",
    muted: "#6b6a66",
    rule: "#ded8cc",
    paper: "#fbfaf7",
    panel: "#f4f1ea",
    router: "#8b3a3a",
    token: "#2f5d7c",
    token2: "#5e7a3a",
    drop: "#a23b32",
    expert: ["#7a3f8f", "#2f6f89", "#7b8b3a", "#b06a2d", "#a23b32", "#437c58", "#6b5aa6", "#8c6f3f", "#3f7a75", "#9a4f75"]
  };

  let seed = 17;
  let running = !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  let t0 = performance.now();
  let state = null;
  let layout = null;

  function random(index) {
    let x = Math.sin(index * 12.9898 + seed * 78.233) * 43758.5453;
    return x - Math.floor(x);
  }

  function readConfig() {
    return {
      tokens: Number(controls.tokens.value),
      experts: Number(controls.experts.value),
      topK: Number(controls.topK.value),
      capacityFactor: Number(controls.capacity.value),
      balance: Number(controls.balance.value)
    };
  }

  function updateOutputs(config) {
    outputs.tokens.value = String(config.tokens);
    outputs.experts.value = String(config.experts);
    outputs.capacity.value = config.capacityFactor.toFixed(2);
    outputs.balance.value = config.balance.toFixed(2);
  }

  function buildState() {
    const config = readConfig();
    updateOutputs(config);

    const capacity = Math.max(1, Math.floor((config.tokens * config.topK / config.experts) * config.capacityFactor));
    const loads = Array(config.experts).fill(0);
    const assignments = [];

    for (let i = 0; i < config.tokens; i += 1) {
      const scores = [];
      const preferredA = Math.floor(random(i + 100) * config.experts);
      const preferredB = Math.floor(random(i + 200) * config.experts);

      for (let e = 0; e < config.experts; e += 1) {
        const locality = e === preferredA ? 0.78 : e === preferredB ? 0.42 : 0;
        const noise = random(i * 31 + e * 19);
        const imbalancePenalty = config.balance * (loads[e] / capacity) * 0.9;
        scores.push({
          expert: e,
          score: noise * 0.55 + locality - imbalancePenalty
        });
      }

      scores.sort((a, b) => b.score - a.score);

      for (let k = 0; k < config.topK; k += 1) {
        const expert = scores[k].expert;
        const dropped = loads[expert] >= capacity;
        const slot = dropped ? capacity : loads[expert]++;
        assignments.push({
          token: i,
          expert,
          rank: k,
          slot,
          dropped,
          jitter: random(i * 101 + k * 13)
        });
      }
    }

    const expected = config.tokens * config.topK / config.experts;
    const variance = loads.reduce((sum, load) => sum + Math.pow(load - expected, 2), 0) / config.experts;
    const imbalance = Math.sqrt(variance) / Math.max(1, expected);

    stats.routed.textContent = String(assignments.filter((a) => !a.dropped).length);
    stats.dropped.textContent = String(assignments.filter((a) => a.dropped).length);
    stats.capacity.textContent = String(capacity);
    stats.imbalance.textContent = imbalance.toFixed(2);

    state = { config, capacity, loads, assignments };
    t0 = performance.now();
    resize();
  }

  function resize() {
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(320, Math.floor(rect.width));
    const height = width < 560 ? 520 : 560;
    canvas.style.height = `${height}px`;
    canvas.width = Math.floor(width * dpr);
    canvas.height = Math.floor(height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    layout = makeLayout(width, height);
    draw(performance.now());
  }

  function makeLayout(width, height) {
    const left = width < 560 ? 38 : 56;
    const routerX = width < 560 ? width * 0.43 : width * 0.42;
    const expertX = width < 560 ? width - 72 : width - 108;
    const top = 78;
    const bottom = height - 72;
    const laneHeight = (bottom - top) / Math.max(1, state.config.experts);
    return { width, height, left, routerX, expertX, top, bottom, laneHeight };
  }

  function ease(x) {
    return x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2;
  }

  function lerp(a, b, p) {
    return a + (b - a) * p;
  }

  function tokenStart(token) {
    const rows = Math.ceil(Math.sqrt(state.config.tokens));
    const col = token % rows;
    const row = Math.floor(token / rows);
    const spreadX = layout.width < 560 ? 42 : 70;
    const spreadY = layout.bottom - layout.top - 30;
    return {
      x: layout.left + (col / Math.max(1, rows - 1)) * spreadX,
      y: layout.top + 16 + (row / Math.max(1, Math.ceil(state.config.tokens / rows) - 1)) * spreadY
    };
  }

  function routerPoint(token) {
    return {
      x: layout.routerX + (random(token + 4) - 0.5) * 20,
      y: layout.height * 0.5 + (random(token + 8) - 0.5) * 84
    };
  }

  function expertSlot(assignment) {
    const centerY = layout.top + layout.laneHeight * (assignment.expert + 0.5);
    const stacked = Math.max(1, state.capacity);
    const offset = ((assignment.slot % stacked) / stacked - 0.5) * Math.max(12, layout.laneHeight - 18);
    const rankOffset = assignment.rank === 0 ? -5 : 5;
    const overflowX = Math.min(layout.expertX + 46, layout.width - 34);
    return {
      x: assignment.dropped ? overflowX : layout.expertX + rankOffset,
      y: assignment.dropped ? layout.bottom + 24 : centerY + offset
    };
  }

  function positionAt(assignment, progress) {
    const start = tokenStart(assignment.token);
    const mid = routerPoint(assignment.token);
    const end = expertSlot(assignment);
    const local = (progress + assignment.token * 0.011 + assignment.rank * 0.035) % 1;

    if (local < 0.44) {
      const p = ease(local / 0.44);
      return { x: lerp(start.x, mid.x, p), y: lerp(start.y, mid.y, p), phase: "router", alpha: 0.55 + p * 0.35 };
    }

    const p = ease((local - 0.44) / 0.56);
    return { x: lerp(mid.x, end.x, p), y: lerp(mid.y, end.y, p), phase: "expert", alpha: 0.92 };
  }

  function drawLabel(text, x, y, align) {
    ctx.font = "12px Inter, system-ui, sans-serif";
    ctx.fillStyle = palette.muted;
    ctx.textAlign = align || "left";
    ctx.fillText(text, x, y);
  }

  function drawCurve(from, to, color, alpha) {
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    const c1x = lerp(from.x, to.x, 0.36);
    const c2x = lerp(from.x, to.x, 0.72);
    ctx.moveTo(from.x, from.y);
    ctx.bezierCurveTo(c1x, from.y, c2x, to.y, to.x, to.y);
    ctx.stroke();
    ctx.restore();
  }

  function draw(time) {
    if (!state || !layout) return;
    const elapsed = running ? (time - t0) / 4600 : 0;
    const progress = running ? elapsed % 1 : 0.66;

    ctx.clearRect(0, 0, layout.width, layout.height);
    ctx.fillStyle = palette.paper;
    ctx.fillRect(0, 0, layout.width, layout.height);

    drawLabel("tokens", layout.left, 38);
    drawLabel("router", layout.routerX, 38, "center");
    drawLabel("experts", layout.expertX, 38, "center");

    ctx.strokeStyle = palette.rule;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(layout.left + 100, 44);
    ctx.lineTo(layout.routerX - 42, 44);
    ctx.moveTo(layout.routerX + 42, 44);
    ctx.lineTo(layout.expertX - 64, 44);
    ctx.stroke();

    ctx.save();
    ctx.translate(layout.routerX, layout.height * 0.5);
    ctx.fillStyle = "rgba(139,58,58,0.10)";
    ctx.strokeStyle = "rgba(139,58,58,0.60)";
    ctx.lineWidth = 1.4;
    roundedRect(ctx, -36, -50, 72, 100, 8);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = palette.router;
    ctx.font = "600 12px Inter, system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("top-k", 0, -6);
    ctx.fillText("gate", 0, 12);
    ctx.restore();

    for (let e = 0; e < state.config.experts; e += 1) {
      const y = layout.top + layout.laneHeight * e + 4;
      const h = Math.max(18, layout.laneHeight - 8);
      const load = state.loads[e];
      const saturation = Math.min(1, load / state.capacity);
      ctx.fillStyle = "rgba(244,241,234,0.85)";
      ctx.strokeStyle = load >= state.capacity ? "rgba(162,59,50,0.80)" : "rgba(222,216,204,1)";
      roundedRect(ctx, layout.expertX - 44, y, 88, h, 6);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = palette.expert[e % palette.expert.length];
      ctx.globalAlpha = 0.22 + saturation * 0.34;
      roundedRect(ctx, layout.expertX - 40, y + 4, 80 * saturation, h - 8, 5);
      ctx.fill();
      ctx.globalAlpha = 1;
      ctx.fillStyle = palette.ink;
      ctx.font = "600 11px Inter, system-ui, sans-serif";
      ctx.textAlign = "left";
      ctx.fillText(`E${e}`, layout.expertX - 34, y + h / 2 + 4);
      ctx.fillStyle = palette.muted;
      ctx.textAlign = "right";
      ctx.fillText(`${load}/${state.capacity}`, layout.expertX + 35, y + h / 2 + 4);
    }

    const sample = state.assignments.filter((_, index) => index % Math.ceil(state.assignments.length / 34) === 0);
    sample.forEach((assignment) => {
      const start = tokenStart(assignment.token);
      const end = expertSlot(assignment);
      drawCurve(start, end, assignment.dropped ? palette.drop : palette.expert[assignment.expert % palette.expert.length], assignment.dropped ? 0.18 : 0.11);
    });

    state.assignments.forEach((assignment) => {
      const pos = positionAt(assignment, progress);
      const color = assignment.dropped
        ? palette.drop
        : assignment.rank === 0
          ? palette.token
          : palette.token2;

      ctx.save();
      ctx.globalAlpha = pos.alpha;
      ctx.fillStyle = color;
      ctx.strokeStyle = "rgba(255,255,255,0.72)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, assignment.rank === 0 ? 4.2 : 3.3, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.restore();
    });

    const dropped = state.assignments.filter((a) => a.dropped).length;
    if (dropped > 0) {
      const overflowWidth = Math.min(96, layout.width - 24);
      const overflowX = Math.min(layout.expertX + 12, layout.width - overflowWidth - 12);
      ctx.fillStyle = "rgba(162,59,50,0.10)";
      ctx.strokeStyle = "rgba(162,59,50,0.45)";
      roundedRect(ctx, overflowX, layout.bottom + 8, overflowWidth, 34, 6);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = palette.drop;
      ctx.font = "600 11px Inter, system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(`${dropped} overflow`, overflowX + overflowWidth / 2, layout.bottom + 30);
    }

    if (running) requestAnimationFrame(draw);
  }

  function roundedRect(context, x, y, width, height, radius) {
    const r = Math.min(radius, width / 2, height / 2);
    context.beginPath();
    context.moveTo(x + r, y);
    context.arcTo(x + width, y, x + width, y + height, r);
    context.arcTo(x + width, y + height, x, y + height, r);
    context.arcTo(x, y + height, x, y, r);
    context.arcTo(x, y, x + width, y, r);
    context.closePath();
  }

  function restartAnimation() {
    t0 = performance.now();
    if (running) requestAnimationFrame(draw);
    else draw(performance.now());
  }

  Object.values(controls).forEach((control) => {
    control.addEventListener("input", buildState);
    control.addEventListener("change", buildState);
  });

  toggle.addEventListener("click", () => {
    running = !running;
    toggle.textContent = running ? "Pause" : "Play";
    toggle.setAttribute("aria-pressed", String(running));
    restartAnimation();
  });

  reroll.addEventListener("click", () => {
    seed += 1;
    buildState();
  });

  window.addEventListener("resize", resize);

  buildState();
  restartAnimation();
})();
