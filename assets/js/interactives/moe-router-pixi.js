(function () {
  const root = document.querySelector("[data-moe-pixi]");
  if (!root) return;

  const stageEl = root.querySelector("[data-stage]");
  const controls = {
    tokens: root.querySelector('[data-control="tokens"]'),
    experts: root.querySelector('[data-control="experts"]'),
    topK: root.querySelector('[data-control="topK"]'),
    capacity: root.querySelector('[data-control="capacity"]'),
    collapse: root.querySelector('[data-control="collapse"]'),
    balance: root.querySelector('[data-control="balance"]')
  };
  const outputs = {
    tokens: root.querySelector('[data-output="tokens"]'),
    experts: root.querySelector('[data-output="experts"]'),
    capacity: root.querySelector('[data-output="capacity"]'),
    collapse: root.querySelector('[data-output="collapse"]'),
    balance: root.querySelector('[data-output="balance"]')
  };
  const stats = {
    routed: root.querySelector('[data-stat="routed"]'),
    dropped: root.querySelector('[data-stat="dropped"]'),
    capacity: root.querySelector('[data-stat="capacity"]'),
    hotExpert: root.querySelector('[data-stat="hotExpert"]')
  };
  const toggle = root.querySelector('[data-action="toggle"]');
  const reroll = root.querySelector('[data-action="reroll"]');

  const colors = {
    bg: 0xfbfaf7,
    panel: 0xf4f1ea,
    ink: 0x1a1a1a,
    muted: 0x6b6a66,
    rule: 0xe8e1d5,
    gate: 0x9b3430,
    blue: 0x23627f,
    green: 0x5b7a42,
    red: 0xa23b32,
    expert: [0x7a4e9e, 0x2e718c, 0x7d8f42, 0xc57a39, 0xb34a44, 0x3f7f5f, 0x675ab0, 0x9a567f, 0x3e817c, 0xa08342, 0x4f6da8, 0xa85835]
  };

  const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  let running = !prefersReducedMotion;
  let seed = 43;
  let app;
  let layers;
  let state;
  let layout;
  let elapsed = 0;

  function rand(n) {
    const x = Math.sin(n * 12.9898 + seed * 78.233) * 43758.5453123;
    return x - Math.floor(x);
  }

  function config() {
    return {
      tokens: Number(controls.tokens.value),
      experts: Number(controls.experts.value),
      topK: Number(controls.topK.value),
      capacityFactor: Number(controls.capacity.value),
      collapse: Number(controls.collapse.value),
      balance: Number(controls.balance.value)
    };
  }

  function updateOutputs(c) {
    outputs.tokens.value = String(c.tokens);
    outputs.experts.value = String(c.experts);
    outputs.capacity.value = c.capacityFactor.toFixed(2);
    outputs.collapse.value = c.collapse.toFixed(2);
    outputs.balance.value = c.balance.toFixed(2);
  }

  function buildState() {
    const c = config();
    updateOutputs(c);

    const capacity = Math.max(1, Math.floor((c.tokens * c.topK / c.experts) * c.capacityFactor));
    const loads = Array(c.experts).fill(0);
    const assignments = [];
    const hotBias = Math.max(0, Math.min(c.experts - 1, Math.floor(rand(900) * c.experts)));

    for (let token = 0; token < c.tokens; token += 1) {
      const naturalA = Math.floor(rand(token + 11) * c.experts);
      const naturalB = Math.floor(rand(token + 37) * c.experts);
      const scores = [];

      for (let expert = 0; expert < c.experts; expert += 1) {
        const semantic = expert === naturalA ? 0.56 : expert === naturalB ? 0.32 : 0;
        const collapse = expert === hotBias ? c.collapse * 0.72 : c.collapse * rand(expert + 301) * 0.16;
        const pressure = c.balance * (loads[expert] / capacity) * 0.86;
        scores.push({
          expert,
          score: semantic + collapse + rand(token * 71 + expert * 17) * 0.48 - pressure
        });
      }

      scores.sort((a, b) => b.score - a.score);

      for (let rank = 0; rank < c.topK; rank += 1) {
        const expert = scores[rank].expert;
        const dropped = loads[expert] >= capacity;
        const slot = dropped ? capacity : loads[expert]++;
        assignments.push({ token, expert, rank, slot, dropped });
      }
    }

    const routed = assignments.filter((a) => !a.dropped).length;
    const dropped = assignments.length - routed;
    const hotLoad = Math.max(...loads);
    const hotExpert = loads.indexOf(hotLoad);

    stats.routed.textContent = String(routed);
    stats.dropped.textContent = String(dropped);
    stats.capacity.textContent = String(capacity);
    stats.hotExpert.textContent = `E${hotExpert}`;

    state = { c, capacity, loads, assignments, hotExpert };
    elapsed = 0;
    resize();
  }

  function makeLayout(width, height) {
    const isNarrow = width < 620;
    const top = isNarrow ? 42 : 52;
    const bottom = height - (isNarrow ? 42 : 46);
    return {
      width,
      height,
      isNarrow,
      tokenX: isNarrow ? 44 : 72,
      gateX: width * (isNarrow ? 0.44 : 0.42),
      expertX: width - (isNarrow ? 70 : 122),
      top,
      bottom,
      lane: (bottom - top) / state.c.experts
    };
  }

  function clearLayer(layer) {
    layer.removeChildren();
  }

  function addText(layer, text, x, y, style, anchorX, anchorY) {
    const node = new PIXI.Text(text, new PIXI.TextStyle(style));
    node.anchor.set(anchorX == null ? 0 : anchorX, anchorY == null ? 0 : anchorY);
    node.x = x;
    node.y = y;
    layer.addChild(node);
    return node;
  }

  function drawStatic() {
    if (!app || !state || !layout) return;
    Object.values(layers).forEach(clearLayer);

    const bg = new PIXI.Graphics();
    bg.beginFill(colors.bg);
    bg.drawRect(0, 0, layout.width, layout.height);
    bg.endFill();

    bg.beginFill(colors.panel, 0.9);
    bg.lineStyle(1, colors.rule, 1);
    bg.drawRoundedRect(16, 16, layout.width - 32, layout.height - 32, 14);
    bg.endFill();

    bg.lineStyle(1, colors.rule, 1);
    bg.moveTo(layout.tokenX + 54, layout.top - 12);
    bg.lineTo(layout.gateX - 62, layout.top - 12);
    bg.moveTo(layout.gateX + 62, layout.top - 12);
    bg.lineTo(layout.expertX - 60, layout.top - 12);
    layers.background.addChild(bg);

    addText(layers.labels, "incoming tokens", layout.isNarrow ? 18 : layout.tokenX, 24, labelStyle(11), layout.isNarrow ? 0 : 0.5, 0);
    addText(layers.labels, "router logits", layout.gateX, 24, labelStyle(11), 0.5, 0);
    addText(layers.labels, "expert buffers", layout.expertX, 24, labelStyle(11), 0.5, 0);

    const gate = new PIXI.Graphics();
    gate.lineStyle(1.5, colors.gate, 0.55);
    gate.beginFill(colors.gate, 0.08);
    gate.drawRoundedRect(layout.gateX - 46, layout.height * 0.5 - 58, 92, 116, 16);
    gate.endFill();
    gate.lineStyle(1, colors.gate, 0.25);
    for (let i = 0; i < 5; i += 1) {
      gate.moveTo(layout.gateX - 28, layout.height * 0.5 - 34 + i * 17);
      gate.lineTo(layout.gateX + 28, layout.height * 0.5 - 34 + i * 17);
    }
    layers.background.addChild(gate);
    addText(layers.labels, "gate", layout.gateX, layout.height * 0.5 - 9, {
      fill: colors.gate,
      fontFamily: "Inter, system-ui, sans-serif",
      fontSize: 14,
      fontWeight: "700"
    }, 0.5, 0.5);
    addText(layers.labels, `top-${state.c.topK}`, layout.gateX, layout.height * 0.5 + 13, {
      fill: colors.gate,
      fontFamily: "JetBrains Mono, monospace",
      fontSize: 11,
      fontWeight: "500"
    }, 0.5, 0.5);

    for (let expert = 0; expert < state.c.experts; expert += 1) {
      const y = layout.top + expert * layout.lane + 4;
      const h = Math.max(26, layout.lane - 8);
      const color = colors.expert[expert % colors.expert.length];
      const full = state.loads[expert] >= state.capacity;
      const hot = expert === state.hotExpert;

      const card = new PIXI.Graphics();
      card.lineStyle(hot ? 2.5 : 1.4, full ? colors.red : color, full ? 0.85 : 0.55);
      card.beginFill(0xffffff, 0.52);
      card.drawRoundedRect(layout.expertX - 50, y, 100, h, 10);
      card.endFill();
      card.beginFill(color, 0.23 + Math.min(0.38, state.loads[expert] / Math.max(1, state.capacity) * 0.32));
      card.drawRoundedRect(layout.expertX - 45, y + 5, 90, h - 10, 8);
      card.endFill();
      layers.experts.addChild(card);

      addText(layers.labels, `E${expert}`, layout.expertX - 38, y + h / 2, {
        fill: colors.ink,
        fontFamily: "JetBrains Mono, monospace",
        fontSize: 12,
        fontWeight: "700"
      }, 0, 0.5);
      addText(layers.labels, `${state.loads[expert]}/${state.capacity}`, layout.expertX + 36, y + h / 2, {
        fill: full ? colors.red : colors.muted,
        fontFamily: "JetBrains Mono, monospace",
        fontSize: 11,
        fontWeight: "600"
      }, 1, 0.5);
    }

    const dropped = state.assignments.filter((a) => a.dropped).length;
    if (dropped > 0) {
      const x = Math.min(layout.expertX - 2, layout.width - 114);
      const y = Math.min(layout.bottom + 8, layout.height - 34);
      const badge = new PIXI.Graphics();
      badge.lineStyle(1.2, colors.red, 0.65);
      badge.beginFill(colors.red, 0.09);
      badge.drawRoundedRect(x, y, 96, 28, 8);
      badge.endFill();
      layers.experts.addChild(badge);
      addText(layers.labels, `${dropped} overflow`, x + 48, y + 14, {
        fill: colors.red,
        fontFamily: "Inter, system-ui, sans-serif",
        fontSize: 11,
        fontWeight: "700"
      }, 0.5, 0.5);
    }
  }

  function labelStyle(size) {
    return {
      fill: colors.muted,
      fontFamily: "Inter, system-ui, sans-serif",
      fontSize: size,
      fontWeight: "500"
    };
  }

  function tokenStart(token) {
    const rows = layout.isNarrow ? 10 : 12;
    const row = token % rows;
    const col = Math.floor(token / rows);
    const x = layout.tokenX - 28 + col * (layout.isNarrow ? 4 : 7);
    const y = layout.top + 18 + row * ((layout.bottom - layout.top - 36) / Math.max(1, rows - 1));
    return { x, y };
  }

  function gatePoint(token, rank) {
    return {
      x: layout.gateX + (rand(token * 9 + rank * 23) - 0.5) * 58,
      y: layout.height * 0.5 + (rand(token * 13 + rank * 41) - 0.5) * 88
    };
  }

  function expertPoint(a) {
    const y0 = layout.top + a.expert * layout.lane + layout.lane / 2;
    const capacity = Math.max(1, state.capacity);
    const slotOffset = ((a.slot % capacity) / capacity - 0.5) * Math.max(8, layout.lane - 20);
    const x = a.dropped ? Math.min(layout.expertX + 42, layout.width - 34) : layout.expertX - 6 + a.rank * 12;
    const y = a.dropped ? layout.bottom + 2 : y0 + slotOffset;
    return { x, y };
  }

  function ease(p) {
    return p < 0.5 ? 4 * p * p * p : 1 - Math.pow(-2 * p + 2, 3) / 2;
  }

  function lerp(a, b, p) {
    return a + (b - a) * p;
  }

  function bezier(p0, p1, p2, p3, t) {
    const a = Math.pow(1 - t, 3);
    const b = 3 * Math.pow(1 - t, 2) * t;
    const c = 3 * (1 - t) * t * t;
    const d = t * t * t;
    return {
      x: a * p0.x + b * p1.x + c * p2.x + d * p3.x,
      y: a * p0.y + b * p1.y + c * p2.y + d * p3.y
    };
  }

  function drawCurve(g, from, to, color, alpha, width) {
    const c1 = { x: lerp(from.x, to.x, 0.36), y: from.y };
    const c2 = { x: lerp(from.x, to.x, 0.72), y: to.y };
    g.lineStyle(width, color, alpha);
    g.moveTo(from.x, from.y);
    g.bezierCurveTo(c1.x, c1.y, c2.x, c2.y, to.x, to.y);
  }

  function drawFrame(delta) {
    if (!state || !layout) return;
    if (running) elapsed += delta / 60;
    clearLayer(layers.routes);
    clearLayer(layers.tokens);

    const routeG = new PIXI.Graphics();
    const tokenG = new PIXI.Graphics();
    const phase = (elapsed / 5.2) % 1;

    state.assignments.forEach((a, index) => {
      if (index % Math.max(1, Math.floor(state.assignments.length / 56)) === 0) {
        const from = tokenStart(a.token);
        const to = expertPoint(a);
        drawCurve(routeG, from, to, a.dropped ? colors.red : colors.expert[a.expert % colors.expert.length], a.dropped ? 0.18 : 0.12, a.dropped ? 1.3 : 1);
      }
    });

    state.assignments.forEach((a, index) => {
      const offset = (a.token * 0.009 + a.rank * 0.04) % 1;
      const p = (phase + offset) % 1;
      const start = tokenStart(a.token);
      const gate = gatePoint(a.token, a.rank);
      const end = expertPoint(a);
      let pos;
      let alpha = 0.9;

      if (p < 0.44) {
        const q = ease(p / 0.44);
        pos = bezier(start, { x: lerp(start.x, gate.x, 0.35), y: start.y }, { x: lerp(start.x, gate.x, 0.72), y: gate.y }, gate, q);
        alpha = 0.55 + q * 0.35;
      } else {
        const q = ease((p - 0.44) / 0.56);
        pos = bezier(gate, { x: lerp(gate.x, end.x, 0.28), y: gate.y }, { x: lerp(gate.x, end.x, 0.74), y: end.y }, end, q);
      }

      const color = a.dropped ? colors.red : a.rank === 0 ? colors.blue : colors.green;
      tokenG.beginFill(color, alpha);
      tokenG.lineStyle(1, 0xffffff, 0.75);
      tokenG.drawCircle(pos.x, pos.y, a.rank === 0 ? 4.4 : 3.5);
      tokenG.endFill();
    });

    layers.routes.addChild(routeG);
    layers.tokens.addChild(tokenG);
  }

  function resize() {
    if (!app || !state) return;
    const width = Math.max(320, Math.floor(stageEl.clientWidth));
    const height = width < 620 ? 580 : 520;
    app.renderer.resize(width, height);
    app.view.style.width = `${width}px`;
    app.view.style.height = `${height}px`;
    layout = makeLayout(width, height);
    drawStatic();
    drawFrame(0);
  }

  function setupPixi() {
    if (!window.PIXI) {
      stageEl.innerHTML = "<p class=\"moe-pixi__fallback\">PixiJS did not load. Check your network connection or use the Canvas version above.</p>";
      return;
    }

    app = new PIXI.Application({
      width: Math.max(320, stageEl.clientWidth),
      height: 520,
      backgroundAlpha: 0,
      antialias: true,
      autoDensity: true,
      resolution: Math.min(window.devicePixelRatio || 1, 2)
    });

    stageEl.appendChild(app.view);
    layers = {
      background: new PIXI.Container(),
      routes: new PIXI.Container(),
      experts: new PIXI.Container(),
      labels: new PIXI.Container(),
      tokens: new PIXI.Container()
    };
    Object.values(layers).forEach((layer) => app.stage.addChild(layer));
    app.ticker.add(drawFrame);

    buildState();
  }

  Object.values(controls).forEach((control) => {
    control.addEventListener("input", buildState);
    control.addEventListener("change", buildState);
  });

  toggle.addEventListener("click", () => {
    running = !running;
    toggle.textContent = running ? "Pause" : "Play";
  });

  reroll.addEventListener("click", () => {
    seed += 1;
    buildState();
  });

  window.addEventListener("resize", resize);

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", setupPixi);
  } else {
    setupPixi();
  }
})();
