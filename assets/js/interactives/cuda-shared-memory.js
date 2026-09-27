(function () {
  const root = document.querySelector("[data-cuda-smem]");
  if (!root) return;

  const canvas = root.querySelector("canvas");
  const ctx = canvas.getContext("2d");
  const controls = {
    blockSize: root.querySelector('[data-control="blockSize"]'),
    step: root.querySelector('[data-control="step"]'),
    pattern: root.querySelector('[data-control="pattern"]')
  };
  const outputs = {
    step: root.querySelector('[data-output="step"]')
  };
  const stats = {
    active: root.querySelector('[data-stat="active"]'),
    stride: root.querySelector('[data-stat="stride"]'),
    banks: root.querySelector('[data-stat="banks"]'),
    bytes: root.querySelector('[data-stat="bytes"]')
  };
  const playButton = root.querySelector('[data-action="play"]');

  const colors = {
    bg: "#fbfaf7",
    panel: "#f4f1ea",
    rule: "#e8e1d5",
    ink: "#1a1a1a",
    muted: "#6b6a66",
    red: "#9b3430",
    redSoft: "rgba(155,52,48,0.12)",
    blue: "#2f6f89",
    blueSoft: "rgba(47,111,137,0.13)",
    green: "#5e7a3a",
    greenSoft: "rgba(94,122,58,0.14)",
    amber: "#b06a2d",
    purple: "#7457a6"
  };

  const stepNames = [
    "load globals",
    "write shared",
    "__syncthreads",
    "reduce stride",
    "reduce again",
    "write partial"
  ];

  let playing = false;
  let timer = null;
  let layout = null;

  function config() {
    return {
      blockSize: Number(controls.blockSize.value),
      step: Number(controls.step.value),
      pattern: controls.pattern.value
    };
  }

  function resize() {
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(320, Math.floor(rect.width));
    const height = width < 620 ? 760 : 620;
    canvas.style.height = `${height}px`;
    canvas.width = Math.floor(width * dpr);
    canvas.height = Math.floor(height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    layout = makeLayout(width, height);
    draw();
  }

  function makeLayout(width, height) {
    const narrow = width < 620;
    const pad = narrow ? 18 : 28;
    const stageTop = narrow ? 36 : 42;
    const globalY = stageTop + (narrow ? 24 : 18);
    const threadY = narrow ? 190 : 168;
    const sharedY = narrow ? 352 : 304;
    const bankY = narrow ? 570 : 486;
    const outputY = height - 58;
    return {
      width,
      height,
      narrow,
      pad,
      globalY,
      threadY,
      sharedY,
      bankY,
      outputY,
      left: pad,
      right: width - pad
    };
  }

  function updateReadout(c) {
    outputs.step.value = stepNames[c.step];
    const active = activeThreads(c);
    const stride = reductionStride(c);
    const pressure = bankPressure(c);

    stats.active.textContent = String(active);
    stats.stride.textContent = stride ? String(stride) : "-";
    stats.banks.textContent = pressure;
    stats.bytes.textContent = `${c.blockSize * 4} B`;
  }

  function reductionStride(c) {
    if (c.step === 3) return Math.max(1, c.blockSize / 2);
    if (c.step === 4) return Math.max(1, c.blockSize / 4);
    return 0;
  }

  function activeThreads(c) {
    const stride = reductionStride(c);
    if (stride) return stride;
    if (c.step === 5) return 1;
    return c.blockSize;
  }

  function accessIndex(thread, c) {
    if (c.pattern === "broadcast") return 0;
    if (c.pattern === "stride2") return thread * 2;
    if (c.pattern === "stride4") return thread * 4;
    return thread;
  }

  function bankPressure(c) {
    const warp = Math.min(32, c.blockSize);
    const counts = new Map();
    for (let t = 0; t < warp; t += 1) {
      const bank = accessIndex(t, c) % 32;
      counts.set(bank, (counts.get(bank) || 0) + 1);
    }
    if (c.pattern === "broadcast") return "broadcast";
    const worst = Math.max(...counts.values());
    return `${worst}-way`;
  }

  function draw() {
    if (!layout) return;
    const c = config();
    updateReadout(c);
    ctx.clearRect(0, 0, layout.width, layout.height);
    ctx.fillStyle = colors.bg;
    ctx.fillRect(0, 0, layout.width, layout.height);

    drawTitleRow(c);
    drawGlobalMemory(c);
    drawThreads(c);
    drawSharedMemory(c);
    drawReductionArrows(c);
    drawBanks(c);
    drawOutput(c);
  }

  function text(value, x, y, style) {
    ctx.save();
    ctx.font = style.font || "12px Inter, system-ui, sans-serif";
    ctx.fillStyle = style.fill || colors.ink;
    ctx.textAlign = style.align || "left";
    ctx.textBaseline = style.baseline || "alphabetic";
    ctx.fillText(value, x, y);
    ctx.restore();
  }

  function rounded(x, y, w, h, r) {
    const radius = Math.min(r, w / 2, h / 2);
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.arcTo(x + w, y, x + w, y + h, radius);
    ctx.arcTo(x + w, y + h, x, y + h, radius);
    ctx.arcTo(x, y + h, x, y, radius);
    ctx.arcTo(x, y, x + w, y, radius);
    ctx.closePath();
  }

  function drawTitleRow(c) {
    text("one CUDA block", layout.left, 24, {
      font: "600 12px Inter, system-ui, sans-serif",
      fill: colors.muted
    });
    text(`step ${c.step + 1}: ${stepNames[c.step]}`, layout.right, 24, {
      font: "600 12px Inter, system-ui, sans-serif",
      fill: colors.red,
      align: "right"
    });
  }

  function cellMetrics(count, y, preferred) {
    const gap = layout.narrow ? 3 : 5;
    const available = layout.right - layout.left;
    const cell = Math.min(preferred, Math.floor((available - gap * (count - 1)) / count));
    const total = cell * count + gap * (count - 1);
    return {
      x: layout.left + (available - total) / 2,
      y,
      cell,
      gap
    };
  }

  function gridMetrics(count, y, preferred, cellHeight) {
    if (!layout.narrow || count <= 8) {
      const line = cellMetrics(count, y, preferred);
      return {
        x: line.x,
        y,
        cell: line.cell,
        gap: line.gap,
        cols: count,
        rows: 1,
        cellHeight,
        pos(index) {
          return {
            x: line.x + index * (line.cell + line.gap),
            y
          };
        }
      };
    }

    const cols = Math.min(8, count);
    const rows = Math.ceil(count / cols);
    const gap = 4;
    const available = layout.right - layout.left;
    const cell = Math.min(preferred, Math.floor((available - gap * (cols - 1)) / cols));
    const total = cell * cols + gap * (cols - 1);
    const x0 = layout.left + (available - total) / 2;
    return {
      x: x0,
      y,
      cell,
      gap,
      cols,
      rows,
      cellHeight,
      pos(index) {
        const col = index % cols;
        const row = Math.floor(index / cols);
        return {
          x: x0 + col * (cell + gap),
          y: y + row * (cellHeight + gap)
        };
      }
    };
  }

  function drawGlobalMemory(c) {
    const shown = Math.min(c.blockSize * 2, layout.narrow ? 16 : 32);
    const m = gridMetrics(shown, layout.globalY, layout.narrow ? 30 : 22, 28);

    text("global memory: x[i]", layout.left, layout.globalY - 14, {
      font: "600 12px Inter, system-ui, sans-serif",
      fill: colors.muted
    });

    for (let i = 0; i < shown; i += 1) {
      const p = m.pos(i);
      const x = p.x;
      const loaded = c.step >= 0 && i < c.blockSize * 2;
      ctx.fillStyle = loaded ? colors.blueSoft : colors.panel;
      ctx.strokeStyle = colors.rule;
      rounded(x, p.y, m.cell, 28, 5);
      ctx.fill();
      ctx.stroke();
      text(String(i), x + m.cell / 2, p.y + 18, {
          font: "10px JetBrains Mono, monospace",
          fill: colors.muted,
          align: "center"
      });
    }
    if (layout.narrow && c.blockSize * 2 > shown) {
      text(`+${c.blockSize * 2 - shown} more`, layout.right, layout.globalY + m.rows * 32 + 14, {
        font: "10px Inter, system-ui, sans-serif",
        fill: colors.muted,
        align: "right"
      });
    }
  }

  function drawThreads(c) {
    const m = gridMetrics(c.blockSize, layout.threadY, layout.narrow ? 32 : 30, 42);
    text("threads read two values each, then hold a register sum", layout.left, layout.threadY - 16, {
      font: "600 12px Inter, system-ui, sans-serif",
      fill: colors.muted
    });

    for (let t = 0; t < c.blockSize; t += 1) {
      const p = m.pos(t);
      const x = p.x;
      const active = t < activeThreads(c);
      ctx.fillStyle = active ? colors.greenSoft : colors.panel;
      ctx.strokeStyle = active ? colors.green : colors.rule;
      rounded(x, p.y, m.cell, 42, 7);
      ctx.fill();
      ctx.stroke();
      text(`t${t}`, x + m.cell / 2, p.y + 16, {
        font: "10px JetBrains Mono, monospace",
        fill: active ? colors.ink : colors.muted,
        align: "center"
      });
      text(registerLabel(t, c), x + m.cell / 2, p.y + 32, {
        font: "9px JetBrains Mono, monospace",
        fill: colors.muted,
        align: "center"
      });
    }
  }

  function registerLabel(t, c) {
    if (c.step < 1) return `x${t}`;
    if (c.step < 3) return `r${t}`;
    const stride = reductionStride(c);
    if (stride && t < stride) return `+${t + stride}`;
    if (c.step === 5 && t === 0) return "out";
    return "-";
  }

  function drawSharedMemory(c) {
    const m = gridMetrics(c.blockSize, layout.sharedY, layout.narrow ? 32 : 30, 46);
    text("shared memory: sdata[threadIdx.x]", layout.left, layout.sharedY - 16, {
      font: "600 12px Inter, system-ui, sans-serif",
      fill: colors.muted
    });

    for (let i = 0; i < c.blockSize; i += 1) {
      const p = m.pos(i);
      const x = p.x;
      const touched = sharedTouched(i, c);
      const active = i < activeThreads(c);
      ctx.fillStyle = touched ? colors.redSoft : colors.panel;
      ctx.strokeStyle = touched ? colors.red : colors.rule;
      rounded(x, p.y, m.cell, 46, 6);
      ctx.fill();
      ctx.stroke();
      text(`s${i}`, x + m.cell / 2, p.y + 17, {
        font: "10px JetBrains Mono, monospace",
        fill: active ? colors.ink : colors.muted,
        align: "center"
      });
      text(valueLabel(i, c), x + m.cell / 2, p.y + 33, {
        font: "9px JetBrains Mono, monospace",
        fill: colors.muted,
        align: "center"
      });
    }
  }

  function sharedTouched(index, c) {
    if (c.step === 1 || c.step === 2) return true;
    const stride = reductionStride(c);
    if (stride) return index < stride || (index >= stride && index < stride * 2);
    if (c.step === 5) return index === 0;
    return false;
  }

  function valueLabel(index, c) {
    if (c.step < 1) return "";
    if (c.step < 3) return `r${index}`;
    if (c.step === 3) return index < c.blockSize / 2 ? "sum" : "read";
    if (c.step === 4) return index < c.blockSize / 4 ? "sum" : index < c.blockSize / 2 ? "read" : "";
    if (c.step === 5) return index === 0 ? "block" : "";
    return "";
  }

  function drawReductionArrows(c) {
    const stride = reductionStride(c);
    if (!stride) {
      if (c.step === 2) {
        drawBarrier(layout.threadY + 82, "__syncthreads(): every store to sdata is visible before reduction starts");
      }
      return;
    }

    const m = cellMetrics(c.blockSize, layout.sharedY, layout.narrow ? 24 : 30);
    for (let t = 0; t < stride; t += 1) {
      if (layout.narrow) continue;
      const fromX = m.x + (t + stride) * (m.cell + m.gap) + m.cell / 2;
      const toX = m.x + t * (m.cell + m.gap) + m.cell / 2;
      const y = m.y + 58 + (t % 2) * 10;
      drawArrow(fromX, y, toX, y, colors.red, 0.36);
    }
  }

  function drawBarrier(y, label) {
    ctx.strokeStyle = colors.red;
    ctx.setLineDash([7, 6]);
    ctx.beginPath();
    ctx.moveTo(layout.left, y);
    ctx.lineTo(layout.right, y);
    ctx.stroke();
    ctx.setLineDash([]);
    text(label, (layout.left + layout.right) / 2, y + 20, {
      font: "12px Inter, system-ui, sans-serif",
      fill: colors.red,
      align: "center"
    });
  }

  function drawArrow(x1, y1, x2, y2, color, alpha) {
    ctx.save();
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.globalAlpha = alpha;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
    const angle = Math.atan2(y2 - y1, x2 - x1);
    ctx.beginPath();
    ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - Math.cos(angle - 0.45) * 7, y2 - Math.sin(angle - 0.45) * 7);
    ctx.lineTo(x2 - Math.cos(angle + 0.45) * 7, y2 - Math.sin(angle + 0.45) * 7);
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }

  function drawBanks(c) {
    const bankCount = layout.narrow ? 16 : 32;
    const m = gridMetrics(bankCount, layout.bankY, layout.narrow ? 30 : 18, 42);
    const hits = Array(32).fill(0);
    for (let t = 0; t < Math.min(32, c.blockSize); t += 1) {
      hits[accessIndex(t, c) % 32] += 1;
    }

    text("shared memory banks for one warp access", layout.left, layout.bankY - 16, {
      font: "600 12px Inter, system-ui, sans-serif",
      fill: colors.muted
    });

    for (let b = 0; b < bankCount; b += 1) {
      const p = m.pos(b);
      const x = p.x;
      const count = hits[b];
      const hot = count > 1 && c.pattern !== "broadcast";
      ctx.fillStyle = hot ? colors.redSoft : count ? colors.blueSoft : colors.panel;
      ctx.strokeStyle = hot ? colors.red : colors.rule;
      rounded(x, p.y, m.cell, 42, 4);
      ctx.fill();
      ctx.stroke();
      text(String(b), x + m.cell / 2, p.y + 16, {
        font: "9px JetBrains Mono, monospace",
        fill: colors.muted,
        align: "center"
      });
      if (count) {
        text(String(count), x + m.cell / 2, p.y + 32, {
          font: "10px JetBrains Mono, monospace",
          fill: hot ? colors.red : colors.ink,
          align: "center"
        });
      }
    }
  }

  function drawOutput(c) {
    const y = layout.outputY;
    const x = (layout.left + layout.right) / 2 - 82;
    ctx.fillStyle = c.step === 5 ? colors.greenSoft : colors.panel;
    ctx.strokeStyle = c.step === 5 ? colors.green : colors.rule;
    rounded(x, y, 164, 34, 8);
    ctx.fill();
    ctx.stroke();
    text("partial[blockIdx.x] = sdata[0]", x + 82, y + 21, {
      font: "11px JetBrains Mono, monospace",
      fill: c.step === 5 ? colors.ink : colors.muted,
      align: "center"
    });
  }

  function schedulePlay() {
    clearInterval(timer);
    if (!playing) return;
    timer = setInterval(() => {
      const next = (Number(controls.step.value) + 1) % stepNames.length;
      controls.step.value = String(next);
      draw();
    }, 1200);
  }

  Object.values(controls).forEach((control) => {
    control.addEventListener("input", draw);
    control.addEventListener("change", draw);
  });

  playButton.addEventListener("click", () => {
    playing = !playing;
    playButton.textContent = playing ? "Pause" : "Play";
    schedulePlay();
  });

  window.addEventListener("resize", resize);
  resize();
})();
