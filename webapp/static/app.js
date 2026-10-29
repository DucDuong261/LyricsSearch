/* =========================================================
   AI Music Matcher — SPA frontend
   - Router theo hash (#/hybrid, #/lyrics, ...) → hỗ trợ Back/Forward & deep link
   - Mọi thay đổi giao diện đi qua morph() → View Transitions API (0.2–0.45s)
   ========================================================= */
(() => {
  "use strict";

  // ---------------- Icons (Lucide, ISC) ----------------
  const ICONS = {
    sparkles: '<path d="M9.937 15.5A2 2 0 0 0 8.5 14.063l-6.135-1.582a.5.5 0 0 1 0-.962L8.5 9.936A2 2 0 0 0 9.937 8.5l1.582-6.135a.5.5 0 0 1 .963 0L14.063 8.5A2 2 0 0 0 15.5 9.937l6.135 1.581a.5.5 0 0 1 0 .964L15.5 14.063a2 2 0 0 0-1.437 1.437l-1.582 6.135a.5.5 0 0 1-.963 0z"/>',
    mic: '<path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" x2="12" y1="19" y2="22"/>',
    music: '<path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/>',
    pen: '<path d="M12 20h9"/><path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4Z"/>',
    activity: '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
    chart: '<path d="M3 3v18h18"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/>',
    search: '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
    upload: '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" x2="12" y1="3" y2="15"/>',
    rotate: '<path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/>',
    chevron: '<path d="m6 9 6 6 6-6"/>',
    alert: '<circle cx="12" cy="12" r="10"/><line x1="12" x2="12" y1="8" y2="12"/><line x1="12" x2="12.01" y1="16" y2="16"/>',
    info: '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>',
    download: '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" x2="12" y1="15" y2="3"/>',
    volume: '<polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M15.54 8.46a5 5 0 0 1 0 7.07"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14"/>',
    file: '<path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/><polyline points="14 2 14 8 20 8"/><circle cx="10" cy="16" r="2"/><path d="M12 16V11l3 1"/>',
  };
  const FILLED = {
    play: '<path d="M7 4.5v15a1 1 0 0 0 1.5.86l12.5-7.5a1 1 0 0 0 0-1.72L8.5 3.64A1 1 0 0 0 7 4.5z"/>',
    pause: '<rect x="6" y="4" width="4" height="16" rx="1.2"/><rect x="14" y="4" width="4" height="16" rx="1.2"/>',
    stop: '<rect x="6" y="6" width="12" height="12" rx="2.5"/>',
  };
  const icon = (name) => FILLED[name]
    ? `<svg class="i fill" viewBox="0 0 24 24" aria-hidden="true">${FILLED[name]}</svg>`
    : `<svg class="i" viewBox="0 0 24 24" aria-hidden="true">${ICONS[name]}</svg>`;

  // ---------------- Helpers ----------------
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const pct = (v, d = 1) => `${(Number(v || 0) * 100).toFixed(d)}%`;
  const fmtTime = (s) => {
    if (!isFinite(s) || s < 0) s = 0;
    return `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
  };
  const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)");
  const supportsVT = typeof document.startViewTransition === "function";
  if (!supportsVT) document.documentElement.classList.add("no-vt");

  /** Cập nhật DOM kèm hiệu ứng morph (View Transitions) — fallback: animation CSS. */
  function morph(update, kind = "swap") {
    if (!supportsVT || reduceMotion.matches) {
      update();
      return Promise.resolve();
    }
    const root = document.documentElement;
    root.classList.add(`vt-${kind}`);
    const vt = document.startViewTransition(update);
    return vt.finished.catch(() => {}).finally(() => root.classList.remove(`vt-${kind}`));
  }

  function hashCode(str) {
    let h = 0;
    for (const ch of String(str)) h = (h * 31 + ch.codePointAt(0)) | 0;
    return Math.abs(h);
  }
  const initialsOf = (title) =>
    (String(title || "").match(/[\p{L}\p{N}]+/gu) || ["♪"]).slice(0, 2).map((w) => w[0]).join("").toUpperCase();
  /** Ảnh bìa sinh tự động: dải màu xanh lá → xanh ngọc, chữ cái đầu của tên bài. */
  function coverAttrs(title) {
    const h = hashCode(title);
    const hue = 105 + (h % 70);
    const light = 42 + (h % 14);
    const a = `hsl(${hue} 72% ${light}%)`;
    const b = `hsl(${(hue + 40) % 360} 60% 14%)`;
    const initials = initialsOf(title);
    return { style: `--cover-a:${a};--cover-b:${b}`, initials: esc(initials) };
  }
  const cover = (title, cls = "") => {
    const c = coverAttrs(title);
    return `<div class="cover ${cls}" style="${c.style}" aria-hidden="true">${c.initials}</div>`;
  };

  async function api(url, opts = {}) {
    const res = await fetch(url, opts);
    let data;
    try { data = await res.json(); } catch { data = { error: `Lỗi máy chủ (${res.status})` }; }
    if (!res.ok && !data.error) data.error = `Lỗi máy chủ (${res.status})`;
    return data;
  }

  // ---------------- Routes ----------------
  const ROUTES = {
    hybrid: {
      group: "Nhận diện", label: "Nhận diện thông minh", short: "Lời hát + Âm thanh", icon: "sparkles", tint: "#1ed760",
      eyebrow: "Hybrid Fusion", title: "Nhận diện thông minh",
      desc: "Hát vài câu, ngân nga hay bật một đoạn nhạc — hệ thống chạy song song Whisper (lời hát) và CLAP Audio Encoder (âm thanh) rồi dung hợp điểm số để chọn bài chính xác nhất.",
      kind: "audio", cta: "Nhận diện ngay", recHint: "Chạm để bắt đầu hát hoặc ngân nga",
    },
    lyrics: {
      group: "Nhận diện", label: "Tìm qua lời hát", short: "Whisper ASR + SBERT", icon: "mic", tint: "#18a84c",
      eyebrow: "Speech → Lyrics", title: "Tìm qua lời hát",
      desc: "Hát hoặc đọc vài câu bạn nhớ. Whisper chuyển giọng nói thành văn bản, sau đó SBERT + FAISS tìm câu hát gần nghĩa nhất trong kho dữ liệu.",
      kind: "audio", cta: "Xử lý bản thu", recHint: "Chạm để bắt đầu thu lời hát",
    },
    text: {
      group: "Nhận diện", label: "Gõ lời bài hát", short: "Tìm theo ngữ nghĩa", icon: "pen", tint: "#2e9b55",
      eyebrow: "Semantic Search", title: "Bạn nhớ câu hát nào?",
      desc: "Gõ một đoạn lời bất kỳ — không cần chính xác từng chữ. Hệ thống so khớp theo ngữ nghĩa nên vẫn tìm được khi bạn nhớ sai vài từ.",
      kind: "text",
    },
    agent: {
      group: "Kho nhạc", label: "Thu thập nhạc mới", short: "AI Data Agent", icon: "download", tint: "#1a9a63",
      eyebrow: "AI Data Agent", title: "Thu thập nhạc mới",
      desc: "Agent tự quét YouTube Music (BXH, xu hướng hoặc ca sĩ / chủ đề bạn chọn), lọc nhạc Việt, tải 30 giây âm thanh, đồng bộ lời hát (LRCLIB) và cập nhật vector Qdrant + FAISS.",
      kind: "agent",
    },
    analysis: {
      group: "Phòng lab", label: "Phân tích tín hiệu", short: "Spectrogram, MFCC, Chroma", icon: "activity", tint: "#15805a",
      eyebrow: "Signal Lab", title: "Phân tích tín hiệu âm thanh",
      desc: "Trực quan hóa 7 đặc trưng số của một bản thu: dạng sóng, Mel-spectrogram, MFCC, Chromagram, năng lượng RMS, trọng tâm phổ và ZCR.",
      kind: "analysis",
    },
    wer: {
      group: "Phòng lab", label: "Đánh giá WER/CER", short: "Chất lượng Whisper", icon: "chart", tint: "#4f8a1d",
      eyebrow: "Evaluation", title: "Đánh giá WER / CER",
      desc: "Chất lượng nhận diện giọng nói của Whisper trên toàn bộ dataset: tỉ lệ lỗi từ (WER), lỗi ký tự (CER) và phân tích loại lỗi.",
      kind: "wer",
    },
  };
  const DEFAULT_ROUTE = "hybrid";

  // ---------------- App state ----------------
  const state = {
    route: null,
    ready: false,
    minDuration: 5,
    audio: {},          // trạng thái riêng cho từng chế độ audio (giữ lại khi chuyển trang)
    text: { query: "", html: "" },
    agent: { limit: 10, source: "all", query: "", html: "" },
    analysis: { q: "", tracks: null, html: "" },
    wer: null,
  };
  const audioState = (mode) => (state.audio[mode] ||= { tab: "mic", phase: "idle", take: null, html: "" });

  const view = $("#view");
  const nav = $("#nav");

  // ---------------- Nav ----------------
  function renderNav() {
    let html = "", group = null;
    for (const [key, r] of Object.entries(ROUTES)) {
      if (r.group !== group) {
        group = r.group;
        html += `<div class="nav-group">${esc(group)}</div>`;
      }
      html += `<a class="nav-item" href="#/${key}" data-route="${key}" title="${esc(r.label)}">
        <span class="nav-ico">${icon(r.icon)}</span>
        <span class="nav-label"><b>${esc(r.label)}</b><small>${esc(r.short)}</small></span>
      </a>`;
    }
    nav.innerHTML = html;
  }
  function updateNav(route) {
    $(".nav-pill", nav)?.remove();
    for (const a of $$(".nav-item", nav)) {
      const active = a.dataset.route === route;
      if (active) {
        a.setAttribute("aria-current", "page");
        a.insertAdjacentHTML("afterbegin", '<span class="nav-pill" aria-hidden="true"></span>');
      } else {
        a.removeAttribute("aria-current");
      }
    }
  }

  // ---------------- Router ----------------
  function currentRoute() {
    const key = location.hash.replace(/^#\/?/, "");
    return ROUTES[key] ? key : DEFAULT_ROUTE;
  }
  function navigate() {
    const route = currentRoute();
    if (route === state.route) return;
    if (recorder.active) recorder.stop(true);
    morph(() => {
      state.route = route;
      document.documentElement.style.setProperty("--tint", ROUTES[route].tint);
      updateNav(route);
      renderView(route);
      $("#main").scrollTop = 0;
      if (!supportsVT) {
        view.classList.remove("enter");
        void view.offsetWidth;
        view.classList.add("enter");
      }
    }, "page");
    document.title = `${ROUTES[route].label} · AI Music Matcher`;
  }

  function heroHTML(r) {
    return `<header class="hero">
      <div class="hero-eyebrow">${icon(r.icon)}<span>${esc(r.eyebrow)}</span></div>
      <h1>${esc(r.title)}</h1>
      <p>${esc(r.desc)}</p>
    </header>`;
  }

  function renderView(route) {
    const r = ROUTES[route];
    const body = { audio: audioViewHTML, text: textViewHTML, agent: agentViewHTML, analysis: analysisViewHTML, wer: werViewHTML }[r.kind](route);
    view.innerHTML = heroHTML(r) + `<div class="content">${body}</div>`;
    ({ audio: bindAudioView, text: bindTextView, agent: bindAgentView, analysis: bindAnalysisView, wer: bindWerView })[r.kind](route);
  }

  // =========================================================
  // AUDIO VIEWS (Hybrid / Lyrics / Melody)
  // =========================================================
  const RING_C = 2 * Math.PI * 78;

  function audioViewHTML(mode) {
    const s = audioState(mode);
    return `<section class="card capture" aria-label="Đầu vào âm thanh">
      <div class="capture-head">
        <div class="segmented" role="tablist" aria-label="Nguồn âm thanh" style="--seg-count:2;--seg-index:${s.tab === "mic" ? 0 : 1}">
          <button role="tab" aria-selected="${s.tab === "mic"}" data-tab="mic">${icon("mic")}Thu âm</button>
          <button role="tab" aria-selected="${s.tab === "file"}" data-tab="file">${icon("upload")}Tải file</button>
        </div>
      </div>
      <div class="stage" id="stage">${stageHTML(mode)}</div>
    </section>
    <div class="slot" id="slot" aria-live="polite">${s.html}</div>`;
  }

  function stageHTML(mode) {
    const s = audioState(mode), r = ROUTES[mode];
    if (s.take) {
      return `<div class="take">
          <button class="btn-play sm" data-act="preview" aria-label="Nghe lại bản thu">${icon("play")}</button>
          <div class="take-meta"><b>${esc(s.take.name)}</b><small>${fmtTime(s.take.duration)} · ${s.take.duration < state.minDuration ? `<span style="color:var(--warn)">ngắn hơn ${state.minDuration}s</span>` : "đủ độ dài"}</small></div>
        </div>
        <div class="actions">
          <button class="btn btn-primary" data-act="submit" ${s.phase === "loading" ? "disabled" : ""}>${icon("sparkles")}${esc(r.cta)}</button>
          <button class="btn btn-ghost" data-act="reset" ${s.phase === "loading" ? "disabled" : ""}>${icon("rotate")}${s.tab === "mic" ? "Thu lại" : "Chọn file khác"}</button>
        </div>`;
    }
    if (s.tab === "file") {
      return `<label class="dropzone" id="dropzone">
          <span class="dz-ico">${icon("upload")}</span>
          <strong>Kéo thả file vào đây hoặc bấm để chọn</strong>
          <span class="hint">Hỗ trợ WAV, MP3 · tối thiểu <b>${state.minDuration} giây</b></span>
          <input type="file" accept=".wav,.mp3,audio/wav,audio/mpeg" class="visually-hidden" id="file-input">
        </label>`;
    }
    return `<div class="orb-wrap">
        <svg class="orb-ring" viewBox="0 0 168 168" aria-hidden="true">
          <circle class="track" cx="84" cy="84" r="78"/>
          <circle class="bar" id="ring" cx="84" cy="84" r="78" stroke-dasharray="${RING_C}" stroke-dashoffset="${RING_C}"/>
        </svg>
        <button class="orb" id="orb" aria-label="Bắt đầu thu âm">${icon("mic")}</button>
      </div>
      <div class="timer" id="timer">0:00</div>
      <canvas class="viz" id="viz" width="1040" height="128" hidden></canvas>
      <p class="hint" id="rec-hint">${esc(r.recHint)} · tối thiểu <b>${state.minDuration} giây</b></p>`;
  }

  function refreshStage(mode) {
    return morph(() => {
      const stage = $("#stage");
      if (stage) stage.innerHTML = stageHTML(mode);
      bindStage(mode);
    });
  }

  function setSlot(mode, html, stateObj) {
    stateObj.html = html;
    return morph(() => {
      const slot = $("#slot");
      if (slot) {
        slot.innerHTML = html;
        slot.firstElementChild?.classList.add("swap-in");
        animateMeters(slot);
      }
    });
  }

  function bindAudioView(mode) {
    const s = audioState(mode);
    const seg = $(".segmented", view);
    seg.addEventListener("click", (e) => {
      const btn = e.target.closest("[data-tab]");
      if (!btn || btn.dataset.tab === s.tab || s.phase === "loading") return;
      if (recorder.active) recorder.stop(true);
      s.tab = btn.dataset.tab;
      s.take = null;
      s.phase = "idle";
      seg.style.setProperty("--seg-index", s.tab === "mic" ? 0 : 1);
      $$("[data-tab]", seg).forEach((b) => b.setAttribute("aria-selected", b.dataset.tab === s.tab));
      refreshStage(mode);
    });
    bindStage(mode);
    bindResults($("#slot"));
    animateMeters($("#slot"));
  }

  function bindStage(mode) {
    const s = audioState(mode);
    const stage = $("#stage");
    if (!stage) return;

    $("#orb", stage)?.addEventListener("click", () => {
      if (recorder.active) recorder.stop();
      else recorder.start(mode);
    });

    const input = $("#file-input", stage);
    if (input) {
      const dz = $("#dropzone", stage);
      input.addEventListener("change", () => input.files[0] && useFile(mode, input.files[0]));
      ["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
      ["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
      dz.addEventListener("drop", (e) => e.dataTransfer.files[0] && useFile(mode, e.dataTransfer.files[0]));
    }

    stage.addEventListener("click", (e) => {
      const act = e.target.closest("[data-act]")?.dataset.act;
      if (act === "preview" && s.take) {
        player.play({ title: s.take.name, artist: "Bản thu của bạn", audio_url: s.take.url });
      } else if (act === "reset") {
        s.take = null;
        s.phase = "idle";
        refreshStage(mode);
      } else if (act === "submit") {
        submitAudio(mode);
      }
    });
  }

  async function useFile(mode, file) {
    if (!/\.(wav|mp3)$/i.test(file.name)) {
      setSlot(mode, alertHTML("Chỉ hỗ trợ file WAV hoặc MP3."), audioState(mode));
      return;
    }
    const url = URL.createObjectURL(file);
    const duration = await probeDuration(url);
    const s = audioState(mode);
    s.take = { blob: file, name: file.name, duration, url };
    refreshStage(mode);
  }

  function probeDuration(url) {
    return new Promise((resolve) => {
      const a = new Audio();
      a.preload = "metadata";
      a.onloadedmetadata = () => resolve(a.duration);
      a.onerror = () => resolve(0);
      a.src = url;
    });
  }

  async function submitAudio(mode) {
    const s = audioState(mode);
    if (!s.take || s.phase === "loading") return;
    s.phase = "loading";
    const btns = $$("#stage [data-act='submit'], #stage [data-act='reset']");
    btns.forEach((b) => (b.disabled = true));
    const msg = {
      hybrid: "AI đang phân tích song song lời hát (Whisper) và âm thanh (CLAP + Qdrant)...",
      lyrics: "AI đang bóc tách âm thanh và nhận dạng câu hát...",
    }[mode];
    await setSlot(mode, loadingHTML(msg), s);
    $("#slot")?.scrollIntoView({ behavior: reduceMotion.matches ? "auto" : "smooth", block: "nearest" });

    const fd = new FormData();
    fd.append("audio", s.take.blob, s.take.name.endsWith(".mp3") ? s.take.name : s.take.name.replace(/\.\w+$/, "") + ".wav");
    let data;
    try {
      data = await api(`/api/search/audio/${mode}`, { method: "POST", body: fd });
    } catch {
      data = { error: "Không kết nối được tới máy chủ." };
    }
    s.phase = "done";
    btns.forEach((b) => (b.disabled = false));
    if (state.route !== mode) {
      s.html = resultHTML(data, mode);
      return;
    }
    await setSlot(mode, resultHTML(data, mode), s);
    bindResults($("#slot"));
  }

  // ---------------- Recorder (MediaRecorder → WAV 16kHz mono) ----------------
  const recorder = {
    active: false,
    async start(mode) {
      const hint = $("#rec-hint");
      if (!navigator.mediaDevices?.getUserMedia) {
        setSlot(mode, alertHTML("Trình duyệt không hỗ trợ thu âm. Hãy dùng tab “Tải file”."), audioState(mode));
        return;
      }
      try {
        this.stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: true } });
      } catch {
        setSlot(mode, alertHTML("Không truy cập được micro. Hãy cấp quyền micro cho trang này rồi thử lại."), audioState(mode));
        return;
      }
      this.mode = mode;
      this.active = true;
      this.chunks = [];
      this.ctx = new (window.AudioContext || window.webkitAudioContext)();
      const src = this.ctx.createMediaStreamSource(this.stream);
      this.analyser = this.ctx.createAnalyser();
      this.analyser.fftSize = 256;
      this.analyser.smoothingTimeConstant = 0.75;
      src.connect(this.analyser);
      this.mr = new MediaRecorder(this.stream);
      this.mr.ondataavailable = (e) => e.data.size && this.chunks.push(e.data);
      this.mr.onstop = () => this.finish();
      this.mr.start();
      this.t0 = performance.now();

      const orb = $("#orb");
      morph(() => {
        orb.classList.add("recording");
        orb.innerHTML = icon("stop");
        orb.setAttribute("aria-label", "Dừng thu âm");
        $("#viz").hidden = false;
        if (hint) hint.innerHTML = "Đang thu... nhấn nút để dừng";
      });
      this.tick();
    },
    tick() {
      if (!this.active) return;
      const el = (performance.now() - this.t0) / 1000;
      const timer = $("#timer"), ring = $("#ring"), canvas = $("#viz");
      if (timer) timer.textContent = fmtTime(el);
      if (ring) {
        const p = Math.min(el / state.minDuration, 1);
        ring.setAttribute("stroke-dashoffset", RING_C * (1 - p));
        if (p >= 1 && !ring.classList.contains("done")) {
          ring.classList.add("done");
          const hint = $("#rec-hint");
          if (hint) hint.innerHTML = "Đã đủ độ dài · nhấn nút để dừng";
        }
      }
      if (canvas) drawBars(canvas, this.analyser);
      this.raf = requestAnimationFrame(() => this.tick());
    },
    stop(discard = false) {
      if (!this.active) return;
      this.active = false;
      this.discard = discard;
      cancelAnimationFrame(this.raf);
      this.mr.state !== "inactive" && this.mr.stop();
      this.stream.getTracks().forEach((t) => t.stop());
    },
    async finish() {
      const ctx = this.ctx;
      const mode = this.mode;
      if (this.discard) { ctx.close(); return; }
      const blob = new Blob(this.chunks, { type: this.mr.mimeType || "audio/webm" });
      let wav, duration;
      try {
        ({ wav, duration } = await toWav16k(blob, ctx));
      } catch {
        ctx.close();
        setSlot(mode, alertHTML("Không giải mã được bản thu. Vui lòng thử lại."), audioState(mode));
        return;
      }
      ctx.close();
      const s = audioState(mode);
      const stamp = new Date().toLocaleTimeString("vi-VN", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
      s.take = { blob: wav, name: `Bản thu lúc ${stamp}`, duration, url: URL.createObjectURL(wav) };
      if (state.route === mode) refreshStage(mode);
    },
  };

  function drawBars(canvas, analyser) {
    const g = canvas.getContext("2d");
    const data = new Uint8Array(analyser.frequencyBinCount);
    analyser.getByteFrequencyData(data);
    const W = canvas.width, H = canvas.height, n = 48, gap = 8;
    const bw = (W - gap * (n - 1)) / n;
    g.clearRect(0, 0, W, H);
    g.fillStyle = "#1ed760";
    for (let i = 0; i < n; i++) {
      const v = data[Math.floor((i / n) * data.length * 0.7)] / 255;
      const h = Math.max(6, v * H);
      const x = i * (bw + gap), y = (H - h) / 2;
      g.beginPath();
      g.roundRect ? g.roundRect(x, y, bw, h, bw / 2) : g.rect(x, y, bw, h);
      g.fill();
    }
  }

  async function toWav16k(blob, ctx) {
    const buf = await ctx.decodeAudioData(await blob.arrayBuffer());
    const rate = 16000;
    const off = new OfflineAudioContext(1, Math.ceil(buf.duration * rate), rate);
    const src = off.createBufferSource();
    src.buffer = buf;
    src.connect(off.destination);
    src.start();
    const rendered = await off.startRendering();
    const pcm = rendered.getChannelData(0);
    const out = new DataView(new ArrayBuffer(44 + pcm.length * 2));
    const wstr = (o, s) => [...s].forEach((c, i) => out.setUint8(o + i, c.charCodeAt(0)));
    wstr(0, "RIFF"); out.setUint32(4, 36 + pcm.length * 2, true); wstr(8, "WAVE");
    wstr(12, "fmt "); out.setUint32(16, 16, true); out.setUint16(20, 1, true); out.setUint16(22, 1, true);
    out.setUint32(24, rate, true); out.setUint32(28, rate * 2, true); out.setUint16(32, 2, true); out.setUint16(34, 16, true);
    wstr(36, "data"); out.setUint32(40, pcm.length * 2, true);
    for (let i = 0; i < pcm.length; i++) {
      const v = Math.max(-1, Math.min(1, pcm[i]));
      out.setInt16(44 + i * 2, v < 0 ? v * 0x8000 : v * 0x7fff, true);
    }
    return { wav: new Blob([out], { type: "audio/wav" }), duration: buf.duration };
  }

  // =========================================================
  // RESULT RENDERING (dùng chung)
  // =========================================================
  const playables = new Map();
  let playSeq = 0;
  const registerPlayable = (t) => { const id = `p${++playSeq}`; playables.set(id, t); return id; };

  const alertHTML = (msg, kind = "") =>
    `<div class="alert ${kind}" role="${kind ? "status" : "alert"}">${icon(kind === "info" ? "info" : "alert")}<p>${esc(msg)}</p></div>`;
  const loadingHTML = (msg) =>
    `<div class="card analyzing"><div class="eq" aria-hidden="true"><span></span><span></span><span></span><span></span><span></span></div><b>Đang phân tích</b><p>${esc(msg)}</p></div>`;

  function meterHTML(label, value, muted = false) {
    return `<div class="meter ${muted ? "muted" : ""}">
      <div class="meter-top"><span>${esc(label)}</span><b>${pct(value)}</b></div>
      <div class="meter-bar"><i data-w="${Math.max(0, Math.min(1, value)) * 100}%"></i></div>
    </div>`;
  }
  function animateMeters(root) {
    if (!root) return;
    requestAnimationFrame(() => $$(".meter-bar i[data-w], .gbar i[data-w]", root).forEach((i) => (i.style.width = i.dataset.w)));
  }

  function highlightLyrics(full, matched) {
    const m = (matched || "").trim().toLowerCase();
    return full.split("\n").map((line) => {
      const l = line.trim().toLowerCase();
      const hit = m && l && (l.length >= 4 ? (m.includes(l) || l.includes(m)) : l === m);
      return hit ? `<mark>${esc(line)}</mark>` : esc(line);
    }).join("\n");
  }

  function playBtn(t, cls = "btn-play") {
    if (!t.audio_url) return `<button class="${cls}" disabled aria-label="Không có file gốc để phát" title="Không tìm thấy file gốc trên máy">${icon("play")}</button>`;
    const id = registerPlayable(t);
    return `<button class="${cls}" data-play="${id}" aria-label="Phát ${esc(t.title)}">${icon("play")}</button>`;
  }

  function resultHTML(data, mode) {
    if (data.error) return alertHTML(data.error);
    const results = data.results || [];
    if (!results.length) return alertHTML("Không tìm thấy bài hát phù hợp!");
    const top = results[0];
    const c = coverAttrs(top.title);

    let badge, chips = [], meters = "";
    if (mode === "hybrid") {
      badge = `${icon("sparkles")}Kết quả dung hợp tối ưu`;
      chips.push(esc(top.fusion_desc));
      if (data.elapsed) chips.push(esc(data.elapsed));
      meters = meterHTML("Độ tin cậy tổng hợp", top.final_score)
        + meterHTML("Điểm lời hát", top.lyrics_score, !top.lyrics_score)
        + meterHTML("Điểm âm thanh (CLAP)", top.audio_score, !top.audio_score);
    } else {
      badge = `${icon("mic")}Phù hợp nhất`;
      chips.push(`Ngưỡng hệ thống <b>${(data.threshold ?? 0.6).toFixed(2)}</b>`);
      meters = meterHTML("Độ tin cậy ngữ nghĩa", top.score);
    }

    let html = `<article class="result reveal-stagger">
      <div class="result-hero" style="${c.style}">
        <div class="cover" style="${c.style}" aria-hidden="true">${c.initials}</div>
        <div>
          <span class="badge">${badge}</span>
          <h2 class="result-title">${esc(top.title)}</h2>
          <p class="result-artist">${esc(top.artist)}</p>
          <div class="result-row">${playBtn(top)}<div class="chips">${chips.map((x) => `<span class="chip">${x}</span>`).join("")}</div></div>
        </div>
      </div>
      <div class="meters">${meters}</div>`;

    if (data.transcription) {
      html += `<blockquote class="quote"><small>AI nghe được</small>“${esc(data.transcription)}”</blockquote>`;
    } else if (mode === "hybrid") {
      html += alertHTML("Không phát hiện câu từ rõ nghĩa — hệ thống đã tự động dồn trọng số sang nhánh âm thanh (CLAP).", "info");
    }
    if (top.matched_line) {
      html += `<blockquote class="quote"><small>Câu hát khớp nhất</small>“${esc(top.matched_line)}”</blockquote>`;
    }
    if (top.full_lyrics) {
      html += `<details class="panel"><summary><span>Xem toàn bộ lời bài hát để xác thực</span>${icon("chevron")}</summary>
        <div class="panel-body"><div class="lyrics">${highlightLyrics(top.full_lyrics, top.matched_line)}</div></div></details>`;
    }
    if (data.pitch_plot) {
      html += `<details class="panel" open><summary><span>Biểu đồ dóng hàng cao độ (Pitch Alignment)</span>${icon("chevron")}</summary>
        <div class="panel-body"><img class="plot" src="${data.pitch_plot}" alt="Biểu đồ so sánh đường cong cao độ của bản thu với đoạn tương ứng trong bài ${esc(top.title)}" loading="lazy"></div></details>`;
    }
    if (results.length > 1) {
      html += `<section><h3 class="section-title">Các kết quả tham khảo khác</h3><div class="tracklist">`;
      results.slice(1).forEach((r, i) => {
        const score = mode === "hybrid" ? r.final_score : r.score;
        const extra = mode === "hybrid" ? r.fusion_desc : r.artist;
        html += `<div class="track">
          <span class="num">${i + 2}</span>${cover(r.title, "cover-xs")}
          <div class="t-main"><b>${esc(r.title)}</b><small>${esc(mode === "lyrics" || mode === "text" ? r.artist : `${r.artist} · ${extra}`)}</small></div>
          <span class="t-score">${mode === "lyrics" || mode === "text" ? Number(score).toFixed(4) : pct(score)}</span>
          ${playBtn(r, "btn-icon")}
        </div>`;
      });
      html += `</div></section>`;
    }
    return html + `</article>`;
  }

  function bindResults(root) {
    if (!root || root.dataset.bound) return;
    root.dataset.bound = "1";
    root.addEventListener("click", (e) => {
      const btn = e.target.closest("[data-play]");
      if (!btn) return;
      const t = playables.get(btn.dataset.play);
      if (t) player.play(t);
    });
  }

  // =========================================================
  // TEXT VIEW
  // =========================================================
  function textViewHTML() {
    return `<section class="card">
      <form id="text-form" role="search">
        <label class="field-label" for="q">Gõ trực tiếp câu hát bạn nhớ</label>
        <div class="searchbox">
          ${icon("search")}
          <input id="q" name="q" type="search" autocomplete="off" placeholder="Ví dụ: Bên em bao lâu..." value="${esc(state.text.query)}" minlength="3" required>
          <button class="btn btn-primary" type="submit">Tìm bài hát</button>
        </div>
        <p class="field-help">Nhập ít nhất 3 ký tự. Hỗ trợ cả tiếng Việt có dấu và không dấu.</p>
      </form>
    </section>
    <div class="slot" id="slot" aria-live="polite">${state.text.html}</div>`;
  }
  function bindTextView() {
    const form = $("#text-form"), input = $("#q");
    bindResults($("#slot"));
    animateMeters($("#slot"));
    if (!state.text.html) input.focus({ preventScroll: true });
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const query = input.value.trim();
      state.text.query = query;
      if (query.length < 3) {
        setSlot("text", alertHTML("Vui lòng nhập lời bài hát dài hơn một chút!"), state.text);
        return;
      }
      const btn = $("button[type=submit]", form);
      btn.disabled = true;
      await setSlot("text", loadingHTML("Đang rà soát dữ liệu lời bài hát..."), state.text);
      let data;
      try {
        data = await api("/api/search/text", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ query }) });
      } catch {
        data = { error: "Không kết nối được tới máy chủ." };
      }
      btn.disabled = false;
      const html = resultHTML(data, "text");
      if (state.route !== "text") { state.text.html = html; return; }
      await setSlot("text", html, state.text);
    });
  }

  // =========================================================
  // AGENT VIEW (Thu thập nhạc mới)
  // =========================================================
  const AGENT_SOURCES = [["all", "Tất cả"], ["charts", "Bảng xếp hạng"], ["search", "Tìm kiếm"]];

  function agentViewHTML() {
    const a = state.agent;
    const idx = AGENT_SOURCES.findIndex(([k]) => k === a.source);
    return `<section class="card">
      <form id="agent-form" class="agent-form">
        <div>
          <label class="field-label" for="agent-limit">Số bài hát mới muốn thu thập: <b id="agent-limit-val">${a.limit}</b></label>
          <input type="range" id="agent-limit" min="3" max="50" step="1" value="${a.limit}">
        </div>
        <div>
          <span class="field-label">Nguồn dữ liệu</span>
          <div class="segmented" role="tablist" id="agent-src" style="--seg-count:3;--seg-index:${idx}">
            ${AGENT_SOURCES.map(([k, l]) => `<button type="button" role="tab" aria-selected="${k === a.source}" data-src="${k}">${l}</button>`).join("")}
          </div>
        </div>
        <div>
          <label class="field-label" for="agent-q">Chủ đề / tên ca sĩ (tùy chọn)</label>
          <div class="searchbox">${icon("search")}<input id="agent-q" type="search" placeholder="Ví dụ: Sơn Tùng M-TP, Đen Vâu, Rap Việt..." value="${esc(a.query)}"></div>
          <p class="field-help">Nhập tên ca sĩ để Agent gom hàng loạt bài của nghệ sĩ đó. Quá trình có thể mất 1–3 phút.</p>
        </div>
        <div><button class="btn btn-primary" type="submit">${icon("download")}Khởi chạy AI Data Agent</button></div>
      </form>
    </section>
    <div class="slot" id="slot" aria-live="polite">${a.html}</div>`;
  }

  function bindAgentView() {
    const a = state.agent;
    const range = $("#agent-limit"), seg = $("#agent-src");
    const fill = () => range.style.setProperty("--fill", `${((range.value - range.min) / (range.max - range.min)) * 100}%`);
    fill();
    range.addEventListener("input", () => { a.limit = +range.value; $("#agent-limit-val").textContent = range.value; fill(); });
    seg.addEventListener("click", (e) => {
      const b = e.target.closest("[data-src]");
      if (!b) return;
      a.source = b.dataset.src;
      seg.style.setProperty("--seg-index", AGENT_SOURCES.findIndex(([k]) => k === a.source));
      $$("[data-src]", seg).forEach((x) => x.setAttribute("aria-selected", x === b));
    });
    $("#agent-form").addEventListener("submit", async (e) => {
      e.preventDefault();
      a.query = $("#agent-q").value.trim();
      const btn = $("#agent-form button[type=submit]");
      btn.disabled = true;
      await setSlot("agent", loadingHTML("AI Data Agent đang quét và nạp dữ liệu (1–3 phút)..."), a);
      let data;
      try {
        data = await api("/api/agent", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ limit: a.limit, source: a.source, query: a.query }) });
      } catch {
        data = { error: "Không kết nối được tới máy chủ." };
      }
      btn.disabled = false;
      let html = data.error ? alertHTML(data.error)
        : `<div class="result"><div class="stat-grid">
            <div class="stat kpi"><span>Bài hát mới</span><b>+${data.added}</b><small>vừa được nạp vào kho</small></div>
            <div class="stat kpi"><span>Tổng số bài hát</span><b>${data.after.toLocaleString("vi-VN")}</b><small>trước đó ${data.before.toLocaleString("vi-VN")}</small></div>
          </div>${alertHTML("Đã cập nhật MongoDB, Qdrant (âm thanh) và FAISS (lời hát). Bạn có thể sang mục tìm kiếm để thử ngay.", "info")}</div>`;
      if (data.log) html += `<details class="panel"><summary><span>Nhật ký chạy Agent</span>${icon("chevron")}</summary><div class="panel-body"><pre class="log">${esc(data.log)}</pre></div></details>`;
      if (state.route !== "agent") { a.html = html; return; }
      await setSlot("agent", html, a);
    });
  }

  // =========================================================
  // ANALYSIS VIEW
  // =========================================================
  const SUMMARY_LABELS = {
    sample_rate: "Tần số lấy mẫu", duration: "Thời lượng", total_samples: "Tổng số mẫu",
    channels: "Số kênh", bit_depth: "Độ sâu bit", max_amplitude: "Biên độ cực đại",
    rms_energy_mean: "Năng lượng RMS TB", spectral_centroid_mean: "Trọng tâm phổ TB",
    zcr_mean: "Zero-crossing rate TB", mfcc_shape: "Kích thước MFCC", mel_spec_shape: "Kích thước Mel-spectrogram",
  };

  function analysisViewHTML() {
    return `<div class="split">
      <aside class="card picker" aria-label="Chọn bài hát">
        <h2 class="section-title" style="margin:0">Chọn bài hát</h2>
        <div class="searchbox">${icon("search")}<input id="track-q" type="search" placeholder="Tìm theo tên bài hoặc ca sĩ" value="${esc(state.analysis.q)}" aria-label="Tìm bài hát để phân tích"></div>
        <div class="picker-list tracklist" id="track-list">${state.analysis.tracks ? trackListHTML(state.analysis.tracks) : skeletonList()}</div>
        <label class="btn btn-ghost" style="width:100%">${icon("upload")}Hoặc tải file của bạn<input type="file" id="an-file" accept=".wav,.mp3,audio/wav,audio/mpeg" class="visually-hidden"></label>
      </aside>
      <div class="slot" id="slot" aria-live="polite">${state.analysis.html || `<div class="card empty">${icon("activity")}<p>Chọn một bài hát ở danh sách bên cạnh hoặc tải file lên để xem phân tích tín hiệu.</p></div>`}</div>
    </div>`;
  }
  const skeletonList = () => Array.from({ length: 6 }, () => `<div class="skeleton" style="height:52px;margin:4px 0"></div>`).join("");
  function trackListHTML(tracks) {
    if (!tracks.length) return `<p class="empty">Không có bài hát nào khớp.</p>`;
    return tracks.map((t) => `<button class="track" data-track="${esc(t.id)}">
      ${cover(t.title, "cover-xs")}
      <span class="t-main"><b>${esc(t.title)}</b><small>${esc(t.artist)}</small></span>
      <span class="t-score">${icon("activity")}</span>
    </button>`).join("");
  }

  async function loadTracks(q) {
    const data = await api(`/api/tracks?q=${encodeURIComponent(q)}`).catch(() => ({ tracks: [] }));
    state.analysis.tracks = data.tracks || [];
    const list = $("#track-list");
    if (list && state.analysis.q === q) list.innerHTML = trackListHTML(state.analysis.tracks);
  }

  function bindAnalysisView() {
    const q = $("#track-q");
    let timer;
    if (!state.analysis.tracks) loadTracks(state.analysis.q);
    q.addEventListener("input", () => {
      clearTimeout(timer);
      timer = setTimeout(() => { state.analysis.q = q.value.trim(); loadTracks(state.analysis.q); }, 250);
    });
    $("#track-list").addEventListener("click", (e) => {
      const b = e.target.closest("[data-track]");
      if (!b) return;
      $$("#track-list .track").forEach((x) => x.classList.toggle("playing", x === b));
      runAnalysis(() => api("/api/analyze", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ track_id: b.dataset.track }) }));
    });
    $("#an-file").addEventListener("change", (e) => {
      const f = e.target.files[0];
      if (!f) return;
      const fd = new FormData();
      fd.append("audio", f, f.name);
      runAnalysis(() => api("/api/analyze", { method: "POST", body: fd }), URL.createObjectURL(f));
    });
    bindResults($("#slot"));
  }

  async function runAnalysis(request, localUrl) {
    await setSlot("analysis", loadingHTML("Đang trích xuất đặc trưng tín hiệu (Mel, MFCC, Chroma...)"), state.analysis);
    let data;
    try { data = await request(); } catch { data = { error: "Không kết nối được tới máy chủ." }; }
    let html;
    if (data.error) {
      html = alertHTML(data.error);
    } else {
      const t = { title: data.title, artist: "Phân tích tín hiệu", audio_url: data.audio_url || localUrl };
      const c = coverAttrs(data.title);
      html = `<article class="result reveal-stagger">
        <div class="result-hero" style="${c.style}">
          <div class="cover" style="${c.style}" aria-hidden="true">${c.initials}</div>
          <div><span class="badge">${icon("activity")}Thông số kỹ thuật</span>
            <h2 class="result-title">${esc(data.title)}</h2>
            <div class="result-row">${playBtn(t)}<span class="chip">${esc(data.summary.duration)} · ${esc(data.summary.sample_rate)}</span></div>
          </div>
        </div>
        <div class="stat-grid">${Object.entries(data.summary).map(([k, v]) => `<div class="stat"><span>${esc(SUMMARY_LABELS[k] || k)}</span><b>${esc(v)}</b></div>`).join("")}</div>
        <section class="card"><h3 class="section-title">Biểu đồ phân tích 7 đặc trưng</h3>
          <img class="plot" src="${data.plot}" alt="Bảng phân tích gồm dạng sóng, Mel-spectrogram, MFCC, Chromagram, RMS, trọng tâm phổ và ZCR" loading="lazy"></section>
      </article>`;
    }
    if (state.route !== "analysis") { state.analysis.html = html; return; }
    await setSlot("analysis", html, state.analysis);
  }

  // =========================================================
  // WER VIEW
  // =========================================================
  const LANG = { vietnamese: "Tiếng Việt", english_or_other: "Tiếng Anh / khác" };
  const werColor = (v) => v < 0.2 ? "#1ed760" : v < 0.4 ? "#7fdca0" : v < 0.6 ? "#f0c94a" : v < 0.8 ? "#ff9f43" : "#f3727f";

  function werViewHTML() {
    return `<div class="slot" id="slot">${state.wer ? werHTML(state.wer) : werSkeleton()}</div>`;
  }
  const werSkeleton = () => `<div class="stat-grid">${Array.from({ length: 4 }, () => `<div class="skeleton" style="height:96px"></div>`).join("")}</div>
    <div class="skeleton" style="height:260px;margin-top:24px"></div>`;

  async function bindWerView() {
    if (!state.wer) {
      const data = await api("/api/wer").catch(() => ({ error: "Không kết nối được tới máy chủ." }));
      state.wer = data;
      if (state.route !== "wer") return;
      await setSlot("wer", werHTML(data), {});
    } else {
      animateMeters($("#slot"));
    }
    bindWerTabs();
  }

  function histogramSVG(values, median) {
    const bins = 30, max = 2, counts = new Array(bins).fill(0);
    values.forEach((v) => counts[Math.min(bins - 1, Math.floor((Math.min(v, max) / max) * bins))]++);
    const W = 800, H = 260, m = { t: 16, r: 12, b: 36, l: 44 };
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const top = Math.max(...counts);
    const yMax = Math.ceil(top / 10) * 10 || 10;
    const bw = iw / bins;
    let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Biểu đồ phân bố WER của ${values.length} bài hát">`;
    for (let i = 0; i <= 4; i++) {
      const y = m.t + ih - (ih * i) / 4;
      s += `<line class="gridline" x1="${m.l}" x2="${W - m.r}" y1="${y}" y2="${y}"/><text x="${m.l - 8}" y="${y + 4}" text-anchor="end">${Math.round((yMax * i) / 4)}</text>`;
    }
    counts.forEach((c, i) => {
      const h = (c / yMax) * ih;
      s += `<rect class="bar" x="${m.l + i * bw + 1.5}" y="${m.t + ih - h}" width="${bw - 3}" height="${h}" rx="2"><title>WER ${pct((i / bins) * max, 0)}–${pct(((i + 1) / bins) * max, 0)}: ${c} bài</title></rect>`;
    });
    for (let v = 0; v <= max; v += 0.5) {
      const x = m.l + (v / max) * iw;
      s += `<text x="${x}" y="${H - 14}" text-anchor="middle">${v * 100}%${v === max ? "+" : ""}</text>`;
    }
    const mx = m.l + (Math.min(median, max) / max) * iw;
    s += `<line class="median" x1="${mx}" x2="${mx}" y1="${m.t}" y2="${m.t + ih}"/><text x="${mx + 6}" y="${m.t + 12}" style="fill:var(--text)">Trung vị ${pct(median)}</text>`;
    return s + `</svg>`;
  }

  function werTable(rows) {
    return `<div class="table-wrap"><table>
      <thead><tr><th>Bài hát</th><th>Ca sĩ</th><th>Ngôn ngữ</th><th class="num">WER</th><th class="num">CER</th></tr></thead>
      <tbody>${rows.map((r) => `<tr>
        <td><span class="t-title" title="${esc(r.title)}">${esc(r.title)}</span></td><td>${esc(r.artist)}</td><td>${esc(LANG[r.language] || r.language)}</td>
        <td class="num"><span class="pill" style="background:${werColor(r.wer)}26;color:${werColor(r.wer)}">${pct(r.wer)}</span></td>
        <td class="num">${pct(r.cer)}</td></tr>`).join("")}</tbody></table></div>`;
  }

  function werHTML(data) {
    if (data.error) return alertHTML(data.error);
    const st = data.statistics, o = st.overall;
    const totalErr = o.total_substitutions + o.total_insertions + o.total_deletions;
    const errs = [
      ["Thay thế (S)", o.total_substitutions, "#1ed760"],
      ["Chèn thêm (I)", o.total_insertions, "#f0c94a"],
      ["Xóa mất (D)", o.total_deletions, "#f3727f"],
    ];
    const d = st.wer_distribution;
    const groups = [
      ["Xuất sắc (<20%)", d.excellent_0_20, "#1ed760"], ["Tốt (20–40%)", d.good_20_40, "#7fdca0"],
      ["Trung bình (40–60%)", d.fair_40_60, "#f0c94a"], ["Kém (60–80%)", d.poor_60_80, "#ff9f43"],
      ["Rất kém (>80%)", d.bad_80_plus, "#f3727f"],
    ];
    const gMax = Math.max(...groups.map((g) => g[1])) || 1;

    return `<div class="result reveal-stagger">
      <div class="stat-grid">
        <div class="stat kpi"><span>Tổng bài đánh giá</span><b>${o.total_tracks.toLocaleString("vi-VN")}</b><small>bài hát có lời chuẩn</small></div>
        <div class="stat kpi"><span>WER trung bình</span><b>${pct(o.wer_mean)}</b><small>tỉ lệ lỗi từ</small></div>
        <div class="stat kpi"><span>WER trung vị</span><b>${pct(o.wer_median)}</b><small>ít bị ảnh hưởng bởi ngoại lai</small></div>
        <div class="stat kpi"><span>CER trung bình</span><b>${pct(o.cer_mean)}</b><small>tỉ lệ lỗi ký tự</small></div>
      </div>
      ${totalErr ? `<section class="card"><h3 class="section-title">Phân tích loại lỗi</h3>
        <div class="stack" role="img" aria-label="Tỉ lệ các loại lỗi">${errs.map(([, v, c]) => `<i style="width:${(v / totalErr) * 100}%;background:${c}"></i>`).join("")}</div>
        <div class="legend">${errs.map(([l, v, c]) => `<span><i style="background:${c}"></i>${l}: <b>${v.toLocaleString("vi-VN")}</b> (${pct(v / totalErr)})</span>`).join("")}
          <span>Từ đúng: <b>${o.total_hits.toLocaleString("vi-VN")}</b></span></div></section>` : ""}
      <section class="card chart"><h3 class="section-title">Phân bố WER trên toàn bộ dataset</h3>${histogramSVG(data.wer_values, o.wer_median)}</section>
      <section class="card"><h3 class="section-title">Phân nhóm chất lượng</h3><div class="groups">
        ${groups.map(([l, v, c]) => `<div class="group-row"><span>${l}</span><div class="gbar"><i data-w="${(v / gMax) * 100}%" style="background:${c}"></i></div><b>${v}</b></div>`).join("")}
      </div></section>
      <section><h3 class="section-title">So sánh theo ngôn ngữ</h3><div class="table-wrap"><table>
        <thead><tr><th>Ngôn ngữ</th><th class="num">Số bài</th><th class="num">WER TB</th><th class="num">CER TB</th></tr></thead>
        <tbody>${Object.entries(st.by_language).map(([k, v]) => `<tr><td>${esc(LANG[k] || k)}</td><td class="num">${v.count}</td><td class="num">${pct(v.wer_mean)}</td><td class="num">${pct(v.cer_mean)}</td></tr>`).join("")}</tbody>
      </table></div></section>
      <section>
        <div style="display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;gap:12px;margin-bottom:14px">
          <h3 class="section-title" style="margin:0">Top 10 bài hát</h3>
          <div class="segmented" role="tablist" id="wer-tabs" style="--seg-count:2;--seg-index:0">
            <button role="tab" aria-selected="true" data-wt="best">WER tốt nhất</button>
            <button role="tab" aria-selected="false" data-wt="worst">WER kém nhất</button>
          </div>
        </div>
        <div id="wer-table">${werTable(st.top_10_best)}</div>
      </section>
    </div>`;
  }

  function bindWerTabs() {
    const tabs = $("#wer-tabs");
    if (!tabs || !state.wer?.statistics) return;
    tabs.addEventListener("click", (e) => {
      const b = e.target.closest("[data-wt]");
      if (!b || b.getAttribute("aria-selected") === "true") return;
      const best = b.dataset.wt === "best";
      tabs.style.setProperty("--seg-index", best ? 0 : 1);
      $$("[data-wt]", tabs).forEach((x) => x.setAttribute("aria-selected", x === b));
      morph(() => {
        $("#wer-table").innerHTML = werTable(best ? state.wer.statistics.top_10_best : state.wer.statistics.top_10_worst);
      });
    });
  }

  // =========================================================
  // PLAYER
  // =========================================================
  const player = (() => {
    const audio = $("#audio"), toggle = $("#player-toggle"), seek = $("#player-seek"), vol = $("#player-vol");
    const fill = (el) => el.style.setProperty("--fill", `${((el.value - el.min) / (el.max - el.min)) * 100}%`);
    const setIcon = () => {
      toggle.innerHTML = icon(audio.paused ? "play" : "pause");
      toggle.setAttribute("aria-label", audio.paused ? "Phát" : "Tạm dừng");
    };
    let seeking = false;
    $("#vol-icon").innerHTML = icon("volume");
    audio.volume = vol.value;
    fill(vol);
    setIcon();
    toggle.addEventListener("click", () => (audio.paused ? audio.play() : audio.pause()));
    audio.addEventListener("play", setIcon);
    audio.addEventListener("pause", setIcon);
    audio.addEventListener("ended", setIcon);
    audio.addEventListener("loadedmetadata", () => { $("#player-dur").textContent = fmtTime(audio.duration); });
    audio.addEventListener("timeupdate", () => {
      $("#player-cur").textContent = fmtTime(audio.currentTime);
      if (!seeking && audio.duration) { seek.value = (audio.currentTime / audio.duration) * 100; fill(seek); }
    });
    seek.addEventListener("input", () => { seeking = true; fill(seek); });
    seek.addEventListener("change", () => { if (audio.duration) audio.currentTime = (seek.value / 100) * audio.duration; seeking = false; });
    vol.addEventListener("input", () => { audio.volume = vol.value; fill(vol); });
    document.addEventListener("keydown", (e) => {
      if (e.code === "Space" && !toggle.disabled && !e.target.closest("input, textarea, button, [contenteditable]")) {
        e.preventDefault();
        toggle.click();
      }
    });

    return {
      play(t) {
        if (!t.audio_url) return;
        if (audio.dataset.src === t.audio_url) {
          audio.paused ? audio.play() : audio.pause();
          return;
        }
        audio.dataset.src = t.audio_url;
        audio.src = t.audio_url;
        const c = coverAttrs(t.title);
        const pc = $("#player-cover");
        pc.setAttribute("style", c.style);
        pc.textContent = initialsOf(t.title);
        $("#player-title").textContent = t.title;
        $("#player-artist").textContent = t.artist || "";
        toggle.disabled = false;
        seek.disabled = false;
        audio.play().catch(() => {});
      },
    };
  })();

  // =========================================================
  // STATUS
  // =========================================================
  async function pollStatus() {
    const el = $("#status"), txt = $(".status-text", el);
    try {
      const s = await api("/api/status");
      state.minDuration = s.min_duration || 5;
      state.ready = !!s.ready;
      el.classList.toggle("ready", state.ready);
      el.classList.toggle("error", !!s.error);
      txt.textContent = s.error ? `Lỗi: ${s.error}` : state.ready ? "AI sẵn sàng" : s.stage;
      if (!state.ready && !s.error) setTimeout(pollStatus, 2000);
    } catch {
      el.classList.add("error");
      txt.textContent = "Mất kết nối máy chủ";
      setTimeout(pollStatus, 4000);
    }
  }

  // ---------------- Boot ----------------
  renderNav();
  window.addEventListener("hashchange", navigate);
  if (!location.hash) history.replaceState(null, "", `#/${DEFAULT_ROUTE}`);
  navigate();
  pollStatus();
})();
