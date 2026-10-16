/**
 * Gemini Live bidirectional WebSocket audio client (/ws/live).
 * Captures 16kHz 16-bit mono PCM from microphone, plays 24kHz 16-bit mono PCM from model,
 * updates level meter & transcript log, and dispatches locate sessions to map polling.
 */
(function () {
  const TARGET_SR = 16000;
  const PLAY_SR = 24000;
  const CHUNK_SAMPLES = 1600; // 100ms at 16kHz = 3200 bytes Int16
  const RMS_THRESHOLD = 0.02;
  const SILENCE_MS = 6000;

  const Live = {
    active: false,
    capCtx: null,
    playCtx: null,
    stream: null,
    source: null,
    proc: null,
    ws: null,
    nextPlayTime: 0,
    playingCount: 0,
    heardSpeech: false,
    lastSpeechAt: 0,
    endSent: false,
    pcmBuf: new Int16Array(CHUNK_SAMPLES),
    pcmFill: 0,
    levels: [],
    lastModelEl: null,
    closeTimer: null,

    toggle() {
      if (this.active) {
        this.stop();
      } else {
        this.start();
      }
    },

    async start() {
      if (this.active) return;
      if (this.closeTimer) {
        clearTimeout(this.closeTimer);
        this.closeTimer = null;
      }

      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (!AudioCtx) {
        if (typeof window.showError === "function") {
          window.showError("마이크를 쓸 수 없어요: AudioContext unsupported");
        }
        return;
      }

      // (a) Create and resume AudioContexts inside click handler BEFORE getUserMedia
      try {
        this.capCtx = new AudioCtx();
        if (this.capCtx.state === "suspended") {
          this.capCtx.resume();
        }
        this.playCtx = new AudioCtx({ sampleRate: PLAY_SR });
        if (this.playCtx.state === "suspended") {
          this.playCtx.resume();
        }
      } catch (e) {
        if (typeof window.showError === "function") {
          window.showError("마이크를 쓸 수 없어요: " + e.message);
        }
        this.cleanupAudio();
        return;
      }

      // (b) Request microphone stream
      try {
        this.stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            channelCount: 1,
            echoCancellation: true,
            noiseSuppression: true,
          },
        });
      } catch (e) {
        if (typeof window.showError === "function") {
          window.showError("마이크를 쓸 수 없어요: " + e.message);
        }
        this.cleanupAudio();
        return;
      }

      this.active = true;
      this.nextPlayTime = 0;
      this.playingCount = 0;
      this.heardSpeech = false;
      this.lastSpeechAt = 0;
      this.endSent = false;
      this.pcmFill = 0;
      this.levels = [];
      this.lastModelEl = null;

      this.setButtonOn(true);
      const meter = document.getElementById("liveMeter");
      const stateEl = document.getElementById("liveState");
      if (meter) {
        meter.hidden = false;
        meter.classList.remove("speaking");
      }
      if (stateEl) stateEl.textContent = "듣는 중";

      // (c) Open WebSocket connection
      const proto = location.protocol === "https:" ? "wss" : "ws";
      const ws = new WebSocket(proto + "://" + location.host + "/ws/live");
      ws.binaryType = "arraybuffer";
      this.ws = ws;

      ws.onmessage = (ev) => this.onMessage(ev);
      ws.onerror = () => this.onDisconnect();
      ws.onclose = () => this.onDisconnect();

      // (d) Downsample mic Float32 to 16000Hz Int16 via ScriptProcessorNode(4096, 1, 1)
      const inRate = this.capCtx.sampleRate || 48000;
      this.source = this.capCtx.createMediaStreamSource(this.stream);
      this.proc = this.capCtx.createScriptProcessor(4096, 1, 1);

      this.proc.onaudioprocess = (e) => {
        if (!this.active) return;
        const input = e.inputBuffer.getChannelData(0);
        if (!input || input.length === 0) return;

        // Calculate RMS of current frame
        let sumSq = 0;
        for (let i = 0; i < input.length; i++) {
          sumSq += input[i] * input[i];
        }
        const rms = Math.sqrt(sumSq / input.length);
        this.updateMeter(rms);

        const now = Date.now();
        if (rms > RMS_THRESHOLD) {
          this.heardSpeech = true;
          this.lastSpeechAt = now;
        } else if (
          this.heardSpeech &&
          !this.endSent &&
          now - this.lastSpeechAt > SILENCE_MS
        ) {
          this.endSent = true;
          if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type: "end" }));
          }
        }

        // Linear interpolation downsample to 16000Hz
        const ratio = inRate / TARGET_SR;
        const outLen = Math.floor(input.length / ratio);
        for (let i = 0; i < outLen; i++) {
          const pos = i * ratio;
          const idx = Math.floor(pos);
          const frac = pos - idx;
          const s0 = input[idx] || 0;
          const s1 = idx + 1 < input.length ? input[idx + 1] : s0;
          const sample = s0 + (s1 - s0) * frac;
          const clamped = Math.max(-1, Math.min(1, sample));
          this.pcmBuf[this.pcmFill++] =
            clamped < 0 ? Math.round(clamped * 32768) : Math.round(clamped * 32767);

          if (this.pcmFill >= CHUNK_SAMPLES) {
            if (
              !this.endSent &&
              this.ws &&
              this.ws.readyState === WebSocket.OPEN
            ) {
              const out = new Int16Array(this.pcmBuf);
              this.ws.send(out.buffer);
            }
            this.pcmFill = 0;
          }
        }
      };

      this.source.connect(this.proc);
      this.proc.connect(this.capCtx.destination);
    },

    updateMeter(rms) {
      const meter = document.getElementById("liveMeter");
      const stateEl = document.getElementById("liveState");
      if (meter && meter.hidden) meter.hidden = false;

      const lv = Math.max(0, Math.min(1, rms * 7));
      const lmBars = meter ? meter.querySelector(".lm-bars") : null;
      if (lmBars) {
        lmBars.style.setProperty("--lv", lv.toFixed(3));
        const bars = lmBars.querySelectorAll("i");
        if (bars.length > 0) {
          this.levels.push(lv);
          if (this.levels.length > bars.length) this.levels.shift();
          for (let i = 0; i < bars.length; i++) {
            const v = this.levels[i] !== undefined ? this.levels[i] : lv;
            const h = Math.round(4 + v * 18);
            bars[i].style.height = h + "px";
          }
        }
      }

      if (this.playingCount > 0) {
        if (meter) meter.classList.add("speaking");
        if (stateEl) stateEl.textContent = "Gemini 말하는 중";
      } else {
        if (meter) meter.classList.remove("speaking");
        if (stateEl) {
          stateEl.textContent = rms > RMS_THRESHOLD ? "말소리 감지" : "듣는 중";
        }
      }
    },

    onMessage(ev) {
      if (ev.data instanceof ArrayBuffer) {
        this.playPcm(ev.data);
        return;
      }
      if (typeof ev.data === "string") {
        let msg;
        try {
          msg = JSON.parse(ev.data);
        } catch (_) {
          return;
        }
        this.handleJson(msg);
      }
    },

    playPcm(buf) {
      if (!this.playCtx || !buf || buf.byteLength < 2) return;
      if (this.playCtx.state === "suspended") {
        this.playCtx.resume();
      }
      const int16 = new Int16Array(buf);
      const float32 = new Float32Array(int16.length);
      for (let i = 0; i < int16.length; i++) {
        float32[i] = int16[i] / 32768;
      }

      const audioBuf = this.playCtx.createBuffer(1, float32.length, PLAY_SR);
      audioBuf.getChannelData(0).set(float32);

      const src = this.playCtx.createBufferSource();
      src.buffer = audioBuf;
      src.connect(this.playCtx.destination);

      const now = this.playCtx.currentTime;
      if (this.nextPlayTime < now) {
        this.nextPlayTime = now;
      }
      src.start(this.nextPlayTime);
      this.nextPlayTime += audioBuf.duration;

      this.playingCount++;
      const meter = document.getElementById("liveMeter");
      const stateEl = document.getElementById("liveState");
      if (meter) meter.classList.add("speaking");
      if (stateEl) stateEl.textContent = "Gemini 말하는 중";

      src.onended = () => {
        this.playingCount = Math.max(0, this.playingCount - 1);
        if (this.playingCount === 0) {
          if (meter) meter.classList.remove("speaking");
          if (stateEl && this.active) {
            stateEl.textContent = "듣는 중";
          }
        }
      };
    },

    handleJson(msg) {
      if (!msg || !msg.type) return;
      const stateEl = document.getElementById("liveState");
      const meter = document.getElementById("liveMeter");

      if (msg.type === "transcript") {
        const log = document.getElementById("liveLog");
        if (!log) return;
        log.hidden = false;
        const text = msg.text || "";
        if (msg.role === "model") {
          if (this.lastModelEl && this.lastModelEl.parentNode === log) {
            this.lastModelEl.textContent += text;
          } else {
            const line = document.createElement("div");
            line.className = "ll-model";
            line.textContent = "Gemini: " + text;
            log.appendChild(line);
            this.lastModelEl = line;
          }
        } else {
          this.lastModelEl = null;
          const line = document.createElement("div");
          line.className = "ll-user";
          line.textContent = "나: " + text;
          log.appendChild(line);
        }
        log.scrollTop = log.scrollHeight;
      } else if (msg.type === "locate") {
        this.lastModelEl = null;
        if (msg.session_id) {
          const st = window.S || (typeof S !== "undefined" ? S : null);
          if (st) st.sid = msg.session_id;
          if (typeof window.startPolling === "function") {
            window.startPolling();
          } else if (typeof startPolling === "function") {
            startPolling();
          }
        }
      } else if (msg.type === "turn_complete") {
        this.lastModelEl = null;
        if (this.playingCount === 0) {
          if (meter) meter.classList.remove("speaking");
          if (stateEl) stateEl.textContent = "듣는 중";
        }
      } else if (msg.type === "error") {
        const errFn =
          window.showError || (typeof showError === "function" ? showError : null);
        if (errFn) errFn(msg.message || "Live 오류");
      }
    },

    stop() {
      this.active = false;
      this.stopCapture();
      this.setButtonOn(false);

      const meter = document.getElementById("liveMeter");
      if (meter) {
        meter.hidden = true;
        meter.classList.remove("speaking");
      }

      const ws = this.ws;
      this.ws = null;
      if (ws) {
        try {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: "end" }));
          }
        } catch (_) {}
        this.closeTimer = setTimeout(() => {
          try {
            ws.close();
          } catch (_) {}
          this.cleanupPlay();
        }, 1000);
      } else {
        this.cleanupPlay();
      }
    },

    onDisconnect() {
      const wasActive = this.active;
      this.active = false;
      this.stopCapture();
      this.cleanupPlay();
      this.setButtonOn(false);
      const meter = document.getElementById("liveMeter");
      if (meter) meter.classList.remove("speaking");
      const stateEl = document.getElementById("liveState");
      if (stateEl) stateEl.textContent = "연결 끝";
      if (!wasActive && meter) {
        meter.hidden = true;
      }
    },

    stopCapture() {
      if (this.proc) {
        try {
          this.proc.onaudioprocess = null;
          this.proc.disconnect();
        } catch (_) {}
        this.proc = null;
      }
      if (this.source) {
        try {
          this.source.disconnect();
        } catch (_) {}
        this.source = null;
      }
      if (this.stream) {
        try {
          this.stream.getTracks().forEach((t) => t.stop());
        } catch (_) {}
        this.stream = null;
      }
      if (this.capCtx) {
        try {
          this.capCtx.close();
        } catch (_) {}
        this.capCtx = null;
      }
    },

    cleanupPlay() {
      this.playingCount = 0;
      this.nextPlayTime = 0;
      if (this.playCtx) {
        try {
          this.playCtx.close();
        } catch (_) {}
        this.playCtx = null;
      }
    },

    cleanupAudio() {
      this.stopCapture();
      this.cleanupPlay();
    },

    setButtonOn(on) {
      const btn = document.getElementById("live");
      if (!btn) return;
      btn.classList.toggle("on", !!on);
      const sp = btn.querySelector("span");
      if (sp) sp.textContent = on ? "대화 끝" : "Live 대화";
    },
  };

  window.Live = Live;

  document.addEventListener("DOMContentLoaded", () => {
    const btn = document.getElementById("live");
    if (btn) {
      btn.addEventListener("click", () => Live.toggle());
    }
  });
})();
