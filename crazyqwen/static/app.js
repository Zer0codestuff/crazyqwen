const $ = (id) => document.getElementById(id);
const ASPECTS = ["1:1", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16"];
const MEGAPIXEL = 1024 * 1024;
const STORE = "crazyqwen.form.v1";

const state = {
  images: [],
  selected: null,
  job: null,
  stepTimes: [],
  pollTimer: null,
};

/* ---------- form ---------- */

function buildAspects() {
  $("aspects").innerHTML = ASPECTS.map(
    (a, i) => `<label><input type="radio" name="aspect" value="${a}"${i === 0 ? " checked" : ""}><span>${a}</span></label>`
  ).join("");
}

function radio(name) {
  return document.querySelector(`input[name="${name}"]:checked`).value;
}

function setRadio(name, value) {
  const input = document.querySelector(`input[name="${name}"][value="${value}"]`);
  if (input) input.checked = true;
}

function dimensions() {
  const [aw, ah] = radio("aspect").split(":").map(Number);
  const area = Number(radio("size")) * MEGAPIXEL;
  const ratio = aw / ah;
  const snap = (v) => Math.max(256, Math.min(2048, Math.round(v / 16) * 16));
  return { width: snap(Math.sqrt(area * ratio)), height: snap(Math.sqrt(area / ratio)) };
}

function syncForm() {
  const { width, height } = dimensions();
  $("dims").textContent = `${width} × ${height}`;
  $("steps-out").textContent = $("steps").value;
  $("guidance-out").textContent = Number($("guidance").value).toFixed(1);
  const quality = radio("mode") === "quality";
  $("advanced").hidden = !quality;
  $("negative").disabled = Number($("guidance").value) <= 1;
  localStorage.setItem(STORE, JSON.stringify(readForm()));
}

function readForm() {
  const seed = $("seed").value.trim();
  return {
    prompt: $("prompt").value,
    mode: radio("mode"),
    aspect: radio("aspect"),
    size: radio("size"),
    seed: seed === "" ? null : Number(seed),
    steps: Number($("steps").value),
    guidance: Number($("guidance").value),
    negative_prompt: $("negative").value,
  };
}

function restoreForm(saved) {
  if (!saved) return;
  if (saved.prompt != null) $("prompt").value = saved.prompt;
  if (saved.mode) setRadio("mode", saved.mode);
  if (saved.aspect) setRadio("aspect", saved.aspect);
  if (saved.size) setRadio("size", saved.size);
  $("seed").value = saved.seed ?? "";
  if (saved.steps) $("steps").value = saved.steps;
  if (saved.guidance) $("guidance").value = saved.guidance;
  if (saved.negative_prompt != null) $("negative").value = saved.negative_prompt;
}

/* ---------- generation ---------- */

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    const detail = Array.isArray(body.detail) ? body.detail.map((d) => d.msg).join(", ") : body.detail;
    throw new Error(detail || `Request failed (${res.status})`);
  }
  return res.json();
}

async function submit(event) {
  event?.preventDefault();
  if (state.job) return cancel();
  const form = readForm();
  if (!form.prompt.trim()) return $("prompt").focus();
  const { width, height } = dimensions();
  try {
    const job = await api("/api/generate", {
      method: "POST",
      body: JSON.stringify({ ...form, width, height }),
    });
    track(job);
  } catch (err) {
    setStatus(err.message, true);
  }
}

async function cancel() {
  if (!state.job) return;
  $("generate").disabled = true;
  setStatus("Cancelling after the current step");
  await api(`/api/jobs/${state.job.id}/cancel`, { method: "POST" }).catch(() => {});
}

function track(job) {
  state.job = job;
  state.stepTimes = [];
  state.lastStep = 0;
  state.lastStepAt = performance.now();
  $("generate").textContent = "Cancel";
  $("generate").classList.add("cancel");
  $("canvas").classList.add("busy");
  $("progress").hidden = false;
  render(job);
  clearInterval(state.pollTimer);
  state.pollTimer = setInterval(poll, 700);
}

async function poll() {
  if (!state.job) return;
  try {
    const job = await api(`/api/jobs/${state.job.id}`);
    render(job);
    if (["done", "error", "cancelled"].includes(job.status)) finish(job);
  } catch (err) {
    setStatus(err.message, true);
  }
}

function render(job) {
  const p = job.progress || {};
  const total = p.total || job.request.steps;
  if (p.phase === "denoising" && p.step > state.lastStep) {
    if (p.step_seconds) state.stepTimes.push(p.step_seconds);
    state.lastStep = p.step;
  }
  const phases = {
    queued: "Waiting for the previous image",
    encoding: "Reading the prompt",
    loading: "Loading the model into memory",
    decoding: "Finishing the image",
  };
  let text = phases[p.phase] || "";
  let fraction = 0.04;
  if (p.phase === "denoising") {
    const avg = state.stepTimes.length
      ? state.stepTimes.reduce((a, b) => a + b, 0) / state.stepTimes.length
      : null;
    const left = avg ? `, about ${duration(avg * (total - p.step))} left` : "";
    text = `Step ${p.step} of ${total}${left}`;
    fraction = 0.08 + 0.87 * (p.step / total);
  } else if (p.phase === "decoding") {
    fraction = 0.97;
  }
  $("progress-fill").style.width = `${Math.round(fraction * 100)}%`;
  setStatus(text);
}

async function finish(job) {
  clearInterval(state.pollTimer);
  state.job = null;
  $("generate").textContent = "Generate";
  $("generate").classList.remove("cancel");
  $("generate").disabled = false;
  $("canvas").classList.remove("busy");
  $("progress").hidden = true;
  $("progress-fill").style.width = "0";
  if (job.status === "done") {
    await loadImages();
    select(job.image.id);
    setStatus(`Done in ${duration(job.image.timings.total)}`);
  } else if (job.status === "error") {
    setStatus(job.error, true);
  } else {
    setStatus("Cancelled");
  }
}

function setStatus(text, isError = false) {
  $("status").textContent = text;
  $("status").classList.toggle("error", isError);
}

function duration(seconds) {
  seconds = Math.max(0, Math.round(seconds));
  if (seconds < 60) return `${seconds} s`;
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return s ? `${m} min ${s} s` : `${m} min`;
}

/* ---------- gallery ---------- */

async function loadImages() {
  state.images = await api("/api/images");
  $("history").innerHTML = state.images
    .map(
      (img) =>
        `<button type="button" data-id="${img.id}" title="${escapeHtml(img.prompt)}">` +
        `<img src="/outputs/${img.file}" alt="" loading="lazy"></button>`
    )
    .join("");
}

function select(id) {
  const img = state.images.find((i) => i.id === id);
  state.selected = img || null;
  document.querySelectorAll(".history button").forEach((b) => {
    b.setAttribute("aria-current", String(b.dataset.id === id));
  });
  if (!img) {
    $("current").hidden = true;
    $("empty").hidden = false;
    $("details").hidden = true;
    return;
  }
  $("current").src = `/outputs/${img.file}`;
  $("current").alt = img.prompt;
  $("current").hidden = false;
  $("empty").hidden = true;
  $("details").hidden = false;
  $("d-prompt").textContent = img.prompt;
  const rows = [
    ["Mode", img.mode === "fast" ? "Fast" : "Quality"],
    ["Size", `${img.width} × ${img.height}`],
    ["Seed", img.seed],
    ["Steps", img.steps],
    img.guidance > 1 ? ["Guidance", img.guidance] : null,
    img.negative_prompt ? ["Negative", img.negative_prompt] : null,
    ["Time", duration(img.timings.total)],
  ].filter(Boolean);
  $("d-meta").innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${escapeHtml(String(v))}</dd>`).join("");
  $("d-download").href = `/outputs/${img.file}`;
  $("d-download").download = img.file;
}

function reuse() {
  const img = state.selected;
  if (!img) return;
  $("prompt").value = img.prompt;
  setRadio("mode", img.mode);
  $("seed").value = img.seed;
  if (img.mode === "quality") {
    $("steps").value = img.steps;
    $("guidance").value = img.guidance;
    $("negative").value = img.negative_prompt || "";
  }
  syncForm();
  $("prompt").focus();
}

async function remove() {
  const img = state.selected;
  if (!img) return;
  const index = state.images.indexOf(img);
  await api(`/api/images/${img.id}`, { method: "DELETE" });
  await loadImages();
  const next = state.images[Math.min(index, state.images.length - 1)];
  select(next ? next.id : null);
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

/* ---------- boot ---------- */

async function boot() {
  buildAspects();
  try {
    restoreForm(JSON.parse(localStorage.getItem(STORE)));
  } catch {}
  syncForm();

  $("form").addEventListener("submit", submit);
  $("form").addEventListener("input", syncForm);
  $("form").addEventListener("change", syncForm);
  $("seed-clear").addEventListener("click", () => {
    $("seed").value = "";
    syncForm();
  });
  $("prompt").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) submit(e);
  });
  $("history").addEventListener("click", (e) => {
    const button = e.target.closest("button[data-id]");
    if (button) select(button.dataset.id);
  });
  document.addEventListener("keydown", (e) => {
    if (e.target.matches("textarea, input")) return;
    if (!state.images.length || !["ArrowLeft", "ArrowRight"].includes(e.key)) return;
    const i = state.images.indexOf(state.selected);
    const next = state.images[Math.max(0, Math.min(state.images.length - 1, i + (e.key === "ArrowRight" ? 1 : -1)))];
    if (next) select(next.id);
  });
  $("d-reuse").addEventListener("click", reuse);
  $("d-delete").addEventListener("click", remove);

  await loadImages();
  if (state.images.length) select(state.images[0].id);

  const status = await api("/api/status");
  if (status.missing) {
    setStatus(`Not ready: ${status.missing}. Run ./run.sh in the project folder to download them.`, true);
  }
  const running = status.jobs.find((j) => j.status === "running") || status.jobs[0];
  if (running) track(running);
}

boot();
