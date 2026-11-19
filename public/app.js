(() => {
  const $ = (s, r = document) => r.querySelector(s);
  const messages = $("#messages"), chat = $("#chat"), input = $("#input"), send = $("#send"), hero = $("#hero");
  const store = {
    get(k) { try { return localStorage.getItem(k); } catch { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch {} },
    del(k) { try { localStorage.removeItem(k); } catch {} },
  };
  let sessionId = store.get("scout.session");
  let busy = false;

  const PALETTE = ["#7df9c9", "#8b7cff", "#ffc36b", "#64d6f5", "#ff9ec7", "#b6f36b"];
  const PREF_LABELS = { role: "role", location: "location", remote: "remote", min_salary: "min salary", skills: "skills", seniority: "level", job_type: "type", avoid: "avoid" };
  const TOOL_LABELS = { search_jobs: "semantic search", filter_jobs: "filter", get_job_details: "job details", save_preference: "remember" };
  const STEPS = ["Understanding your request", "Searching the job index", "Ranking the best matches", "Checking every pick"];

  const el = (tag, cls, text) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  };

  function toast(msg) {
    const t = $("#toast");
    t.textContent = msg;
    t.classList.add("show");
    clearTimeout(toast.t);
    toast.t = setTimeout(() => t.classList.remove("show"), 4200);
  }

  function scrollDown() { requestAnimationFrame(() => { chat.scrollTop = chat.scrollHeight; }); }

  function money(n, cur) {
    if (!n) return "";
    const sym = { USD: "$", EUR: "€", GBP: "£", INR: "₹" }[cur] || (cur ? cur + " " : "");
    return n >= 1000 ? `${sym}${Math.round(n / 1000)}k` : `${sym}${n}`;
  }

  function ago(iso) {
    if (!iso) return "";
    const d = (Date.now() - new Date(iso).getTime()) / 864e5;
    if (d < 1) return "today";
    if (d < 2) return "yesterday";
    if (d < 30) return `${Math.floor(d)} days ago`;
    return new Date(iso).toLocaleDateString(undefined, { day: "numeric", month: "short" });
  }

  function hash(s) { let h = 0; for (const c of s) h = (h * 31 + c.charCodeAt(0)) | 0; return Math.abs(h); }

  function renderPrefs(prefs = {}) {
    const box = $("#prefs");
    const keys = Object.keys(prefs);
    $("#prefCount").textContent = keys.length;
    box.innerHTML = "";
    if (!keys.length) {
      box.append(el("p", "muted small", "Tell me your skills, location or what to avoid and I'll keep it in mind."));
      return;
    }
    keys.forEach((k) => {
      const chip = el("span", "pref" + (k === "avoid" ? " pref--avoid" : ""));
      chip.append(el("b", null, PREF_LABELS[k] || k), document.createTextNode(prefs[k]));
      box.append(chip);
    });
  }

  function jobCard(job, why) {
    const node = $("#tplJob").content.firstElementChild.cloneNode(true);
    const color = PALETTE[hash(job.company) % PALETTE.length];
    const logo = $(".job__logo", node);
    logo.textContent = (job.company || "?").trim().charAt(0).toUpperCase();
    logo.style.background = `linear-gradient(135deg, ${color}, ${color}99)`;
    $(".job__title", node).textContent = job.title;
    $(".job__company", node).textContent = job.company;
    $(".job__why", node).textContent = why || "";
    if (!why) $(".job__why", node).remove();

    const meta = $(".job__meta", node);
    if (job.remote) meta.append(el("span", "chip chip--remote", "● Remote"));
    if (job.location && !(job.remote && /^remote$/i.test(job.location))) meta.append(el("span", "chip", job.location));
    if (job.job_type) meta.append(el("span", "chip", job.job_type.replace(/_/g, " ")));
    if (job.salary_min || job.salary_max) {
      const lo = money(job.salary_min, job.salary_currency), hi = money(job.salary_max, job.salary_currency);
      meta.append(el("span", "chip chip--money", lo && hi && lo !== hi ? `${lo} – ${hi}` : lo || hi));
    }
    const tags = $(".job__tags", node);
    (job.tags || []).slice(0, 5).forEach((t) => tags.append(el("span", null, t)));

    const score = $(".job__score", node);
    if (typeof job.score === "number") {
      const pct = Math.max(0, Math.min(100, Math.round(job.score * 100)));
      $("span", score).textContent = pct;
      score.title = `Match score ${pct}/100`;
      requestAnimationFrame(() => requestAnimationFrame(() => {
        $(".arc", score).style.strokeDashoffset = String(97.4 * (1 - pct / 100));
      }));
    } else score.remove();

    $(".job__posted", node).textContent = ago(job.posted_at);
    const apply = $(".job__apply", node);
    if (job.url) apply.href = job.url; else apply.remove();

    node.addEventListener("pointermove", (e) => {
      const b = node.getBoundingClientRect();
      node.style.setProperty("--mx", `${e.clientX - b.left}px`);
      node.style.setProperty("--my", `${e.clientY - b.top}px`);
    });
    return node;
  }

  function addUser(text) {
    const row = el("div", "msg msg--user");
    row.append(el("div", "bubble", text));
    messages.append(row);
    scrollDown();
  }

  function addTyping() {
    const row = el("div", "msg typing");
    row.append(el("div", "avatar", "S"));
    const wrap = el("div", "reply");
    const box = el("div", "typing__box");
    const dots = el("span", "typing__dots");
    dots.append(el("i"), el("i"), el("i"));
    const label = el("span", null, STEPS[0]);
    box.append(dots, label);
    const sk = el("div", "skeleton");
    sk.append(el("div"), el("div"), el("div"));
    wrap.append(box, sk);
    row.append(wrap);
    messages.append(row);
    let i = 0;
    const timer = setInterval(() => { i = Math.min(i + 1, STEPS.length - 1); label.textContent = STEPS[i]; }, 1700);
    scrollDown();
    return () => { clearInterval(timer); row.remove(); };
  }

  function addReply(data) {
    const row = el("div", "msg");
    row.append(el("div", "avatar", "S"));
    const body = el("div", "reply");
    body.append(el("p", "reply__text", data.reply.message));

    if (data.jobs.length) {
      const grid = el("div", "jobs");
      const why = Object.fromEntries(data.reply.picks.map((p) => [p.job_id, p.why]));
      data.jobs.forEach((job, i) => {
        const card = jobCard(job, why[job.id]);
        card.style.animationDelay = `${i * 90}ms`;
        grid.append(card);
      });
      body.append(grid);
    }

    if (data.reply.follow_up) {
      const f = el("button", "follow", data.reply.follow_up);
      f.prepend(el("span", null, "↳"));
      f.addEventListener("click", () => { input.value = ""; input.placeholder = data.reply.follow_up; input.focus(); });
      body.append(f);
    }

    if (data.trace.length || data.dropped_picks.length) {
      const d = el("details", "trace");
      const total = data.trace.reduce((s, t) => s + t.ms, 0);
      d.append(el("summary", null, `How I found these · ${data.trace.length} tool call${data.trace.length === 1 ? "" : "s"} · ${total} ms${data.model ? " · " + data.model : ""}`));
      const ol = el("ol");
      data.trace.forEach((t) => {
        const li = el("li", t.ok ? "" : "bad");
        li.append(el("b", null, TOOL_LABELS[t.tool] || t.tool), el("code", null, JSON.stringify(t.args)), el("span", null, `${t.results} · ${t.ms}ms`));
        ol.append(li);
      });
      if (data.dropped_picks.length) {
        const li = el("li");
        li.append(el("b", "guard", "guard"), el("code", null, `removed ${data.dropped_picks.length} pick(s) not found in search results`), el("span", null, ""));
        ol.append(li);
      }
      d.append(ol);
      body.append(d);
    }

    row.append(body);
    messages.append(row);
    scrollDown();
  }

  async function ask(text) {
    if (busy || !text.trim()) return;
    busy = true;
    send.disabled = true;
    hero.classList.add("gone");
    addUser(text.trim());
    input.value = "";
    autosize();
    const stop = addTyping();
    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text.trim(), session_id: sessionId }),
      });
      const data = await res.json().catch(() => ({}));
      stop();
      if (!res.ok) throw new Error(data.detail || `Request failed (${res.status})`);
      sessionId = data.session_id;
      store.set("scout.session", sessionId);
      addReply(data);
      renderPrefs(data.preferences);
    } catch (err) {
      stop();
      toast(err.message || "Something went wrong. Try again.");
    } finally {
      busy = false;
      send.disabled = false;
      input.focus();
    }
  }

  async function loadStats() {
    try {
      const res = await fetch("/api/stats");
      if (!res.ok) throw new Error();
      const s = await res.json();
      $('[data-stat="jobs"]').textContent = s.jobs.toLocaleString();
      $('[data-stat="remote"]').textContent = s.remote.toLocaleString();
      $("#statsMeta").textContent = `${s.indexed.toLocaleString()} embedded · ${s.sources.join(", ") || "no sources yet"}${s.latest ? " · newest " + ago(s.latest) : ""}`;
    } catch {
      $("#status").classList.add("off");
      $("#status").lastChild.textContent = "Offline";
      $("#statsMeta").textContent = "Job index unavailable right now.";
    }
  }

  async function restore() {
    if (!sessionId) return;
    try {
      const res = await fetch(`/api/session/${sessionId}`);
      if (!res.ok) throw new Error();
      const s = await res.json();
      renderPrefs(s.preferences);
      if (s.history.length) {
        hero.classList.add("gone");
        s.history.forEach((m) => {
          if (m.role === "user") addUser(m.content);
          else {
            const row = el("div", "msg");
            row.append(el("div", "avatar", "S"));
            const r = el("div", "reply");
            r.append(el("p", "reply__text", m.content.replace(/\nShown: .*/s, "")));
            row.append(r);
            messages.append(row);
          }
        });
        scrollDown();
      }
    } catch {
      sessionId = null;
      store.del("scout.session");
    }
  }

  function reset() {
    sessionId = null;
    store.del("scout.session");
    messages.innerHTML = "";
    hero.classList.remove("gone");
    renderPrefs({});
    closeSide();
    input.focus();
  }

  function autosize() {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 160) + "px";
    send.disabled = busy || !input.value.trim();
  }

  const side = $("#side"), scrim = $("#scrim");
  const closeSide = () => { side.classList.remove("open"); scrim.classList.remove("show"); };
  $("#menu").addEventListener("click", () => { side.classList.add("open"); scrim.classList.add("show"); });
  scrim.addEventListener("click", closeSide);

  $("#composer").addEventListener("submit", (e) => { e.preventDefault(); ask(input.value); });
  input.addEventListener("input", autosize);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(input.value); }
  });
  $("#suggest").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) ask(b.dataset.q); });
  $("#newChat").addEventListener("click", reset);
  $("#newChatTop").addEventListener("click", reset);

  autosize();
  loadStats();
  restore();
})();
