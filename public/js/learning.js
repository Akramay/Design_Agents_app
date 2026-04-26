/**
 * learning.js
 * ───────────
 * Connects your existing EJS UI to the Python Flask backend.
 * Your learning.ejs stays 100% unchanged.
 *
 * Flow:
 *   1. Student drops file  → calls /api/setup   → Python ParserAgent runs
 *   2. Student answers     → calls /api/answer  → BKT + IRT + LLM run
 *   3. UI updates with question, thinking log, feedback, videos
 */

// ── Python API base URL ─────────────────────────────────────
// Change port if your Flask runs on a different one
const API_BASE = "http://localhost:3000";

// ── App state ───────────────────────────────────────────────
let currentFile      = null;
let sessionActive    = false;
let questionStartTime = null;   // when current question was shown
let currentQuestion  = null;    // full question object from Python
let isDark           = false;

// ═══════════════════════════════════════════════════════════
//  THEME
// ═══════════════════════════════════════════════════════════
function toggleTheme() {
  isDark = !isDark;
  document.documentElement.setAttribute("data-theme", isDark ? "dark" : "light");
  document.getElementById("themeLabel").textContent = isDark ? "Light mode" : "Dark mode";
  document.getElementById("themeIcon").innerHTML = isDark
    ? '<circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>'
    : '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>';
}

// ═══════════════════════════════════════════════════════════
//  FILE HANDLING
// ═══════════════════════════════════════════════════════════
function onDragOver(e) {
  e.preventDefault();
  document.getElementById("dropArea").classList.add("drag-over");
}
function onDragLeave(e) {
  document.getElementById("dropArea").classList.remove("drag-over");
}
function onDrop(e) {
  e.preventDefault();
  document.getElementById("dropArea").classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file) onFileSelect(file);
}

function onFileSelect(file) {
  if (!file) return;

  const allowed = ["pdf","ppt","pptx","doc","docx","txt"];
  const ext = file.name.split(".").pop().toLowerCase();
  if (!allowed.includes(ext)) {
    showNotif("Unsupported file type. Use PDF, PPT, PPTX, DOC, DOCX or TXT.");
    return;
  }

  currentFile = file;

  // show file preview card
  document.getElementById("fileName").textContent  = file.name;
  document.getElementById("fileSize").textContent  = (file.size / (1024*1024)).toFixed(2) + " MB";
  document.getElementById("fileIcon").textContent  = ext.toUpperCase();
  document.getElementById("filePreview").classList.remove("hidden");
  document.getElementById("genArea").classList.remove("hidden");
}

function removeFile() {
  currentFile = null;
  document.getElementById("filePreview").classList.add("hidden");
  document.getElementById("genArea").classList.add("hidden");
  document.getElementById("fileInput").value = "";
}

// ═══════════════════════════════════════════════════════════
//  START PROCESSING — uploads file to Python, gets first question
// ═══════════════════════════════════════════════════════════
async function startProcessing() {
  if (!currentFile) return;

  // switch to processing state
  showState("stateProcessing");
  animateProcessingSteps();

  try {
    // ── send file to Python backend ──────────────────────
    const formData = new FormData();
    formData.append("file", currentFile);

    const response = await fetch(`${API_BASE}/setup`, {
      method: "POST",
      body:   formData,
    });

    if (!response.ok) {
      throw new Error(`Server error: ${response.status}`);
    }

    const data = await response.json();

    // ── store session state ──────────────────────────────
    sessionActive   = true;
    currentQuestion = data.question;

    // ── populate the results view ────────────────────────
    populateResults(data);

    // switch to results state
    showState("stateResults");
    showNotif("Questions generated!");

  } catch (err) {
    console.error("Setup error:", err);
    showState("stateUpload");
    showNotif("Error connecting to Python server. Is Flask running?");
  }
}

// ═══════════════════════════════════════════════════════════
//  POPULATE RESULTS VIEW
// ═══════════════════════════════════════════════════════════
function populateResults(data) {
  // ── left panel: lecture info ─────────────────────────
  document.getElementById("docTitle").textContent     = currentFile.name;
  document.getElementById("docBadge").textContent     = currentFile.name.split(".").pop().toUpperCase();
  document.getElementById("metaWords").textContent    = data.word_count   || "—";
  document.getElementById("metaPages").textContent    = data.slide_count  || "—";
  document.getElementById("metaDiff").textContent     = "Adaptive";

  // show concept list in topics area
  const graph = data.concept_graph || [];
  const topicsList = document.getElementById("topicsList");
  topicsList.innerHTML = graph.map(c =>
    `<div class="topic-item">
       <span class="topic-dot"></span>
       <span>${c.concept}</span>
       <span class="topic-diff">${"★".repeat(c.difficulty)}${"☆".repeat(5 - c.difficulty)}</span>
     </div>`
  ).join("");

  document.getElementById("metaQCount").textContent  = graph.length;

  // ── right panel: first question ──────────────────────
  renderQuestion(data.question, data.current_concept, data.agent_thinking);
}

// ═══════════════════════════════════════════════════════════
//  RENDER CURRENT QUESTION + AGENT THINKING
// ═══════════════════════════════════════════════════════════
function renderQuestion(question, concept, thinkingLog) {
  if (!question) return;

  currentQuestion  = question;
  questionStartTime = Date.now();  // start timer

  const qList = document.getElementById("qList");
  document.getElementById("qCountBadge").textContent =
    `Concept: ${concept || "—"}`;

  // ── agent thinking section ───────────────────────────
  const thinkingHTML = thinkingLog && thinkingLog.length
    ? `<div class="thinking-log">
        <div class="thinking-title">🧠 Agent Thinking</div>
        ${thinkingLog.map(t =>
          `<div class="thinking-entry">
             <span class="thinking-agent">${t.agent}</span>
             <span class="thinking-msg">${t.message}</span>
           </div>`
        ).join("")}
       </div>`
    : "";

  // ── the question card ────────────────────────────────
  qList.innerHTML = `
    ${thinkingHTML}

    <div class="q-card active" id="activeQuestion">
      <div class="q-header">
        <div class="q-num">Current Question</div>
        <div class="q-diff-badge">${question.difficulty || "medium"}</div>
      </div>
      <div class="q-text">${question.question}</div>

      <div class="q-timer" id="qTimer">⏱ 0s / ${question.expected_time_seconds}s</div>

      <textarea
        id="studentAnswer"
        class="answer-input"
        placeholder="Type your answer here..."
        rows="4"
      ></textarea>

      <div class="q-actions">
        <button class="act-btn primary" onclick="submitAnswer()">
          Submit Answer
        </button>
        <button class="act-btn" onclick="requestHint()">
          💡 Hint
        </button>
      </div>

      <div id="feedbackArea" class="feedback-area hidden"></div>
    </div>

    <div class="q-card" id="historyLog">
      <div class="q-header">
        <div class="q-num">Progress</div>
      </div>
      <div id="progressBar" class="progress-wrap">
        <div class="progress-fill" id="progressFill" style="width:0%"></div>
      </div>
      <div id="historyEntries"></div>
    </div>
  `;

  // start the timer
  startTimer(question.expected_time_seconds);
}

// ═══════════════════════════════════════════════════════════
//  TIMER
// ═══════════════════════════════════════════════════════════
let timerInterval = null;

function startTimer(expectedSeconds) {
  if (timerInterval) clearInterval(timerInterval);
  let elapsed = 0;

  timerInterval = setInterval(() => {
    elapsed++;
    const el = document.getElementById("qTimer");
    if (!el) { clearInterval(timerInterval); return; }
    el.textContent = `⏱ ${elapsed}s / ${expectedSeconds}s`;

    // auto-show hint reminder if student is taking too long
    if (elapsed === Math.round(expectedSeconds * 1.5)) {
      el.style.color = "var(--color-text-warning)";
      el.textContent += "  — take your time or click Hint";
    }
  }, 1000);
}

function getElapsedSeconds() {
  if (!questionStartTime) return 30;
  return Math.round((Date.now() - questionStartTime) / 1000);
}

// ═══════════════════════════════════════════════════════════
//  SUBMIT ANSWER — sends to Python, renders response
// ═══════════════════════════════════════════════════════════
async function submitAnswer() {
  const answerText = document.getElementById("studentAnswer")?.value?.trim();
  if (!answerText) {
    showNotif("Please write an answer first.");
    return;
  }

  clearInterval(timerInterval);
  const timeTaken = getElapsedSeconds();

  // disable button while waiting
  const btn = document.querySelector("#activeQuestion .act-btn.primary");
  if (btn) { btn.disabled = true; btn.textContent = "Thinking..."; }

  try {
    const response = await fetch(`${API_BASE}/answer`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({
        answer:     answerText,
        time_taken: timeTaken,
      }),
    });

    const data = await response.json();

    // ── show feedback in the card ────────────────────────
    renderFeedback(data);

    // ── update history log ───────────────────────────────
    addToHistory(answerText, data);

    // ── update agent thinking ────────────────────────────
    if (data.agent_thinking) {
      const thinkEl = document.querySelector(".thinking-log");
      if (thinkEl) {
        thinkEl.innerHTML = `
          <div class="thinking-title">🧠 Agent Thinking</div>
          ${data.agent_thinking.map(t =>
            `<div class="thinking-entry">
               <span class="thinking-agent">${t.agent}</span>
               <span class="thinking-msg">${t.message}</span>
             </div>`
          ).join("")}`;
      }
    }

    // ── if next question available, show it after a delay ─
    if (data.action !== "SESSION_COMPLETE" &&
        data.action !== "SHOW_HINT" &&
        data.action !== "SHOW_EXPLANATION" &&
        data.action !== "RECOMMEND_VIDEO") {

      setTimeout(() => {
        renderQuestion(data.next_question, data.concept, data.agent_thinking);
      }, 3000);
    }

    if (data.action === "SESSION_COMPLETE") {
      setTimeout(() => showSessionComplete(data), 2000);
    }

  } catch (err) {
    console.error("Answer error:", err);
    showNotif("Error submitting answer. Check Python server.");
    if (btn) { btn.disabled = false; btn.textContent = "Submit Answer"; }
  }
}

// ═══════════════════════════════════════════════════════════
//  RENDER FEEDBACK (hint / explanation / videos)
// ═══════════════════════════════════════════════════════════
function renderFeedback(data) {
  const area = document.getElementById("feedbackArea");
  if (!area) return;

  area.classList.remove("hidden");
  area.innerHTML = "";

  // ── result badge ─────────────────────────────────────
  const isCorrect = data.action === "INCREASE_DIFFICULTY" ||
                    data.action === "NEXT_CONCEPT"         ||
                    data.action === "KEEP_LEVEL";

  const badge = document.createElement("div");
  badge.className = `result-badge ${isCorrect ? "correct" : "incorrect"}`;
  badge.textContent = isCorrect ? "✓ Correct" : "✗ Needs review";
  area.appendChild(badge);

  // ── agent message ────────────────────────────────────
  if (data.message) {
    const msg = document.createElement("div");
    msg.className = "agent-message";
    msg.textContent = data.message;
    area.appendChild(msg);
  }

  // ── hint ─────────────────────────────────────────────
  if (data.hint) {
    const h = document.createElement("div");
    h.className = "feedback-block hint-block";
    h.innerHTML = `<div class="fb-label">💡 Hint</div><div class="fb-text">${data.hint}</div>`;
    area.appendChild(h);
  }

  // ── explanation ──────────────────────────────────────
  if (data.explanation) {
    const e = document.createElement("div");
    e.className = "feedback-block explanation-block";
    e.innerHTML = `<div class="fb-label">📖 Explanation</div><div class="fb-text">${data.explanation}</div>`;
    area.appendChild(e);
  }

  // ── YouTube videos ───────────────────────────────────
  if (data.videos && data.videos.length > 0) {
    const v = document.createElement("div");
    v.className = "feedback-block video-block";
    v.innerHTML = `
      <div class="fb-label">🎬 Recommended Videos</div>
      ${data.videos.map(vid => `
        <a href="${vid.url}" target="_blank" class="video-link">
          ${vid.thumbnail
            ? `<img src="${vid.thumbnail}" class="video-thumb" alt="">`
            : ""}
          <span>${vid.title}</span>
        </a>
      `).join("")}
    `;
    area.appendChild(v);
  }
}

// ═══════════════════════════════════════════════════════════
//  REQUEST HINT (before submitting)
// ═══════════════════════════════════════════════════════════
async function requestHint() {
  const timeTaken = getElapsedSeconds();

  // send a placeholder answer just to trigger SHOW_HINT
  try {
    const response = await fetch(`${API_BASE}/hint`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ time_taken: timeTaken }),
    });
    const data = await response.json();
    if (data.hint) {
      const area = document.getElementById("feedbackArea");
      area.classList.remove("hidden");
      area.innerHTML = `
        <div class="feedback-block hint-block">
          <div class="fb-label">💡 Hint</div>
          <div class="fb-text">${data.hint}</div>
        </div>`;
    }
  } catch (err) {
    showNotif("Could not get hint right now.");
  }
}

// ═══════════════════════════════════════════════════════════
//  HISTORY LOG
// ═══════════════════════════════════════════════════════════
let historyCount   = 0;
let correctCount   = 0;

function addToHistory(answer, data) {
  historyCount++;
  const wasCorrect = !["SHOW_EXPLANATION","RECOMMEND_VIDEO","DECREASE_DIFFICULTY"]
                      .includes(data.action);
  if (wasCorrect) correctCount++;

  // update progress bar
  const pct = Math.round((correctCount / historyCount) * 100);
  const fill = document.getElementById("progressFill");
  if (fill) fill.style.width = pct + "%";

  // add entry
  const entries = document.getElementById("historyEntries");
  if (!entries) return;

  const entry = document.createElement("div");
  entry.className = `history-entry ${wasCorrect ? "correct" : "incorrect"}`;
  entry.innerHTML = `
    <span class="h-num">#${historyCount}</span>
    <span class="h-answer">${answer.substring(0, 60)}${answer.length > 60 ? "…" : ""}</span>
    <span class="h-result">${wasCorrect ? "✓" : "✗"}</span>
  `;
  entries.prepend(entry);
}

// ═══════════════════════════════════════════════════════════
//  SESSION COMPLETE
// ═══════════════════════════════════════════════════════════
function showSessionComplete(data) {
  const qList = document.getElementById("qList");
  qList.innerHTML = `
    <div class="q-card complete-card">
      <div class="complete-icon">🎉</div>
      <div class="complete-title">All Concepts Mastered!</div>
      <div class="complete-stats">
        <div class="stat-item">
          <div class="stat-val">${historyCount}</div>
          <div class="stat-label">Questions answered</div>
        </div>
        <div class="stat-item">
          <div class="stat-val">${correctCount}</div>
          <div class="stat-label">Correct answers</div>
        </div>
        <div class="stat-item">
          <div class="stat-val">${Math.round((correctCount/historyCount)*100)}%</div>
          <div class="stat-label">Accuracy</div>
        </div>
      </div>
    </div>
  `;
}

// ═══════════════════════════════════════════════════════════
//  CHAT (ask about lecture)
// ═══════════════════════════════════════════════════════════
function onChatKey(e) {
  if (e.key === "Enter") sendChat();
}

async function sendChat() {
  const input = document.getElementById("chatInput");
  const text  = input.value.trim();
  if (!text) return;
  input.value = "";

  try {
    const response = await fetch(`${API_BASE}/chat`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ message: text }),
    });
    const data = await response.json();
    showNotif("Chat: " + (data.reply || "").substring(0, 60) + "…");
  } catch {
    showNotif("Chat unavailable right now.");
  }
}

// ═══════════════════════════════════════════════════════════
//  TOOLBAR BUTTONS
// ═══════════════════════════════════════════════════════════
function regenerate() {
  if (!sessionActive) return;
  showNotif("Regenerating questions…");
  startProcessing();
}

function makeHarder() {
  showNotif("Increasing difficulty for next question…");
  fetch(`${API_BASE}/harder`, { method: "POST" })
    .then(r => r.json())
    .then(data => {
      if (data.question) renderQuestion(data.question, data.concept, data.agent_thinking);
    })
    .catch(() => showNotif("Could not change difficulty."));
}

function revealAll() {
  document.querySelectorAll(".answer-reveal").forEach(el => {
    el.classList.remove("hidden");
  });
  showNotif("Answers revealed!");
}

function expandAll() {
  document.querySelectorAll(".q-card").forEach(el => {
    el.classList.add("expanded");
  });
}

function exportPDF() {
  showNotif("Export feature coming soon!");
}

function resetApp() {
  sessionActive    = false;
  currentFile      = null;
  currentQuestion  = null;
  historyCount     = 0;
  correctCount     = 0;
  clearInterval(timerInterval);

  document.getElementById("fileInput").value = "";
  document.getElementById("filePreview").classList.add("hidden");
  document.getElementById("genArea").classList.add("hidden");
  document.getElementById("qList").innerHTML = "";

  showState("stateUpload");

  // tell Python to reset session
  fetch(`${API_BASE}/reset`, { method: "POST" }).catch(() => {});
}

// ═══════════════════════════════════════════════════════════
//  STATE SWITCHER
// ═══════════════════════════════════════════════════════════
function showState(id) {
  document.querySelectorAll(".state").forEach(el => el.classList.remove("active"));
  document.getElementById(id).classList.add("active");
}

// ═══════════════════════════════════════════════════════════
//  PROCESSING ANIMATION
// ═══════════════════════════════════════════════════════════
function animateProcessingSteps() {
  const steps = document.querySelectorAll(".proc-step");
  let i = 0;

  const advance = () => {
    if (i > 0 && steps[i-1]) {
      steps[i-1].classList.remove("active");
      steps[i-1].classList.add("done");
      steps[i-1].querySelector(".spin-ring, .pending-dot").outerHTML =
        '<svg class="step-check" viewBox="0 0 24 24"><polyline points="20 6 9 17 4 12"/></svg>';
    }
    if (i < steps.length) {
      steps[i].classList.add("active");
      const dot = steps[i].querySelector(".pending-dot");
      if (dot) dot.className = "spin-ring";
      i++;
      setTimeout(advance, 900);
    }
  };
  advance();
}

// ═══════════════════════════════════════════════════════════
//  NOTIFICATION TOAST
// ═══════════════════════════════════════════════════════════
let notifTimer = null;
function showNotif(text) {
  const el = document.getElementById("notif");
  document.getElementById("notifText").textContent = text;
  el.classList.remove("hide");
  clearTimeout(notifTimer);
  notifTimer = setTimeout(() => el.classList.add("hide"), 3500);
}
