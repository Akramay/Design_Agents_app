'use strict';

const App = {
  selectedFile: null,
  questionStartedAt: null,
  procTimers: [],
  notifTimer: null,
  session: null,
  sessionId: null,  // Current session ID
  hintUsed: false,
  confirmAction: null,  // Function to execute when modal is confirmed
};

// ═══════════════════════════════════════════════════════════
//  SESSION MANAGEMENT (auto-clears when browser closes)
// ═══════════════════════════════════════════════════════════

function getSessionId() {
  // sessionStorage clears when browser/tab closes - perfect for our use case!
  return sessionStorage.getItem('adaptive_tutor_session_id');
}

function setSessionId(sessionId) {
  sessionStorage.setItem('adaptive_tutor_session_id', sessionId);
  App.sessionId = sessionId;
}

function clearSessionId() {
  sessionStorage.removeItem('adaptive_tutor_session_id');
  App.sessionId = null;
}

function hasActiveSession() {
  return getSessionId() !== null;
}

// ═══════════════════════════════════════════════════════════
//  UI FUNCTIONS
// ═══════════════════════════════════════════════════════════

function toggleTheme() {
  const isDark = document.body.classList.toggle('dark');
  document.getElementById('themeLabel').textContent = isDark ? 'Light mode' : 'Dark mode';
  localStorage.setItem('color-theme', isDark ? 'dark' : 'light');
  const icon = document.getElementById('themeIcon');
  icon.innerHTML = isDark
    ? `<circle cx="12" cy="12" r="5"/>
       <line x1="12" y1="1" x2="12" y2="3"/>
       <line x1="12" y1="21" x2="12" y2="23"/>
       <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/>
       <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>
       <line x1="1" y1="12" x2="3" y2="12"/>
       <line x1="21" y1="12" x2="23" y2="12"/>
       <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/>
       <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>`
    : `<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>`;
}

function onDragOver(e) {
  e.preventDefault();
  document.getElementById('dropArea').classList.add('dragging');
}

function onDragLeave() {
  document.getElementById('dropArea').classList.remove('dragging');
}

function onDrop(e) {
  e.preventDefault();
  document.getElementById('dropArea').classList.remove('dragging');
  const file = e.dataTransfer.files[0];
  if (file) onFileSelect(file);
}

function onFileSelect(file) {
  if (!file) return;
  App.selectedFile = file;

  const ext = file.name.split('.').pop().toLowerCase();
  const extUpper = ext.toUpperCase();
  const sizeMB = (file.size / 1_048_576).toFixed(1);
  const iconClass =
    ext === 'pdf' ? 'pdf'
      : ext === 'ppt' || ext === 'pptx' ? 'ppt'
        : ext === 'doc' || ext === 'docx' ? 'doc'
          : 'default';

  const iconEl = document.getElementById('fileIcon');
  iconEl.textContent = extUpper;
  iconEl.className = `file-icon-box ${iconClass}`;

  document.getElementById('fileName').textContent = file.name;
  document.getElementById('fileSize').textContent = `${sizeMB} MB`;
  document.getElementById('filePreview').classList.remove('hidden');
  document.getElementById('genArea').classList.remove('hidden');
}

function removeFile() {
  App.selectedFile = null;
  document.getElementById('fileInput').value = '';
  document.getElementById('filePreview').classList.add('hidden');
  document.getElementById('genArea').classList.add('hidden');
}

function setState(id) {
  ['stateUpload', 'stateProcessing', 'stateResults'].forEach((stateId) => {
    document.getElementById(stateId).classList.remove('active');
  });
  document.getElementById(id).classList.add('active');
}

function resetProcessingAnimation() {
  App.procTimers.forEach((timer) => clearTimeout(timer));
  App.procTimers = [];
}

function runProcessingAnimation() {
  const stepIds = ['procStep1', 'procStep2', 'procStep3', 'procStep4'];
  const delays = [0, 1100, 2200, 3300];

  stepIds.forEach((id, index) => {
    const el = document.getElementById(id);
    el.classList.remove('active');
    el.style.opacity = index === 0 ? '1' : '0.4';
    el.querySelector('.step-icon').innerHTML = `<div class="${index === 0 ? 'spin-ring' : 'pending-dot'}"></div>`;
  });

  stepIds.forEach((id, index) => {
    const activate = setTimeout(() => {
      const el = document.getElementById(id);
      el.classList.add('active');
      el.style.opacity = '1';
      el.querySelector('.step-icon').innerHTML = '<div class="spin-ring"></div>';
    }, delays[index]);

    const complete = setTimeout(() => {
      const el = document.getElementById(id);
      el.classList.remove('active');
      el.querySelector('.step-icon').innerHTML =
        `<div class="check-circle">
           <svg viewBox="0 0 24 24"><polyline points="20 6 9 17 4 12"/></svg>
         </div>`;
    }, delays[index] + 900);

    App.procTimers.push(activate, complete);
  });
}

// ═══════════════════════════════════════════════════════════
//  API FUNCTIONS
// ═══════════════════════════════════════════════════════════

async function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = String(reader.result || '');
      const base64 = result.includes(',') ? result.split(',')[1] : result;
      resolve(base64);
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

async function apiFetch(url, options = {}) {
  const response = await fetch(url, {
    headers: {
      'Content-Type': 'application/json',
      ...(options.headers || {}),
    },
    ...options,
  });

  const data = await response.json();
  if (!response.ok || !data.ok) {
    throw new Error(data.error || 'Request failed.');
  }

  return data;
}

async function startProcessing() {
  if (!App.selectedFile) {
    showNotif('Choose a lecture file first.');
    return;
  }

  const button = document.getElementById('genBtn');
  button.disabled = true;
  setState('stateProcessing');
  resetProcessingAnimation();
  runProcessingAnimation();

  try {
    const base64 = await fileToBase64(App.selectedFile);
    const data = await apiFetch('/learn/api/session/setup', {
      method: 'POST',
      body: JSON.stringify({
        fileName: App.selectedFile.name,
        fileData: base64,
      }),
    });

    // Store the new session ID
    setSessionId(data.session_id);
    
    applySession(data.session, data.decision);
    setState('stateResults');
    showNotif('Adaptive session is ready.');
  } catch (error) {
    setState('stateUpload');
    showNotif(error.message || 'Could not start the session.');
  } finally {
    button.disabled = false;
  }
}

function applySession(session, decision) {
  App.session = session;
  App.questionStartedAt = Date.now();
  App.hintUsed = false;

  const question = session.current_question || {};
  const conceptState = session.current_concept_state || {};
  const progress = session.progress || {};

  document.getElementById('docBadge').textContent = fileExtensionLabel(session.lecture_title);
  document.getElementById('docTitle').textContent = session.lecture_title || 'Lecture session';
  document.getElementById('sessionSummary').textContent =
    session.current_summary || 'The parser extracted concepts and the tutor is ready to adapt question difficulty.';

  document.getElementById('metaConcepts').textContent = progress.concept_count ?? '-';
  document.getElementById('metaMastered').textContent = `${progress.mastered_count ?? 0}`;
  document.getElementById('metaAnswered').textContent = session.total_questions ?? 0;
  document.getElementById('metaAccuracy').textContent = `${Math.round((progress.accuracy || 0) * 100)}%`;

  document.getElementById('currentConceptName').textContent = session.current_concept || '-';
  document.getElementById('bktValue').textContent = formatNumber(conceptState.bkt_p_learned);
  document.getElementById('thetaValue').textContent = formatSigned(conceptState.irt_theta);
  document.getElementById('difficultyValue').textContent = formatSigned(question.b);

  document.getElementById('questionConceptPill').textContent = session.current_concept || 'Concept';
  document.getElementById('questionDifficultyPill').textContent = question.difficulty || 'Adaptive';
  
  // Render question based on type
  renderQuestion(question);

  document.getElementById('qCountBadge').textContent =
    `Expected ${question.expected_time_seconds || 0}s`;

  // Show feedback message with correct/wrong indicator
  renderFeedbackMessage(session, decision);

  document.getElementById('resultsTitle').textContent =
    session.current_concept ? `Working on ${session.current_concept}` : 'Adaptive session ready';
  document.getElementById('resultsSubtitle').textContent =
    `Current question difficulty is ${question.difficulty || 'adaptive'} and the next step depends on the student's answer.`;

  // HIDE concept graph as requested
  hideConceptGraph();
  
  renderThinking(session.agent_thinking || []);
  renderFeedback(session);
  
  updateTimerHint(question.expected_time_seconds);
  updateHintButton(session.hint_available);
}

function renderQuestion(question) {
  const questionType = question.type || 'essay';
  const promptDiv = document.getElementById('questionPrompt');
  const contextDiv = document.getElementById('questionContext');
  const answerArea = document.getElementById('answerInput');

  promptDiv.textContent = question.question || 'Upload a lecture to begin.';

  // Always fully reset both elements first so MCQ->essay (or vice versa)
  // never leaves stale radio buttons or a hidden textarea behind.
  contextDiv.innerHTML = '';
  if (answerArea) {
    answerArea.value = '';
    answerArea.style.display = 'none';
  }

  if (questionType === 'mcq') {
    const options = question.options || [];
    contextDiv.innerHTML = `
      <div class="mcq-options" id="mcqOptions">
        ${options.map((opt, idx) => `
          <label class="mcq-option">
            <input type="radio" name="mcq_answer" value="${escapeAttribute(opt)}">
            <span class="option-letter">${String.fromCharCode(65 + idx)}</span>
            <span class="option-text">${escapeHtml(opt)}</span>
          </label>
        `).join('')}
      </div>
    `;
    // answerArea stays hidden (already set above)
  } else {
    // Essay - clear any leftover MCQ markup and show textarea
    const keyPoints = question.key_points || [];
    contextDiv.textContent = keyPoints.length
      ? `Your answer should address: ${keyPoints.join(', ')}.`
      : 'Write your answer in the box below.';

    if (answerArea) {
      answerArea.style.display = 'block';
      answerArea.focus();
    }
  }
}

function renderFeedbackMessage(session, decision) {
  const feedbackDiv = document.getElementById('feedbackMessage');
  
  // Check if this is a response to an answer
  if (session.last_answer_correct !== null && session.last_answer_correct !== undefined) {
    const correct = session.last_answer_correct;
    const correctClass = correct ? 'feedback-correct' : 'feedback-wrong';
    const correctText = correct ? 'CORRECT' : 'WRONG';
    
    const decisionMessage = decision?.message_to_student
      || session.llm_decision?.message_to_student
      || (correct ? 'Good work! Moving forward.' : 'Let\'s review this concept.');

    // Build correct answer block (shown for both correct and wrong answers)
    const question = session.current_question || session.last_question || {};
    let correctAnswerHtml = '';
    if (question.correct_answer) {
      correctAnswerHtml = `
        <div class="correct-answer-block">
          <span class="correct-answer-label">Correct answer:</span>
          <span class="correct-answer-text">${escapeHtml(question.correct_answer)}</span>
        </div>
      `;
    }
    
    feedbackDiv.innerHTML = `
      <div class="answer-result ${correctClass}">
        <div class="result-content">
          <div class="result-badge">${correctText}</div>
          <div class="result-message">${escapeHtml(decisionMessage)}</div>
          ${correctAnswerHtml}
        </div>
      </div>
    `;
  } else {
    // No answer submitted yet
    const decisionMessage = decision?.message_to_student
      || session.llm_decision?.message_to_student
      || 'Answer the question to let the tutor adapt.';
    feedbackDiv.innerHTML = `<div class="feedback-waiting">${escapeHtml(decisionMessage)}</div>`;
  }
}

function hideConceptGraph() {
  const conceptSection = document.querySelector('.section-block');
  if (conceptSection && conceptSection.querySelector('.concept-list')) {
    conceptSection.style.display = 'none';
  }
}

function renderThinking(entries) {
  const container = document.getElementById('thinkingList');
  container.innerHTML = '';

  if (!entries.length) {
    container.innerHTML = '<div class="empty-note">Agent reasoning will appear here after setup.</div>';
    return;
  }

  entries.forEach((entry) => {
    const item = document.createElement('div');
    item.className = 'thinking-item';
    item.innerHTML = `
      <div class="thinking-head">
        <strong>${escapeHtml(entry.agent || 'Agent')}</strong>
        <span>${escapeHtml(entry.timestamp || '')}</span>
      </div>
      <div class="thinking-body">${escapeHtml(entry.message || '')}</div>
    `;
    container.appendChild(item);
  });
}

function renderFeedback(session) {
  const hintBlock = document.getElementById('hintBlock');
  const explanationBlock = document.getElementById('explanationBlock');
  const videoList = document.getElementById('videoList');

  if (session.hint) {
    hintBlock.classList.remove('hidden');
    hintBlock.innerHTML = `
      <div class="feedback-label">
        <svg viewBox="0 0 24 24" width="16" height="16">
          <circle cx="12" cy="12" r="10"/>
          <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/>
          <line x1="12" y1="17" x2="12.01" y2="17"/>
        </svg>
        Hint
      </div>
      <div class="feedback-content">${escapeHtml(session.hint)}</div>
    `;
  } else {
    hintBlock.classList.add('hidden');
    hintBlock.innerHTML = '';
  }

  if (session.explanation) {
    explanationBlock.classList.remove('hidden');
    explanationBlock.innerHTML = `
      <div class="feedback-label">
        <svg viewBox="0 0 24 24" width="16" height="16">
          <circle cx="12" cy="12" r="10"/>
          <line x1="12" y1="16" x2="12" y2="12"/>
          <line x1="12" y1="8" x2="12.01" y2="8"/>
        </svg>
        Explanation
      </div>
      <div class="feedback-content">${escapeHtml(session.explanation)}</div>
    `;
  } else {
    explanationBlock.classList.add('hidden');
    explanationBlock.innerHTML = '';
  }

  if ((session.videos || []).length) {
    videoList.classList.remove('hidden');
    videoList.innerHTML = `
      <div class="feedback-label">
        <svg viewBox="0 0 24 24" width="16" height="16">
          <polygon points="5 3 19 12 5 21 5 3"/>
        </svg>
        Recommended Videos
      </div>
      ${session.videos.map((video) => `
        <a class="video-card" href="${escapeAttribute(video.url || '#')}" target="_blank" rel="noreferrer">
          <strong>${escapeHtml(video.title || 'Video')}</strong>
          <span>${escapeHtml(video.channel || 'Open recommendation')}</span>
        </a>
      `).join('')}
    `;
  } else {
    videoList.classList.add('hidden');
    videoList.innerHTML = '';
  }
}

function updateHintButton(hintAvailable) {
  let hintBtn = document.getElementById('hintBtn');
  if (!hintBtn) {
    const answerActions = document.querySelector('.answer-actions');
    if (answerActions) {
      hintBtn = document.createElement('button');
      hintBtn.id = 'hintBtn';
      hintBtn.className = 'act-btn hint-btn';
      hintBtn.onclick = requestHint;
      hintBtn.innerHTML = `
        <svg viewBox="0 0 24 24">
          <circle cx="12" cy="12" r="10"/>
          <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/>
          <line x1="12" y1="17" x2="12.01" y2="17"/>
        </svg>
        Get Hint
      `;
      answerActions.insertBefore(hintBtn, answerActions.querySelector('.primary'));
    }
  }
  
  if (hintBtn) {
    hintBtn.disabled = !hintAvailable;
    hintBtn.style.display = hintAvailable ? 'flex' : 'none';
  }
}

async function requestHint() {
  const sessionId = getSessionId();
  if (!sessionId) {
    showNotif('No active session.');
    return;
  }

  try {
    const data = await apiFetch('/learn/api/session/hint', {
      method: 'POST',
      body: JSON.stringify({ session_id: sessionId }),
    });

    App.hintUsed = true;
    applySession(data.session);
    showNotif('Hint provided. This will affect your learning score.');
  } catch (error) {
    showNotif(error.message || 'Could not get hint.');
  }
}

async function submitAnswer() {
  const sessionId = getSessionId();
  if (!sessionId) {
    showNotif('No active session. Please start a new session.');
    return;
  }

  if (!App.session?.current_question) {
    showNotif('No question available.');
    return;
  }

  const question = App.session.current_question;
  const questionType = question.type || 'essay';
  
  let answer = '';
  if (questionType === 'mcq') {
    const selected = document.querySelector('input[name="mcq_answer"]:checked');
    if (!selected) {
      showNotif('Please select an answer.');
      return;
    }
    answer = selected.value;
  } else {
    answer = document.getElementById('answerInput').value.trim();
    if (!answer) {
      showNotif('Write an answer before submitting.');
      return;
    }
  }

  const button = document.getElementById('submitAnswerBtn');
  button.disabled = true;

  try {
    const elapsedSeconds = Math.max(1, Math.round((Date.now() - (App.questionStartedAt || Date.now())) / 1000));
    const data = await apiFetch('/learn/api/session/answer', {
      method: 'POST',
      body: JSON.stringify({
        session_id: sessionId,
        answer,
        timeTaken: elapsedSeconds,
        hintUsed: App.hintUsed,
      }),
    });

    // Store next session data but don't advance yet — show feedback + Next Question button
    App.pendingNextSession = data;

    // Show the feedback for the CURRENT question using the returned session data
    renderFeedbackMessage(data.session, data.decision);
    renderFeedback(data.session);

    // Disable answer inputs so student can't re-submit
    const answerInput = document.getElementById('answerInput');
    if (answerInput) answerInput.disabled = true;
    document.querySelectorAll('input[name="mcq_answer"]').forEach(r => r.disabled = true);

    // Hide Submit/Hint buttons and show Next Question button
    button.style.display = 'none';
    const hintBtn = document.getElementById('hintBtn');
    if (hintBtn) hintBtn.style.display = 'none';

    let nextBtn = document.getElementById('nextQuestionBtn');
    if (!nextBtn) {
      nextBtn = document.createElement('button');
      nextBtn.id = 'nextQuestionBtn';
      nextBtn.className = 'act-btn primary';
      nextBtn.onclick = advanceToNextQuestion;
      nextBtn.innerHTML = `
        <svg viewBox="0 0 24 24" width="16" height="16">
          <line x1="5" y1="12" x2="19" y2="12"/>
          <polyline points="12 5 19 12 12 19"/>
        </svg>
        Next Question
      `;
      const answerActions = document.querySelector('.answer-actions');
      if (answerActions) answerActions.appendChild(nextBtn);
    }
    nextBtn.style.display = 'flex';

    showNotif('Answer processed by the tutor.');
  } catch (error) {
    showNotif(error.message || 'Could not submit answer.');
    button.disabled = false;
  }
}

function advanceToNextQuestion() {
  if (!App.pendingNextSession) return;

  const { session, decision } = App.pendingNextSession;
  App.pendingNextSession = null;

  // Re-enable submit button and restore UI
  const submitBtn = document.getElementById('submitAnswerBtn');
  if (submitBtn) {
    submitBtn.disabled = false;
    submitBtn.style.display = 'flex';
  }

  // Hide Next Question button
  const nextBtn = document.getElementById('nextQuestionBtn');
  if (nextBtn) nextBtn.style.display = 'none';

  // Re-enable answer inputs
  const answerInput = document.getElementById('answerInput');
  if (answerInput) answerInput.disabled = false;

  applySession(session, decision);
}

async function refreshSession() {
  const sessionId = getSessionId();
  if (!sessionId) {
    showNotif('No active session.');
    return;
  }

  try {
    const data = await apiFetch('/learn/api/session/state', {
      method: 'POST',
      body: JSON.stringify({ session_id: sessionId }),
    });
    
    applySession(data.session);
    setState('stateResults');
    showNotif('Session state refreshed.');
  } catch (error) {
    showNotif(error.message || 'Could not refresh session.');
    // If session not found, clear it and reset
    clearSessionId();
    setState('stateUpload');
  }
}

async function endSession() {
  showConfirmModal(
    'End Session',
    'Are you sure you want to end this session and return home?',
    'End Session',
    async () => {
      const sessionId = getSessionId();
      if (sessionId) {
        try {
          await apiFetch('/learn/api/session/reset', {
            method: 'POST',
            body: JSON.stringify({ session_id: sessionId }),
          });
        } catch (error) {
          // Ignore errors
        }
      }
      
      // Clear session and redirect
      clearSessionId();
      window.location.href = '/';
    }
  );
}

async function resetApp() {
  showConfirmModal(
    'New Lecture',
    'Are you sure you want to end this session and start a new one?',
    'Start New',
    async () => {
      const sessionId = getSessionId();
      if (sessionId) {
        try {
          await apiFetch('/learn/api/session/reset', {
            method: 'POST',
            body: JSON.stringify({ session_id: sessionId }),
          });
        } catch (error) {
          // Continue with reset even if backend fails
        }
      }

      // Clear session
      clearSessionId();
      
      App.procTimers.forEach((timer) => clearTimeout(timer));
      App.procTimers = [];
      App.selectedFile = null;
      App.session = null;
      App.questionStartedAt = null;
      App.hintUsed = false;

      document.getElementById('fileInput').value = '';
      document.getElementById('filePreview').classList.add('hidden');
      document.getElementById('genArea').classList.add('hidden');
      document.getElementById('answerInput').value = '';
      setState('stateUpload');
      showNotif('Ready for another lecture.');
    }
  );
}

function showConfirmModal(title, message, buttonText, action) {
  document.getElementById('confirmTitle').textContent = title;
  document.getElementById('confirmMessage').textContent = message;
  document.getElementById('confirmActionBtn').textContent = buttonText;
  App.confirmAction = action;
  document.getElementById('confirmModal').classList.remove('hidden');
}

function closeConfirmModal() {
  document.getElementById('confirmModal').classList.add('hidden');
  App.confirmAction = null;
}

async function executeConfirmAction() {
  if (App.confirmAction) {
    await App.confirmAction();
  }
  closeConfirmModal();
}

function updateTimerHint(expectedSeconds) {
  const hint = expectedSeconds
    ? `Expected answering time for this question is about ${expectedSeconds} seconds.`
    : 'Timer starts when the question is shown.';
  document.getElementById('timerHint').textContent = hint;
}

function fileExtensionLabel(fileName) {
  if (!fileName || !fileName.includes('.')) return 'FILE';
  return fileName.split('.').pop().toUpperCase();
}

function formatNumber(value) {
  return Number.isFinite(Number(value)) ? Number(value).toFixed(3) : '-';
}

function formatSigned(value) {
  return Number.isFinite(Number(value)) ? Number(value).toFixed(2) : '-';
}

function showNotif(msg) {
  const notif = document.getElementById('notif');
  document.getElementById('notifText').textContent = msg;
  notif.classList.remove('hide');
  clearTimeout(App.notifTimer);
  App.notifTimer = setTimeout(() => notif.classList.add('hide'), 3200);
}

function escapeHtml(str) {
  if (typeof str !== 'string') return '';
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function escapeAttribute(str) {
  return escapeHtml(str);
}

// ═══════════════════════════════════════════════════════════
//  INITIALIZATION
// ═══════════════════════════════════════════════════════════

window.addEventListener('load', async () => {
  // Check if there's an active session in sessionStorage
  const sessionId = getSessionId();
  
  if (sessionId) {
    // Try to resume session
    try {
      const data = await apiFetch('/learn/api/session/state', {
        method: 'POST',
        body: JSON.stringify({ session_id: sessionId }),
      });
      
      applySession(data.session);
      setState('stateResults');
      console.log('Resumed session:', sessionId);
    } catch (error) {
      // Session expired or invalid - clear it
      console.log('Session not found, starting fresh');
      clearSessionId();
      setState('stateUpload');
    }
  } else {
    // No session - start fresh
    setState('stateUpload');
  }
  
  // Add end session button
  const resultsActions = document.querySelector('.results-actions');
  if (resultsActions && !document.getElementById('endSessionBtn')) {
    const endBtn = document.createElement('button');
    endBtn.id = 'endSessionBtn';
    endBtn.className = 'act-btn danger';
    endBtn.onclick = endSession;
    endBtn.innerHTML = `
      <svg viewBox="0 0 24 24">
        <line x1="18" y1="6" x2="6" y2="18"/>
        <line x1="6" y1="6" x2="18" y2="18"/>
      </svg>
      End Session
    `;
    resultsActions.appendChild(endBtn);
  }
});