const express = require('express');
const fs = require('fs');
const path = require('path');
const { spawnSync } = require('child_process');

const router = express.Router();

const ROOT_DIR = path.join(__dirname, '..');
const PYTHON_DIR = path.join(ROOT_DIR, 'adaptive_tutor');
const PYTHON_BRIDGE = path.join(PYTHON_DIR, 'api_bridge.py');
const UPLOADS_DIR = path.join(ROOT_DIR, 'uploads');

function log(message, ...args) {
  console.log(`[TutorAPI] ${new Date().toISOString()} - ${message}`, ...args);
}

function logError(message, error) {
  console.error(`[TutorAPI] ${new Date().toISOString()} - ${message}`, error);
}

function createResponsePayload(response, operation) {
  return {
    operation,
    ...response,
  };
}

function buildErrorBody(error, operation) {
  const message = String(error?.message || error || 'Unexpected backend error.');
  const body = {
    ok: false,
    operation,
    error: message,
    user_message: 'An internal backend error occurred. Check the terminal for details.',
    source: 'backend',
  };

  const lower = message.toLowerCase();

  if (/no api key found|groq_api_key not set|gemini_api_key not set/i.test(message)) {
    body.user_message = 'LLM is not configured. Set GROQ_API_KEY or GEMINI_API_KEY in .env.';
    body.source = 'llm';
  } else if (/both llm providers failed/i.test(message)) {
    body.user_message = 'AI generation failed on both Groq and Gemini. Check the terminal logs.';
    body.source = 'llm';
    body.provider = 'Groq+Gemini';
  } else if (/groq/i.test(message) && /gemini/i.test(message)) {
    body.user_message = 'AI generation failed. It tried Groq and Gemini.';
    body.source = 'llm';
    body.provider = 'Groq+Gemini';
  } else if (/groq/i.test(message)) {
    body.user_message = 'AI generation failed on Groq. Check your Groq API key and terminal logs.';
    body.source = 'llm';
    body.provider = 'Groq';
  } else if (/gemini/i.test(message)) {
    body.user_message = 'AI generation failed on Gemini. Check your Gemini API key and terminal logs.';
    body.source = 'llm';
    body.provider = 'Gemini';
  } else if (/invalid json/i.test(lower) || /could not turn response into a dictionary/i.test(lower)) {
    body.user_message = 'The assistant sent an invalid response. Try again in a moment.';
    body.source = 'llm';
  } else if (/session expired or not found/i.test(lower)) {
    body.user_message = 'Session expired or not found. Upload the lecture again to start a new session.';
  } else if (/no session_id provided/i.test(lower)) {
    body.user_message = 'No active session was found. Start by uploading a lecture.';
  } else if (/could not load session/i.test(lower)) {
    body.user_message = 'Could not restore session state. Restart the lecture upload.';
  } else if (/no active question/i.test(lower)) {
    body.user_message = 'No active question is available. Start a new lecture session.';
  } else if (/python bridge/i.test(lower)) {
    body.user_message = 'The backend bridge failed to parse the Python response. Check the terminal logs.';
  } else if (/python interpreter was not found|unable to start python bridge/i.test(lower)) {
    body.user_message = 'Backend bridge failed to start. Check the terminal for Python interpreter issues.';
  } else if (/timed out|timeout/i.test(lower)) {
    body.user_message = 'A backend request timed out. Try again or check your connection.';
  }

  return body;
}


if (!fs.existsSync(UPLOADS_DIR)) {
  fs.mkdirSync(UPLOADS_DIR, { recursive: true });
}

function sanitizeFileName(name) {
  return String(name || 'lecture.txt').replace(/[^a-zA-Z0-9._-]/g, '_');
}

function runPythonBridge(payload) {
  log('Starting Python bridge', payload);
  const candidates = process.env.PYTHON_BIN
    ? [process.env.PYTHON_BIN]
    : ['python', 'py'];

  let lastError = null;

  for (const command of candidates) {
    log(`Trying interpreter: ${command}`);
    const result = spawnSync(command, [PYTHON_BRIDGE], {
      cwd: PYTHON_DIR,
      encoding: 'utf8',
      input: JSON.stringify(payload),
      timeout: 600000,
    });

    if (result.error) {
      lastError = result.error;
      logError(`Interpreter failed: ${command}`, result.error);
      continue;
    }

    if (result.status !== 0 && !result.stdout) {
      const stderr = (result.stderr || '').trim();
      logError('Python bridge returned non-zero status without stdout.', stderr);
      throw new Error(stderr || `Python bridge exited with code ${result.status}`);
    }

    const stdout = (result.stdout || '').trim();
    if (!stdout) {
      throw new Error((result.stderr || '').trim() || 'Python bridge returned no data.');
    }

    try {
      const parsed = JSON.parse(stdout);
      log(`Python bridge action '${payload.action}' completed successfully.`);
      return parsed;
    } catch (error) {
      logError('Invalid JSON from Python bridge', stdout.slice(0, 400));
      throw new Error(`Invalid JSON from Python bridge: ${stdout.slice(0, 400)}`);
    }
  }

  throw new Error(
    lastError
      ? `Unable to start Python bridge: ${lastError.message}`
      : 'Python interpreter was not found.'
  );
}


router.post('/session/state', (req, res) => {
  const operation = 'session_state';
  try {
    const { session_id } = req.body || {};
    log(`Received ${operation}`, { session_id });
    const response = runPythonBridge({
      action: 'state',
      session_id: session_id,
    });
    res.status(response.ok ? 200 : 404).json(createResponsePayload(response, operation));
  } catch (error) {
    logError(`Error ${operation}`, error);
    res.status(500).json(buildErrorBody(error, operation));
  }
});

router.post('/session/setup', (req, res) => {
  const operation = 'session_setup';
  try {
    const { fileName, fileData } = req.body || {};
    log(`Received ${operation}`, { fileName, length: fileData ? fileData.length : 0 });

    if (!fileName || !fileData) {
      return res.status(400).json({
        ok: false,
        error: 'fileName and fileData are required.',
        operation,
      });
    }

    const safeName = `${Date.now()}-${sanitizeFileName(fileName)}`;
    const savedPath = path.join(UPLOADS_DIR, safeName);
    const buffer = Buffer.from(fileData, 'base64');

    fs.writeFileSync(savedPath, buffer);
    log(`Saved uploaded lecture to ${savedPath}`);

    const response = runPythonBridge({
      action: 'setup',
      file_path: savedPath,
    });

    res.status(response.ok ? 200 : 500).json(createResponsePayload(response, operation));
  } catch (error) {
    logError(`Error ${operation}`, error);
    res.status(500).json(buildErrorBody(error, operation));
  }
});

router.post('/session/answer', (req, res) => {
  const operation = 'session_answer';
  try {
    const { session_id, answer, timeTaken, hintUsed } = req.body || {};
    log(`Received ${operation}`, { session_id, answer, timeTaken, hintUsed });
    const response = runPythonBridge({
      action: 'answer',
      session_id: session_id,
      answer: answer || '',
      time_taken: Number.isFinite(Number(timeTaken)) ? Number(timeTaken) : 30,
      hint_used: Boolean(hintUsed),
    });

    res.status(response.ok ? 200 : 500).json(createResponsePayload(response, operation));
  } catch (error) {
    logError(`Error ${operation}`, error);
    res.status(500).json(buildErrorBody(error, operation));
  }
});

router.post('/session/hint', (req, res) => {
  const operation = 'session_hint';
  try {
    const { session_id } = req.body || {};
    log(`Received ${operation}`, { session_id });
    const response = runPythonBridge({
      action: 'get_hint',
      session_id: session_id,
    });

    res.status(response.ok ? 200 : 500).json(createResponsePayload(response, operation));
  } catch (error) {
    logError(`Error ${operation}`, error);
    res.status(500).json(buildErrorBody(error, operation));
  }
});

router.post('/session/reset', (req, res) => {
  const operation = 'session_reset';
  try {
    const { session_id } = req.body || {};
    log(`Received ${operation}`, { session_id });
    const response = runPythonBridge({
      action: 'reset',
      session_id: session_id,
    });
    res.status(200).json(createResponsePayload(response, operation));
  } catch (error) {
    logError(`Error ${operation}`, error);
    res.status(500).json(buildErrorBody(error, operation));
  }
});

router.post('/session/suggest-videos', (req, res) => {
  const operation = 'session_suggest_videos';
  try {
    const { session_id } = req.body || {};
    log(`Received ${operation}`, { session_id });
    const response = runPythonBridge({
      action: 'suggest_videos',
      session_id: session_id,
    });
    res.status(response.ok ? 200 : 500).json(createResponsePayload(response, operation));
  } catch (error) {
    logError(`Error ${operation}`, error);
    res.status(500).json(buildErrorBody(error, operation));
  }
});

router.post('/session/explain', (req, res) => {
  const operation = 'session_explain';
  try {
    const { session_id } = req.body || {};
    log(`Received ${operation}`, { session_id });
    const response = runPythonBridge({
      action: 'explain',
      session_id: session_id,
    });
    res.status(response.ok ? 200 : 500).json(createResponsePayload(response, operation));
  } catch (error) {
    logError(`Error ${operation}`, error);
    res.status(500).json(buildErrorBody(error, operation));
  }
});

module.exports = router;