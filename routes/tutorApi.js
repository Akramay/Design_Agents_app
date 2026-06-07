const express = require('express');
const fs = require('fs');
const path = require('path');
const { spawnSync } = require('child_process');

const router = express.Router();

const ROOT_DIR = path.join(__dirname, '..');
const PYTHON_DIR = path.join(ROOT_DIR, 'adaptive_tutor');
const PYTHON_BRIDGE = path.join(PYTHON_DIR, 'api_bridge.py');
const UPLOADS_DIR = path.join(ROOT_DIR, 'uploads');

if (!fs.existsSync(UPLOADS_DIR)) {
  fs.mkdirSync(UPLOADS_DIR, { recursive: true });
}

function sanitizeFileName(name) {
  return String(name || 'lecture.txt').replace(/[^a-zA-Z0-9._-]/g, '_');
}

function runPythonBridge(payload) {
  const candidates = process.env.PYTHON_BIN
    ? [process.env.PYTHON_BIN]
    : ['python', 'py'];

  let lastError = null;

  for (const command of candidates) {
    const result = spawnSync(command, [PYTHON_BRIDGE], {
      cwd: PYTHON_DIR,
      encoding: 'utf8',
      input: JSON.stringify(payload),
      timeout: 600000,
    });

    if (result.error) {
      lastError = result.error;
      continue;
    }

    if (result.status !== 0 && !result.stdout) {
      const stderr = (result.stderr || '').trim();
      throw new Error(stderr || `Python bridge exited with code ${result.status}`);
    }

    const stdout = (result.stdout || '').trim();
    if (!stdout) {
      throw new Error((result.stderr || '').trim() || 'Python bridge returned no data.');
    }

    try {
      return JSON.parse(stdout);
    } catch (error) {
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
  try {
    const { session_id } = req.body || {};
    const response = runPythonBridge({
      action: 'state',
      session_id: session_id,
    });
    res.status(response.ok ? 200 : 404).json(response);
  } catch (error) {
    res.status(500).json({ ok: false, error: error.message });
  }
});

router.post('/session/setup', (req, res) => {
  try {
    const { fileName, fileData } = req.body || {};

    if (!fileName || !fileData) {
      return res.status(400).json({
        ok: false,
        error: 'fileName and fileData are required.',
      });
    }

    const safeName = `${Date.now()}-${sanitizeFileName(fileName)}`;
    const savedPath = path.join(UPLOADS_DIR, safeName);
    const buffer = Buffer.from(fileData, 'base64');

    fs.writeFileSync(savedPath, buffer);

    const response = runPythonBridge({
      action: 'setup',
      file_path: savedPath,
    });

    res.status(response.ok ? 200 : 500).json(response);
  } catch (error) {
    res.status(500).json({ ok: false, error: error.message });
  }
});

router.post('/session/answer', (req, res) => {
  try {
    const { session_id, answer, timeTaken, hintUsed } = req.body || {};
    const response = runPythonBridge({
      action: 'answer',
      session_id: session_id,
      answer: answer || '',
      time_taken: Number.isFinite(Number(timeTaken)) ? Number(timeTaken) : 30,
      hint_used: Boolean(hintUsed),
    });

    res.status(response.ok ? 200 : 500).json(response);
  } catch (error) {
    res.status(500).json({ ok: false, error: error.message });
  }
});

router.post('/session/hint', (req, res) => {
  try {
    const { session_id } = req.body || {};
    const response = runPythonBridge({
      action: 'get_hint',
      session_id: session_id,
    });

    res.status(response.ok ? 200 : 500).json(response);
  } catch (error) {
    res.status(500).json({ ok: false, error: error.message });
  }
});

router.post('/session/reset', (req, res) => {
  try {
    const { session_id } = req.body || {};
    const response = runPythonBridge({
      action: 'reset',
      session_id: session_id,
    });
    res.status(200).json(response);
  } catch (error) {
    res.status(500).json({ ok: false, error: error.message });
  }
});

router.post('/session/suggest-videos', (req, res) => {
  try {
    const { session_id } = req.body || {};
    const response = runPythonBridge({
      action: 'suggest_videos',
      session_id: session_id,
    });
    res.status(response.ok ? 200 : 500).json(response);
  } catch (error) {
    res.status(500).json({ ok: false, error: error.message });
  }
});

router.post('/session/explain', (req, res) => {
  try {
    const { session_id } = req.body || {};
    const response = runPythonBridge({
      action: 'explain',
      session_id: session_id,
    });
    res.status(response.ok ? 200 : 500).json(response);
  } catch (error) {
    res.status(500).json({ ok: false, error: error.message });
  }
});

module.exports = router;