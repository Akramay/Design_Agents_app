import io
import json
import os
import sys
import traceback
import time
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta

from blackboard import Blackboard
from main import build_system, YOUTUBE_API_KEY

# Session files directory
SESSION_DIR = "sessions"
SESSION_TIMEOUT_HOURS = 24  # Auto-delete sessions older than this

# Create sessions directory if it doesn't exist
os.makedirs(SESSION_DIR, exist_ok=True)


def _get_session_path(session_id):
    """Get the file path for a specific session."""
    return os.path.join(SESSION_DIR, f"student_state_{session_id}.json")


def _cleanup_old_sessions():
    """Delete session files older than SESSION_TIMEOUT_HOURS."""
    try:
        cutoff_time = time.time() - (SESSION_TIMEOUT_HOURS * 3600)
        for filename in os.listdir(SESSION_DIR):
            if filename.startswith("student_state_") and filename.endswith(".json"):
                filepath = os.path.join(SESSION_DIR, filename)
                if os.path.getmtime(filepath) < cutoff_time:
                    os.remove(filepath)
                    print(f"  [Cleanup] Deleted old session: {filename}")
    except Exception as e:
        print(f"  [Cleanup] Error during cleanup: {e}")


def _serialize_session(bb, session_id):
    """Serialize the current session state for the frontend."""
    graph = bb.read("concept_graph") or []
    model = bb.read("student_model") or {}
    current_concept = bb.read("current_concept")
    current_state = model.get(current_concept, {}) if current_concept else {}

    mastered_count = sum(
        1 for state in model.values() if state.get("mastered")
    )
    total_questions = bb.read("total_questions") or 0
    total_correct = bb.read("total_correct") or 0
    accuracy = (total_correct / total_questions) if total_questions else 0.0

    current_summary = ""
    for concept in graph:
        if concept.get("concept") == current_concept:
            current_summary = concept.get("summary", "")
            break

    return {
        "session_id": session_id,
        "file_path": bb.read("file_path"),
        "lecture_title": os.path.basename(bb.read("file_path") or ""),
        "current_concept": current_concept,
        "current_summary": current_summary,
        "current_question": bb.read("current_question"),
        "current_concept_state": current_state,
        "concept_graph": graph,
        "student_model": model,
        "hint": bb.read("hint"),
        "hint_available": bb.read("hint_available"),
        "explanation": bb.read("explanation"),
        "videos": bb.read("videos") or [],
        "agent_thinking": bb.read("agent_thinking") or [],
        "llm_decision": bb.read("llm_decision"),
        "last_answer_correct": bb.read("last_answer_correct"),
        "last_answer_text": bb.read("last_answer_text"),
        "total_questions": total_questions,
        "total_correct": total_correct,
        "progress": {
            "mastered_count": mastered_count,
            "concept_count": len(graph),
            "accuracy": accuracy,
        },
    }


def _handle_request(payload):
    action = payload.get("action")

    # Clean up old sessions on every request
    _cleanup_old_sessions()

    if action == "reset":
        session_id = payload.get("session_id")
        if session_id:
            session_path = _get_session_path(session_id)
            if os.path.exists(session_path):
                os.remove(session_path)
                return {"ok": True, "message": "Session deleted."}
        return {"ok": True, "message": "No session to delete."}

    if action == "setup":
        # Generate new session ID
        session_id = f"{int(time.time())}_{os.urandom(4).hex()}"
        session_path = _get_session_path(session_id)
        
        lecture_path = payload["file_path"]
        orchestrator, bb = build_system(resume=False, save_path=session_path)
        orchestrator.setup_session(lecture_path)
        bb.save()
        
        return {
            "ok": True,
            "message": "Session created.",
            "session_id": session_id,
            "session": _serialize_session(bb, session_id),
        }

    if action == "answer":
        session_id = payload.get("session_id")
        if not session_id:
            raise RuntimeError("No session_id provided.")
        
        session_path = _get_session_path(session_id)
        if not os.path.exists(session_path):
            raise RuntimeError("Session expired or not found. Please start a new session.")
        
        answer_text = payload.get("answer", "")
        time_taken = float(payload.get("time_taken", 30))
        hint_used = bool(payload.get("hint_used", False))
        
        # Build system and resume session
        orchestrator, bb = build_system(resume=True, save_path=session_path)
        
        if not bb.read("current_question"):
            raise RuntimeError("No active question. Session may be corrupted.")
        
        # Process answer
        decision = orchestrator.process_answer(answer_text, time_taken, hint_used)
        
        return {
            "ok": True,
            "message": "Answer processed.",
            "decision": decision,
            "session": _serialize_session(bb, session_id),
        }

    if action == "get_hint":
        session_id = payload.get("session_id")
        if not session_id:
            raise RuntimeError("No session_id provided.")
        
        session_path = _get_session_path(session_id)
        if not os.path.exists(session_path):
            raise RuntimeError("Session expired or not found.")
        
        bb = Blackboard(save_path=session_path)
        if not bb.load():
            raise RuntimeError("Could not load session.")
        
        hint_available = bb.read("hint_available")
        if not hint_available:
            return {"ok": False, "error": "Hint already used for this question."}
        
        # Generate hint
        from question_feedback_agents import FeedbackAgent
        feedback_agent = FeedbackAgent(bb)
        bb.write("llm_decision", {"action": "SHOW_HINT"})
        feedback_agent.run()
        bb.save()
        
        return {
            "ok": True,
            "message": "Hint generated.",
            "session": _serialize_session(bb, session_id),
        }

    if action == "state":
        session_id = payload.get("session_id")
        if not session_id:
            return {"ok": False, "error": "No session_id provided."}
        
        session_path = _get_session_path(session_id)
        if not os.path.exists(session_path):
            return {"ok": False, "error": "Session expired or not found."}
        
        bb = Blackboard(save_path=session_path)
        if not bb.load():
            return {"ok": False, "error": "Could not load session."}
        
        return {"ok": True, "session": _serialize_session(bb, session_id)}

    if action == "suggest_videos":
        session_id = payload.get("session_id")
        if not session_id:
            raise RuntimeError("No session_id provided.")
        
        session_path = _get_session_path(session_id)
        if not os.path.exists(session_path):
            raise RuntimeError("Session expired or not found.")
        
        bb = Blackboard(save_path=session_path)
        if not bb.load():
            raise RuntimeError("Could not load session.")
        
        # Generate video suggestions
        from video_suggestion_agent import VideoSuggestionAgent
        video_agent = VideoSuggestionAgent(bb, youtube_api_key=YOUTUBE_API_KEY)
        perception = video_agent.perceive()
        decision = video_agent.reason(perception)
        video_agent.act(decision)
        bb.save()
        
        return {
            "ok": True,
            "message": "Video suggestions generated.",
            "session": _serialize_session(bb, session_id),
        }

    if action == "explain":
        session_id = payload.get("session_id")
        if not session_id:
            raise RuntimeError("No session_id provided.")

        session_path = _get_session_path(session_id)
        if not os.path.exists(session_path):
            raise RuntimeError("Session expired or not found.")

        bb = Blackboard(save_path=session_path)
        if not bb.load():
            raise RuntimeError("Could not load session.")

        graph    = bb.read("concept_graph") or []
        contexts = bb.read("concept_contexts") or {}

        from llm_provider import call_llm, parse_json as _parse_json

        concepts_out = []
        for node in graph:
            concept = node.get("concept", "")
            ctx     = contexts.get(concept, node.get("summary", ""))[:800]
            try:
                raw = call_llm(
                    f"You are a teaching assistant. Explain the concept below in 3-5 clear sentences "
                    f"suitable for a student who just read the lecture. "
                    f"Then list 3-5 key points as short bullet phrases.\n\n"
                    f"Concept: {concept}\n"
                    f"Lecture context: {ctx}\n\n"
                    f"Respond ONLY with valid JSON, no markdown, no extra text:\n"
                    f'{{ "explanation": "...", "key_points": ["...", "..."] }}',
                    max_tokens=400,
                )
                parsed = _parse_json(raw)
                concepts_out.append({
                    "concept":    concept,
                    "explanation": parsed.get("explanation", ctx[:300]),
                    "key_points":  parsed.get("key_points", []),
                })
            except Exception:
                concepts_out.append({
                    "concept":    concept,
                    "explanation": ctx[:300],
                    "key_points":  [],
                })

        return {"ok": True, "concepts": concepts_out}

        raise ValueError(f"Unsupported action: {action}")


def main():
    raw_request = sys.stdin.read().strip() or "{}"
    payload = json.loads(raw_request)
    log_buffer = io.StringIO()

    try:
        with redirect_stdout(log_buffer), redirect_stderr(log_buffer):
            response = _handle_request(payload)
        response["logs"] = log_buffer.getvalue()
    except Exception as exc:
        response = {
            "ok": False,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "logs": log_buffer.getvalue(),
        }

    sys.stdout.write(json.dumps(response))


if __name__ == "__main__":
    main()