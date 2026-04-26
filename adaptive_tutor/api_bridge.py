import io
import json
import os
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout

from blackboard import Blackboard
from main import STUDENT_STATE_FILE, build_system


def _serialize_session(bb):
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
        "file_path": bb.read("file_path"),
        "lecture_title": os.path.basename(bb.read("file_path") or ""),
        "current_concept": current_concept,
        "current_summary": current_summary,
        "current_question": bb.read("current_question"),
        "current_concept_state": current_state,
        "concept_graph": graph,
        "student_model": model,
        "hint": bb.read("hint"),
        "explanation": bb.read("explanation"),
        "videos": bb.read("videos") or [],
        "agent_thinking": bb.read("agent_thinking") or [],
        "llm_decision": bb.read("llm_decision"),
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

    if action == "reset":
        if os.path.exists(STUDENT_STATE_FILE):
            os.remove(STUDENT_STATE_FILE)
        return {"ok": True, "message": "Session reset."}

    if action == "setup":
        lecture_path = payload["file_path"]
        orchestrator, bb = build_system(resume=False)
        orchestrator.setup_session(lecture_path)
        bb.save()
        return {
            "ok": True,
            "message": "Session created.",
            "session": _serialize_session(bb),
        }

    if action == "answer":
        answer_text = payload.get("answer", "")
        time_taken = float(payload.get("time_taken", 30))
        orchestrator, bb = build_system(resume=True, include_parser=False)
        if not bb.read("current_question"):
            raise RuntimeError("No active session found. Start with setup first.")
        decision = orchestrator.process_answer(answer_text, time_taken)
        return {
            "ok": True,
            "message": "Answer processed.",
            "decision": decision,
            "session": _serialize_session(bb),
        }

    if action == "state":
        bb = Blackboard(save_path=STUDENT_STATE_FILE)
        if not bb.load():
            return {"ok": False, "error": "No active session found."}
        return {"ok": True, "session": _serialize_session(bb)}

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