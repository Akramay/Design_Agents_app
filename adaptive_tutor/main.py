"""
main.py
───────
Entry point for the Adaptive Tutoring System.

Two modes:
  1. setup_session(file_path) — parse lecture, initialize, get first question
  2. process_answer(text, time) — handle student answer, get next action

UPDATED: Support for custom save paths (session-based files)
"""

import json
import os

from blackboard                import Blackboard
from parser_agent              import ParserAgent
from bkt_agent                 import BKTAgent
from irt_agent                 import IRTAgent
from question_feedback_agents  import QuestionAgent, FeedbackAgent
from orchestrator_agent        import OrchestratorAgent


# ── Configuration ─────────────────────────────────────────────
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")  # set in environment
STUDENT_STATE_FILE = "student_state.json"  # legacy default


def build_system(resume: bool = False, save_path: str = None) -> tuple:
    """
    Wire all agents together and return the orchestrator.
    
    Args:
        resume: Whether to resume from saved state
        save_path: Custom path for state file (if None, uses STUDENT_STATE_FILE)
    
    Returns:
        (orchestrator, blackboard) tuple
    """
    # Use custom save path if provided, otherwise use default
    state_file = save_path if save_path else STUDENT_STATE_FILE
    
    # shared blackboard
    bb = Blackboard(save_path=state_file)

    # try to resume previous session
    if resume and bb.load():
        print(f"  [main] Resuming session from {state_file}")

    # build all agents
    agents = {
        "parser":   ParserAgent(bb),
        "bkt":      BKTAgent(bb),
        "irt":      IRTAgent(bb),
        "question": QuestionAgent(bb),
        "feedback": FeedbackAgent(bb, youtube_api_key=YOUTUBE_API_KEY),
    }

    orchestrator = OrchestratorAgent(bb, agents)
    return orchestrator, bb


def print_state_summary(bb: Blackboard):
    """Print a readable summary of the current student model."""
    model = bb.read("student_model") or {}
    print(f"\n{'═'*55}")
    print(f"  CURRENT STUDENT STATE")
    print(f"{'═'*55}")
    for concept, state in model.items():
        mastered = "✓ MASTERED" if state.get("mastered") else ""
        bar_len  = int(state.get("bkt_p_learned", 0) * 20)
        bar      = "█" * bar_len + "░" * (20 - bar_len)
        print(f"  {concept[:30]:<30} P(L)={state.get('bkt_p_learned', 0):.3f} |{bar}| θ={state.get('irt_theta', 0):+.2f} {mastered}")
    print(f"{'═'*55}")
    print(f"  Total questions: {bb.read('total_questions')}")
    print(f"  Total correct  : {bb.read('total_correct')}")
    print()


def print_agent_thinking(bb: Blackboard):
    """Print what each agent was thinking — this goes to the UI too."""
    thinking = bb.read("agent_thinking") or []
    if not thinking:
        return
    print(f"\n{'─'*55}")
    print(f"  AGENT THINKING LOG (shown in UI):")
    print(f"{'─'*55}")
    for entry in thinking:
        print(f"  [{entry['timestamp']}] {entry['agent']}: {entry['message']}")
    print(f"{'─'*55}")


# ══════════════════════════════════════════════════════════════
#  FLASK API INTEGRATION (for your web app)
# ══════════════════════════════════════════════════════════════

def get_flask_app(orchestrator, bb):
    """
    If you're using Flask, use this function.
    It returns a Flask app with endpoints.
    """
    try:
        from flask import Flask, request, jsonify
        app = Flask(__name__)

        @app.route("/setup", methods=["POST"])
        def setup():
            data = request.json
            orchestrator.setup_session(data["file_path"])
            question = bb.read("current_question")
            thinking = bb.read("agent_thinking")
            return jsonify({
                "question":      question,
                "concept":       bb.read("current_concept"),
                "agent_thinking": thinking,
            })

        @app.route("/answer", methods=["POST"])
        def answer():
            data       = request.json
            answer_txt = data.get("answer", "")
            time_taken = float(data.get("time_taken", 30))
            hint_used  = bool(data.get("hint_used", False))

            decision = orchestrator.process_answer(answer_txt, time_taken, hint_used)

            return jsonify({
                "action":          decision["action"],
                "reasoning":       decision.get("reasoning", ""),
                "message":         decision.get("message_to_student", ""),
                "next_question":   bb.read("current_question"),
                "hint":            bb.read("hint"),
                "explanation":     bb.read("explanation"),
                "videos":          bb.read("videos"),
                "concept":         bb.read("current_concept"),
                "agent_thinking":  bb.read("agent_thinking"),
                "student_model":   bb.read("student_model"),
            })

        @app.route("/state", methods=["GET"])
        def state():
            return jsonify({
                "student_model":   bb.read("student_model"),
                "current_concept": bb.read("current_concept"),
                "concept_graph":   bb.read("concept_graph"),
                "total_questions": bb.read("total_questions"),
                "total_correct":   bb.read("total_correct"),
                "agent_thinking":  bb.read("agent_thinking"),
            })

        return app

    except ImportError:
        print("  Flask not installed. Run: pip install flask")
        return None


# ══════════════════════════════════════════════════════════════
#  DIRECT TEST (run this file to test without a UI)
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"\n{'═'*55}")
    print(f"  ADAPTIVE TUTORING SYSTEM — Test Run")
    print(f"{'═'*55}")

    # ── build the system ──────────────────────────────────
    orchestrator, bb = build_system(resume=False)

    # ── use a sample lecture text file for testing ────────
    test_file = "sample_lecture.txt"

    # create a minimal test file if it doesn't exist
    if not os.path.exists(test_file):
        print(f"\n  Creating sample lecture file: {test_file}")
        with open(test_file, "w") as f:
            f.write("""
Introduction to Intelligent Agents
====================================

An intelligent agent is a system that perceives its environment 
through sensors and acts upon it through actuators to achieve goals.

Agent Architecture
==================
Agent architecture defines the internal design of an agent.
Types include: reactive agents, deliberative agents, BDI agents,
and layered architectures.

Reactive Agents
===============
Reactive agents use simple condition-action rules.
They respond immediately to stimuli without memory or planning.
Example: Thermostat, Braitenberg vehicles.

BDI Agents (Belief Desire Intention)
=====================================
BDI agents maintain three mental states:
- Beliefs: what the agent thinks is true about the world
- Desires: goals the agent wants to achieve  
- Intentions: the goals the agent is currently committed to

Multi-Agent Systems (MAS)
==========================
A multi-agent system contains multiple interacting agents.
Types: CMAS (cooperative) and SMAS (self-interested).
Communication: direct (message passing) or indirect (blackboard).

Contract Net Protocol
=====================
Contract Net is a coordination protocol where:
1. Manager recognizes a task
2. Manager announces the task
3. Contractors bid
4. Manager awards to best bidder

Speech Acts
===========
Speech acts are utterances that perform actions:
- Locutionary: the literal words
- Illocutionary: the intent
- Perlocutionary: the effect on the hearer
Types: representatives, directives, commissives, expressives, declarations.
""")

    # ── SESSION SETUP ─────────────────────────────────────
    print(f"\n  PHASE 1: Setting up session with lecture...")
    orchestrator.setup_session(test_file)

    # show what question was generated
    q = bb.read("current_question")
    print(f"\n{'═'*55}")
    print(f"  FIRST QUESTION:")
    print(f"  Concept : {bb.read('current_concept')}")
    print(f"  Type    : {q.get('type', 'essay').upper()}")
    print(f"  Q: {q['question']}")
    if q.get('type') == 'mcq':
        print(f"  Options:")
        for i, opt in enumerate(q.get('options', [])):
            print(f"    {chr(65+i)}. {opt}")
    print(f"  Expected time: {q['expected_time_seconds']}s")
    print(f"{'═'*55}")

    print_agent_thinking(bb)
    print_state_summary(bb)

    # ── SIMULATE ANSWERS ──────────────────────────────────
    test_answers = [
        {
            "answer": "An agent is a system that perceives its environment and takes actions to achieve goals",
            "time":   18.0,
            "hint":   False,
            "label":  "Good answer, normal time"
        },
        {
            "answer": "BDI stands for Belief Desire Intention and these are the three mental states of a BDI agent",
            "time":   25.0,
            "hint":   False,
            "label":  "Correct answer"
        },
        {
            "answer": "I don't know",
            "time":   45.0,
            "hint":   False,
            "label":  "Wrong answer, slow time — should trigger explanation"
        },
    ]

    for i, test in enumerate(test_answers):
        print(f"\n{'═'*55}")
        print(f"  SIMULATED ANSWER #{i+1}: {test['label']}")
        print(f"  Student types: \"{test['answer']}\"")
        print(f"  Time taken:     {test['time']}s")
        print(f"  Hint used:      {test['hint']}")
        print(f"{'═'*55}")

        input(f"\n  Press ENTER to process this answer...")

        decision = orchestrator.process_answer(
            answer_text = test["answer"],
            time_taken  = test["time"],
            hint_used   = test["hint"]
        )

        print(f"\n{'─'*55}")
        print(f"  ACTION  : {decision['action']}")
        print(f"  MESSAGE : {decision.get('message_to_student', '')}")

        # show feedback if any
        hint    = bb.read("hint")
        explain = bb.read("explanation")
        videos  = bb.read("videos")

        if hint:
            print(f"\n  HINT: {hint}")
        if explain:
            print(f"\n  EXPLANATION: {explain}")
        if videos:
            print(f"\n  VIDEOS:")
            for v in videos:
                print(f"    • {v['title']}: {v['url']}")

        print_agent_thinking(bb)
        print_state_summary(bb)

        # show next question
        next_q = bb.read("current_question")
        if next_q:
            print(f"\n  NEXT QUESTION:")
            print(f"  Concept: {bb.read('current_concept')}")
            print(f"  Type: {next_q.get('type', 'essay').upper()}")
            print(f"  Q: {next_q['question']}")

    print(f"\n{'═'*55}")
    print(f"  Test session complete!")
    print(f"  State saved to: {bb.save_path}")
    print(f"{'═'*55}")