"""
blackboard.py
─────────────
The shared environment all agents read from and write to.
Agents NEVER talk to each other directly — only through here.
This is the Blackboard Architecture from MAS Lecture 3.

UPDATED: Added fields for hint system and question deduplication
"""

import json
import os
from datetime import datetime


class Blackboard:
    def __init__(self, save_path="student_state.json"):
        self.save_path = save_path
        self.state = {
            # ── set by: user / frontend ──────────────────
            "file_path":            None,   # uploaded lecture path
            "student_id":           "student_001",

            # ── written by: ParserAgent ──────────────────
            "lecture_text":         None,   # raw extracted text
            "concept_graph":        None,   # ordered list of concepts
            "current_concept":      None,   # concept being tested now

            # ── written by: BKT + IRT agents ────────────
            "student_model":        {},     # full per-concept model

            # ── written by: frontend (answer submission) ─
            "last_answer_text":     None,   # what student typed
            "last_answer_correct":  None,   # True / False
            "last_time_taken":      None,   # seconds
            "last_time_ratio":      None,   # actual / expected
            "last_question":        None,   # previous question data for feedback

            # ── written by: QuestionAgent ────────────────
            "current_question":     None,   # full question dict
            "asked_questions":      [],     # list of question hashes (deduplication)
            
            # ── hint system ──────────────────────────────
            "hint_available":       True,   # can student request hint?
            "hint_used_current_question": False,  # did they use hint this time?

            # ── written by: OrchestratorAgent ────────────
            "llm_decision":         None,   # action + reasoning
            "agent_thinking":       [],     # log shown in UI
            "next_b":               None,   # next difficulty selected by IRT

            # ── written by: FeedbackAgent ────────────────
            "hint":                 None,
            "explanation":          None,
            "videos":               [],

            # ── session metadata ─────────────────────────
            "session_started":      datetime.now().isoformat(),
            "total_questions":      0,
            "total_correct":        0,
        }

    # ── core read / write ────────────────────────────────
    def read(self, key):
        return self.state.get(key)

    def write(self, key, value):
        self.state[key] = value

    # ── agent thinking log (shown in UI) ─────────────────
    def log_thinking(self, agent_name, message):
        entry = {
            "agent":     agent_name,
            "message":   message,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        }
        self.state["agent_thinking"].append(entry)
        # also print to terminal so you can follow along
        print(f"  [{agent_name}] {message}")

    def clear_thinking(self):
        self.state["agent_thinking"] = []

    # ── persistence ──────────────────────────────────────
    def save(self):
        """Save student state to JSON after every answer."""
        with open(self.save_path, "w") as f:
            json.dump(self.state, f, indent=2, default=str)
        print(f"\n  [Blackboard] State saved → {self.save_path}")

    def load(self):
        """Resume a previous session if file exists."""
        if os.path.exists(self.save_path):
            with open(self.save_path) as f:
                self.state = json.load(f)
            print(f"  [Blackboard] Session loaded from {self.save_path}")
            return True
        return False