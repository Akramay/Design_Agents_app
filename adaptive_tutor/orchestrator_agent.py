"""
orchestrator_agent.py
─────────────────────
The coordinator agent. It:
  1. Sets up the session (calls ParserAgent, initializes student model)
  2. After every student answer:
       - Grades the answer (LLM)
       - Runs BKT → IRT → LLM reasoning
       - Routes to Feedback or next Question
       - Saves state to JSON

This is the "manager" from Contract Net Protocol in Lecture 3.
It coordinates without doing the specialists' jobs itself.

FIXED ISSUES:
- Grading bug where response was used before assignment
- Added hint_used_current_question tracking
- Added question history to prevent duplicates
"""

import re
import hashlib

from base_agent   import BaseAgent
from llm_provider import call_llm, parse_json
from bkt_agent    import BKT_PARAMS


class OrchestratorAgent(BaseAgent):

    def __init__(self, blackboard, agents: dict):
        super().__init__("Orchestrator", blackboard)
        self.parser   = agents["parser"]
        self.bkt      = agents["bkt"]
        self.irt      = agents["irt"]
        self.question = agents["question"]
        self.feedback = agents["feedback"]

    # ══════════════════════════════════════════════════════════
    #  PHASE 1 — called once when lecture is uploaded
    # ══════════════════════════════════════════════════════════

    def setup_session(self, file_path: str):
        """
        Full initialization:
          1. Parse lecture → concept graph
          2. Build student model from graph
          3. Generate first question
        """
        print(f"\n{'═'*55}")
        print(f"  ORCHESTRATOR: Setting up new session")
        print(f"  File: {file_path}")
        print(f"{'═'*55}")

        self.blackboard.clear_thinking()
        self.blackboard.write("file_path", file_path)
        
        # Initialize question history tracking
        self.blackboard.write("asked_questions", [])

        self.blackboard.log_thinking(
            "Orchestrator",
            f"New session started. Asking ParserAgent to analyze the lecture..."
        )

        # ── Step 1: Parser agent builds concept graph ─────
        self.parser.run()

        # ── Step 2: Initialize student model ─────────────
        graph = self.blackboard.read("concept_graph")
        student_model = self._initialize_student_model(graph)
        self.blackboard.write("student_model", student_model)

        print(f"\n  [Orchestrator] Student model initialized:")
        for concept, state in student_model.items():
            print(f"    {concept}: P(L)={state['bkt_p_learned']:.2f}, θ={state['irt_theta']:.2f}, b={state['irt_b_current']:.1f}")

        self.blackboard.log_thinking(
            "Orchestrator",
            f"Student model initialized for {len(student_model)} concepts. "
            f"Generating first question..."
        )

        # ── Step 3: Generate first question ──────────────
        self.blackboard.write("next_b", -1.5)  # always start easy
        self.blackboard.write("hint_used_current_question", False)
        self.question.run()

        print(f"\n{'═'*55}")
        print(f"  ORCHESTRATOR: Session ready! First question set.")
        print(f"{'═'*55}")

    def _initialize_student_model(self, graph: list) -> dict:
        """Build one entry per concept with starting BKT and IRT values."""
        model = {}
        for concept_info in graph:
            d = concept_info.get("difficulty", 3)
            params = BKT_PARAMS.get(d, BKT_PARAMS[3])
            model[concept_info["concept"]] = {
                # BKT fields
                "bkt_p_learned":  0.10,                    # low prior: assume not known
                "bkt_p_transit":  params["p_transit"],
                "bkt_p_slip":     params["p_slip"],
                "bkt_p_guess":    params["p_guess"],
                "mastered":       False,
                # IRT fields
                "irt_theta":      0.0,                     # neutral starting ability
                "irt_b_current":  (d - 3) * 0.5,          # maps difficulty 1-5 to b range
                # tracking
                "attempts":       0,
                "history":        [],
                "correct_streak": 0,
            }
        return model

    # ══════════════════════════════════════════════════════════
    #  PHASE 2 — called after every student answer
    # ══════════════════════════════════════════════════════════

    def process_answer(self, answer_text: str, time_taken: float, hint_used: bool = False) -> dict:
        """
        Full answer processing pipeline:
          Grade → Update history → BKT → IRT → LLM reason → Route action
        """
        print(f"\n{'═'*55}")
        print(f"  ORCHESTRATOR: Processing student answer")
        print(f"  Answer: \"{answer_text[:60]}...\"")
        print(f"  Time  : {time_taken:.1f}s")
        print(f"  Hint used: {hint_used}")
        print(f"{'═'*55}")

        self.blackboard.clear_thinking()

        concept  = self.blackboard.read("current_concept")
        question = self.blackboard.read("current_question")
        model    = self.blackboard.read("student_model")

        # ── Step 1: Grade the answer (LLM) ───────────────
        print(f"\n  [Orchestrator] Step 1: Grading answer with LLM...")
        correct = self._grade_answer(answer_text, question)

        # ── Step 2: Compute time ratio ────────────────────
        expected_time = question.get("expected_time_seconds", 30)
        time_ratio    = time_taken / expected_time

        print(f"\n  [Orchestrator] Step 2: Time analysis:")
        print(f"  Expected time : {expected_time}s")
        print(f"  Actual time   : {time_taken:.1f}s")
        print(f"  Time ratio    : {time_ratio:.2f}x  ", end="")
        if time_ratio < 0.4:
            print("← Very fast! Possible lucky guess")
        elif time_ratio < 1.5:
            print("← Normal thinking time")
        elif time_ratio < 2.5:
            print("← Slow — student is struggling")
        else:
            print("← Very slow — serious difficulty")

        # ── Step 3: Update student history ───────────────
        model[concept]["attempts"] += 1
        if correct:
            model[concept]["correct_streak"] = model[concept].get("correct_streak", 0) + 1
        else:
            model[concept]["correct_streak"] = 0

        self.blackboard.write("student_model",      model)
        self.blackboard.write("last_answer_text",   answer_text)
        self.blackboard.write("last_answer_correct", correct)
        self.blackboard.write("last_time_taken",    time_taken)
        self.blackboard.write("last_time_ratio",    time_ratio)
        self.blackboard.write("hint_used_current_question", hint_used)

        # update session totals
        total_q = self.blackboard.read("total_questions") or 0
        total_c = self.blackboard.read("total_correct")   or 0
        self.blackboard.write("total_questions", total_q + 1)
        if correct:
            self.blackboard.write("total_correct", total_c + 1)

        self.blackboard.log_thinking(
            "Orchestrator",
            f"Answer graded: {'CORRECT ✓' if correct else 'WRONG ✗'}. "
            f"Time ratio: {time_ratio:.2f}x. "
            f"{'Hint was used. ' if hint_used else ''}"
            f"Running BKT and IRT updates now..."
        )

        # ── Step 4: BKT updates P(learned) ───────────────
        self.bkt.run()

        # ── Step 5: IRT updates theta + picks next b ─────
        self.irt.run()

        # ── Step 6: LLM reasons about what to do next ────
        print(f"\n  [Orchestrator] Step 6: LLM reasoning about next action...")
        decision = self._llm_reason(concept, correct, time_ratio, hint_used)
        self.blackboard.write("llm_decision", decision)

        # ── Step 7: Execute the action ───────────────────
        action = decision["action"]
        print(f"\n  [Orchestrator] Step 7: Executing action → {action}")
        self._execute_action(action)

        # ── Step 8: Save state ────────────────────────────
        self.blackboard.save()

        print(f"\n{'═'*55}")
        print(f"  ORCHESTRATOR: Answer processing complete")
        print(f"  Next action: {action}")
        print(f"{'═'*55}")

        return decision

    # ── PERCEIVE / REASON / ACT (agent interface) ─────────────
    def perceive(self):
        return {}

    def reason(self, perception):
        return {}

    def act(self, decision):
        pass

    # ── PRIVATE: Grade answer ─────────────────────────────────
    def _grade_answer(self, answer_text: str, question: dict) -> bool:
        """
        Use LLM to grade the student's answer.
        Returns True if correct, False if wrong.
        
        FIXED: Proper error handling for LLM call failures
        """
        question_type = question.get("type", "essay")
        
        # For MCQ, simple exact match
        if question_type == "mcq":
            student_answer = answer_text.strip().upper()
            correct_answer = question.get("correct_answer", "").upper()
            correct = (student_answer == correct_answer)
            
            print(f"  [Orchestrator] MCQ Grading:")
            print(f"  Student chose : {student_answer}")
            print(f"  Correct answer: {correct_answer}")
            print(f"  Result        : {'✓ CORRECT' if correct else '✗ WRONG'}")
            
            return correct
        
        # For essay questions, use LLM grading
        prompt = f"""You are grading a student's answer to a university AI course question.

Question: {question.get('question', '')}

Expected key points: {question.get('key_points', [])}

Student's answer: \"\"\"{answer_text}\"\"\"

Does the student's answer cover the key points adequately?
Respond with ONLY one word: "true" if correct, "false" if incorrect.
Be lenient — if they show understanding of the core concept, mark it true."""

        try:
            result = call_llm(prompt, max_tokens=10).strip().lower()
            correct = "true" in result

            print(f"  [Orchestrator] Essay Grading result: {'✓ CORRECT' if correct else '✗ WRONG'}")
            print(f"  [Orchestrator] LLM grader said: \"{result}\"")
            
        except Exception as e:
            print(f"  [Orchestrator] LLM grading failed: {e}")
            print(f"  [Orchestrator] Using fallback lexical grading...")
            
            # Fallback: lexical overlap
            expected = " ".join(question.get("key_points", [])).lower().split()
            answer_words = set(answer_text.lower().split())
            overlap = sum(1 for word in expected if word in answer_words)
            correct = overlap >= max(1, len(expected) // 4)
            result = f"fallback lexical grader (overlap: {overlap}/{len(expected)})"
            
            print(f"  [Orchestrator] Fallback result: {'✓ CORRECT' if correct else '✗ WRONG'}")

        self.blackboard.log_thinking(
            "Orchestrator",
            f"Answer grading: {'CORRECT ✓' if correct else 'WRONG ✗'} — "
            f"{'LLM evaluated key points' if 'fallback' not in result else 'Lexical matching used'}."
        )
        return correct

    # ── PRIVATE: LLM Reasoning ────────────────────────────────
    def _llm_reason(self, concept: str, correct: bool, time_ratio: float, hint_used: bool) -> dict:
        """
        The core intelligence of the system.
        LLM receives all math signals and reasons step by step.
        Returns a structured decision.
        """
        model = self.blackboard.read("student_model")[concept]
        graph = self.blackboard.read("concept_graph") or []

        # check if all concepts are mastered
        all_mastered = all(
            self.blackboard.read("student_model").get(c["concept"], {}).get("mastered", False)
            for c in graph
        )

        bkt_p  = model["bkt_p_learned"]
        theta  = model["irt_theta"]
        next_b = model["irt_b_current"]
        hist   = model["history"][-6:]  # last 6 answers

        prompt = f"""You are an intelligent adaptive tutoring agent for a university AI course.
A student just answered a question. Use the math signals to reason and decide the next action.

═══ SIGNALS ═══
Concept being tested : {concept}
Student answer       : {"CORRECT ✓" if correct else "WRONG ✗"}
Hint was used        : {hint_used}
Time ratio           : {time_ratio:.2f}x expected (< 0.5 = very fast, > 1.5 = struggling)
Recent history       : {hist}  (last {len(hist)} answers)

BKT P(learned)  = {bkt_p:.3f}
  Interpretation:
    < 0.35 = concept not learned
    0.35-0.65 = uncertain
    0.65-0.85 = likely learned  
    > 0.85 = MASTERED

IRT theta (ability) = {theta:.3f}
  Interpretation:
    < -1.0 = weak student
    -1 to 0 = below average
    0 to 1 = average
    > 1.0 = strong student

Next question difficulty b = {next_b:.2f}
All concepts mastered = {all_mastered}
═══════════════

Reason through these questions step by step:
1. Based on BKT, has the student truly learned this concept or are they still uncertain?
2. Based on time ratio and hint usage, is this luck/guess OR genuine understanding?
3. What is the best next action to maximize learning?

Return ONLY this JSON (no markdown, no explanation):
{{
  "reasoning": "your step by step thinking in 2-3 sentences",
  "action": "EXACTLY one of: INCREASE_DIFFICULTY | KEEP_LEVEL | DECREASE_DIFFICULTY | SHOW_HINT | SHOW_EXPLANATION | RECOMMEND_VIDEO | NEXT_CONCEPT | SESSION_COMPLETE",
  "message_to_student": "one encouraging, specific message to show the student (1-2 sentences)"
}}

Action guide:
- NEXT_CONCEPT      → BKT > 0.85 (mastered, move on)
- SESSION_COMPLETE  → all concepts mastered
- INCREASE_DIFFICULTY → correct + fast + BKT rising + not lucky + no hint
- KEEP_LEVEL        → mixed signals, need more evidence
- DECREASE_DIFFICULTY → wrong answer + BKT falling
- SHOW_HINT         → time_ratio > 1.5 and student requested help
- SHOW_EXPLANATION  → wrong answer + BKT < 0.4
- RECOMMEND_VIDEO   → wrong + BKT < 0.35 + repeated failures"""

        print(f"  [Orchestrator] Sending reasoning prompt to LLM...")

        try:
            raw = call_llm(prompt, max_tokens=300)
            decision = parse_json(raw)
            
        except Exception as e:
            print(f"  [Orchestrator] LLM reasoning failed: {e}")
            print(f"  [Orchestrator] Using fallback decision logic...")
            
            # Fallback decision logic based on signals
            if all_mastered:
                action = "SESSION_COMPLETE"
                msg = "You have mastered all detected concepts in this lecture."
            elif model["mastered"]:
                action = "NEXT_CONCEPT"
                msg = f"You have likely mastered {concept}. Moving to the next concept."
            elif not correct and bkt_p < 0.35:
                action = "SHOW_EXPLANATION"
                msg = f"Let's slow down and reinforce {concept} with an explanation."
            elif not correct:
                action = "DECREASE_DIFFICULTY"
                msg = f"We'll stay with {concept} and make the next question more approachable."
            elif time_ratio < 0.8 and theta >= 0 and not hint_used:
                action = "INCREASE_DIFFICULTY"
                msg = f"Nice work on {concept}. The next question can be a little harder."
            else:
                action = "KEEP_LEVEL"
                msg = f"Good progress on {concept}. Let's gather one more signal at a similar level."
                
            decision = {
                "reasoning": "Fallback tutor policy selected the next action from BKT, IRT, and answer correctness.",
                "action": action,
                "message_to_student": msg,
            }

        print(f"\n  [Orchestrator] LLM Decision:")
        print(f"  Action   : {decision['action']}")
        print(f"  Reasoning: {decision['reasoning']}")
        print(f"  Message  : {decision['message_to_student']}")

        self.blackboard.log_thinking(
            "Orchestrator",
            f"LLM Decision: {decision['action']}. "
            f"Reasoning: {decision['reasoning']}"
        )

        return decision

    # ── PRIVATE: Route action ─────────────────────────────────
    def _execute_action(self, action: str):
        # Reset hint flag
        self.blackboard.write("hint_used_current_question", False)

        if action in ["SHOW_HINT", "SHOW_EXPLANATION", "RECOMMEND_VIDEO"]:
            self.feedback.run()
            # CRITICAL: Force a new question after feedback so we don't loop
            self.question.run() 
        elif action == "NEXT_CONCEPT":
            self._advance_to_next_concept()
            self.question.run()
        else:
            self.question.run()

    def _advance_to_next_concept(self):
        """Move to the next unmastered concept in dependency order."""
        graph   = self.blackboard.read("concept_graph") or []
        model   = self.blackboard.read("student_model") or {}
        current = self.blackboard.read("current_concept")

        # find next unmastered concept
        for concept_info in graph:
            name = concept_info["concept"]
            if name == current:
                continue
            if not model.get(name, {}).get("mastered", False):
                # check all dependencies are mastered
                deps = concept_info.get("depends_on", [])
                deps_met = all(
                    model.get(d, {}).get("mastered", False)
                    for d in deps
                )
                if deps_met:
                    self.blackboard.write("current_concept", name)
                    self.blackboard.write("next_b", -1.0)  # start easy on new concept
                    print(f"\n  [Orchestrator] Advanced → new concept: {name}")
                    self.blackboard.log_thinking(
                        "Orchestrator",
                        f"Concept '{current}' mastered! Moving to '{name}'."
                    )
                    return

        print(f"\n  [Orchestrator] All concepts complete!")