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

KEY IMPROVEMENTS over original:
  - MCQ grading compares full option text case-insensitively (the only correct approach
    since correct_answer is now always the full option text, never a letter).
  - Essay grading uses a rubric-style prompt that references key_points AND
    the concept context for more accurate, lenient-where-appropriate scoring.
  - setup_session passes concept_contexts from ParserAgent to blackboard.
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
          1. Parse lecture → concept graph + concept_contexts
          2. Build student model from graph
          3. Generate first question
        """
        print(f"\n{'═'*55}")
        print(f"  ORCHESTRATOR: Setting up new session")
        print(f"  File: {file_path}")
        print(f"{'═'*55}")

        self.blackboard.clear_thinking()
        self.blackboard.write("file_path", file_path)
        
        # Extract and store lecture title from file path
        import os
        lecture_title = os.path.basename(file_path)
        self.blackboard.write("lecture_title", lecture_title)

        # Initialize question history tracking
        self.blackboard.write("asked_questions", [])

        self.blackboard.log_thinking(
            "Orchestrator",
            f"New session started. Asking ParserAgent to analyze the lecture..."
        )

        # Step 1: Parser agent builds concept graph + contexts
        # NOTE: parser.act() now writes both "concept_graph" and "concept_contexts"
        self.parser.run()

        # Step 2: Initialize student model
        graph         = self.blackboard.read("concept_graph")
        student_model = self._initialize_student_model(graph)
        self.blackboard.write("student_model", student_model)

        print(f"\n  [Orchestrator] Student model initialized:")
        for concept, state in student_model.items():
            print(f"    {concept}: P(L)={state['bkt_p_learned']:.2f}, "
                  f"θ={state['irt_theta']:.2f}, b={state['irt_b_current']:.1f}")

        self.blackboard.log_thinking(
            "Orchestrator",
            f"Student model initialized for {len(student_model)} concepts. "
            f"Generating first question..."
        )

        # Step 3: Generate first question (always start easy)
        self.blackboard.write("next_b", -1.5)
        self.blackboard.write("hint_used_current_question", False)
        self.question.run()

        print(f"\n{'═'*55}")
        print(f"  ORCHESTRATOR: Session ready! First question set.")
        print(f"{'═'*55}")

    def _initialize_student_model(self, graph: list) -> dict:
        model = {}
        for concept_info in graph:
            d      = concept_info.get("difficulty", 3)
            params = BKT_PARAMS.get(d, BKT_PARAMS[3])
            model[concept_info["concept"]] = {
                "bkt_p_learned":  0.10,
                "bkt_p_transit":  params["p_transit"],
                "bkt_p_slip":     params["p_slip"],
                "bkt_p_guess":    params["p_guess"],
                "mastered":       False,
                "irt_theta":      0.0,
                "irt_b_current":  (d - 3) * 0.5,
                "attempts":       0,
                "history":        [],
                "correct_streak": 0,
            }
        return model

    # ══════════════════════════════════════════════════════════
    #  PHASE 2 — called after every student answer
    # ══════════════════════════════════════════════════════════

    def process_answer(self, answer_text: str, time_taken: float, hint_used: bool = False) -> dict:
        print(f"\n{'═'*55}")
        print(f"  ORCHESTRATOR: Processing student answer")
        print(f"  Answer: \"{answer_text[:80]}...\"")
        print(f"  Time  : {time_taken:.1f}s  |  Hint used: {hint_used}")
        print(f"{'═'*55}")

        self.blackboard.clear_thinking()

        concept  = self.blackboard.read("current_concept")
        question = self.blackboard.read("current_question")
        model    = self.blackboard.read("student_model")

        # Step 1: Grade
        print(f"\n  [Orchestrator] Step 1: Grading answer...")
        correct = self._grade_answer(answer_text, question)

        # Step 2: Time ratio
        expected_time = question.get("expected_time_seconds", 30)
        time_ratio    = time_taken / max(expected_time, 1)

        print(f"\n  [Orchestrator] Step 2: Time analysis:")
        print(f"  Expected: {expected_time}s  |  Actual: {time_taken:.1f}s  |  Ratio: {time_ratio:.2f}x", end="")
        if time_ratio < 0.4:
            print(" ← Very fast (possible lucky guess)")
        elif time_ratio < 1.5:
            print(" ← Normal thinking time")
        elif time_ratio < 2.5:
            print(" ← Slow — student is struggling")
        else:
            print(" ← Very slow — serious difficulty")

        # Step 3: Update history
        model[concept]["attempts"] += 1
        if correct:
            model[concept]["correct_streak"] = model[concept].get("correct_streak", 0) + 1
        else:
            model[concept]["correct_streak"] = 0

        self.blackboard.write("student_model",       model)
        self.blackboard.write("last_answer_text",    answer_text)
        self.blackboard.write("last_answer_correct", correct)
        self.blackboard.write("last_time_taken",     time_taken)
        self.blackboard.write("last_time_ratio",     time_ratio)
        self.blackboard.write("hint_used_current_question", hint_used)

        total_q = self.blackboard.read("total_questions") or 0
        total_c = self.blackboard.read("total_correct")   or 0
        self.blackboard.write("total_questions", total_q + 1)
        if correct:
            self.blackboard.write("total_correct", total_c + 1)

        self.blackboard.log_thinking(
            "Orchestrator",
            f"Answer graded: {'CORRECT ✓' if correct else 'WRONG ✗'}. "
            f"Time ratio: {time_ratio:.2f}x. "
            f"{'Hint used. ' if hint_used else ''}"
            f"Running BKT and IRT updates..."
        )

        # Step 4: BKT
        self.bkt.run()

        # Step 5: IRT
        self.irt.run()

        # Step 6: LLM reasoning
        print(f"\n  [Orchestrator] Step 6: LLM reasoning about next action...")
        decision = self._llm_reason(concept, correct, time_ratio, hint_used)
        self.blackboard.write("llm_decision", decision)

        # Step 7: Execute
        action = decision["action"]
        print(f"\n  [Orchestrator] Step 7: Executing action → {action}")
        self._execute_action(action)

        # Step 8: Save
        self.blackboard.save()

        print(f"\n{'═'*55}")
        print(f"  ORCHESTRATOR: Done. Next action: {action}")
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
        question_type = question.get("type", "essay")

        if question_type == "mcq":
            # correct_answer is always the FULL TEXT of the correct option.
            # The frontend sends back the full option text the student selected.
            student_ans = answer_text.strip().lower()
            correct_ans = question.get("correct_answer", "").strip().lower()

            # Exact match (after normalisation)
            if student_ans == correct_ans:
                correct = True
            else:
                # Fallback: check if student sent just a letter (A/B/C/D) —
                # map it to the option at that index and compare.
                opts = question.get("options", [])
                letter_map = {"a": 0, "b": 1, "c": 2, "d": 3}
                if student_ans in letter_map and opts:
                    idx = letter_map[student_ans]
                    if idx < len(opts):
                        correct = opts[idx].strip().lower() == correct_ans
                    else:
                        correct = False
                else:
                    # Partial containment: student's answer contains the correct text
                    correct = correct_ans in student_ans or student_ans in correct_ans

            print(f"  [Orchestrator] MCQ Grading:")
            print(f"  Student chose : {answer_text[:80]!r}")
            print(f"  Correct answer: {question.get('correct_answer', '')[:80]!r}")
            print(f"  Result        : {'✓ CORRECT' if correct else '✗ WRONG'}")
            return correct

        # ── Essay grading ─────────────────────────────────────
        concept  = self.blackboard.read("current_concept")
        contexts = self.blackboard.read("concept_contexts") or {}
        ctx      = contexts.get(concept, "")

        prompt = f"""You are grading a student's short-answer question for a university course.

Concept being tested: {concept}

Lecture context (source of truth):
\"\"\"
{ctx[:600]}
\"\"\"

Question: {question.get('question', '')}

Key points the answer should cover:
{question.get('key_points', [])}

Model answer:
{question.get('expected_answer', '')}

Student's answer:
\"\"\"
{answer_text}
\"\"\"

Grading rules:
- Mark TRUE if the student demonstrates understanding of at least 60% of the key points.
- Be lenient with wording — paraphrasing is fine.
- Do NOT penalise for minor spelling mistakes.
- Mark FALSE only if the student clearly misunderstands the concept or provides an empty/irrelevant answer.

Respond with ONLY one word: true  OR  false"""

        try:
            result  = call_llm(prompt, max_tokens=10).strip().lower()
            correct = "true" in result
            print(f"  [Orchestrator] Essay Grading: {'✓ CORRECT' if correct else '✗ WRONG'} (LLM said: {result!r})")
        except Exception as e:
            print(f"  [Orchestrator] LLM grading failed: {e} — using lexical fallback")
            expected     = " ".join(question.get("key_points", [])).lower().split()
            answer_words = set(answer_text.lower().split())
            overlap      = sum(1 for word in expected if word in answer_words)
            correct      = overlap >= max(1, len(expected) // 4)
            print(f"  [Orchestrator] Lexical fallback: {'✓ CORRECT' if correct else '✗ WRONG'} (overlap {overlap}/{len(expected)})")

        self.blackboard.log_thinking(
            "Orchestrator",
            f"Essay grading: {'CORRECT ✓' if correct else 'WRONG ✗'}."
        )
        return correct

    # ── PRIVATE: LLM Reasoning ────────────────────────────────
    def _llm_reason(self, concept: str, correct: bool, time_ratio: float, hint_used: bool) -> dict:
        model = self.blackboard.read("student_model")[concept]
        graph = self.blackboard.read("concept_graph") or []

        all_mastered = all(
            self.blackboard.read("student_model").get(c["concept"], {}).get("mastered", False)
            for c in graph
        )

        bkt_p  = model["bkt_p_learned"]
        theta  = model["irt_theta"]
        next_b = model["irt_b_current"]
        hist   = model["history"][-6:]

        prompt = f"""You are an intelligent adaptive tutoring agent for a university course.
A student just answered a question. Use the signals below to decide the next action.

═══ SIGNALS ═══
Concept             : {concept}
Answer              : {"CORRECT ✓" if correct else "WRONG ✗"}
Hint used           : {hint_used}
Time ratio          : {time_ratio:.2f}x expected  (< 0.5 = very fast, > 1.5 = struggling)
Recent history      : {hist}

BKT P(learned) = {bkt_p:.3f}
  < 0.35 = not learned  |  0.35-0.65 = uncertain  |  0.65-0.85 = likely learned  |  > 0.85 = MASTERED

IRT theta (ability) = {theta:.3f}
  < -1.0 = weak  |  -1 to 0 = below avg  |  0 to 1 = avg  |  > 1.0 = strong

Next question b = {next_b:.2f}   |   All concepts mastered = {all_mastered}
═══════════════

Reason step by step:
1. Has the student truly learned this concept (BKT) or are they still uncertain?
2. Does the time ratio and hint usage suggest genuine understanding or a lucky guess?
3. What is the best next action to maximise learning?

Return ONLY this JSON (no markdown):
{{
  "reasoning": "your step-by-step thinking in 2-3 sentences",
  "action": "EXACTLY one of: INCREASE_DIFFICULTY | KEEP_LEVEL | DECREASE_DIFFICULTY | SHOW_HINT | SHOW_EXPLANATION | RECOMMEND_VIDEO | NEXT_CONCEPT | SESSION_COMPLETE",
  "message_to_student": "one encouraging, specific message to show the student (1-2 sentences)"
}}

Action guide:
- NEXT_CONCEPT       → BKT > 0.85 (mastered, move on)
- SESSION_COMPLETE   → all concepts mastered
- INCREASE_DIFFICULTY → correct + fast + BKT rising + no hint used
- KEEP_LEVEL         → mixed signals, need more evidence
- DECREASE_DIFFICULTY → wrong answer + BKT falling
- SHOW_EXPLANATION   → wrong answer + BKT < 0.4
- RECOMMEND_VIDEO    → wrong + BKT < 0.35 + repeated failures"""

        print(f"  [Orchestrator] Sending reasoning prompt to LLM...")

        try:
            raw      = call_llm(prompt, max_tokens=300)
            decision = parse_json(raw)
        except Exception as e:
            print(f"  [Orchestrator] LLM reasoning failed: {e} — using fallback logic")

            if all_mastered:
                action = "SESSION_COMPLETE"
                msg    = "You have mastered all detected concepts in this lecture. Well done!"
            elif model["mastered"]:
                action = "NEXT_CONCEPT"
                msg    = f"You have mastered {concept}. Moving to the next concept."
            elif not correct and bkt_p < 0.35:
                action = "SHOW_EXPLANATION"
                msg    = f"Let's slow down and reinforce {concept} with a detailed explanation."
            elif not correct:
                action = "DECREASE_DIFFICULTY"
                msg    = f"We'll keep working on {concept} with a more approachable question."
            elif time_ratio < 0.8 and theta >= 0 and not hint_used:
                action = "INCREASE_DIFFICULTY"
                msg    = f"Great work on {concept}! Time to try something harder."
            else:
                action = "KEEP_LEVEL"
                msg    = f"Good progress on {concept}. One more question at this level."

            decision = {
                "reasoning": "Fallback policy applied based on BKT/IRT/correctness signals.",
                "action":    action,
                "message_to_student": msg,
            }

        print(f"\n  [Orchestrator] Decision → {decision['action']}")
        print(f"  Reasoning: {decision['reasoning']}")
        print(f"  Message  : {decision['message_to_student']}")

        self.blackboard.log_thinking(
            "Orchestrator",
            f"LLM Decision: {decision['action']}. Reasoning: {decision['reasoning']}"
        )
        return decision

    # ── PRIVATE: Route action ─────────────────────────────────
    def _execute_action(self, action: str):
        self.blackboard.write("hint_used_current_question", False)

        if action in ["SHOW_HINT", "SHOW_EXPLANATION", "RECOMMEND_VIDEO"]:
            self.feedback.run()
            self.question.run()       # always generate a new question after feedback
        elif action == "NEXT_CONCEPT":
            self._advance_to_next_concept()
            self.question.run()
        else:
            self.question.run()

    def _advance_to_next_concept(self):
        graph   = self.blackboard.read("concept_graph") or []
        model   = self.blackboard.read("student_model") or {}
        current = self.blackboard.read("current_concept")

        for concept_info in graph:
            name = concept_info["concept"]
            if name == current:
                continue
            if not model.get(name, {}).get("mastered", False):
                deps     = concept_info.get("depends_on", [])
                deps_met = all(model.get(d, {}).get("mastered", False) for d in deps)
                if deps_met:
                    self.blackboard.write("current_concept", name)
                    self.blackboard.write("next_b", -1.0)   # start easy on new concept
                    print(f"\n  [Orchestrator] Advanced → new concept: {name}")
                    self.blackboard.log_thinking(
                        "Orchestrator",
                        f"Concept '{current}' mastered! Moving to '{name}'."
                    )
                    return

        print(f"\n  [Orchestrator] All concepts complete!")