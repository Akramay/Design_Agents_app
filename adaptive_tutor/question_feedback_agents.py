"""
question_feedback_agents.py
────────────────────────────
KEY IMPROVEMENTS over previous version:
  1. MCQ generation embeds "answer_key" directly in the question object.
     No second LLM call needed to grade MCQ — grading is a pure string comparison.
  2. Essay generation embeds a "grading_rubric" (list of required key points defined
     by the LLM at generation time). Essay grading uses ONE LLM call against this
     rubric, not an open-ended evaluation.
  3. _verify_mcq() second LLM call is REMOVED. Quality checks are folded into the
     generation prompt itself (rule 9 asks the LLM to self-check before returning).
     This saves one LLM call per question.
  4. New GradingAgent:
       - Grades MCQ by direct string comparison against stored answer_key.
       - Grades essay by rubric coverage (one LLM call).
       - Writes last_answer_correct (bool) and last_answer_feedback (str) to blackboard.
       - Writes next_difficulty_hint ("harder" | "easier" | "same") so IRT/BKT agents
         have a clean, explicit signal to update theta / mastery.
  5. FeedbackAgent now reads next_difficulty_hint to include adaptive context in
     explanations ("you're ready for harder questions" or "let's reinforce this").
  6. All prompts remain grounded in lecture context only — no hallucinated content.
"""

import json
import re
import hashlib
import random
import requests as http_requests

from base_agent   import BaseAgent
from irt_agent    import DIFFICULTY_LABELS
from llm_provider import call_llm, parse_json


# ══════════════════════════════════════════════════════════════
class QuestionAgent(BaseAgent):

    def __init__(self, blackboard):
        super().__init__("QuestionAgent", blackboard)
        self.type_distribution = {
            (-2.0, -0.5): (0.80, 0.20),
            (-0.5,  0.5): (0.50, 0.50),
            ( 0.5,  2.5): (0.20, 0.80),
        }

    # ── PERCEIVE ─────────────────────────────────────────────
    def perceive(self) -> dict:
        concept  = self.blackboard.read("current_concept")
        graph    = self.blackboard.read("concept_graph") or []
        next_b   = self.blackboard.read("next_b") or 0.0
        contexts = self.blackboard.read("concept_contexts") or {}

        summary = ""
        for c in graph:
            if c["concept"] == concept:
                summary = c.get("summary", "")
                break

        # Full lecture context for this concept (richer than just the summary)
        lecture_context = contexts.get(concept, summary)

        return {
            "concept":         concept,
            "summary":         summary,
            "lecture_context": lecture_context,
            "b":               next_b,
        }

    # ── REASON ───────────────────────────────────────────────
    def reason(self, p: dict) -> dict:
        b             = p["b"]
        question_type = self._select_question_type(b)
        closest_b     = min(DIFFICULTY_LABELS.keys(), key=lambda x: abs(x - b))
        expected_time = {
            -2.0: 20, -1.5: 25, -1.0: 30, -0.5: 35,
             0.0: 40,  0.5: 50,  1.0: 60,  1.5: 75, 2.0: 90,
        }.get(closest_b, 40)

        print(f"\n  [QuestionAgent] Generating {question_type.upper()} at b={b:.2f}")

        question = None
        for attempt in range(6):          # 6 tries before giving up
            print(f"  [QuestionAgent] Attempt {attempt+1}/6 ...")
            q = (
                self._generate_mcq(p, b, expected_time)
                if question_type == "mcq"
                else self._generate_essay(p, b, expected_time)
            )
            if q is None:
                print(f"  [QuestionAgent] Attempt {attempt+1}: generation returned None")
                continue
            if not self._is_good_quality(q):
                print(f"  [QuestionAgent] Attempt {attempt+1}: failed quality check — question: {q.get('question','')!r}")
                continue
            if self._is_duplicate(q, p["concept"]):
                print(f"  [QuestionAgent] Attempt {attempt+1}: failed duplicate check — question: {q.get('question','')!r}")
                continue
            question = q
            break

        if question is None:
            raise RuntimeError(
                f"[QuestionAgent] Could not generate a valid {question_type} question "
                f"for '{p['concept']}' after 6 attempts. "
                f"Check LLM API key, rate limits, and concept context quality."
            )

        question["concept"]    = p["concept"]
        question["b"]          = b
        question["difficulty"] = DIFFICULTY_LABELS[closest_b]
        question["type"]       = question_type

        self.blackboard.log_thinking(
            "QuestionAgent",
            f"Generated {question_type} question for '{p['concept']}' (b={b:.1f})"
        )
        return question

    # ── ACT ──────────────────────────────────────────────────
    def act(self, decision: dict):
        self.blackboard.write("current_question", decision)

        asked = self.blackboard.read("asked_questions") or []
        asked.append({
            "concept":   decision["concept"],
            "type":      decision["type"],
            "text_hash": hashlib.md5(decision["question"].lower().encode()).hexdigest(),
            "keywords":  self._extract_keywords(decision["question"]),
        })
        self.blackboard.write("asked_questions", asked)

        self.blackboard.write("hint",                        None)
        self.blackboard.write("explanation",                 None)
        self.blackboard.write("videos",                      [])
        self.blackboard.write("hint_available",              True)
        self.blackboard.write("hint_used_current_question",  False)
        # Clear any previous grading result when a new question is served
        self.blackboard.write("last_answer_correct",         None)
        self.blackboard.write("last_answer_feedback",        None)
        self.blackboard.write("next_difficulty_hint",        None)

    # ── MCQ GENERATION ───────────────────────────────────────
    def _generate_mcq(self, p: dict, b: float, expected_time: int) -> dict | None:
        """
        Single-pass MCQ generation.
        The LLM produces the question, 4 options, correct_answer (exact copy of
        the correct option), an explanation, and an answer_key field.

        answer_key is stored in the question object on the blackboard.
        GradingAgent reads it directly — no second LLM call needed for grading.

        The prompt includes a self-check instruction (rule 9) so the LLM validates
        its own output before returning, replacing the old _verify_mcq() second call.
        """
        style = (
            "a straightforward definition question (What is X?)"  if b < -1.0 else
            "a how-it-works question (How does X work / what does it do?)" if b < 0.5 else
            "an application, comparison, or tradeoff question (When/Why/Compare)"
        )

        context_block = p["lecture_context"] or p["summary"] or f"Concept: {p['concept']}"

        prompt_gen = f"""You are a university exam question writer.

LECTURE CONTEXT for the concept "{p['concept']}":
\"\"\"
{context_block[:1500]}
\"\"\"

TASK: Write {style} as a 4-option multiple-choice question.

STRICT RULES — violating any rule means the question is REJECTED:
1. The question and ALL answer options MUST be grounded in the lecture context above.
   Do NOT invent facts, examples, or definitions that are not in the context.
2. Write ONE clear question sentence (do not embed the answer in the question).
3. Write EXACTLY 4 answer options as complete, standalone sentences.
4. Exactly ONE option must be correct and verifiable from the context above.
5. The 3 wrong options must be plausible but clearly incorrect on reflection.
   Wrong options must NOT be obviously absurd or unrelated to the topic.
6. Do NOT include option letters (A/B/C/D) inside the text fields.
7. The "correct_answer" field must be copied EXACTLY (character for character)
   from one of the entries in "options". Any mismatch will cause a crash.
8. The "explanation" must quote or paraphrase from the lecture context above.
9. SELF-CHECK before returning: confirm that correct_answer is an exact copy of
   one of the 4 options. If it is not, fix it before returning.

Return ONLY valid JSON (no markdown, no extra text):
{{
  "question": "...",
  "options": [
    "First full option sentence.",
    "Second full option sentence.",
    "Third full option sentence.",
    "Fourth full option sentence."
  ],
  "correct_answer": "Exact copy of whichever option above is correct.",
  "answer_key": "Exact copy of whichever option above is correct.",
  "explanation": "One sentence from the lecture context that proves the correct answer."
}}"""

        try:
            raw_gen  = call_llm(prompt_gen, max_tokens=500)
            question = parse_json(raw_gen)
        except Exception as e:
            print(f"  [QuestionAgent] MCQ gen error: {e}")
            return None

        # ── Alignment repair ──────────────────────────────────
        import difflib

        opts = question.get("options", [])
        ca   = question.get("correct_answer", "")

        if not opts or len(opts) != 4:
            print(f"  [QuestionAgent] MCQ has {len(opts)} options (need 4), skipping")
            return None

        if ca not in opts:
            ca_norm = ca.strip().lower().rstrip(".")

            # 1) Case-insensitive + strip punctuation match
            match = next(
                (o for o in opts if o.strip().lower().rstrip(".") == ca_norm),
                None
            )
            if not match:
                # 2) correct_answer is a substring of an option (or vice versa)
                match = next(
                    (o for o in opts
                     if ca_norm in o.lower() or o.strip().lower().rstrip(".") in ca_norm),
                    None
                )
            if not match:
                # 3) Fuzzy best-match — accept if similarity >= 0.75
                scored = [
                    (difflib.SequenceMatcher(None, ca_norm, o.strip().lower()).ratio(), o)
                    for o in opts
                ]
                best_score, best_opt = max(scored, key=lambda x: x[0])
                if best_score >= 0.75:
                    match = best_opt
                    print(f"  [QuestionAgent] Fuzzy repair matched (score={best_score:.2f}): {match!r}")

            if match:
                question["correct_answer"] = match
                question["answer_key"]     = match   # keep answer_key in sync
                ca = match
            else:
                print(f"  [QuestionAgent] correct_answer not in options after all repairs:\n"
                      f"    CA   : {ca!r}\n"
                      f"    Opts : {opts}")
                return None

        # Ensure answer_key is always set (LLM may have omitted it)
        question["answer_key"] = ca

        # Shuffle so correct answer is at a random position
        random.shuffle(question["options"])
        question["key_points"]            = [p["concept"]]
        question["expected_time_seconds"] = expected_time
        return question

    # ── ESSAY GENERATION ─────────────────────────────────────
    def _generate_essay(self, p: dict, b: float, expected_time: int) -> dict | None:
        """
        Essay generation with embedded grading_rubric.
        The LLM defines at generation time exactly which key points a correct
        answer must cover. GradingAgent uses this rubric directly — no open-ended
        LLM evaluation of the student's response.
        """
        verb    = "Define" if b < -1.0 else "Explain" if b < 0.5 else "Analyze"
        context = p["lecture_context"] or p["summary"] or f"Concept: {p['concept']}"

        prompt = f"""You are writing a university short-answer exam question.

LECTURE CONTEXT for "{p['concept']}":
\"\"\"
{context[:1200]}
\"\"\"

TASK: Write a "{verb}..." short-answer question strictly based on the context above.

RULES:
1. Start the question with "{verb}".
2. Be specific — reference something concrete from the context, not a generic restatement.
3. Do NOT write "the concept of" or "the idea of".
4. expected_answer: 2-4 sentences a student could write from the context alone.
5. key_points: 3-5 specific things from the context the student must mention.
6. grading_rubric: same list as key_points — these are the exact criteria used to
   grade the student's response. Be concrete (e.g. "mentions that X causes Y")
   not vague (e.g. "understands the concept").

Return ONLY valid JSON:
{{
  "question": "{verb} ...",
  "expected_answer": "model answer using the context",
  "key_points": ["point1", "point2", "point3"],
  "grading_rubric": ["criterion1", "criterion2", "criterion3"]
}}"""

        try:
            raw      = call_llm(prompt, max_tokens=400)
            question = parse_json(raw)
        except Exception as e:
            print(f"  [QuestionAgent] Essay LLM error: {e}")
            return None

        if not question.get("question") or not question.get("expected_answer"):
            print(f"  [QuestionAgent] Essay response missing required fields, will retry")
            return None

        # Fallback: if LLM omitted grading_rubric, use key_points
        if not question.get("grading_rubric"):
            question["grading_rubric"] = question.get("key_points", [])

        question["expected_time_seconds"] = expected_time
        return question

    # ── HELPERS ───────────────────────────────────────────────
    def _select_question_type(self, b: float) -> str:
        for (b_min, b_max), (mcq_w, essay_w) in self.type_distribution.items():
            if b_min <= b < b_max:
                return random.choices(["mcq", "essay"], weights=[mcq_w, essay_w])[0]
        return "essay"

    def _is_good_quality(self, q: dict) -> bool:
        if q is None:
            return False
        text = q.get("question", "").lower()
        bad  = ["widget button", "...", "concept name", "[concept]", "[topic]",
                "your concept here", "insert concept"]
        return len(text) >= 20 and not any(b in text for b in bad)

    def _mcq_is_aligned(self, q: dict) -> bool:
        """Return True iff correct_answer is an exact member of options."""
        return q.get("correct_answer", "") in q.get("options", [])

    def _is_duplicate(self, question: dict, concept: str) -> bool:
        asked    = self.blackboard.read("asked_questions") or []
        new_kw   = set(self._extract_keywords(question["question"]))
        new_hash = hashlib.md5(question["question"].lower().encode()).hexdigest()

        for old in asked:
            if old.get("concept") != concept:
                continue
            if old.get("text_hash") == new_hash:
                return True
            if old.get("type") == question.get("type"):
                if len(new_kw & set(old.get("keywords", []))) >= 6:
                    return True
        return False

    def _extract_keywords(self, text: str) -> list:
        stop = {
            "the","a","an","is","are","was","were","what","how","why","when",
            "where","which","who","and","or","but","in","on","at","of","to",
            "for","with","that","this",
        }
        words = re.findall(r'\b[a-z]{4,}\b', text.lower())
        return sorted(set(w for w in words if w not in stop))[:8]


# ══════════════════════════════════════════════════════════════
class GradingAgent(BaseAgent):
    """
    Grades the student's answer against the stored answer_key (MCQ) or
    grading_rubric (essay) that the LLM embedded at question-generation time.

    MCQ grading: pure string comparison — zero LLM calls.
    Essay grading: one LLM call that checks rubric coverage only, not open-ended
                   evaluation.  This is deterministic and context-grounded.

    Writes to blackboard:
      last_answer_correct   (bool)   — True if the answer is correct
      last_answer_feedback  (str)    — message shown to the student
      next_difficulty_hint  (str)    — "harder" | "easier" | "same"
                                       consumed by IRT/BKT agents to update theta

    The difficulty hint is the bridge between grading and the adaptive engine:
      correct  → "harder"  (push IRT b up, raise BKT mastery)
      wrong    → "easier"  (push IRT b down, lower BKT mastery estimate)
    """

    def perceive(self) -> dict:
        question    = self.blackboard.read("current_question") or {}
        user_answer = self.blackboard.read("last_answer_text") or ""
        return {
            "question":    question,
            "user_answer": user_answer,
            "qtype":       question.get("type", "essay"),
        }

    def reason(self, p: dict) -> dict:
        q           = p["question"]
        user_answer = p["user_answer"].strip()
        qtype       = p["qtype"]

        if qtype == "mcq":
            correct, feedback = self._grade_mcq(q, user_answer)
        else:
            correct, feedback = self._grade_essay(q, user_answer)

        difficulty_hint = "harder" if correct else "easier"

        print(f"\n  [GradingAgent] Result: {'CORRECT' if correct else 'WRONG'} "
              f"→ next difficulty: {difficulty_hint}")

        return {
            "correct":          correct,
            "feedback":         feedback,
            "difficulty_hint":  difficulty_hint,
        }

    def act(self, decision: dict):
        self.blackboard.write("last_answer_correct",  decision["correct"])
        self.blackboard.write("last_answer_feedback", decision["feedback"])
        self.blackboard.write("next_difficulty_hint", decision["difficulty_hint"])
        self.blackboard.log_thinking(
            "GradingAgent",
            f"Graded answer: correct={decision['correct']}, "
            f"hint={decision['difficulty_hint']}"
        )

    # ── MCQ GRADING ───────────────────────────────────────────
    def _grade_mcq(self, question: dict, user_answer: str) -> tuple[bool, str]:
        """
        Pure string comparison — no LLM call.
        Compares the student's answer against answer_key stored at generation time.
        Case-insensitive, stripped of leading/trailing whitespace.
        """
        answer_key = question.get("answer_key") or question.get("correct_answer", "")
        correct    = user_answer.strip().lower() == answer_key.strip().lower()

        if correct:
            feedback = (
                f"Correct! {question.get('explanation', '')}".strip()
            )
        else:
            feedback = (
                f"Not quite. The correct answer is: {answer_key}\n\n"
                f"{question.get('explanation', '')}".strip()
            )
        return correct, feedback

    # ── ESSAY GRADING ─────────────────────────────────────────
    def _grade_essay(self, question: dict, user_answer: str) -> tuple[bool, str]:
        """
        Rubric-based essay grading — ONE LLM call.
        The rubric was defined by the LLM at question-generation time and stored
        in question["grading_rubric"]. The grader only checks whether the student's
        answer covers each criterion — it does not do open-ended evaluation.

        Passing threshold: student must cover at least 60% of rubric criteria.
        """
        rubric = question.get("grading_rubric") or question.get("key_points", [])

        if not rubric:
            # No rubric available — fall back to a simple length-based pass
            correct  = len(user_answer.split()) >= 20
            feedback = (
                "Good effort!"
                if correct
                else f"Try to be more detailed. A model answer would cover: "
                     f"{question.get('expected_answer', '')}"
            )
            return correct, feedback

        rubric_lines = "\n".join(f"  {i+1}. {r}" for i, r in enumerate(rubric))

        prompt = f"""You are grading a student's short-answer response.

Question: {question.get('question', '')}

Grading rubric — the student must cover these points:
{rubric_lines}

Student's answer:
\"\"\"{user_answer[:800]}\"\"\"

For each rubric criterion, decide if the student's answer addresses it (yes/no).
Then compute: covered = number of "yes" answers, total = {len(rubric)}.

Reply ONLY with valid JSON — no markdown, no extra text:
{{
  "criteria": {json.dumps([{"criterion": r, "covered": "yes or no"} for r in rubric])},
  "covered_count": <integer>,
  "total_count": {len(rubric)},
  "missing": ["list of criteria the student did NOT cover"]
}}"""

        try:
            raw    = call_llm(prompt, max_tokens=300)
            result = parse_json(raw)
        except Exception as e:
            print(f"  [GradingAgent] Essay grading LLM error: {e}, using fallback")
            # Fallback: pass if answer is at least 25 words
            correct  = len(user_answer.split()) >= 25
            feedback = (
                "Answer accepted (automated check unavailable)."
                if correct
                else f"Please expand your answer. Expected coverage: {question.get('expected_answer', '')}"
            )
            return correct, feedback

        covered_count = result.get("covered_count", 0)
        total_count   = result.get("total_count", len(rubric))
        missing       = result.get("missing", [])
        ratio         = covered_count / total_count if total_count > 0 else 0
        correct       = ratio >= 0.60   # pass at 60% rubric coverage

        if correct:
            feedback = (
                f"Well done! You covered {covered_count}/{total_count} key points."
                + (f" You could also mention: {'; '.join(missing[:2])}." if missing else "")
            )
        else:
            feedback = (
                f"You covered {covered_count}/{total_count} key points. "
                f"A complete answer should also address: {'; '.join(missing)}.\n\n"
                f"Model answer: {question.get('expected_answer', '')}"
            )
        return correct, feedback


# ══════════════════════════════════════════════════════════════
class FeedbackAgent(BaseAgent):

    def __init__(self, blackboard, youtube_api_key: str = ""):
        super().__init__("FeedbackAgent", blackboard)
        self.youtube_api_key = youtube_api_key

    def perceive(self) -> dict:
        decision = self.blackboard.read("llm_decision") or {}
        return {
            "action":           decision.get("action", "SHOW_EXPLANATION"),
            "concept":          self.blackboard.read("current_concept"),
            "question":         self.blackboard.read("current_question") or {},
            "correct":          self.blackboard.read("last_answer_correct"),
            "answer":           self.blackboard.read("last_answer_text"),
            "difficulty_hint":  self.blackboard.read("next_difficulty_hint"),
        }

    def reason(self, p: dict) -> dict:
        result = {"action": p["action"]}
        if p["action"] == "SHOW_HINT":
            result["hint"] = self._generate_hint(p)
        elif p["action"] == "SHOW_EXPLANATION":
            result["explanation"] = self._generate_explanation(p)
        elif p["action"] == "RECOMMEND_VIDEO":
            result["explanation"] = self._generate_explanation(p)
            result["videos"]      = self._search_youtube(p["concept"])
        return result

    def act(self, decision: dict):
        if "hint" in decision:
            self.blackboard.write("hint",           decision["hint"])
            self.blackboard.write("hint_available", False)
        if "explanation" in decision:
            self.blackboard.write("explanation", decision["explanation"])
        if "videos" in decision:
            self.blackboard.write("videos", decision["videos"])

    def _generate_hint(self, p: dict) -> str:
        q_text  = p["question"].get("question", "this question")
        context = self._get_concept_context(p["concept"])
        try:
            return call_llm(
                f"Give a 2-sentence hint for the question below WITHOUT revealing the answer.\n"
                f"Use only information from the context provided.\n\n"
                f"Context: {context[:600]}\n"
                f"Question: {q_text}\n"
                f"Topic: {p['concept']}",
                max_tokens=120
            )
        except Exception:
            return f"Think carefully about what makes {p['concept']} unique and how the lecture defines it."

    def _generate_explanation(self, p: dict) -> str:
        q                = p["question"]
        qtype            = q.get("type", "essay")
        context          = self._get_concept_context(p["concept"])
        difficulty_hint  = p.get("difficulty_hint")   # "harder" | "easier" | None

        # Adaptive closing line based on performance
        adaptive_note = ""
        if difficulty_hint == "harder":
            adaptive_note = "\nEnd with one encouraging sentence noting the student is ready for more challenging questions on this topic."
        elif difficulty_hint == "easier":
            adaptive_note = "\nEnd with one supportive sentence noting the student should review this concept before moving on."

        if qtype == "mcq":
            prompt = (
                f"Using the lecture context below, explain in 3-4 sentences why the "
                f"correct answer is right and briefly note why the other options are wrong.\n\n"
                f"Lecture context: {context[:600]}\n"
                f"Question: {q.get('question', '')}\n"
                f"Correct answer: {q.get('correct_answer', '')}\n"
                f"Explanation hint: {q.get('explanation', '')}"
                f"{adaptive_note}"
            )
        else:
            prompt = (
                f"Using the lecture context below, explain in 3-4 sentences what a "
                f"good answer should cover. Be specific and reference the context.\n\n"
                f"Lecture context: {context[:600]}\n"
                f"Question: {q.get('question', '')}\n"
                f"Key points expected: {q.get('key_points', [])}\n"
                f"Model answer: {q.get('expected_answer', '')}"
                f"{adaptive_note}"
            )
        try:
            return call_llm(prompt, max_tokens=250)
        except Exception:
            kp = q.get("key_points", [])
            return f"A good answer should cover: {', '.join(kp[:3])}."

    def _get_concept_context(self, concept: str) -> str:
        """Fetch the richer concept context stored by the parser, fall back to summary."""
        contexts = self.blackboard.read("concept_contexts") or {}
        if concept in contexts:
            return contexts[concept]
        graph = self.blackboard.read("concept_graph") or []
        for c in graph:
            if c["concept"] == concept:
                return c.get("summary", "")
        return ""

    def _search_youtube(self, concept: str) -> list:
        if not self.youtube_api_key:
            return [{
                "title":     f"{concept} explained",
                "url":       f"https://www.youtube.com/results?search_query={concept.replace(' ', '+')}+explained",
                "channel":   "YouTube Search",
                "thumbnail": "",
            }]
        try:
            params = {
                "part": "snippet", "q": f"{concept} explained tutorial",
                "type": "video",   "maxResults": 3,
                "key":  self.youtube_api_key, "relevanceLanguage": "en",
            }
            items = http_requests.get(
                "https://www.googleapis.com/youtube/v3/search",
                params=params, timeout=10
            ).json().get("items", [])
            return [{
                "title":     item["snippet"]["title"],
                "url":       f"https://youtube.com/watch?v={item['id']['videoId']}",
                "thumbnail": item["snippet"]["thumbnails"]["medium"]["url"],
                "channel":   item["snippet"]["channelTitle"],
            } for item in items]
        except Exception:
            return [{
                "title":     f"{concept} tutorial",
                "url":       f"https://www.youtube.com/results?search_query={concept.replace(' ', '+')}",
                "channel":   "YouTube Search",
                "thumbnail": "",
            }]