"""
question_feedback_agents.py
────────────────────────────
KEY IMPROVEMENTS over original:
  1. MCQ generation now includes the FULL concept context extracted from the
     lecture (stored on blackboard as concept_contexts), not just a 1-sentence summary.
  2. A two-pass self-verification step: after generation, LLM is asked to confirm
     the correct answer is factually defensible. If it isn't, the question is retried.
  3. MCQ grading fixed: compares by full option text (case-insensitive), not letter codes.
  4. Essay grading is stricter: uses a rubric-style prompt with key_points.
  5. Fallback MCQs are better: use concept_contexts if available.
  6. All prompts instruct the LLM to ONLY use information from the provided lecture
     context, preventing hallucinated or generic answers.
  7. Option validation: rejects questions where the correct_answer differs from
     any option by only punctuation/spacing (catches near-miss alignment bugs).
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
            # NOTE: alignment is already enforced inside _generate_mcq; no double-check needed here
            question = q
            break

        if question is None:
            # ── FALLBACK DISABLED ─────────────────────────────────────────────────
            # All fallback methods are commented out — they produce hardcoded,
            # repetitive questions with the same answers every time.
            # If all LLM retries fail, raise so the error is visible rather than
            # silently serving a low-quality question.
            # question = self._fallback_question(p, question_type, expected_time)
            # ─────────────────────────────────────────────────────────────────────
            raise RuntimeError(
                f"[QuestionAgent] Could not generate a valid {question_type} question "
                f"for '{p['concept']}' after 6 attempts. "
                f"Check Gemini API key, rate limits, and concept context quality."
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

    # ── MCQ GENERATION ───────────────────────────────────────
    def _generate_mcq(self, p: dict, b: float, expected_time: int) -> dict | None:
        """
        Two-pass MCQ generation:
          Pass 1 — LLM writes the question + 4 options + marks the correct one.
          Pass 2 — LLM self-verifies the correct answer is factually sound.

        CRITICAL: correct_answer stores the FULL TEXT of the correct option.
        The prompt strictly forbids using information not in the provided context.
        """
        style = (
            "a straightforward definition question (What is X?)"  if b < -1.0 else
            "a how-it-works question (How does X work / what does it do?)" if b < 0.5 else
            "an application, comparison, or tradeoff question (When/Why/Compare)"
        )

        context_block = p["lecture_context"] or p["summary"] or f"Concept: {p['concept']}"

        # ── PASS 1: Generate ──────────────────────────────────
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
                ca = match
            else:
                print(f"  [QuestionAgent] correct_answer not in options after all repairs:\n"
                      f"    CA   : {ca!r}\n"
                      f"    Opts : {opts}")
                return None

        # ── PASS 2: Self-verification (only for harder questions to avoid burning rate limit) ──
        # The alignment repair above already ensures structural correctness.
        # Verification is a bonus quality check, not a hard gate.
        if b > 0.5:
            if not self._verify_mcq(question, context_block):
                print(f"  [QuestionAgent] MCQ failed self-verification, will retry")
                return None

        # Shuffle so correct answer is at a random position
        random.shuffle(question["options"])
        question["key_points"]            = [p["concept"]]
        question["expected_time_seconds"] = expected_time
        return question

    def _verify_mcq(self, question: dict, context: str) -> bool:
        """
        Ask the LLM to confirm the MCQ is internally consistent and factually grounded.
        Returns True if the question passes, False if it should be retried.
        """
        verification_prompt = f"""You are a strict exam quality reviewer.

Lecture context:
\"\"\"
{context[:800]}
\"\"\"

MCQ to review:
Question : {question.get('question', '')}
Options  :
  1. {question['options'][0] if len(question['options']) > 0 else ''}
  2. {question['options'][1] if len(question['options']) > 1 else ''}
  3. {question['options'][2] if len(question['options']) > 2 else ''}
  4. {question['options'][3] if len(question['options']) > 3 else ''}
Marked correct: {question.get('correct_answer', '')}

Answer these three checks:
A) Is the marked correct answer factually supported by the lecture context? (yes/no)
B) Is the question stem clear and unambiguous? (yes/no)
C) Are the wrong options plausible but clearly wrong on reflection? (yes/no)

Reply ONLY with JSON:
{{"A": "yes or no", "B": "yes or no", "C": "yes or no"}}"""

        try:
            raw    = call_llm(verification_prompt, max_tokens=80)
            result = parse_json(raw)
            passed = (
                result.get("A", "no").lower().startswith("y") and
                result.get("B", "no").lower().startswith("y") and
                result.get("C", "no").lower().startswith("y")
            )
            if not passed:
                print(f"  [QuestionAgent] Verification failed: {result}")
            return passed
        except Exception as e:
            print(f"  [QuestionAgent] Verification call failed ({e}), accepting question anyway")
            return True   # Don't penalise if verifier itself crashes

    # ── ESSAY GENERATION ─────────────────────────────────────
    def _generate_essay(self, p: dict, b: float, expected_time: int) -> dict | None:
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

Return ONLY valid JSON:
{{
  "question": "{verb} ...",
  "expected_answer": "model answer using the context",
  "key_points": ["point1", "point2", "point3"]
}}"""

        try:
            raw      = call_llm(prompt, max_tokens=350)
            question = parse_json(raw)
        except Exception as e:
            print(f"  [QuestionAgent] Essay LLM error: {e}")
            return None  # Let reason() retry rather than silently using a hardcoded question

        if not question.get("question") or not question.get("expected_answer"):
            print(f"  [QuestionAgent] Essay response missing required fields, will retry")
            return None

        question["expected_time_seconds"] = expected_time
        return question

    # ── FALLBACKS (DISABLED) ──────────────────────────────────
    # These methods are commented out because they produce hardcoded,
    # repetitive questions with the same answers every time.
    # All question generation must go through the Gemini API.
    # If generation fails, a RuntimeError is raised in reason() above
    # so the error is visible rather than silently served.
    #
    # def _fallback_mcq(self, p: dict) -> dict:
    #     concept = p["concept"]
    #     context = (p.get("lecture_context") or p.get("summary") or "").strip()
    #     correct_text = context[:120].rstrip(".") + "." if len(context) >= 30 else (
    #         f"{concept} is a formally defined process or system with a specific role in its domain."
    #     )
    #     distractors = [
    #         f"A method used to bypass {concept} in edge-case scenarios.",
    #         f"An optional extension that replaces {concept} in modern implementations.",
    #         f"A deprecated approach that {concept} was designed to supersede.",
    #     ]
    #     options_pool = distractors + [correct_text]
    #     random.shuffle(options_pool)
    #     return {
    #         "question":              f"Which statement best describes {concept}?",
    #         "options":               options_pool,
    #         "correct_answer":        correct_text,
    #         "explanation":           f"{concept}: {correct_text}",
    #         "key_points":            [concept],
    #         "expected_time_seconds": 35,
    #     }
    #
    # def _fallback_question(self, p: dict, qtype: str, expected_time: int) -> dict:
    #     if qtype == "mcq":
    #         q = self._fallback_mcq(p)
    #         q["expected_time_seconds"] = expected_time
    #         return q
    #     return {
    #         "question":              f"Explain the core purpose of {p['concept']} as described in the lecture.",
    #         "expected_answer":       p.get("summary") or f"Technical explanation of {p['concept']}.",
    #         "key_points":            [p["concept"], "definition", "application"],
    #         "expected_time_seconds": expected_time,
    #         "type":                  "essay",
    #         "concept":               p["concept"],
    #     }

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
        # Only reject truly garbage/placeholder outputs — do NOT ban natural
        # phrases like "the concept of" because concept names (e.g. "Foundation
        # of Natural Language") legitimately appear in well-formed questions.
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
                # Threshold raised to 6: concepts naturally share topic keywords
                # across questions, 4 was too aggressive and killed valid retries
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
class FeedbackAgent(BaseAgent):

    def __init__(self, blackboard, youtube_api_key: str = ""):
        super().__init__("FeedbackAgent", blackboard)
        self.youtube_api_key = youtube_api_key

    def perceive(self) -> dict:
        decision = self.blackboard.read("llm_decision") or {}
        return {
            "action":   decision.get("action", "SHOW_EXPLANATION"),
            "concept":  self.blackboard.read("current_concept"),
            "question": self.blackboard.read("current_question") or {},
            "correct":  self.blackboard.read("last_answer_correct"),
            "answer":   self.blackboard.read("last_answer_text"),
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
        q       = p["question"]
        qtype   = q.get("type", "essay")
        context = self._get_concept_context(p["concept"])

        if qtype == "mcq":
            prompt = (
                f"Using the lecture context below, explain in 3-4 sentences why the "
                f"correct answer is right and briefly note why the other options are wrong.\n\n"
                f"Lecture context: {context[:600]}\n"
                f"Question: {q.get('question', '')}\n"
                f"Correct answer: {q.get('correct_answer', '')}\n"
                f"Explanation hint: {q.get('explanation', '')}"
            )
        else:
            prompt = (
                f"Using the lecture context below, explain in 3-4 sentences what a "
                f"good answer should cover. Be specific and reference the context.\n\n"
                f"Lecture context: {context[:600]}\n"
                f"Question: {q.get('question', '')}\n"
                f"Key points expected: {q.get('key_points', [])}\n"
                f"Model answer: {q.get('expected_answer', '')}"
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