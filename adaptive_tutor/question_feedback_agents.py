"""
question_feedback_agents.py
────────────────────────────
Uses the free llm_provider module (Gemini primary, Groq fallback).

Key fixes vs. original:
  - correct_answer stores full option TEXT, not a letter like "A"
  - options are shuffled so correct answer isn't always first
  - fallback MCQs randomise which slot holds the correct answer
  - duplicate threshold raised (4 keywords, 8 keyword slots)
  - essay grading uses expected_answer + key_points properly
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
        concept = self.blackboard.read("current_concept")
        graph   = self.blackboard.read("concept_graph") or []
        next_b  = self.blackboard.read("next_b") or 0.0
        summary = next(
            (c.get("summary", "") for c in graph if c["concept"] == concept), ""
        )
        return {"concept": concept, "summary": summary, "b": next_b}

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
        for attempt in range(3):
            q = (
                self._generate_mcq(p, b, expected_time)
                if question_type == "mcq"
                else self._generate_essay(p, b, expected_time)
            )
            if not self._is_good_quality(q):
                print(f"  [QuestionAgent] Low quality, retry {attempt+1}/3")
                continue
            if self._is_duplicate(q, p["concept"]):
                print(f"  [QuestionAgent] Duplicate, retry {attempt+1}/3")
                continue
            question = q
            break

        if question is None:
            question = self._fallback_question(p, question_type, expected_time)

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
    def _generate_mcq(self, p: dict, b: float, expected_time: int) -> dict:
        """
        IMPORTANT: correct_answer = full text of the correct option (not "A").
        Options are shuffled so the correct one is at a random position.
        """
        style = (
            "a definition question (what is it?)"                    if b < -1.0 else
            "a how-it-works question (how does it work?)"             if b <  0.5 else
            "an application or tradeoff question (when/why/compare)"
        )

        prompt = f"""You are writing a university exam question about: {p['concept']}

Background: {p['summary']}

Write {style} as a multiple-choice question.

STRICT RULES:
1. Write ONE clear question sentence.
2. Write EXACTLY 4 answer options as complete sentences.
3. Exactly ONE option must be correct and clearly defensible.
4. The 3 wrong options must be plausible but clearly incorrect on reflection.
5. Do NOT use option letters (A/B/C/D) inside the text.
6. Do NOT write "the concept of" or "the idea of" anywhere.
7. The correct_answer field must contain the EXACT TEXT of the correct option.

Return ONLY valid JSON, no markdown, no explanation:
{{
  "question": "...",
  "options": ["full text 1", "full text 2", "full text 3", "full text 4"],
  "correct_answer": "exact copy of the correct option text",
  "explanation": "one sentence explaining why the correct answer is right"
}}"""

        try:
            raw      = call_llm(prompt, max_tokens=400)
            question = parse_json(raw)

            opts = question.get("options", [])
            ca   = question.get("correct_answer", "")

            if ca not in opts:
                # Case-insensitive repair
                match = next(
                    (o for o in opts if o.strip().lower() == ca.strip().lower()), None
                )
                if match:
                    question["correct_answer"] = match
                else:
                    raise ValueError(
                        f"correct_answer not in options.\ncorrect_answer: {ca!r}\noptions: {opts}"
                    )

            random.shuffle(question["options"])

        except Exception as e:
            print(f"  [QuestionAgent] MCQ LLM error: {e}")
            question = self._fallback_mcq(p)

        question["key_points"]            = [p["concept"]]
        question["expected_time_seconds"] = expected_time
        return question

    # ── ESSAY GENERATION ─────────────────────────────────────
    def _generate_essay(self, p: dict, b: float, expected_time: int) -> dict:
        verb = "Define" if b < -1.0 else "Explain" if b < 0.5 else "Analyze"

        prompt = f"""You are writing a university short-answer question about: {p['concept']}

Background: {p['summary']}

Write a {verb.lower()} question.

RULES:
1. Start the question with "{verb}".
2. Be specific — do NOT write "the concept of" or "the idea of".
3. expected_answer: 2-4 sentences a student could realistically write.
4. key_points: 3-5 distinct things the student must mention.

Return ONLY valid JSON:
{{
  "question": "...",
  "expected_answer": "model answer",
  "key_points": ["point1", "point2", "point3"]
}}"""

        try:
            raw      = call_llm(prompt, max_tokens=300)
            question = parse_json(raw)
        except Exception as e:
            print(f"  [QuestionAgent] Essay LLM error: {e}")
            question = {
                "question":        f"{verb} {p['concept']} and provide a concrete example.",
                "expected_answer": p["summary"] or f"A clear explanation of {p['concept']}.",
                "key_points":      [p["concept"], "definition", "example"],
            }

        question["expected_time_seconds"] = expected_time
        return question

    # ── FALLBACKS ─────────────────────────────────────────────
    def _fallback_mcq(self, p: dict) -> dict:
        concept = p["concept"]
        summary = (p.get("summary") or "").strip().lstrip("•–-– ").strip()

        # Reject the summary if it looks like garbage, an email, a placeholder,
        # or a sentence fragment (anything that would make a nonsense answer option).
        _bad_markers = [
            "@", ".edu", ".com", "http", "core technical topic",
            "key concept covered", "fundamental mechanism",
            "dr.", "prof", "csc", "miuegypt",
        ]
        _summary_ok = (
            summary
            and len(summary) >= 30
            and " " in summary                          # must have multiple words
            and summary[0].isalpha()                    # starts with a real letter
            and not any(m in summary.lower() for m in _bad_markers)
        )

        if _summary_ok:
            correct_text = summary[:120].rstrip(".") + "."
        else:
            correct_text = (
                f"{concept} is a formal system or process with a defined structure "
                f"and a specific role in language analysis."
            )

        distractors = [
            f"A method used to bypass {concept} in edge-case scenarios.",
            f"An optional extension that replaces {concept} in modern implementations.",
            f"A deprecated approach that {concept} was designed to supersede.",
        ]

        options_pool = distractors + [correct_text]
        random.shuffle(options_pool)

        return {
            "question":              f"Which statement best describes the role of {concept}?",
            "options":               options_pool,
            "correct_answer":        correct_text,
            "explanation":           f"{concept} is central to this topic. The other options describe things it does not do.",
            "key_points":            [concept],
            "expected_time_seconds": 35,
        }

    def _fallback_question(self, p: dict, qtype: str, expected_time: int) -> dict:
    # Force the question to change so the duplicate detector doesn't loop
        templates = [
            f"Analyze the core principles of {p['concept']} and their impact on the field.",
            f"What are the defining characteristics that separate {p['concept']} from similar topics?",
                f"Provide a detailed technical breakdown of how {p['concept']} is implemented.",
        f"Explain the historical development and current state of {p['concept']}."
        ]
    
        chosen_q = random.choice(templates)
    
        return {
        "question": chosen_q,
        "expected_answer": f"Technical explanation of {p['concept']}.",
        "key_points": [p["concept"], "analysis", "application"],
        "expected_time_seconds": expected_time,
        "type": "essay", # Force essay for fallbacks to avoid broken MCQ options
        "concept": p["concept"]
        }

    # ── HELPERS ───────────────────────────────────────────────
    def _select_question_type(self, b: float) -> str:
        for (b_min, b_max), (mcq_w, essay_w) in self.type_distribution.items():
            if b_min <= b < b_max:
                return random.choices(["mcq", "essay"], weights=[mcq_w, essay_w])[0]
        return "essay"

    def _is_good_quality(self, q: dict) -> bool:
        text = q.get("question", "").lower()
        bad  = ["the concept of", "the idea of", "widget button"]
        return len(text) >= 15 and not any(b in text for b in bad)

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
                if len(new_kw & set(old.get("keywords", []))) >= 4:
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
        q_text = p["question"].get("question", "this question")
        try:
            return call_llm(
                f"Give a 2-sentence hint for this question without revealing the answer:\n"
                f"Question: {q_text}\nTopic: {p['concept']}",
                max_tokens=100
            )
        except Exception:
            return f"Think about what makes {p['concept']} unique and how it's applied."

    def _generate_explanation(self, p: dict) -> str:
        q     = p["question"]
        qtype = q.get("type", "essay")
        if qtype == "mcq":
            prompt = (
                f"Explain in 3-4 sentences why this is the correct answer.\n"
                f"Question: {q.get('question', '')}\n"
                f"Correct answer: {q.get('correct_answer', '')}\n"
                f"Explanation hint: {q.get('explanation', '')}"
            )
        else:
            prompt = (
                f"Explain what a good answer should cover (3-4 sentences).\n"
                f"Question: {q.get('question', '')}\n"
                f"Topic: {p['concept']}\n"
                f"Key points: {q.get('key_points', [])}\n"
                f"Model answer: {q.get('expected_answer', '')}"
            )
        try:
            return call_llm(prompt, max_tokens=200)
        except Exception:
            kp = q.get("key_points", [])
            return f"A good answer should cover: {', '.join(kp[:3])}."

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