"""
question_agent.py
─────────────────
Generates a question at the exact difficulty level IRT selected.

Perceives:  current_concept, next_b, concept summary
Acts:       current_question  (written to blackboard)

feedback_agent.py is also in this file:
Handles wrong answers — generates hints, explanations, YouTube videos.

Perceives:  llm_decision, current_concept, current_question
Acts:       hint, explanation, videos  (written to blackboard)
"""

import json
import re
import requests
import ollama

from base_agent import BaseAgent
from irt_agent import DIFFICULTY_LABELS


# ══════════════════════════════════════════════════════════════
#  QUESTION AGENT
# ══════════════════════════════════════════════════════════════

class QuestionAgent(BaseAgent):

    def __init__(self, blackboard):
        super().__init__("QuestionAgent", blackboard)

    # ── PERCEIVE ─────────────────────────────────────────────
    def perceive(self) -> dict:
        """Read the concept and difficulty from the blackboard."""
        concept = self.blackboard.read("current_concept")
        graph   = self.blackboard.read("concept_graph") or []
        next_b  = self.blackboard.read("next_b")
        if next_b is None:
            next_b = 0.0  # default: medium difficulty

        # get summary for this concept from the graph
        summary = ""
        for c in graph:
            if c["concept"] == concept:
                summary = c.get("summary", "")
                break

        print(f"\n  [QuestionAgent] PERCEIVE:")
        print(f"  Concept   : {concept}")
        print(f"  next_b    : {next_b:.2f}  ← difficulty IRT selected")
        print(f"  Summary   : {summary[:80]}...")

        self.blackboard.log_thinking(
            "QuestionAgent",
            f"I need to write a question about '{concept}' "
            f"at difficulty b={next_b:.1f}. "
            f"Calling LLM to generate question..."
        )

        return {"concept": concept, "summary": summary, "b": next_b}

    # ── REASON ───────────────────────────────────────────────
    def reason(self, p: dict) -> dict:
        """Call LLM to generate a question at the specified difficulty."""
        b = p["b"]

        # find closest difficulty label
        closest_b = min(DIFFICULTY_LABELS.keys(), key=lambda x: abs(x - b))
        difficulty_desc = DIFFICULTY_LABELS[closest_b]

        # expected time varies with difficulty
        expected_time = {
            -2.0: 15, -1.5: 18, -1.0: 22,
            -0.5: 27,  0.0: 32,  0.5: 38,
             1.0: 45,  1.5: 55,  2.0: 65,
        }.get(closest_b, 30)

        print(f"\n  [QuestionAgent] REASON:")
        print(f"  Difficulty desc   : {difficulty_desc}")
        print(f"  Expected time     : {expected_time}s")
        print(f"  Calling LLM...")

        prompt = f"""You are a university tutor for an Intelligent Agents and AI course.

Concept to test: {p['concept']}
Concept context: {p['summary']}

Required difficulty: {difficulty_desc}
(Scale: -2=very easy definition, 0=medium explanation, +2=very hard analysis)

Write EXACTLY ONE question at this difficulty level.

Rules:
- The question must be answerable from lecture knowledge alone
- Do not give hints or the answer in the question text
- Make it specific, not vague

Return ONLY this JSON (no markdown, no explanation):
{{
  "question": "the full question text here",
  "expected_answer": "what a correct answer must cover",
  "key_points": ["point 1", "point 2", "point 3"],
  "expected_time_seconds": {expected_time}
}}"""

        response = ollama.chat(
            model="llama3",
            messages=[{"role": "user", "content": prompt}]
        )

        raw = response["message"]["content"].strip()
        # strip markdown if present
        raw = re.sub(r"```json\s*", "", raw)
        raw = re.sub(r"```\s*", "", raw)

        try:
            question = json.loads(raw)
        except json.JSONDecodeError:
            # fallback question if LLM returns invalid JSON
            question = {
                "question": f"Explain the concept of {p['concept']} in your own words.",
                "expected_answer": f"A clear explanation of {p['concept']}",
                "key_points": [p['concept']],
                "expected_time_seconds": expected_time
            }

        question["concept"]    = p["concept"]
        question["b"]          = b
        question["difficulty"] = difficulty_desc

        print(f"\n  [QuestionAgent] Question generated:")
        print(f"  Q: {question['question']}")
        print(f"  Expected time: {question['expected_time_seconds']}s")
        print(f"  Key points: {question['key_points']}")

        self.blackboard.log_thinking(
            "QuestionAgent",
            f"Question generated at b={b:.1f}: \"{question['question'][:80]}...\""
        )

        return question

    # ── ACT ──────────────────────────────────────────────────
    def act(self, decision: dict):
        """Write the generated question to the blackboard."""
        self.blackboard.write("current_question", decision)
        # clear previous feedback when new question is set
        self.blackboard.write("hint",        None)
        self.blackboard.write("explanation", None)
        self.blackboard.write("videos",      [])

        print(f"\n  [QuestionAgent] ACT → current_question written to blackboard")


# ══════════════════════════════════════════════════════════════
#  FEEDBACK AGENT
# ══════════════════════════════════════════════════════════════

class FeedbackAgent(BaseAgent):

    def __init__(self, blackboard, youtube_api_key: str = ""):
        super().__init__("FeedbackAgent", blackboard)
        self.youtube_api_key = youtube_api_key

    # ── PERCEIVE ─────────────────────────────────────────────
    def perceive(self) -> dict:
        """Read the LLM decision and current context."""
        decision = self.blackboard.read("llm_decision") or {}
        concept  = self.blackboard.read("current_concept")
        question = self.blackboard.read("current_question") or {}

        action   = decision.get("action",    "SHOW_EXPLANATION")
        reasoning= decision.get("reasoning", "")

        print(f"\n  [FeedbackAgent] PERCEIVE:")
        print(f"  Action required  : {action}")
        print(f"  Concept          : {concept}")
        print(f"  LLM reasoning    : {reasoning[:100]}...")

        self.blackboard.log_thinking(
            "FeedbackAgent",
            f"Action needed: {action} for concept '{concept}'. "
            f"I will generate appropriate feedback for the student."
        )

        return {
            "action":    action,
            "concept":   concept,
            "question":  question,
            "reasoning": reasoning,
        }

    # ── REASON ───────────────────────────────────────────────
    def reason(self, p: dict) -> dict:
        """
        Based on the action, generate the appropriate feedback:
          SHOW_HINT         → short nudge, no answer
          SHOW_EXPLANATION  → full concept explanation
          RECOMMEND_VIDEO   → explanation + YouTube search
        """
        result = {"action": p["action"]}
        action = p["action"]

        print(f"\n  [FeedbackAgent] REASON:")

        if action == "SHOW_HINT":
            print(f"  Generating hint (no answer given)...")
            result["hint"] = self._generate_hint(p)

        elif action == "SHOW_EXPLANATION":
            print(f"  Generating full explanation...")
            result["explanation"] = self._generate_explanation(p)

        elif action == "RECOMMEND_VIDEO":
            print(f"  Generating explanation + searching YouTube...")
            result["explanation"] = self._generate_explanation(p)
            result["videos"]      = self._search_youtube(p["concept"])

        return result

    # ── ACT ──────────────────────────────────────────────────
    def act(self, decision: dict):
        """Write feedback to the blackboard for the UI to display."""
        if "hint"        in decision:
            self.blackboard.write("hint",        decision["hint"])
            print(f"\n  [FeedbackAgent] ACT → hint written to blackboard")
            print(f"  Hint: {decision['hint'][:100]}...")

        if "explanation" in decision:
            self.blackboard.write("explanation", decision["explanation"])
            print(f"\n  [FeedbackAgent] ACT → explanation written to blackboard")
            print(f"  Explanation: {decision['explanation'][:100]}...")

        if "videos"      in decision:
            self.blackboard.write("videos",      decision["videos"])
            print(f"\n  [FeedbackAgent] ACT → {len(decision['videos'])} videos written to blackboard")
            for v in decision["videos"]:
                print(f"    • {v['title']}")
                print(f"      {v['url']}")

        self.blackboard.log_thinking(
            "FeedbackAgent",
            f"Feedback ready: {decision['action']}. "
            + (f"Hint provided." if "hint" in decision else "")
            + (f"Explanation provided." if "explanation" in decision else "")
            + (f"{len(decision.get('videos', []))} videos recommended." if "videos" in decision else "")
        )

    # ── PRIVATE HELPERS ───────────────────────────────────────

    def _generate_hint(self, p: dict) -> str:
        """Generate a hint that points toward the answer without revealing it."""
        prompt = f"""A student is struggling with this question:
"{p['question'].get('question', '')}"

The concept being tested: {p['concept']}
Diagnosis from tutoring agent: {p['reasoning']}

Write ONE short hint (2-3 sentences max).
- Point them toward the right way of thinking
- Do NOT give the answer
- Do NOT repeat the question
Just the hint text, nothing else."""

        response = ollama.chat(
            model="llama3",
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"].strip()

    def _generate_explanation(self, p: dict) -> str:
        """Generate a clear explanation of the concept the student got wrong."""
        prompt = f"""A student answered incorrectly on this concept: {p['concept']}

The question was: "{p['question'].get('question', '')}"
Correct answer should cover: {p['question'].get('key_points', [])}

Tutoring agent diagnosis: {p['reasoning']}

Write a clear, friendly explanation (3-5 sentences) that:
1. Explains what {p['concept']} actually means
2. Addresses the likely misunderstanding
3. Gives a simple real-world example

Just the explanation text, nothing else."""

        response = ollama.chat(
            model="llama3",
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"].strip()

    def _search_youtube(self, concept: str) -> list:
        """Search YouTube for videos explaining the weak concept."""
        if not self.youtube_api_key:
            print(f"  [FeedbackAgent] No YouTube API key — returning placeholder")
            return [{
                "title": f"Search YouTube: '{concept} explained'",
                "url":   f"https://www.youtube.com/results?search_query={concept.replace(' ', '+')}+explained",
                "thumbnail": ""
            }]

        try:
            url = "https://www.googleapis.com/youtube/v3/search"
            params = {
                "part":       "snippet",
                "q":          f"{concept} explained tutorial",
                "type":       "video",
                "maxResults": 3,
                "key":        self.youtube_api_key,
                "relevanceLanguage": "en",
            }
            response = requests.get(url, params=params, timeout=10)
            items    = response.json().get("items", [])

            videos = []
            for item in items:
                videos.append({
                    "title":     item["snippet"]["title"],
                    "url":       f"https://youtube.com/watch?v={item['id']['videoId']}",
                    "thumbnail": item["snippet"]["thumbnails"]["medium"]["url"],
                    "channel":   item["snippet"]["channelTitle"],
                })
            return videos

        except Exception as e:
            print(f"  [FeedbackAgent] YouTube search failed: {e}")
            return [{
                "title": f"{concept} explained",
                "url":   f"https://www.youtube.com/results?search_query={concept.replace(' ', '+')}",
                "thumbnail": ""
            }]
