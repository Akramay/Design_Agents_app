"""
question_feedback_agents.py
────────────────────────────
QuestionAgent: Generates questions at exact difficulty (MCQ or Essay)
FeedbackAgent: Generates hints, explanations, and recommends videos

NEW FEATURES:
- Dynamic question type selection (MCQ vs Essay)
- Question deduplication via hashing
- Hint generation system
- Improved explanations for wrong answers
- Question type distribution based on difficulty
"""

import json
import re
import hashlib
import random
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
        # Question type distribution by difficulty
        # Lower b = easier = more MCQ
        # Higher b = harder = more Essay
        self.type_distribution = {
            (-2.0, -1.0): (0.7, 0.3),   # 70% MCQ, 30% Essay
            (-1.0,  0.0): (0.6, 0.4),   # 60% MCQ, 40% Essay
            ( 0.0,  1.0): (0.4, 0.6),   # 40% MCQ, 60% Essay
            ( 1.0,  2.5): (0.2, 0.8),   # 20% MCQ, 80% Essay
        }

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

        return {"concept": concept, "summary": summary, "b": next_b}

    # ── REASON ───────────────────────────────────────────────
    def reason(self, p: dict) -> dict:
        """
        Generate a question at the specified difficulty.
        Randomly select MCQ or Essay based on difficulty.
        """
        b = p["b"]
        
        # Determine question type based on difficulty
        question_type = self._select_question_type(b)
        
        # Find closest difficulty label
        closest_b = min(DIFFICULTY_LABELS.keys(), key=lambda x: abs(x - b))
        difficulty_desc = DIFFICULTY_LABELS[closest_b]

        # Expected time varies with difficulty
        expected_time = {
            -2.0: 15, -1.5: 18, -1.0: 22,
            -0.5: 27,  0.0: 32,  0.5: 38,
             1.0: 45,  1.5: 55,  2.0: 65,
        }.get(closest_b, 30)

        print(f"\n  [QuestionAgent] REASON:")
        print(f"  Question type     : {question_type.upper()}")
        print(f"  Difficulty desc   : {difficulty_desc}")
        print(f"  Expected time     : {expected_time}s")
        print(f"  Calling LLM...")

        # Try to generate unique question (max 3 attempts)
        for attempt in range(3):
            if question_type == "mcq":
                question = self._generate_mcq(p, difficulty_desc, expected_time)
            else:
                question = self._generate_essay(p, difficulty_desc, expected_time)
            
            # Check if this question was already asked
            if not self._is_duplicate(question):
                break
            print(f"  [QuestionAgent] Duplicate detected, regenerating... (attempt {attempt + 1}/3)")
        else:
            print(f"  [QuestionAgent] Warning: Could not generate unique question after 3 attempts")

        question["concept"]    = p["concept"]
        question["b"]          = b
        question["difficulty"] = difficulty_desc
        question["type"]       = question_type

        print(f"\n  [QuestionAgent] Question generated:")
        print(f"  Type: {question_type.upper()}")
        print(f"  Q: {question['question']}")
        if question_type == "mcq":
            print(f"  Options: {question['options']}")
            print(f"  Correct: {question['correct_answer']}")
        else:
            print(f"  Key points: {question['key_points']}")

        self.blackboard.log_thinking(
            "QuestionAgent",
            f"Generated {question_type.upper()} question at b={b:.1f}: \"{question['question'][:60]}...\""
        )

        return question

    # ── ACT ──────────────────────────────────────────────────
    def act(self, decision: dict):
        """Write the generated question to the blackboard."""
        self.blackboard.write("current_question", decision)
        
        # Add to question history
        asked_questions = self.blackboard.read("asked_questions") or []
        question_hash = self._hash_question(decision)
        asked_questions.append(question_hash)
        self.blackboard.write("asked_questions", asked_questions)
        
        # Clear previous feedback and hint
        self.blackboard.write("hint",        None)
        self.blackboard.write("explanation", None)
        self.blackboard.write("videos",      [])
        self.blackboard.write("hint_available", True)
        self.blackboard.write("hint_used_current_question", False)

        print(f"\n  [QuestionAgent] ACT → current_question written to blackboard")

    # ── PRIVATE HELPERS ───────────────────────────────────────

    def _select_question_type(self, b: float) -> str:
        """Select MCQ or Essay based on difficulty level."""
        for (b_min, b_max), (mcq_prob, essay_prob) in self.type_distribution.items():
            if b_min <= b < b_max:
                return random.choices(["mcq", "essay"], weights=[mcq_prob, essay_prob])[0]
        # Default for very hard questions
        return "essay"

    def _generate_mcq(self, p: dict, difficulty_desc: str, expected_time: int) -> dict:
        """Generate a multiple choice question with 4 options."""
        prompt = f"""You are a university tutor for an Intelligent Agents and AI course.

Concept to test: {p['concept']}
Concept context: {p['summary']}

Required difficulty: {difficulty_desc}
(Scale: -2=very easy definition, 0=medium explanation, +2=very hard analysis)

Write EXACTLY ONE multiple choice question at this difficulty level with 4 options (A, B, C, D).

Rules:
- The question must test understanding of {p['concept']}
- Create 3 plausible wrong answers (distractors)
- Make distractors believable but clearly wrong to someone who understands
- Don't make it too obvious which is correct
- Mark one option as correct

Return ONLY this JSON (no markdown, no explanation):
{{
  "question": "the full question text here",
  "options": [
    "Option A text",
    "Option B text", 
    "Option C text",
    "Option D text"
  ],
  "correct_answer": "A",
  "explanation": "why this answer is correct and others are wrong",
  "expected_time_seconds": {expected_time}
}}"""

        try:
            response = ollama.chat(
                model="llama3",
                messages=[{"role": "user", "content": prompt}]
            )
            raw = response["message"]["content"].strip()
            raw = re.sub(r"```json\s*", "", raw)
            raw = re.sub(r"```\s*", "", raw)
            question = json.loads(raw)
            
            # Validate structure
            if not all(k in question for k in ["question", "options", "correct_answer"]):
                raise ValueError("Missing required fields")
            if len(question["options"]) != 4:
                raise ValueError("Must have exactly 4 options")
                
        except Exception as e:
            print(f"  [QuestionAgent] MCQ generation failed: {e}")
            print(f"  [QuestionAgent] Using fallback MCQ...")
            
            question = {
                "question": f"Which of the following best describes {p['concept']}?",
                "options": [
                    f"The basic definition of {p['concept']}",
                    f"An unrelated AI concept",
                    f"A programming language feature",
                    f"A hardware component"
                ],
                "correct_answer": "A",
                "explanation": f"Option A correctly identifies {p['concept']}",
                "expected_time_seconds": expected_time
            }

        # Add key points for hint generation
        question["key_points"] = [p['concept'], question.get("explanation", "")]
        return question

    def _generate_essay(self, p: dict, difficulty_desc: str, expected_time: int) -> dict:
        """Generate an open-ended essay question."""
        prompt = f"""You are a university tutor for an Intelligent Agents and AI course.

Concept to test: {p['concept']}
Concept context: {p['summary']}

Required difficulty: {difficulty_desc}
(Scale: -2=very easy definition, 0=medium explanation, +2=very hard analysis)

Write EXACTLY ONE essay question at this difficulty level.

Rules:
- The question must be answerable from lecture knowledge alone
- Do not give hints or the answer in the question text
- Make it specific, not vague
- For easy: ask for definitions or basic explanations
- For hard: ask for analysis, comparison, or application

Return ONLY this JSON (no markdown, no explanation):
{{
  "question": "the full question text here",
  "expected_answer": "what a correct answer must cover",
  "key_points": ["point 1", "point 2", "point 3"],
  "expected_time_seconds": {expected_time}
}}"""

        try:
            response = ollama.chat(
                model="llama3",
                messages=[{"role": "user", "content": prompt}]
            )
            raw = response["message"]["content"].strip()
            raw = re.sub(r"```json\s*", "", raw)
            raw = re.sub(r"```\s*", "", raw)
            question = json.loads(raw)
            
        except Exception as e:
            print(f"  [QuestionAgent] Essay generation failed: {e}")
            print(f"  [QuestionAgent] Using fallback essay question...")
            
            question = {
                "question": f"Explain the concept of {p['concept']} in your own words.",
                "expected_answer": f"A clear explanation of {p['concept']}",
                "key_points": [p['concept']],
                "expected_time_seconds": expected_time
            }

        return question

    def _hash_question(self, question: dict) -> str:
        """Create a hash of the question to detect duplicates."""
        # Hash based on question text and concept
        text = f"{question.get('concept', '')}:{question.get('question', '')}"
        return hashlib.md5(text.encode()).hexdigest()

    def _is_duplicate(self, question: dict) -> bool:
        """Check if this question was already asked."""
        asked = self.blackboard.read("asked_questions") or []
        question_hash = self._hash_question(question)
        return question_hash in asked


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
        correct  = self.blackboard.read("last_answer_correct")
        answer   = self.blackboard.read("last_answer_text")

        action   = decision.get("action",    "SHOW_EXPLANATION")
        reasoning= decision.get("reasoning", "")

        print(f"\n  [FeedbackAgent] PERCEIVE:")
        print(f"  Action required  : {action}")
        print(f"  Concept          : {concept}")
        print(f"  Answer was       : {'CORRECT' if correct else 'WRONG'}")
        print(f"  LLM reasoning    : {reasoning[:100]}...")

        self.blackboard.log_thinking(
            "FeedbackAgent",
            f"Action needed: {action} for concept '{concept}'. "
            f"Generating appropriate feedback for the student."
        )

        return {
            "action":    action,
            "concept":   concept,
            "question":  question,
            "reasoning": reasoning,
            "correct":   correct,
            "answer":    answer,
        }

    # ── REASON ───────────────────────────────────────────────
    def reason(self, p: dict) -> dict:
        """
        Based on the action, generate the appropriate feedback:
          SHOW_HINT         → short nudge, no answer
          SHOW_EXPLANATION  → full concept explanation + correct answer
          RECOMMEND_VIDEO   → explanation + YouTube search
        """
        result = {"action": p["action"]}
        action = p["action"]

        print(f"\n  [FeedbackAgent] REASON:")

        if action == "SHOW_HINT":
            print(f"  Generating hint (no answer given)...")
            result["hint"] = self._generate_hint(p)

        elif action == "SHOW_EXPLANATION":
            print(f"  Generating full explanation with correct answer...")
            result["explanation"] = self._generate_explanation(p)

        elif action == "RECOMMEND_VIDEO":
            print(f"  Generating explanation + searching YouTube...")
            result["explanation"] = self._generate_explanation(p)
            result["videos"]      = self._search_youtube(p["concept"])

        return result

    # ── ACT ──────────────────────────────────────────────────
    def act(self, decision: dict):
        """Write feedback to the blackboard for the UI to display."""
        if "hint" in decision:
            self.blackboard.write("hint", decision["hint"])
            self.blackboard.write("hint_available", False)  # Can't use hint again
            print(f"\n  [FeedbackAgent] ACT → hint written to blackboard")
            print(f"  Hint: {decision['hint'][:100]}...")

        if "explanation" in decision:
            self.blackboard.write("explanation", decision["explanation"])
            print(f"\n  [FeedbackAgent] ACT → explanation written to blackboard")
            print(f"  Explanation: {decision['explanation'][:100]}...")

        if "videos" in decision:
            self.blackboard.write("videos", decision["videos"])
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
        question_type = p["question"].get("type", "essay")
        
        if question_type == "mcq":
            # For MCQ, give a hint about which options to eliminate
            prompt = f"""A student is struggling with this multiple choice question:
"{p['question'].get('question', '')}"

Options:
{chr(10).join(f"{chr(65+i)}. {opt}" for i, opt in enumerate(p['question'].get('options', [])))}

The correct answer is {p['question'].get('correct_answer', 'A')}.

Write ONE short hint (1-2 sentences) that:
- Helps them eliminate 1-2 obviously wrong options
- Points toward the right way of thinking
- Does NOT reveal the correct answer directly
Just the hint text, nothing else."""
        else:
            # For essay, point toward key concepts
            prompt = f"""A student is struggling with this question:
"{p['question'].get('question', '')}"

The concept being tested: {p['concept']}
Key points they should cover: {p['question'].get('key_points', [])}

Write ONE short hint (2-3 sentences max).
- Point them toward the right way of thinking about {p['concept']}
- Mention ONE key point they should consider
- Do NOT give the full answer
- Do NOT repeat the question
Just the hint text, nothing else."""

        try:
            response = ollama.chat(
                model="llama3",
                messages=[{"role": "user", "content": prompt}]
            )
            return response["message"]["content"].strip()
        except Exception:
            if question_type == "mcq":
                return (
                    f"Think about what makes {p['concept']} unique. "
                    f"One or two of these options don't fit the definition at all."
                )
            else:
                return (
                    f"Focus on the core idea behind {p['concept']}. "
                    f"Think about how it connects to {p['question'].get('key_points', ['the main concept'])[0]}."
                )

    def _generate_explanation(self, p: dict) -> str:
        """
        Generate a clear explanation of the correct answer.
        For wrong answers, explain why they were wrong and what's correct.
        """
        question_type = p["question"].get("type", "essay")
        
        if question_type == "mcq":
            # For MCQ, explain the correct answer and why others are wrong
            correct_answer = p["question"].get("correct_answer", "A")
            options = p["question"].get("options", [])
            correct_idx = ord(correct_answer) - ord('A')
            
            prompt = f"""A student {'answered incorrectly' if not p['correct'] else 'answered correctly'} on this multiple choice question:

Question: "{p['question'].get('question', '')}"

Options:
{chr(10).join(f"{chr(65+i)}. {opt}" for i, opt in enumerate(options))}

The correct answer is: {correct_answer}. {options[correct_idx]}
{'The student chose: ' + p['answer'] if p['answer'] and not p['correct'] else ''}

Write a clear, friendly explanation (3-5 sentences) that:
1. Explains why {correct_answer} is the correct answer
2. Briefly explains why the other options are incorrect
3. Connects this back to the core concept: {p['concept']}
4. {'Addresses why their choice was wrong' if not p['correct'] else 'Reinforces their understanding'}

Just the explanation text, nothing else."""
        else:
            # For essay, explain what a good answer should include
            prompt = f"""A student answered {'incorrectly' if not p['correct'] else 'correctly'} on this concept: {p['concept']}

The question was: "{p['question'].get('question', '')}"
Correct answer should cover: {p['question'].get('key_points', [])}
{'Student wrote: "' + p['answer'][:200] + '"' if not p['correct'] else ''}

Write a clear, friendly explanation (4-6 sentences) that:
1. Explains what {p['concept']} actually means
2. {'Addresses what was missing or incorrect in their answer' if not p['correct'] else 'Reinforces the key points'}
3. Provides a simple real-world example
4. Summarizes the key takeaway they should remember

Just the explanation text, nothing else."""

        try:
            response = ollama.chat(
                model="llama3",
                messages=[{"role": "user", "content": prompt}]
            )
            return response["message"]["content"].strip()
        except Exception:
            if question_type == "mcq":
                correct_answer = p["question"].get("correct_answer", "A")
                options = p["question"].get("options", [])
                correct_idx = ord(correct_answer) - ord('A')
                return (
                    f"The correct answer is {correct_answer}: {options[correct_idx]}. "
                    f"This option correctly captures the essence of {p['concept']}. "
                    f"The other options represent common misconceptions or related but distinct concepts."
                )
            else:
                key_points = p['question'].get('key_points', [p['concept']])
                return (
                    f"{p['concept']} is a key concept in this lecture. "
                    f"A strong answer should mention: {', '.join(key_points[:3])}. "
                    f"Think of how this concept applies in practice and why it matters in the broader context of AI systems."
                )

    def _search_youtube(self, concept: str) -> list:
        """Search YouTube for videos explaining the weak concept."""
        if not self.youtube_api_key:
            print(f"  [FeedbackAgent] No YouTube API key — returning search link")
            return [{
                "title": f"Search YouTube: '{concept} explained'",
                "url":   f"https://www.youtube.com/results?search_query={concept.replace(' ', '+')}+explained",
                "channel": "YouTube Search",
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
                "channel": "YouTube Search",
                "thumbnail": ""
            }]