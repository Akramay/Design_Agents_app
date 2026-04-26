"""
irt_agent.py
────────────
Item Response Theory Agent.

Tracks the student's ability estimate θ (theta) and
selects the optimal question difficulty b after every answer.

Perceives:  last_answer_correct, student_model, current_concept
Acts:       updates theta and next_b in student_model

KEY MATH:
  P(correct) = 1 / (1 + e^(−a × (θ − b)))

  θ (theta)  = student ability     (starts at 0, range −3 to +3)
  b          = question difficulty  (range −2 to +2)
  a          = discrimination       (fixed at 1.0)

  When θ = b → P(correct) = 0.50 (perfectly balanced)
  We aim for P(correct) ≈ 0.55–0.70 (challenging but achievable)
"""

from math import exp
from base_agent import BaseAgent


# difficulty levels mapped to descriptive labels for the LLM
DIFFICULTY_LABELS = {
    -2.0: "very easy — just define the term in one sentence",
    -1.5: "easy — identify or recognize which concept applies",
    -1.0: "easy-medium — explain what it is and why it matters",
    -0.5: "medium — describe how it works with a simple example",
     0.0: "medium — explain the concept clearly with an example",
     0.5: "medium-hard — compare to a related concept",
     1.0: "hard — apply the concept to a real scenario",
     1.5: "hard — analyze tradeoffs or design decisions",
     2.0: "very hard — critique, design, or deeply analyze",
}

DISCRIMINATION   = 1.0   # a — fixed, controls curve steepness
LEARNING_RATE    = 0.3   # how fast theta shifts per answer
TARGET_P_MIN     = 0.50  # minimum P(correct) we want for next question
TARGET_P_MAX     = 0.70  # maximum P(correct) we want for next question


class IRTAgent(BaseAgent):

    def __init__(self, blackboard):
        super().__init__("IRTAgent", blackboard)

    # ── PERCEIVE ─────────────────────────────────────────────
    def perceive(self) -> dict:
        """Read student ability and last answer from blackboard."""
        concept = self.blackboard.read("current_concept")
        model   = self.blackboard.read("student_model")
        correct = self.blackboard.read("last_answer_correct")

        concept_state = model.get(concept, {})
        theta  = concept_state.get("irt_theta",     0.0)
        b_used = concept_state.get("irt_b_current", 0.0)

        # compute P(correct) that was predicted BEFORE the student answered
        p_predicted = self._logistic(theta, b_used)

        print(f"\n  [IRTAgent] PERCEIVE:")
        print(f"  Concept      : {concept}")
        print(f"  Answer       : {'✓ CORRECT' if correct else '✗ WRONG'}")
        print(f"  θ (theta)    : {theta:.4f}  ← current ability estimate")
        print(f"  b used       : {b_used:.2f}   ← difficulty of question just answered")
        print(f"  P(predicted) : {p_predicted:.4f}  ← what IRT predicted before answer")
        print(f"  Surprise?    : {'No — expected outcome' if (correct and p_predicted > 0.5) or (not correct and p_predicted < 0.5) else 'YES — unexpected! Bigger theta update'}")

        return {
            "concept":     concept,
            "correct":     correct,
            "theta":       theta,
            "b_used":      b_used,
            "p_predicted": p_predicted,
        }

    # ── REASON ───────────────────────────────────────────────
    def reason(self, p: dict) -> dict:
        """
        Step 1: Update theta based on surprise (unexpectedness of answer).
        Step 2: Find new b value that puts P(correct) in target zone.
        
        KEY INSIGHT on the update formula:
        ───────────────────────────────────
        Correct answer on HARD question (P=0.3) → big theta jump (+0.7 × LR)
        Correct answer on EASY question (P=0.9) → small theta jump (+0.1 × LR)
        Wrong answer on EASY question   (P=0.9) → big theta drop  (−0.9 × LR)
        Wrong answer on HARD question   (P=0.3) → small theta drop (−0.3 × LR)

        This mirrors how a real teacher thinks:
        Acing a hard question is very informative.
        Failing an easy question is also very informative.
        """
        theta       = p["theta"]
        b_used      = p["b_used"]
        p_predicted = p["p_predicted"]

        print(f"\n  [IRTAgent] REASON — Updating theta:")
        print(f"  {'─'*44}")

        if p["correct"]:
            # how surprising was this correct answer?
            # if P was low (hard question) → more surprising → bigger update
            delta = LEARNING_RATE * (1 - p_predicted)
            new_theta = theta + delta

            print(f"  CORRECT answer:")
            print(f"  θ_new = θ + LR × (1 − P(predicted))")
            print(f"        = {theta:.4f} + {LEARNING_RATE} × (1 − {p_predicted:.4f})")
            print(f"        = {theta:.4f} + {LEARNING_RATE} × {(1-p_predicted):.4f}")
            print(f"        = {theta:.4f} + {delta:.4f}")
            print(f"        = {new_theta:.4f}")
            print(f"  Reason: {'Answered a HARD question correctly — big jump!' if p_predicted < 0.4 else 'Expected correct answer — small jump'}")

        else:
            # how surprising was this wrong answer?
            # if P was high (easy question) → more surprising → bigger penalty
            delta     = LEARNING_RATE * p_predicted
            new_theta = theta - delta

            print(f"  WRONG answer:")
            print(f"  θ_new = θ − LR × P(predicted)")
            print(f"        = {theta:.4f} − {LEARNING_RATE} × {p_predicted:.4f}")
            print(f"        = {theta:.4f} − {delta:.4f}")
            print(f"        = {new_theta:.4f}")
            print(f"  Reason: {'Failed an EASY question — big drop!' if p_predicted > 0.6 else 'Failed a hard question — small drop'}")

        # clamp theta to valid range
        new_theta = max(-3.0, min(3.0, new_theta))

        # ── PICK NEXT DIFFICULTY ──────────────────────────
        next_b, p_next = self._pick_difficulty(new_theta)

        print(f"")
        print(f"  θ clamped to range [−3, +3]: {new_theta:.4f}")
        print(f"")
        print(f"  ── Selecting next question difficulty ──")
        print(f"  Scanning b values for target P(correct) ∈ [{TARGET_P_MIN}, {TARGET_P_MAX}]:")

        for b_candidate in [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]:
            p_cand = self._logistic(new_theta, b_candidate)
            marker = "  ← SELECTED" if b_candidate == next_b else ""
            in_zone = TARGET_P_MIN <= p_cand <= TARGET_P_MAX
            zone_marker = "✓" if in_zone else " "
            print(f"    b={b_candidate:+.1f}  P(correct)={p_cand:.4f} {zone_marker}{marker}")

        print(f"")
        print(f"  Next b selected : {next_b:.1f}")
        print(f"  P(correct) next : {p_next:.4f}  ← student should succeed ~{p_next*100:.0f}% of the time")
        print(f"  Difficulty label: {DIFFICULTY_LABELS.get(next_b, 'medium')}")

        # ── INTERPRET THETA ───────────────────────────────
        if new_theta < -1.5:
            ability_desc = "Beginner — needs foundational questions"
        elif new_theta < -0.5:
            ability_desc = "Below average — building understanding"
        elif new_theta < 0.5:
            ability_desc = "Average — solid foundation"
        elif new_theta < 1.5:
            ability_desc = "Above average — ready for challenge"
        else:
            ability_desc = "Expert level — use hardest questions"

        print(f"  Ability status  : {ability_desc}")

        self.blackboard.log_thinking(
            "IRTAgent",
            f"θ updated: {theta:.3f} → {new_theta:.3f} "
            f"({'▲ up' if new_theta > theta else '▼ down'}). "
            f"Next question difficulty: b={next_b:.1f} "
            f"(P(correct)≈{p_next:.2f}). "
            f"Student level: {ability_desc}"
        )

        return {
            "concept":   p["concept"],
            "theta":     new_theta,
            "next_b":    next_b,
            "p_next":    p_next,
        }

    # ── ACT ──────────────────────────────────────────────────
    def act(self, decision: dict):
        """Write updated theta and next_b to the student model."""
        model   = self.blackboard.read("student_model")
        concept = decision["concept"]

        model[concept]["irt_theta"]     = decision["theta"]
        model[concept]["irt_b_current"] = decision["next_b"]

        self.blackboard.write("student_model", model)
        self.blackboard.write("next_b",        decision["next_b"])

        print(f"\n  [IRTAgent] ACT → Blackboard updated")
        print(f"  student_model['{concept}']['irt_theta']     = {decision['theta']:.4f}")
        print(f"  student_model['{concept}']['irt_b_current'] = {decision['next_b']:.2f}")
        print(f"  next_b (for QuestionAgent)                  = {decision['next_b']:.2f}")

    # ── PRIVATE HELPERS ───────────────────────────────────────

    def _logistic(self, theta: float, b: float) -> float:
        """
        The IRT logistic function.
        Returns P(correct answer) given student ability and question difficulty.
        """
        return 1 / (1 + exp(-DISCRIMINATION * (theta - b)))

    def _pick_difficulty(self, theta: float):
        """
        Scan candidate b values and return the one that puts
        P(correct) in the target zone [0.50, 0.70].
        If none found, return the closest to the ideal 0.60.
        """
        candidates = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
        best_b, best_p, best_dist = 0.0, 0.5, float("inf")

        for b in candidates:
            p = self._logistic(theta, b)
            if TARGET_P_MIN <= p <= TARGET_P_MAX:
                return b, p  # first valid one found
            # track closest to ideal 0.60 as fallback
            dist = abs(p - 0.60)
            if dist < best_dist:
                best_b, best_p, best_dist = b, p, dist

        return best_b, best_p  # fallback: closest to 0.60
