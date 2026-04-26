"""
Bayesian Knowledge Tracing agent.
"""

from base_agent import BaseAgent


BKT_PARAMS = {
    1: {"p_transit": 0.25, "p_slip": 0.05, "p_guess": 0.25},
    2: {"p_transit": 0.20, "p_slip": 0.08, "p_guess": 0.22},
    3: {"p_transit": 0.17, "p_slip": 0.10, "p_guess": 0.20},
    4: {"p_transit": 0.13, "p_slip": 0.12, "p_guess": 0.18},
    5: {"p_transit": 0.10, "p_slip": 0.15, "p_guess": 0.15},
}

MASTERY_THRESHOLD = 0.85


class BKTAgent(BaseAgent):
    def __init__(self, blackboard):
        super().__init__("BKTAgent", blackboard)

    def perceive(self) -> dict:
        concept = self.blackboard.read("current_concept")
        model = self.blackboard.read("student_model")
        correct = self.blackboard.read("last_answer_correct")
        hint_used = bool(self.blackboard.read("hint_used_current_question"))

        concept_state = model.get(concept, {})
        return {
            "concept": concept,
            "correct": correct,
            "hint_used": hint_used,
            "p_learned": concept_state.get("bkt_p_learned", 0.10),
            "p_transit": concept_state.get("bkt_p_transit", 0.15),
            "p_slip": concept_state.get("bkt_p_slip", 0.10),
            "p_guess": concept_state.get("bkt_p_guess", 0.20),
            "history": concept_state.get("history", []),
        }

    def reason(self, p: dict) -> dict:
        learned = p["p_learned"]
        transit = p["p_transit"]
        slip = p["p_slip"]
        guess = p["p_guess"]

        if p["correct"]:
            observed = learned * (1 - slip) + (1 - learned) * guess
            posterior = (learned * (1 - slip)) / observed
        else:
            observed = learned * slip + (1 - learned) * (1 - guess)
            posterior = (learned * slip) / observed

        if p["hint_used"]:
            if p["correct"]:
                posterior = learned + (posterior - learned) * 0.55
            else:
                posterior = learned + (posterior - learned) * 0.85
            transit *= 0.75

        new_learned = posterior + (1 - posterior) * transit
        new_learned = max(0.001, min(0.999, new_learned))
        mastered = new_learned >= MASTERY_THRESHOLD
        interpretation = (
            "mastered"
            if mastered
            else "used a hint, so confidence update was softened"
            if p["hint_used"]
            else "needs more evidence"
        )

        self.blackboard.log_thinking(
            "BKTAgent",
            f"P(learned) for '{p['concept']}': {learned:.3f} -> {new_learned:.3f}. "
            f"{'Hint used. ' if p['hint_used'] else ''}{interpretation}.",
        )

        return {
            "concept": p["concept"],
            "p_learned": new_learned,
            "mastered": mastered,
            "history": p["history"] + [
                "correct_with_hint" if p["correct"] and p["hint_used"]
                else "wrong_with_hint" if p["hint_used"]
                else "correct" if p["correct"]
                else "wrong"
            ],
        }

    def act(self, decision: dict):
        model = self.blackboard.read("student_model")
        concept = decision["concept"]
        model[concept]["bkt_p_learned"] = decision["p_learned"]
        model[concept]["mastered"] = decision["mastered"]
        model[concept]["history"] = decision["history"]
        self.blackboard.write("student_model", model)
        self.blackboard.write("bkt_mastered", decision["mastered"])