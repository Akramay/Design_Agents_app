"""
bkt_agent.py
────────────
Bayesian Knowledge Tracing Agent.

Tracks ONE number per concept: P(learned)
Updates it after every student answer using Bayes' theorem.

Perceives:  last_answer_correct, student_model, current_concept
Acts:       updates student_model with new P(learned) + mastered flag

The 4 BKT parameters (set once, never change):
  P(learned)  — prior: how likely they knew it before starting
  P(transit)  — how much each question attempt teaches them
  P(slip)     — chance of answering wrong even if they know it
  P(guess)    — chance of answering right even if they don't know it
"""

from base_agent import BaseAgent


# ── BKT parameter presets per difficulty tier ─────────────────
# These are reasonable educational defaults.
# In a research system you would learn these from data.
BKT_PARAMS = {
    1: {"p_transit": 0.25, "p_slip": 0.05, "p_guess": 0.25},  # very easy
    2: {"p_transit": 0.20, "p_slip": 0.08, "p_guess": 0.22},  # easy
    3: {"p_transit": 0.17, "p_slip": 0.10, "p_guess": 0.20},  # medium
    4: {"p_transit": 0.13, "p_slip": 0.12, "p_guess": 0.18},  # hard
    5: {"p_transit": 0.10, "p_slip": 0.15, "p_guess": 0.15},  # very hard
}

MASTERY_THRESHOLD = 0.85  # P(learned) must exceed this to mark mastered


class BKTAgent(BaseAgent):

    def __init__(self, blackboard):
        super().__init__("BKTAgent", blackboard)

    # ── PERCEIVE ─────────────────────────────────────────────
    def perceive(self) -> dict:
        """Read the answer result and current student knowledge state."""
        concept = self.blackboard.read("current_concept")
        model   = self.blackboard.read("student_model")
        correct = self.blackboard.read("last_answer_correct")

        concept_state = model.get(concept, {})

        perception = {
            "concept":   concept,
            "correct":   correct,
            "p_learned": concept_state.get("bkt_p_learned", 0.10),
            "p_transit": concept_state.get("bkt_p_transit", 0.15),
            "p_slip":    concept_state.get("bkt_p_slip",    0.10),
            "p_guess":   concept_state.get("bkt_p_guess",   0.20),
            "attempts":  concept_state.get("attempts",       0),
            "history":   concept_state.get("history",        []),
        }

        print(f"\n  [BKTAgent] PERCEIVE:")
        print(f"  Concept    : {concept}")
        print(f"  Answer     : {'✓ CORRECT' if correct else '✗ WRONG'}")
        print(f"  P(learned) : {perception['p_learned']:.4f}  ← before update")
        print(f"  P(transit) : {perception['p_transit']:.4f}  (fixed — how fast concept is learned)")
        print(f"  P(slip)    : {perception['p_slip']:.4f}  (fixed — know it but answer wrong)")
        print(f"  P(guess)   : {perception['p_guess']:.4f}  (fixed — don't know but guess right)")
        print(f"  Attempts   : {perception['attempts']}")
        print(f"  History    : {perception['history'][-8:]}")

        return perception

    # ── REASON ───────────────────────────────────────────────
    def reason(self, p: dict) -> dict:
        """
        Run the full BKT update equations step by step.
        
        MATH EXPLANATION:
        ─────────────────
        We want to find: "given that student answered correctly/wrongly,
        what is the updated probability they have learned this?"
        
        This is Bayes' theorem:
          P(learned | evidence) = P(evidence | learned) × P(learned) / P(evidence)
        """
        L  = p["p_learned"]
        PT = p["p_transit"]
        PS = p["p_slip"]
        PG = p["p_guess"]

        print(f"\n  [BKTAgent] REASON — Running Bayes update:")
        print(f"  {'─'*44}")

        if p["correct"]:
            # ── CORRECT ANSWER UPDATE ──────────────────────
            # P(correct) = chance of correct regardless of knowledge
            p_correct_obs = L * (1 - PS) + (1 - L) * PG

            # P(learned | correct) = P(correct | learned) × P(learned) / P(correct)
            posterior = (L * (1 - PS)) / p_correct_obs

            print(f"  Answer was CORRECT:")
            print(f"  P(obs=correct) = P(L)×(1−P(slip)) + (1−P(L))×P(guess)")
            print(f"                 = {L:.4f}×{(1-PS):.4f} + {(1-L):.4f}×{PG:.4f}")
            print(f"                 = {L*(1-PS):.4f} + {(1-L)*PG:.4f}")
            print(f"                 = {p_correct_obs:.4f}")
            print(f"")
            print(f"  P(L|correct)   = P(L)×(1−P(slip)) / P(correct)")
            print(f"                 = {L:.4f}×{(1-PS):.4f} / {p_correct_obs:.4f}")
            print(f"                 = {L*(1-PS):.4f} / {p_correct_obs:.4f}")
            print(f"                 = {posterior:.4f}")

        else:
            # ── WRONG ANSWER UPDATE ───────────────────────
            p_wrong_obs = L * PS + (1 - L) * (1 - PG)
            posterior   = (L * PS) / p_wrong_obs

            print(f"  Answer was WRONG:")
            print(f"  P(obs=wrong) = P(L)×P(slip) + (1−P(L))×(1−P(guess))")
            print(f"               = {L:.4f}×{PS:.4f} + {(1-L):.4f}×{(1-PG):.4f}")
            print(f"               = {L*PS:.4f} + {(1-L)*(1-PG):.4f}")
            print(f"               = {p_wrong_obs:.4f}")
            print(f"")
            print(f"  P(L|wrong)   = P(L)×P(slip) / P(wrong)")
            print(f"               = {L:.4f}×{PS:.4f} / {p_wrong_obs:.4f}")
            print(f"               = {L*PS:.4f} / {p_wrong_obs:.4f}")
            print(f"               = {posterior:.4f}")

        # ── TRANSIT STEP ──────────────────────────────────
        # Even after updating, maybe student learned from this attempt
        new_L = posterior + (1 - posterior) * PT
        new_L = max(0.001, min(0.999, new_L))  # clamp to (0, 1)

        print(f"")
        print(f"  + Transit (learned from attempt):")
        print(f"  P(L|final) = {posterior:.4f} + (1 − {posterior:.4f}) × {PT:.4f}")
        print(f"             = {posterior:.4f} + {(1-posterior)*PT:.4f}")
        print(f"             = {new_L:.4f}")

        # ── MASTERY CHECK ─────────────────────────────────
        mastered     = new_L >= MASTERY_THRESHOLD
        delta        = new_L - L
        direction    = "▲ INCREASED" if delta > 0 else "▼ DECREASED"

        print(f"")
        print(f"  {'─'*44}")
        print(f"  P(learned): {L:.4f} → {new_L:.4f}  ({direction} by {abs(delta):.4f})")
        print(f"  Mastered  : {'YES ✓ (P ≥ 0.85 — move to next concept)' if mastered else f'NO  (need {MASTERY_THRESHOLD - new_L:.4f} more)'}")

        # ── INTERPRET THE NUMBER ──────────────────────────
        if new_L < 0.35:
            interpretation = "Student has NOT learned this — needs easier questions or explanation"
        elif new_L < 0.60:
            interpretation = "Uncertain — need more evidence, keep testing"
        elif new_L < MASTERY_THRESHOLD:
            interpretation = "Likely learned — close to mastery, continue"
        else:
            interpretation = "MASTERED — ready for next concept"

        print(f"  Status    : {interpretation}")

        self.blackboard.log_thinking(
            "BKTAgent",
            f"P(learned) for '{p['concept']}': {L:.3f} → {new_L:.3f} "
            f"({'✓ CORRECT' if p['correct'] else '✗ WRONG'} answer). "
            f"{'Mastered! Moving on.' if mastered else interpretation}"
        )

        return {
            "concept":    p["concept"],
            "p_learned":  new_L,
            "mastered":   mastered,
            "delta":      delta,
            "history":    p["history"] + ["correct" if p["correct"] else "wrong"],
        }

    # ── ACT ──────────────────────────────────────────────────
    def act(self, decision: dict):
        """Write updated P(learned) back into the student model on the blackboard."""
        model   = self.blackboard.read("student_model")
        concept = decision["concept"]

        model[concept]["bkt_p_learned"] = decision["p_learned"]
        model[concept]["mastered"]      = decision["mastered"]
        model[concept]["history"]       = decision["history"]

        self.blackboard.write("student_model", model)
        self.blackboard.write("bkt_mastered",  decision["mastered"])

        print(f"\n  [BKTAgent] ACT → Blackboard updated")
        print(f"  student_model['{concept}']['bkt_p_learned'] = {decision['p_learned']:.4f}")
        print(f"  student_model['{concept}']['mastered']      = {decision['mastered']}")
