"""
base_agent.py
─────────────
The blueprint every agent inherits from.
Enforces the perceive → reason → act loop from Lecture 2.

Every agent in this project is a class that:
  - inherits BaseAgent
  - implements perceive(), reason(), act()
  - calls run() to execute the loop once
"""


class BaseAgent:
    def __init__(self, name: str, blackboard):
        self.name       = name        # agent identity
        self.blackboard = blackboard  # shared environment

    def perceive(self):
        """
        Read relevant data from the blackboard.
        This is the agent's SENSORS — what it can see.
        Must be overridden by each subclass.
        """
        raise NotImplementedError(f"{self.name} must implement perceive()")

    def reason(self, perception: dict):
        """
        Process what was perceived, produce a decision.
        This is the agent's BRAIN — where logic lives.
        Must be overridden by each subclass.
        """
        raise NotImplementedError(f"{self.name} must implement reason()")

    def act(self, decision):
        """
        Write the result back to the blackboard.
        This is the agent's ACTUATOR — how it changes the world.
        Must be overridden by each subclass.
        """
        raise NotImplementedError(f"{self.name} must implement act()")
    def save(
            
            
    def run(self):
        """
        The agent loop: perceive → reason → act.
        Called by the Orchestrator when this agent's turn comes.
        Returns the decision for the Orchestrator to inspect.
        """
        print(f"\n{'─'*50}")
        print(f"  AGENT RUNNING: {self.name}")
        print(f"{'─'*50}")

        perception = self.perceive()
        decision   = self.reason(perception)
        self.act(decision)
        return decision
