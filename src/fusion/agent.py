"""
Backwards-compatibility shim: Agent search loop and decision logging are organized under src.agent.search
"""

from src.agent.search import AutonomousFusionAgent, evaluate_candidate_cv

__all__ = ["AutonomousFusionAgent", "evaluate_candidate_cv"]
