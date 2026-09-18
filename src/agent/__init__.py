"""
Agent Module: Autonomous search loops and decision logging for multimodal pathology foundation model fusion.

Provides 5 generations of autonomous fusion search agents:
- v1: SequentialAgent (Fixed sequential exploration + greedy exploitation)
- v2: ExploitationAgent (Multi-resolution concat [mean; max] first + targeted exploitation)
- v3: HierarchicalAgent (3-stage: pooling isolation -> fusion topologies -> 15-fold stability utility)
- v4: AutonomousAgent (Hypothesis-driven search conditioned on history & 1.0-SE guardrail early stopping)
- v5: UnifiedAgent (Unified architecture: statistical pooling screen -> closed-loop reasoning -> 15-fold stability gate)
"""

from src.agent.search_v1_sequential import AutonomousFusionAgent as SequentialAgent
from src.agent.search_v2_principled_exploitation import PrincipledExploitationAgent as ExploitationAgent
from src.agent.search_v3_hierarchical_stability import HierarchicalStabilityAgent as HierarchicalAgent
from src.agent.search_v4_llm import LLMFusionAgent as AutonomousAgent
from src.agent.search_v5_unified import UnifiedAgent

# Backward-compatible aliases
SequentialGuardrailAgent = SequentialAgent
PrincipledExploitationAgent = ExploitationAgent
HierarchicalStabilityAgent = HierarchicalAgent
LLMFusionAgent = AutonomousAgent
UnifiedProductionAgent = UnifiedAgent

# Default champion agent
AutonomousFusionAgent = UnifiedAgent


def get_agent_cls(version: str = "v5"):
    v = version.lower()
    if v in ("v1", "sequential", "sequential_agent", "sequential_guardrail"):
        return SequentialAgent
    elif v in ("v2", "exploitation", "exploitation_agent", "principled", "principled_exploitation"):
        return ExploitationAgent
    elif v in ("v3", "hierarchical", "hierarchical_agent", "stability", "hierarchical_stability"):
        return HierarchicalAgent
    elif v in ("v4", "autonomous", "autonomous_agent", "llm", "simple_llm"):
        return AutonomousAgent
    elif v in ("v5", "unified", "unified_agent", "production", "unified_production"):
        return UnifiedAgent
    else:
        raise ValueError(f"Unknown agent version: '{version}'. Choose from 'v1', 'v2', 'v3', 'v4', 'v5'.")


__all__ = [
    "AutonomousFusionAgent",
    "SequentialAgent",
    "ExploitationAgent",
    "HierarchicalAgent",
    "AutonomousAgent",
    "UnifiedAgent",
    "SequentialGuardrailAgent",
    "PrincipledExploitationAgent",
    "HierarchicalStabilityAgent",
    "LLMFusionAgent",
    "UnifiedProductionAgent",
    "get_agent_cls",
]
