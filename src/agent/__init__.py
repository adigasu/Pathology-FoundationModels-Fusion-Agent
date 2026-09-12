"""
Agent Module: Autonomous search loop and decision logging for multimodal pathology foundation model fusion.
Provides:
- v1: SequentialGuardrailAgent (archived original sequential search)
- v2: PrincipledExploitationAgent (Agent A: Principled multi-resolution exploitation)
- v3: HierarchicalStabilityAgent (Agent B: User proposed 3-stage pooling -> fusion -> 15-fold stability)
- v4: LLMFusionAgent (Agent C: Simple, principled LLM agent with autonomous reasoning & fallback)
"""

from src.agent.search_v1_sequential import AutonomousFusionAgent as SequentialGuardrailAgent
from src.agent.search_v2_principled_exploitation import PrincipledExploitationAgent
from src.agent.search_v3_hierarchical_stability import HierarchicalStabilityAgent
from src.agent.search_v4_llm import LLMFusionAgent

# Default alias
AutonomousFusionAgent = LLMFusionAgent

def get_agent_cls(version: str = "v4"):
    v = version.lower()
    if v in ("v1", "sequential"):
        return SequentialGuardrailAgent
    elif v in ("v2", "principled", "exploitation"):
        return PrincipledExploitationAgent
    elif v in ("v3", "hierarchical", "stability"):
        return HierarchicalStabilityAgent
    elif v in ("v4", "llm", "simple_llm"):
        return LLMFusionAgent
    else:
        raise ValueError(f"Unknown agent version: {version}. Choose from 'v1', 'v2', 'v3', 'v4', 'llm'.")

__all__ = [
    "AutonomousFusionAgent",
    "SequentialGuardrailAgent",
    "PrincipledExploitationAgent",
    "HierarchicalStabilityAgent",
    "LLMFusionAgent",
    "get_agent_cls"
]
