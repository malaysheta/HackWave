from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict, List, Optional
from enum import Enum

from langgraph.graph import add_messages
from typing_extensions import Annotated
import operator


class QueryType(Enum):
    DOMAIN = "domain"
    UX_UI = "ux_ui"
    TECHNICAL = "technical"
    REVENUE = "revenue"
    GENERAL = "general"


class DebateCategory(Enum):
    DOMAIN_EXPERT = "domain_expert"
    UX_UI_SPECIALIST = "ux_ui_specialist"
    TECHNICAL_ARCHITECT = "technical_architect"
    REVENUE_MODEL_ANALYST = "revenue_model_analyst"
    MODERATOR = "moderator"


class OverallState(TypedDict):
    messages: Annotated[list, add_messages]
    user_query: str
    query_type: QueryType
    debate_category: Optional[DebateCategory]
    domain_expert_analysis: Optional[str]
    ux_ui_specialist_analysis: Optional[str]
    technical_architect_analysis: Optional[str]
    revenue_model_analyst_analysis: Optional[str]
    moderator_aggregation: Optional[str]
    debate_resolution: Optional[str]
    final_answer: Optional[str]
    processing_time: float


class DomainExpertState(TypedDict):
    user_query: str
    domain_analysis: str
    domain_requirements: List[str]
    domain_concerns: List[str]


class UXUISpecialistState(TypedDict):
    user_query: str
    ux_analysis: str
    ui_requirements: List[str]
    user_experience_concerns: List[str]


class TechnicalArchitectState(TypedDict):
    user_query: str
    technical_analysis: str
    technical_requirements: List[str]
    technical_concerns: List[str]


class RevenueModelAnalystState(TypedDict):
    user_query: str
    revenue_analysis: str
    revenue_requirements: List[str]
    revenue_concerns: List[str]


class ModeratorState(TypedDict):
    user_query: str
    domain_analysis: Optional[str]
    ux_analysis: Optional[str]
    technical_analysis: Optional[str]
    revenue_analysis: Optional[str]
    aggregated_requirements: List[str]
    conflict_resolution: Optional[str]
    final_recommendations: List[str]


class DebateAnalysisState(TypedDict):
    user_query: str
    debate_content: str
    debate_category: DebateCategory
    routing_decision: str


@dataclass(kw_only=True)
class ProductRequirementsOutput:
    final_answer: str = field(default=None)
    requirements_summary: str = field(default=None)
    processing_time: float = field(default=0.0)
