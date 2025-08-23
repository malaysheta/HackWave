from typing import List, Optional
from pydantic import BaseModel, Field
from enum import Enum


class QueryType(Enum):
    DOMAIN = "domain"
    UX_UI = "ux_ui"
    TECHNICAL = "technical"
    GENERAL = "general"


class DebateCategory(Enum):
    DOMAIN_EXPERT = "domain_expert"
    UX_UI_SPECIALIST = "ux_ui_specialist"
    TECHNICAL_ARCHITECT = "technical_architect"
    MODERATOR = "moderator"


class DomainExpertAnalysis(BaseModel):
    domain_analysis: str = Field(
        description="Detailed analysis of the domain-specific aspects of the product requirements."
    )
    domain_requirements: List[str] = Field(
        description="List of domain-specific requirements identified from the analysis."
    )
    domain_concerns: List[str] = Field(
        description="List of domain-related concerns or potential issues."
    )
    priority_level: str = Field(
        description="Priority level for domain requirements (High/Medium/Low)."
    )


class UXUISpecialistAnalysis(BaseModel):
    ux_analysis: str = Field(
        description="Detailed analysis of user experience aspects of the product requirements."
    )
    ui_requirements: List[str] = Field(
        description="List of UI/UX requirements identified from the analysis."
    )
    user_experience_concerns: List[str] = Field(
        description="List of UX-related concerns or potential issues."
    )
    accessibility_requirements: List[str] = Field(
        description="List of accessibility requirements that should be considered."
    )


class TechnicalArchitectAnalysis(BaseModel):
    technical_analysis: str = Field(
        description="Detailed analysis of technical architecture aspects of the product requirements."
    )
    technical_requirements: List[str] = Field(
        description="List of technical requirements identified from the analysis."
    )
    technical_concerns: List[str] = Field(
        description="List of technical concerns or potential issues."
    )
    scalability_considerations: List[str] = Field(
        description="List of scalability considerations for the technical implementation."
    )


class ModeratorAggregation(BaseModel):
    aggregated_requirements: List[str] = Field(
        description="Consolidated list of all requirements from different specialists."
    )
    conflict_resolution: Optional[str] = Field(
        description="Resolution of any conflicts between different specialist perspectives.",
        default=None
    )
    final_recommendations: List[str] = Field(
        description="Final prioritized recommendations for the product requirements."
    )
    implementation_priority: List[str] = Field(
        description="Prioritized list of requirements for implementation order."
    )


class DebateAnalysis(BaseModel):
    debate_category: DebateCategory = Field(
        description="Category of the debate to determine which specialist should handle it."
    )
    routing_decision: str = Field(
        description="Decision on how to route the debate for resolution."
    )
    urgency_level: str = Field(
        description="Urgency level of the debate (High/Medium/Low)."
    )
    estimated_resolution_time: str = Field(
        description="Estimated time needed to resolve this debate (in minutes)."
    )


class QueryClassification(BaseModel):
    query_type: QueryType = Field(
        description="Classification of the user query to determine routing."
    )
    confidence_score: float = Field(
        description="Confidence score for the classification (0.0 to 1.0)."
    )
    reasoning: str = Field(
        description="Reasoning behind the classification decision."
    )
