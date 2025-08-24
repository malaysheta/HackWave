import os
from pydantic import BaseModel, Field
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig


class Configuration(BaseModel):
    """The configuration for the multi-agent product requirements refinement system."""

    model: str = Field(
        default="gemini-2.0-flash",
        metadata={
            "description": "The Gemini model to use for all agent interactions."
        },
    )

    max_debate_resolution_time: int = Field(
        default=120,
        metadata={
            "description": "The maximum time in seconds allowed for debate resolution (target: under 2 minutes)."
        },
    )

    enable_parallel_processing: bool = Field(
        default=True,
        metadata={
            "description": "Whether to enable parallel processing of specialist analyses."
        },
    )

    thread_id: Optional[str] = Field(
        default=None,
        metadata={
            "description": "Unique identifier for the conversation thread to enable memory persistence."
        },
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )

        # Get raw values from environment or config
        raw_values: dict[str, Any] = {
            name: os.environ.get(name.upper(), configurable.get(name))
            for name in cls.model_fields.keys()
        }

        # Filter out None values
        values = {k: v for k, v in raw_values.items() if v is not None}

        return cls(**values)
