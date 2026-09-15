"""Token usage and estimated API cost tracking module."""

import logging
from typing import Dict, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Model pricing table in USD per 1,000,000 tokens (Prompt, Completion)
MODEL_PRICING_PER_1M_TOKENS: Dict[str, Dict[str, float]] = {
    "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
    "gpt-4o": {"prompt": 2.50, "completion": 10.00},
    "gpt-4o-2024-08-06": {"prompt": 2.50, "completion": 10.00},
    "gpt-3.5-turbo": {"prompt": 0.50, "completion": 1.50},
}


class UsageMetrics(BaseModel):
    """LLM token usage and estimated cost metadata."""

    model: str = Field(description="OpenAI model used for extraction")
    prompt_tokens: int = Field(default=0, ge=0, description="Number of prompt tokens")
    completion_tokens: int = Field(default=0, ge=0, description="Number of completion tokens")
    total_tokens: int = Field(default=0, ge=0, description="Total tokens used")
    estimated_cost_usd: float = Field(default=0.0, ge=0.0, description="Estimated API cost in USD")


def calculate_llm_cost(
    model: str,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
) -> UsageMetrics:
    """
    Calculate token usage metrics and estimated API cost.
    Gracefully handles missing token counts or unlisted model names.
    """
    p_tokens = prompt_tokens or 0
    c_tokens = completion_tokens or 0
    t_tokens = p_tokens + c_tokens

    cost = 0.0
    model_key = model.lower()

    # Match exact or base model prefix
    matched_pricing = None
    for key, pricing in MODEL_PRICING_PER_1M_TOKENS.items():
        if key in model_key:
            matched_pricing = pricing
            break

    if matched_pricing:
        prompt_cost = (p_tokens / 1_000_000.0) * matched_pricing["prompt"]
        completion_cost = (c_tokens / 1_000_000.0) * matched_pricing["completion"]
        cost = round(prompt_cost + completion_cost, 6)

    return UsageMetrics(
        model=model,
        prompt_tokens=p_tokens,
        completion_tokens=c_tokens,
        total_tokens=t_tokens,
        estimated_cost_usd=cost,
    )
