"""OpenAI client wrapper with Pydantic structured output parsing and retry handling."""

import logging
import re
from typing import List, Tuple, Optional
import openai
from openai import OpenAI, APIError, AuthenticationError, RateLimitError, APIConnectionError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config import get_settings
from src.models import LeadEnrichmentResult, PageChunk
from src.llm.prompts import SYSTEM_PROMPT, format_context_for_llm
from src.llm.cost_tracker import calculate_llm_cost, UsageMetrics

logger = logging.getLogger(__name__)


def sanitize_error_message(msg: str) -> str:
    """Mask any potential API key substrings from error message strings."""
    if not msg:
        return ""
    return re.sub(r"sk-[a-zA-Z0-9\-_]{15,}", "sk-***MASKED***", str(msg))


class LLMExtractor:
    """LLM Structured Extraction engine using OpenAI SDK and Pydantic models."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        settings = get_settings()
        self.api_key = api_key or settings.openai_api_key
        self.model = model or settings.openai_model
        self._client: Optional[OpenAI] = None

    def _get_client(self) -> OpenAI:
        """Instantiate and return OpenAI client safely."""
        if not self.api_key or not self.api_key.strip():
            raise ValueError(
                "OPENAI_API_KEY is not configured in environment or .env file."
            )
        if not self._client:
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    @retry(
        retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _execute_api_call(self, formatted_context: str):
        """Execute OpenAI API call with bounded retries for transient errors."""
        client = self._get_client()
        return client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Please extract company information from the following website content:\n\n{formatted_context}",
                },
            ],
            response_format=LeadEnrichmentResult,
        )

    def extract_lead_data(
        self, chunks: List[PageChunk]
    ) -> Tuple[LeadEnrichmentResult, UsageMetrics]:
        """
        Extract structured lead data from preprocessed website page chunks.

        Returns:
            Tuple[LeadEnrichmentResult, UsageMetrics]: Extracted Pydantic result and token usage metrics.
        """
        formatted_context = format_context_for_llm(chunks)

        try:
            completion = self._execute_api_call(formatted_context)
            parsed_result: Optional[LeadEnrichmentResult] = completion.choices[0].message.parsed

            if not parsed_result:
                refusal = getattr(completion.choices[0].message, "refusal", None)
                err_msg = f"Model output refusal: {refusal}" if refusal else "Failed to parse structured output from model"
                raise ValueError(err_msg)

            # Calculate token usage and cost metrics
            usage = getattr(completion, "usage", None)
            p_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
            c_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

            usage_metrics = calculate_llm_cost(
                model=self.model,
                prompt_tokens=p_tokens,
                completion_tokens=c_tokens,
            )

            return parsed_result, usage_metrics

        except AuthenticationError as auth_err:
            clean_msg = sanitize_error_message(str(auth_err))
            logger.error(f"OpenAI Authentication Error: {clean_msg}")
            raise RuntimeError(f"OpenAI Authentication Error: {clean_msg}") from auth_err

        except (RateLimitError, APIConnectionError, APIError) as api_err:
            clean_msg = sanitize_error_message(str(api_err))
            logger.error(f"OpenAI API Error: {clean_msg}")
            raise RuntimeError(f"OpenAI API Error: {clean_msg}") from api_err

        except Exception as exc:
            clean_msg = sanitize_error_message(str(exc))
            logger.error(f"Unexpected error in LLM extraction: {clean_msg}")
            raise RuntimeError(f"LLM Extraction failed: {clean_msg}") from exc
