"""LLM layer. To add another provider, subclass LLMProvider and register it in create_llm()."""
from __future__ import annotations

from abc import ABC, abstractmethod

from .config import Config


class LLMError(Exception):
    """Friendly, user-displayable LLM failure."""


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, messages: list[dict]) -> str:
        """messages: [{'role': 'system'|'user'|'assistant', 'content': str}, ...] -> reply text."""


class OpenAICompatibleLLM(LLMProvider):
    """Works with any server that speaks the OpenAI chat-completions API."""

    def __init__(self, cfg: Config):
        try:
            import openai
        except ImportError:
            raise LLMError("The 'openai' package is not installed. Run: pip install -r requirements.txt") from None
        self._openai = openai
        self._cfg = cfg
        self._client = openai.OpenAI(
            api_key=cfg.llm_api_key or "not-needed",
            base_url=cfg.llm_base_url,
            timeout=cfg.llm_timeout,
            max_retries=1,
        )

    def generate(self, messages: list[dict]) -> str:
        o = self._openai
        try:
            resp = self._client.chat.completions.create(
                model=self._cfg.llm_model,
                messages=messages,
                temperature=self._cfg.llm_temperature,
                max_tokens=self._cfg.llm_max_tokens,
            )
        except o.AuthenticationError:
            raise LLMError("The LLM provider rejected the API key. Check LLM_API_KEY in .env.") from None
        except o.RateLimitError:
            raise LLMError("The LLM provider says rate limit or quota exceeded. Wait a bit or check your plan.") from None
        except o.NotFoundError:
            raise LLMError(f"Model or endpoint not found. Check LLM_MODEL ({self._cfg.llm_model}) and LLM_BASE_URL.") from None
        except o.APITimeoutError:
            raise LLMError("The LLM request timed out.") from None
        except o.APIConnectionError:
            raise LLMError("Could not reach the LLM provider. Check your internet connection and LLM_BASE_URL.") from None
        except o.APIStatusError as e:
            raise LLMError(f"The LLM provider returned an error (HTTP {e.status_code}).") from None
        except Exception as e:  # last resort: never crash the call loop
            raise LLMError(f"Unexpected LLM error: {type(e).__name__}") from None

        # Providers may return text as a string, a list of content blocks, or
        # an empty message with a finish_reason. Normalize the common formats
        # and include safe diagnostic metadata when no usable text is returned.
        text = ""
        finish_reason = "no_choices"
        if resp.choices:
            choice = resp.choices[0]
            finish_reason = getattr(choice, "finish_reason", None) or "unknown"
            content = getattr(choice.message, "content", None)
            if isinstance(content, str):
                text = content.strip()
            elif isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict):
                        value = block.get("text")
                    else:
                        value = getattr(block, "text", None)
                    if isinstance(value, str) and value.strip():
                        parts.append(value.strip())
                text = " ".join(parts).strip()

            # Some providers use the refusal field instead of message.content.
            if not text:
                refusal = getattr(choice.message, "refusal", None)
                if isinstance(refusal, str) and refusal.strip():
                    text = refusal.strip()

        if not text:
            response_id = getattr(resp, "id", None) or "unavailable"
            model = getattr(resp, "model", None) or self._cfg.llm_model
            raise LLMError(
                "The LLM returned no text "
                f"(finish_reason={finish_reason}, model={model}, response_id={response_id}). "
                "Check provider/model availability; retry once and inspect the provider dashboard."
            )
        return text


def create_llm(cfg: Config) -> LLMProvider:
    cfg.validate_llm()
    return OpenAICompatibleLLM(cfg)
