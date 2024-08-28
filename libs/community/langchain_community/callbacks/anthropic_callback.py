import threading
from typing import Any, Dict, List, Union
import logging

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

logger = logging.getLogger(__name__)

MODEL_COST_PER_1K_INPUT_TOKENS = {
    "claude-instant-v1": 0.0008,
    "claude-v2": 0.008,
    "claude-v2:1": 0.008,
    "claude-3-sonnet-20240229": 0.003,
    "claude-3-5-sonnet-20240620": 0.003,
    "claude-3-haiku-20240307": 0.00025,
}

MODEL_COST_PER_1K_OUTPUT_TOKENS = {
    "claude-instant-v1": 0.0024,
    "claude-v2": 0.024,
    "claude-v2:1": 0.024,
    "claude-3-sonnet-20240229": 0.015,
    "claude-3-5-sonnet-20240620": 0.015,
    "claude-3-haiku-20240307": 0.00125,
}


def _get_anthropic_claude_token_cost(
    prompt_tokens: int, completion_tokens: int, model_id: Union[str, None]
) -> tuple[float]:
    """Get the cost of tokens for the Claude model."""
    if model_id not in MODEL_COST_PER_1K_INPUT_TOKENS:
        logger.warn(
            f"Unknown model: {model_id}. Please provide a valid Anthropic model name."
            "Known models are: " + ", ".join(MODEL_COST_PER_1K_INPUT_TOKENS.keys())
        )
        return 0, 0
    
    prompt_cost =  prompt_tokens / 1000 * MODEL_COST_PER_1K_INPUT_TOKENS[model_id] 
    completion_cost = completion_tokens / 1000 * MODEL_COST_PER_1K_OUTPUT_TOKENS[model_id]
    return prompt_cost, completion_cost

class AnthropicTokenUsageCallbackHandler(BaseCallbackHandler):
    """Callback Handler that tracks anthropic info."""

    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    successful_requests: int = 0
    total_cost: float = 0.0

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return (
            f"Tokens Used: {self.total_tokens}\n"
            f"\tPrompt Tokens: {self.prompt_tokens}\n"
            f"\tCompletion Tokens: {self.completion_tokens}\n"
            f"Successful Requests: {self.successful_requests}\n"
            f"Total Cost (USD): ${self.total_cost}"
        )

    @property
    def always_verbose(self) -> bool:
        """Whether to call verbose callbacks even if verbose is False."""
        return True

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Print out the prompts."""
        pass

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Print out the token."""
        pass

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Collect token usage."""
        # Check for usage_metadata (langchain-core >= 0.2.2)
        # largely based on code for OpenAICallbackHandler
        try:
            generation = response.generations[0][0]
        except IndexError:
            generation = None
        if isinstance(generation, ChatGeneration):
            try:
                message = generation.message
                if isinstance(message, AIMessage):
                    usage_metadata = message.usage_metadata
                    response_metadata = message.response_metadata
                else:
                    usage_metadata = None
                    response_metadata = None
            except AttributeError:
                usage_metadata = None
                response_metadata = None
        else:
            usage_metadata = None
            response_metadata = None
        if usage_metadata:
            token_usage = {"total_tokens": usage_metadata["total_tokens"]}
            completion_tokens = usage_metadata["output_tokens"]
            prompt_tokens = usage_metadata["input_tokens"]
            
            if response_model_name := (response_metadata or {}).get("model"):
                model_name = response_model_name
            elif response.llm_output is None:
                model_name = ""
            else:
                model_name = response.llm_output.get("model_name", "")

        else:
            if response.llm_output is None:
                return None

            if "token_usage" not in response.llm_output:
                with self._lock:
                    self.successful_requests += 1
                return None

            # compute tokens and cost for this request
            token_usage = response.llm_output["token_usage"]
            completion_tokens = token_usage.get("completion_tokens", 0)
            prompt_tokens = token_usage.get("prompt_tokens", 0)
            model_name = response.llm_output.get("model_name", "")
        
        if model_name in MODEL_COST_PER_1K_INPUT_TOKENS:
            prompt_cost, completion_cost = _get_anthropic_claude_token_cost(
                    prompt_tokens, completion_tokens, model_name)
        else:
            if model_name:
                logger.warn(f"'{model_name}' is not a valid model")
            else:
                logger.warn("A model name was not provided. The streaming method does not return a model name")
            
            completion_cost = 0
            prompt_cost = 0

        # update shared state behind lock
        with self._lock:
            self.total_cost += prompt_cost + completion_cost
            self.total_tokens += token_usage.get("total_tokens", 0)
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            self.successful_requests += 1

    def __copy__(self) -> "AnthropicTokenUsageCallbackHandler":
        """Return a copy of the callback handler."""
        return self

    def __deepcopy__(self, memo: Any) -> "AnthropicTokenUsageCallbackHandler":
        """Return a deep copy of the callback handler."""
        return self
