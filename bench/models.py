"""Models for the benchmark. Every rank wraps its model in a MeteredModel, which counts
requests and tokens, applies the run's sampling settings and retries transient API errors.
"""

import asyncio
import os
import random
from dataclasses import dataclass

from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError
from pydantic_ai.models import Model, infer_model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings, merge_model_settings


@dataclass
class Usage:
    """LLM requests made and tokens used"""

    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.requests + other.requests,
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
        )

    def __sub__(self, other: "Usage") -> "Usage":
        return Usage(
            self.requests - other.requests,
            self.input_tokens - other.input_tokens,
            self.output_tokens - other.output_tokens,
        )


class MeteredModel(WrapperModel):
    """Wraps a model to accumulate its usage, apply settings to every request and retry
    connection errors, timeouts, rate limits and server errors with exponential backoff

    Args:
        wrapped: the model that answers requests
        settings: settings such as temperature, applied under any per-request settings
        retries: attempts after the first before an error is raised

    Attributes:
        usage: running total over every request made through this model
    """

    def __init__(
        self, wrapped: Model, settings: ModelSettings | None = None, retries: int = 3
    ):
        super().__init__(wrapped)
        self.extra_settings = settings
        self.retries = retries
        self.usage = Usage()

    @property
    def settings(self) -> ModelSettings | None:
        return merge_model_settings(self.wrapped.settings, self.extra_settings)

    async def request(self, messages, model_settings, model_request_parameters):
        for attempt in range(self.retries + 1):
            try:
                response = await super().request(
                    messages, model_settings, model_request_parameters
                )
                break
            except ModelAPIError as e:
                transient = not isinstance(e, ModelHTTPError) or (
                    e.status_code in (408, 429) or e.status_code >= 500
                )
                if not transient or attempt == self.retries:
                    raise
                await asyncio.sleep(2**attempt)

        self.usage += Usage(
            1, response.usage.input_tokens, response.usage.output_tokens
        )
        return response


class RandomModel(TestModel):
    """Offline stand-in that replies to every text request with a random ``Answer: X``, for
    checking the harness and evaluation end to end at chance accuracy"""

    def __init__(self, seed: int = 0):
        super().__init__(seed=seed, model_name="random")
        self.rng = random.Random(seed)

    async def request(self, messages, model_settings, model_request_parameters):
        # structured requests (ask_list) keep TestModel's schema-generated output
        self.custom_output_text = (
            None
            if model_request_parameters.output_tools
            else f"Answer: {self.rng.choice('ABCD')}"
        )
        return await super().request(messages, model_settings, model_request_parameters)


def build_model(name: str, base_url: str | None = None, seed: int = 0) -> Model:
    """Create the model for name.

    ``test`` is pydantic-ai's TestModel and ``random`` is a RandomModel seeded with seed; both
    run offline. With base_url, name is a model served by an OpenAI-compatible server such as
    llama.cpp, vLLM or Ollama, and structured output is requested as a JSON schema rather than
    a tool call, so the server needs no tool-call parser. Otherwise name is a pydantic-ai model
    id such as ``openai:gpt-5`` or ``anthropic:claude-sonnet-5``, with its API key read from
    the environment.
    """
    if name == "test":
        return TestModel()
    if name == "random":
        return RandomModel(seed)
    if base_url:
        api_key = os.getenv("OPENAI_API_KEY", "not-needed")
        return OpenAIChatModel(
            model_name=name,
            provider=OpenAIProvider(base_url=base_url, api_key=api_key),
            profile=ModelProfile(default_structured_output_mode="native"),
        )
    return infer_model(name)
