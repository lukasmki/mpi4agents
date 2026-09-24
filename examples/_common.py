TEST = True

if TEST:
    from pydantic_ai.models.test import TestModel

    MODEL = TestModel()
else:
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.models.openai import OpenAIChatModel

    MODEL = OpenAIChatModel(
        model_name="unsloth/Qwen3-0.6B-GGUF",
        provider=OpenAIProvider(base_url="http://127.0.0.1:8080"),
    )
