from openai import OpenAI

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential
)

from .base import BaseLLM


class OpenAILLM(BaseLLM):
    """A class that wraps an OpenAI causal language model (LLM) client for text generation."""
 
    def __init__(
        self,
        model_name: str,
        max_new_tokens: int,
    ) -> None:

        self.provider = "openai"

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        self.client = OpenAI()

    def __repr__(self) -> str:
        return (
            f"OpenAILLM("
            f"provider={self.provider}, "
            f"model_name={self.model_name}, "
            f"max_new_tokens={self.max_new_tokens})"
        )

    def generate(
        self,
        prompt: str | dict[str, str],
        temperature: float = 0.0,
    ) -> str:
        """Generate text for the given prompt."""

        # Extract system and user prompts based on the input type
        if isinstance(prompt, str):
            system_prompt = "You are a helpful assistant."
            user_prompt = prompt
        else:
            system_prompt = prompt["system_prompt"]
            user_prompt = prompt["user_prompt"]

        # Call the generate_with_backoff function to handle retries and backoff
        return generate_with_backoff(
            client=self.client,
            model_name=self.model_name,
            max_new_tokens=self.max_new_tokens,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature
        )


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def generate_with_backoff(
    client: OpenAI,
    model_name: str,
    max_new_tokens: int,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
) -> str:
    """Generate text for the given prompt using the OpenAI API with backoff."""

    # Set up the parameters for the OpenAI Chat Completion API call
    params: dict[str, object] = {
        "model": model_name,
        "messages": [
            {
                "role": (
                    "system" if model_name.startswith("gpt-4")
                    else "developer"
                ),
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        "seed": 123,
    }

    # Adjust parameters based on the model type
    if model_name.startswith("gpt-4"):
        params["max_tokens"] = max_new_tokens
        params["temperature"] = temperature
    else:
        params["max_completion_tokens"] = max_new_tokens
        params["reasoning_effort"] = "medium"

    # Send the request to the OpenAI Chat Completion API and get the response
    response = client.chat.completions.create(**params)

    # Extract the generated text from the first completion
    generated_text = response.choices[0].message.content
    if generated_text is None:
        raise RuntimeError("OpenAI response did not contain text output.")

    return generated_text

