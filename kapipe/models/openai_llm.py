from openai import OpenAI

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential
)


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def generate_with_backoff(client, openai_model_name, max_new_tokens, system_prompt, user_prompt, temperature=0.0):
    response = client.chat.completions.create(
        model=openai_model_name,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=temperature,
        seed=123,
        max_tokens=max_new_tokens,
    )
    return response.choices[0].message.content


class OpenAILLM:
    
    def __init__(
        self,
        # Model
        openai_model_name,
        # Generation
        max_new_tokens
    ):
        self.openai_model_name = openai_model_name
        self.max_new_tokens = max_new_tokens

        self.client = OpenAI()

    def generate(self, prompt=None, system_prompt=None, user_prompt=None, temperature=0.0):
        if prompt is not None:
            if user_prompt is not None:
                raise ValueError("Specify either `prompt` or `user_prompt`, not both.")
        else:
            if user_prompt is None:
                raise ValueError("Specify `prompt` or `user_prompt`.")
        user_prompt = prompt if prompt is not None else user_prompt

        if system_prompt is None:
            system_prompt = "You are a helpful assistant."

        return generate_with_backoff(
            client=self.client,
            openai_model_name=self.openai_model_name,
            max_new_tokens=self.max_new_tokens,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature
        )
