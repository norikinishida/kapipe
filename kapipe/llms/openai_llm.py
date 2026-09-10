import json
from typing import Any

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

        # Construct the request body for the OpenAI Chat Completion API
        params = self.make_request_body(prompt=prompt, temperature=temperature)

        # Call the generate_with_backoff function to handle retries and backoff
        return generate_with_backoff(
            client=self.client,
            params=params,
        )

    def make_request_body(
        self,
        prompt: str | dict[str, str],
        temperature: float = 0.0,
    ) -> dict[str, object]:
        """Construct the request body for the OpenAI Chat Completion API."""

        # Extract system and user prompts based on the input type
        if isinstance(prompt, str):
            system_prompt = "You are a helpful assistant."
            user_prompt = prompt
        else:
            system_prompt = prompt["system_prompt"]
            user_prompt = prompt["user_prompt"]

        # Set up the parameters for the OpenAI Chat Completion API call
        params: dict[str, object] = {
            "model": self.model_name,
            "messages": [
                {
                    "role": (
                        "system" if self.model_name.startswith("gpt-4")
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
        if self.model_name.startswith("gpt-4"):
            params["max_tokens"] = self.max_new_tokens
            params["temperature"] = temperature
        else:
            params["max_completion_tokens"] = self.max_new_tokens
            # params["reasoning_effort"] = "medium"

        return params

    def submit_batch(
        self,
        prompts: list[str | dict[str, str]],
        temperature: float = 0.0,
    ) -> str:
        """Submit prompts through the Batch API and return the Batch ID.

        Submit one batch containing 1 to 50,000 requests and at most 200 MB
        of JSONL input. Keep the returned ID for a later fetch_batch() call.
        """

        # Validate the number of requests before uploading any data
        if not 1 <= len(prompts) <= 50000:
            raise ValueError("A batch must contain between 1 and 50000 prompts")

        # Construct one request for each prompt in input order
        request_lines: list[str] = []
        for prompt_i, prompt in enumerate(prompts):
            request_body: dict[str, object] = self.make_request_body(
                prompt=prompt,
                temperature=temperature,
            )
            request: dict[str, object] = {
                "custom_id": f"request-{prompt_i:08d}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": request_body,
            }
            request_lines.append(json.dumps(request, ensure_ascii=False) + "\n")

        # Encode the JSONL input
        content: bytes = "".join(request_lines).encode("utf-8")

        # Validate the upload size before proceeding
        if len(content) > 200000000:
            raise ValueError("The batch input exceeds 200 MB")

        # Upload the request file for Batch API processing
        input_file = self.client.files.create(
            file=("requests.jsonl", content, "application/jsonl"),
            purpose="batch",
        )

        # Submit the uploaded requests without waiting for completion
        batch = self.client.batches.create(
            input_file_id=input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )

        return batch.id

    def fetch_batch(
        self,
        batch_id: str,
    ) -> list[str]:
        """Fetch generated texts in the original prompt order.

        Fetch a batch created by submit_batch(). Raise an error if the batch
        is unfinished or any request failed. Do not wait or resubmit requests.
        """

        # Retrieve the batch status before downloading its output
        batch = self.client.batches.retrieve(batch_id)

        # Validate the batch status and ensure it is ready for fetching results
        if batch.status != "completed":
            raise RuntimeError(f"Batch {batch_id} has status {batch.status}")
        if batch.request_counts.failed != 0:
            raise RuntimeError(
                f"Batch {batch_id} contains failed requests; "
                f"error_file_id={batch.error_file_id}"
            )
        if batch.output_file_id is None:
            raise RuntimeError(f"Batch {batch_id} has no output file")

        # Download the generated responses
        content: bytes = self.client.files.content(batch.output_file_id).read()

        # Associate each generated text with its submitted request ID
        custom_id_to_generated_text: dict[str, str] = {}
        for line in content.decode("utf-8").splitlines():
            # Parse the JSON line into a dictionary and extract the request ID
            record: dict[str, Any] = json.loads(line)
            custom_id: str = record["custom_id"]

            # Validate the response for the current request ID
            if custom_id in custom_id_to_generated_text:
                raise RuntimeError(f"Duplicate response ID: {custom_id}")
            if record["error"] is not None:
                raise RuntimeError(f"Request {custom_id} failed: {record['error']}")
            if record["response"]["status_code"] != 200:
                raise RuntimeError(f"Request {custom_id} returned a non-200 status")

            # Extract the generated text from the first completion
            generated_text: str | None = (
                record["response"]["body"]["choices"][0]["message"]["content"]
            )
            if generated_text is None:
                raise RuntimeError("OpenAI response did not contain text output.")

            # Store the generated text for the current request ID
            custom_id_to_generated_text[custom_id] = generated_text

        # Generate the list of expected request IDs based on 
        # the total number of requests in the batch.
        expected_ids: list[str] = [
            f"request-{request_i:08d}"
            for request_i in range(batch.request_counts.total)
        ]

        # Validate that every expected request has a corresponding response
        if set(custom_id_to_generated_text) != set(expected_ids):
            raise RuntimeError("Response IDs do not match the submitted requests")

        # Restore input order because Batch output order can differ
        generated_texts: list[str] = []
        for custom_id in expected_ids:
            generated_texts.append(custom_id_to_generated_text[custom_id])

        return generated_texts


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def generate_with_backoff(
    client: OpenAI,
    params: dict[str, object],
) -> str:
    """Submit the request and extract generated text with retries and backoff."""

    # Send the request to the OpenAI Chat Completion API and get the response
    response = client.chat.completions.create(**params)

    # Extract the generated text from the first completion
    generated_text = response.choices[0].message.content
    if generated_text is None:
        raise RuntimeError("OpenAI response did not contain text output.")

    return generated_text

