import json
from typing import Any

from openai import OpenAI

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential
)

from .base import BaseLLM


# OpenAI Batch API limits
OPENAI_BATCH_MAX_REQUESTS = 50000
OPENAI_BATCH_MAX_BYTES = 200000000


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
    ) -> list[str]:
        """Submit prompts through the Batch API and return the Batch IDs.

        Split prompts into batches containing at most 50,000 requests and
        200 MB of JSONL input. Keep the returned IDs for a later
        fetch_batch() call.
        """

        # Require at least one request
        if len(prompts) == 0:
            raise ValueError("At least one prompt is required")

        # Construct size-limited groups of JSONL request lines
        request_line_batches: list[list[bytes]] = [[]]
        batch_sizes: list[int] = [0]
        for prompt in prompts:
            # Number request IDs independently within each OpenAI batch
            request_i: int = len(request_line_batches[-1])
            request_body: dict[str, object] = self.make_request_body(
                prompt=prompt,
                temperature=temperature,
            )
            request: dict[str, object] = {
                "custom_id": f"request-{request_i:08d}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": request_body,
            }
            request_line: bytes = (
                json.dumps(request, ensure_ascii=False) + "\n"
            ).encode("utf-8")

            # Reject a request that cannot fit into an otherwise empty batch
            if len(request_line) > OPENAI_BATCH_MAX_BYTES:
                raise ValueError("A single batch request exceeds 200 MB")

            # Start a new batch before exceeding either OpenAI limit
            if (
                len(request_line_batches[-1]) == OPENAI_BATCH_MAX_REQUESTS
                or batch_sizes[-1] + len(request_line)
                > OPENAI_BATCH_MAX_BYTES
            ):
                request_line_batches.append([])
                batch_sizes.append(0)

                # Restart custom IDs because each batch is fetched separately
                request["custom_id"] = "request-00000000"
                request_line = (
                    json.dumps(request, ensure_ascii=False) + "\n"
                ).encode("utf-8")

            # Preserve the prompt order within and across batches
            request_line_batches[-1].append(request_line)
            batch_sizes[-1] += len(request_line)

        # Upload and submit every size-limited request file
        batch_ids: list[str] = []
        for request_lines in request_line_batches:
            # Combine the encoded lines without changing their order
            content: bytes = b"".join(request_lines)

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
            batch_ids.append(batch.id)

        return batch_ids

    def fetch_batch(
        self,
        batch_ids: list[str],
    ) -> list[str]:
        """Fetch generated texts from batches in the original prompt order.

        Fetch batches created by submit_batch(). Raise an error if any batch
        is unfinished or any request failed. Do not wait or resubmit requests.
        """

        generated_texts: list[str] = []
        for batch_id in batch_ids:
            # Retrieve the batch status before downloading its output
            batch = self.client.batches.retrieve(batch_id)

            # Validate the batch status before fetching its responses
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
            content: bytes = self.client.files.content(
                batch.output_file_id
            ).read()

            # Associate each generated text with its submitted request ID
            custom_id_to_generated_text: dict[str, str] = {}
            for line in content.decode("utf-8").splitlines():
                # Parse the JSON line and extract the request ID
                record: dict[str, Any] = json.loads(line)
                custom_id: str = record["custom_id"]

                # Validate the response for the current request ID
                if custom_id in custom_id_to_generated_text:
                    raise RuntimeError(f"Duplicate response ID: {custom_id}")
                if record["error"] is not None:
                    raise RuntimeError(
                        f"Request {custom_id} failed: {record['error']}"
                    )
                if record["response"]["status_code"] != 200:
                    raise RuntimeError(
                        f"Request {custom_id} returned a non-200 status"
                    )

                # Extract the generated text from the first completion
                generated_text: str | None = (
                    record["response"]["body"]["choices"][0]["message"][
                        "content"
                    ]
                )
                if generated_text is None:
                    raise RuntimeError(
                        "OpenAI response did not contain text output."
                    )

                # Store the generated text for the current request ID
                custom_id_to_generated_text[custom_id] = generated_text

            # Generate the expected request IDs for the current batch
            expected_ids: list[str] = [
                f"request-{request_i:08d}"
                for request_i in range(batch.request_counts.total)
            ]

            # Validate that every expected request has one response
            if set(custom_id_to_generated_text) != set(expected_ids):
                raise RuntimeError(
                    "Response IDs do not match the submitted requests"
                )

            # Restore input order within the current batch
            for custom_id in expected_ids:
                generated_texts.append(
                    custom_id_to_generated_text[custom_id]
                )

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
