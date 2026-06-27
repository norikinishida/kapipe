import logging

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


logger = logging.getLogger(__name__)


class HuggingFaceLLM:
    """A class that wraps a Hugging Face causal language model (LLM) for text generation."""

    def __init__(
        self,
        model_name: str,
        max_new_tokens: int,
        # Optional
        quantization_bits: int | None = None,
    ) -> None:

        self.provider = "hf"

        self.model_name = model_name
        self.quantization_bits = quantization_bits
        self.max_new_tokens = max_new_tokens

        # Build the optional bitsandbytes quantization configuration.
        # None means full precision loading with torch.bfloat16 weights.
        # 4 means NF4 4-bit quantization with bfloat16 computation.
        # 8 means bitsandbytes 8-bit quantization.
        self.bnb_config = self._set_quantization(
            quantization_bits=quantization_bits
        )

        # Load the model and tokenizer
        self.llm, self.tokenizer = self._initialize_llm_and_tokenizer(
            model_name=self.model_name
        )

    def _set_quantization(
        self,
        quantization_bits: int | None
    ) -> BitsAndBytesConfig | None:
        """Build the optional bitsandbytes quantization configuration."""

        # Return no configuration when quantization is disabled
        if quantization_bits not in [4, 8]:
            logger.info("No quantization applied (full precision)")
            return None

        # Create the base bitsandbytes configuration
        bnb_config = BitsAndBytesConfig()

        # Configure NF4 quantization for 4-bit inference
        if quantization_bits == 4:
            bnb_config.load_in_4bit = True
            bnb_config.bnb_4bit_quant_type = "nf4"
            bnb_config.bnb_4bit_use_double_quant = True
            bnb_config.bnb_4bit_compute_dtype = torch.bfloat16
            # bnb_config.llm_int8_enable_fp32_cpu_offload = True
            logger.info(
                "Using 4-bit quantization "
                "(quant_type: nf4, double_quant: True, "
                "compute_dtype: bfloat16)"
            )

        # Configure bitsandbytes quantization for 8-bit inference
        if quantization_bits == 8:
            bnb_config.load_in_8bit = True
            logger.info("Using 8-bit quantization")

        return bnb_config

    def _initialize_llm_and_tokenizer(
        self,
        model_name: str
    ) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load the model and tokenizer."""

        logger.info(
            f"Loading a large language model: {model_name}"
        )

        # Define the common model-loading arguments
        model_kwargs: dict[str, object] = {
            "device_map": "auto",
            "torch_dtype": torch.bfloat16,
            "quantization_config": self.bnb_config,
            "trust_remote_code": True,
        }

        # Enable Flash Attention 2 when the environment supports it
        attn_impl = self._get_attn_implementation()
        if attn_impl is not None:
            model_kwargs["attn_implementation"] = attn_impl
            logger.info(
                "Using device_map='auto' (Flash Attention enabled)."
            )
        # else:
        #     model_kwargs["device_map"] = "balanced"
        #     model_kwargs["max_memory"] = self._get_max_memory_for_v100()
        #     logger.info(
        #         "Flash Attention unavailable. "
        #         "Using device_map='balanced' with max_memory for V100."
        #     )

        # Load the actual causal LM
        llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )

        # Switch the language model to inference mode
        llm.eval()

        # Load the tokenizer with left-side padding and truncation
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=True,
        )

        # Reuse the EOS token when the tokenizer has no padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return llm, tokenizer

    def _get_attn_implementation(self) -> str | None:
        """Return Flash Attention 2 only when it is available and useful."""

        # Disable Flash Attention when CUDA is unavailable
        if not torch.cuda.is_available():
            logger.info("CUDA is not available. Disable Flash Attention.")
            return None

        # Disable Flash Attention 2 on GPUs older than the Ampere generation
        major, _ = torch.cuda.get_device_capability()
        if major < 8:
            logger.info(
                "GPU does not support Flash Attention 2 (requires SM80+)."
            )
            return None

        # Check whether the optional flash-attn package is installed
        try:
            import flash_attn  # noqa: F401
        except Exception as error:
            logger.info(f"flash-attn is not available: {error}")
            return None

        # Use the implementation name expected by Transformers
        logger.info("Flash Attention 2 is enabled.")
        return "flash_attention_2"

    # def _get_max_memory_for_v100(self) -> dict[int | str, str]:
    #     """
    #     Return a conservative max_memory setting for DGX-2 (V100 32GB × N).
    #     This is used only when Flash Attention is unavailable.
    #     """
    #     n_gpus = torch.cuda.device_count()
    #     max_memory: dict[int | str, str] = {
    #         i: "28GiB" for i in range(n_gpus)
    #     }
    #     max_memory["cpu"] = "200GiB"
    #     return max_memory

    def generate(
        self,
        prompt: str,
        do_sample: bool = False,
        temperature: float = 0.0,
    ) -> str:
        """Generate text for the given prompt."""

        # Represent the prompt as a single user message
        messages: list[dict[str, str]] = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        # Render the model-specific chat template into plain text first
        text_input = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize the rendered prompt
        model_inputs = self.tokenizer(
            text_input,
            padding=True,
            truncation=True,
            # max_length=min(self.tokenizer.model_max_length, 32768),  
            return_tensors="pt",
        )

        # Move the model inputs to the language model device
        model_inputs = model_inputs.to(self.llm.device)

        # Use the tokenizer EOS token as the default stopping token
        eos_token_ids: list[int] = [self.tokenizer.eos_token_id]

        # Add the end-of-turn token used by Llama 3 when available
        if "<|eot_id|>" in self.tokenizer.get_vocab():
            eos_token_ids.append(
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            )

        # Generate output token IDs without computing gradients
        with torch.no_grad():
            generated_token_ids = self.llm.generate(
                **model_inputs,
                max_new_tokens=self.max_new_tokens,
                # Parameters that control the generation outputs
                num_beams=1,
                do_sample=do_sample,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=eos_token_ids,
            )

        # Remove the input prompt tokens from each generated sequence
        generated_token_ids_trimmed = [
            output_token_ids[len(input_token_ids):]
            for input_token_ids, output_token_ids in zip(
                model_inputs.input_ids,
                generated_token_ids,
            )
        ]

        # Decode only the newly generated tokens
        generated_text: str = self.tokenizer.batch_decode(
            generated_token_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        # Remove the outer whitespace and return it
        return generated_text.strip()
