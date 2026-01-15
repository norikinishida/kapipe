import logging

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch


logger = logging.getLogger(__name__)


class HuggingFaceLLM:
    def __init__(
        self,
        device: str,
        # Model
        llm_name_or_path: str,
        # Generation
        max_new_tokens: int,
        quantization_bits: int | None = None,
    ):
        # self.device = device
        self.llm_name_or_path = llm_name_or_path
        self.max_new_tokens = max_new_tokens

        # Quantization Setup
        self.bnb_config = self._set_quantization(quantization_bits=quantization_bits)

        # LLM, tokenizer
        self.llm, self.tokenizer = self._initialize_llm_and_tokenizer(
            llm_name_or_path=self.llm_name_or_path
        )

    def _set_quantization(
        self,
        quantization_bits: int | None
    ) -> BitsAndBytesConfig | None:
        if quantization_bits in [4, 8]:
            bnb_config = BitsAndBytesConfig()
            if quantization_bits == 4:
                bnb_config.load_in_4bit = True
                bnb_config.bnb_4bit_quant_type = "nf4"
                bnb_config.bnb_4bit_use_double_quant = True
                bnb_config.bnb_4bit_compute_dtype = torch.bfloat16
                # bnb_config.llm_int8_enable_fp32_cpu_offload = True
                logger.info(
                    "Using 4-bit quantization (quant_type: nf4, double_quant: True, compute_dtype: bfloat16)"
                )
            elif quantization_bits == 8:
                bnb_config.load_in_8bit = True
                logger.info("Using 8-bit quantization")
            return bnb_config
        else:
            logger.info("No quantization applied (full precision)")
            return None

    def _initialize_llm_and_tokenizer(
        self,
        llm_name_or_path: str
    ) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
        logger.info(f"Loading a large language model: {llm_name_or_path}")
 
        kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.bfloat16,
            "quantization_config": self.bnb_config,
            "trust_remote_code": True
        }

        attn_impl = self._get_attn_implementation()
        if attn_impl is not None:
            kwargs["attn_implementation"] = attn_impl
            logger.info("Using device_map='auto' (Flash Attention enabled).")
        # else:
        #     kwargs["device_map"] = "balanced"
        #     kwargs["max_memory"] = self._get_max_memory_for_v100()
        #     logger.info(
        #         "Flash Attention unavailable. "
        #         "Using device_map='balanced' with max_memory for V100."
        #     )

        llm = AutoModelForCausalLM.from_pretrained(
            llm_name_or_path,
            **kwargs
            # torch_dtype=torch.bfloat16,
            # device_map="auto",
            # quantization_config=self.bnb_config,
            # trust_remote_code=True,
            # attn_implementation="flash_attention_2"
        )
        llm.eval()

        tokenizer = AutoTokenizer.from_pretrained(
            llm_name_or_path,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
           tokenizer.pad_token = tokenizer.eos_token

        return llm, tokenizer

    def _get_attn_implementation(self) -> str | None:
        if not torch.cuda.is_available():
            logger.info("CUDA is not available. Disable Flash Attention.")
            return None

        major, _ = torch.cuda.get_device_capability()
        if major < 8:
            # FlashAttention2 requires Ampere (SM80) or newer
            logger.info("GPU does not support Flash Attention 2 (requires SM80+).")
            return None

        try:
            import flash_attn  # noqa: F401
        except Exception as e:
            logger.info(f"flash-attn is not available: {e}")
            return None

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
        temperature: float = 0.0
    ) -> str:
        # Apply the chat template
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        text_input = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize
        model_inputs = self.tokenizer(
            text_input,
            padding=True,
            truncation=True,
            # max_length=min(self.tokenizer.model_max_length, 32768),  
            return_tensors="pt",
        ).to(self.llm.device)

        # EOS (Llama3 requires "<|eot_id|>" additionally)
        eos_tokens = [self.tokenizer.eos_token_id]
        if "<|eot_id|>" in self.tokenizer.get_vocab():
            eos_tokens.append(self.tokenizer.convert_tokens_to_ids("<|eot_id|>"))

        # Generate
        generated_ids = self.llm.generate(
            **model_inputs,
            max_new_tokens=self.max_new_tokens,
            # Parameters that control the generation outputs
            num_beams=1,
            do_sample=do_sample,
            temperature=temperature,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=eos_tokens,
        )

        # Trim the prompt part
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        # Decode
        generated_text = self.tokenizer.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return generated_text
