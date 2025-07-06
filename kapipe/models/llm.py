from collections import OrderedDict

import torch
# import torch.nn as nn
# import torch.nn.functional as F
from transformers import (
    AutoConfig, AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, StoppingCriteriaList, StoppingCriteria
)

# from . import shared_functions


class LLM:

    def __init__(
        self,
        device,
        # Model
        llm_name_or_path,
        max_seg_len,
        quantization_bits,
        # Generation
        max_new_tokens,
        beam_size,
        do_sample,
        num_return_sequences,
        stop_list,
        clean_up_tokenization_spaces,
    ):
        """
        Parameters
        ----------
        device : str
        llm_name_or_path : str
        max_seg_len : int
        max_new_tokens : int
        quantization_bits : int
        beam_size : int
        do_sample : bool
        num_return_sequences : int
        stop_list : list[str]
        clean_up_tokenization_spaces : bool
        """
        ########################
        # Hyper parameters
        ########################

        self.device = device

        self.llm_name_or_path = llm_name_or_path
        self.max_seg_len = max_seg_len
        self.quantization_bits = quantization_bits

        self.max_new_tokens = max_new_tokens
        self.beam_size = beam_size
        self.do_sample = do_sample
        self.num_return_sequences = num_return_sequences
        self.stop_list = stop_list
        self.clean_up_tokenization_spaces = clean_up_tokenization_spaces

        ########################
        # Components
        ########################

        self.bnb_config = self._set_quantization(
            quantization_bits=self.quantization_bits
        )
        self.llm, self.tokenizer = self._initialize_llm_and_tokenizer(
            pretrained_model_name_or_path=self.llm_name_or_path
        )
        self.stopping_criteria = self._define_stopping_criteria()

        ########################
        # Preprocessor
        ########################

        self.preprocessor = LLMPreprocessor(
            tokenizer=self.tokenizer,
            max_seg_len=self.max_seg_len
        )

    def _set_quantization(
        self,
        quantization_bits
    ):
        """Configure quantization settings based on the specified bits.

        Parameters
        ----------
        quantization_bits : int

        Returns
        -------
        BitsAndBytesConfig | None
        """
        if quantization_bits in [4, 8]:
            bnb_config = BitsAndBytesConfig()
            if quantization_bits == 4:
                bnb_config.load_in_4bit = True
                bnb_config.bnb_4bit_quant_type = 'nf4'
                bnb_config.bnb_4bit_use_double_quant = True
                bnb_config.bnb_4bit_compute_dtype = torch.bfloat16
            elif quantization_bits == 8:
                bnb_config.load_in_8bit = True
            return bnb_config
        return None

    def _initialize_llm_and_tokenizer(
        self,
        pretrained_model_name_or_path
    ):
        """Initializes the LLM and tokenizer with the given model ID.

        Parameters
        ----------
        pretrained_model_name_or_path : str

        Returns
        -------
        tuple[AutoModelForCausalLM, AutoTokenizer]
        """
        llm_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=True
        )
        llm_config.max_seq_len = self.max_seg_len

        llm = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=True,
            config=llm_config,
            quantization_config=self.bnb_config,
            torch_dtype=torch.bfloat16,
            device_map='auto',
        )
        llm.eval() # Set the model to evaluation mode

        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            padding_side="left",
            truncation_side="left",
            model_max_length=self.max_seg_len,
        )
        # Most LLMs don't have a pad token by default
        tokenizer.pad_token = tokenizer.eos_token

        return llm, tokenizer

    def _define_stopping_criteria(self):
        """Defines stopping criteria for text generation.

        Returns
        -------
        StoppingCriteriaList
        """
        stop_token_ids = [
            self.tokenizer(x)['input_ids']
            for x in self.stop_list
        ]
        stop_token_ids = [
            torch.LongTensor(x).to(self.device)
            for x in stop_token_ids
        ]

        class StopOnTokens(StoppingCriteria):
            def __call__(
                self,
                input_ids: torch.LongTensor,
                scores: torch.FloatTensor,
                **kwargs
            ) -> bool:
                for stop_ids in stop_token_ids:
                    if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                        return True
                return False

        return StoppingCriteriaList([StopOnTokens()])

    ################
    # Forward pass
    ################

    def preprocess(self, prompt):
        """
        Parameters
        ----------
        prompt : str

        Returns
        -------
        dict[str, Any]
        """
        return self.preprocessor.preprocess(prompt=prompt)

    def tensorize(self, preprocessed_data, compute_loss):
        """
        Parameters
        ----------
        preprocessed_data : dict[str, Any]
        compute_loss : bool

        Returns
        -------
        dict[str, Any]
        """
        model_input = {}

        model_input["compute_loss"] = compute_loss

        # (batch_size, max_seg_len)
        model_input["segments_id"] = torch.tensor(
            preprocessed_data["llm_input"]["segments_id"],
            device=self.device
        )

        # (batch_size, max_seg_len)
        model_input["segments_mask"] = torch.tensor(
            preprocessed_data["llm_input"]["segments_mask"],
            device=self.device
        )

        return model_input

    def generate(
        self,
        segments_id,
        segments_mask,
        compute_loss
    ):
        """Generates text based on the given prompt.

        Parameters
        ----------
        segments_id : torch.Tensor
            shape of (n_segments, max_seg_len)
        segments_mask : torch.Tensor
            shape of (n_segments, max_seg_len)
        compute_loss : bool

        Returns
        -------
        list[str]
        """
        # Generate tokens (IDs)
        # (1, n_tokens)
        generated_token_ids = self.llm.generate(
            # **inputs,
            input_ids=segments_id,
            attention_mask=segments_mask,
            # Parameters that control the length of the output
            max_new_tokens=self.max_new_tokens,
            # Parameters that control the generation strategy used
            num_beams=self.beam_size,
            do_sample=self.do_sample,
            # Parameters for manipulation of the model output logits
            # temperature=self.temperature,
            # top_k=self.top_k,
            # top_p=self.top_p,
            repetition_penalty=1.1,
            # Parameters that define the output variables of `generate`
            num_return_sequences=self.num_return_sequences,
            # Special tokens that can be used at generation time
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            # Others
            stopping_criteria=self.stopping_criteria
        )

        # Convert token IDs to tokens
        # list[str]
        generated_texts = self.tokenizer.batch_decode(
            generated_token_ids,
            clean_up_tokenization_spaces=self.clean_up_tokenization_spaces,
            skip_special_tokens=False
        )
        return generated_texts

    def remove_prompt_from_generated_text(self, generated_text):
        """
        Parameters
        ----------
        generated_text : str

        Returns
        -------
        str
        """
        return self.preprocessor.remove_prompt_from_generated_text(
            generated_text=generated_text
        )
 
class LLMPreprocessor:

    def __init__(
        self,
        tokenizer,
        max_seg_len,
    ):
        """
        Parameters
        ----------
        tokenizer : PreTrainedTokenizer
        max_seg_len : int
        """
        self.tokenizer = tokenizer
        self.max_seg_len = max_seg_len

        # Set the EOS marker and the user-turn-end marker
        if "Llama-2" in tokenizer.name_or_path:
            self.eos_marker = "</s>"
            self.user_turn_end_marker = "[/INST]"
        elif "Llama-3" in tokenizer.name_or_path:
            self.eos_marker = "<|eot_id|>"
            self.user_turn_end_marker = "<|start_header_id|>assistant<|end_header_id|>"
        elif "gemma" in tokenizer.name_or_path:
            self.eos_marker = "<eos>"
            self.user_turn_end_marker = "<start_of_turn>model"
        else:
            raise Exception(f"Invalid tokenizer: {tokenizer.name_or_path}")

    def preprocess(self, prompt):
        """
        Parameters
        ----------
        prompt : str

        Returns
        -------
        dict[str, Any]
        """
        preprocessed_data = OrderedDict()

        #####
        # prompt
        #####

        prompt = self.tokenizer.apply_chat_template(
            [
                # {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            tokenize=False,
            add_generation_prompt=True
        )
        preprocessed_data["prompt"] = prompt

        #####
        # llm_input: dict[str, Any]
        # segments: list[list[str]]
        # segments_id: list[list[int]]
        # segments_mask: list[list[int]]
        #####

        inputs = self.tokenizer.encode_plus(
            prompt,
            add_special_tokens=False,
            max_length=self.max_seg_len,
            padding=True,
            truncation=True,
            return_overflowing_tokens=True
        )
        # assert len(inputs["input_ids"]) == 1, len(inputs["input_ids"])
        if len(inputs["input_ids"]) > 1:
            preprocessed_data["skip"] = True

        if len(inputs["input_ids"][0]) >= 4096 - 512 - 5:
            preprocessed_data["skip"] = True
        else:
            preprocessed_data["skip"] = False

        llm_input = {}
        llm_input["segments_id"] = inputs["input_ids"]
        llm_input["segments"] = [
            self.tokenizer.convert_ids_to_tokens(seg)
            for seg in inputs["input_ids"]
        ]
        llm_input["segments_mask"] = inputs["attention_mask"]
        preprocessed_data["llm_input"] = llm_input

        return preprocessed_data

    def remove_prompt_from_generated_text(self, generated_text):
        """
        Parameters
        ----------
        generated_text : str

        Returns
        -------
        str
        """
        start = (
            generated_text.find(self.user_turn_end_marker)
            + len(self.user_turn_end_marker)
        )
        generated_text = generated_text[start:].strip()
        generated_text = generated_text.replace(self.eos_marker, "")
        return generated_text

