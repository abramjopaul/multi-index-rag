"""Generator abstraction for Track C local GPU inference.

Both implementations expose the same interface:
  generate_batch(messages_list) -> list[str]
  unload() -> None

messages_list is a list of conversations, each a list of chat message dicts
as returned by PromptTemplate.render().

VLLMGenerator uses vLLM's LLM.chat() for fast offline batched inference.
HFGenerator uses HuggingFace transformers as a fallback.
"""

from __future__ import annotations

import gc
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class Generator(ABC):
    @abstractmethod
    def generate_batch(self, messages_list: list[list[dict]]) -> list[str]:
        """Generate answers for a batch of prompts.

        Args:
            messages_list: One conversation (list of role/content dicts) per topic.

        Returns:
            One generated answer string per topic, in the same order.
        """

    @abstractmethod
    def unload(self) -> None:
        """Release GPU memory. Call after generate_batch() is done."""


class VLLMGenerator(Generator):
    """Fast batched generator using vLLM offline inference engine.

    All topics are processed in a single LLM.chat() call, which is the
    most efficient path on the L4 GPU.
    """

    def __init__(
        self,
        model: str,
        revision: str,
        dtype: str = "bfloat16",
        quantization: str | None = None,
        decoding=None,
        max_model_len: int | None = None,
    ):
        try:
            from vllm import LLM, SamplingParams
        except ImportError as e:
            raise ImportError(
                "vllm is not installed. Install it with: pip install vllm\n"
                "Or set generator.backend: 'hf' to use HuggingFace transformers."
            ) from e

        logger.info(
            f"Loading vLLM model: {model} (revision={revision}, dtype={dtype}, "
            f"quantization={quantization}, max_model_len={max_model_len})"
        )
        self._llm = LLM(
            model=model,
            revision=revision,
            dtype=dtype,
            quantization=quantization,
            seed=decoding.seed if decoding else 42,
            trust_remote_code=True,
            max_model_len=max_model_len,
        )

        dec = decoding
        self._sampling_params = SamplingParams(
            temperature=dec.temperature if dec else 0.0,
            top_p=dec.top_p if dec else 1.0,
            max_tokens=dec.max_new_tokens if dec else 512,
        )
        logger.info("vLLM model loaded")

    def generate_batch(self, messages_list: list[list[dict]]) -> list[str]:
        logger.info(f"Generating {len(messages_list)} answers with vLLM...")
        outputs = self._llm.chat(
            messages=messages_list,
            sampling_params=self._sampling_params,
            use_tqdm=True,
        )
        answers = [o.outputs[0].text.strip() for o in outputs]
        logger.info("Generation complete")
        return answers

    def unload(self) -> None:
        logger.info("Unloading vLLM model and freeing GPU memory...")
        del self._llm
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("GPU memory freed")


class HFGenerator(Generator):
    """Fallback generator using HuggingFace transformers pipeline.

    Use when vLLM is unavailable or incompatible with the target model.
    Slower than VLLMGenerator due to sequential processing.
    """

    def __init__(
        self,
        model: str,
        revision: str,
        dtype: str = "bfloat16",
        quantization: str | None = None,
        decoding=None,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16

        quant_config = None
        if quantization in ("int4", "int8"):
            from transformers import BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(
                load_in_4bit=(quantization == "int4"),
                load_in_8bit=(quantization == "int8"),
            )

        logger.info(
            f"Loading HF model: {model} (revision={revision}, dtype={dtype}, "
            f"quantization={quantization})"
        )
        self._tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
        self._model = AutoModelForCausalLM.from_pretrained(
            model,
            revision=revision,
            torch_dtype=torch_dtype,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
        )
        self._pipe = pipeline(
            "text-generation",
            model=self._model,
            tokenizer=self._tokenizer,
        )
        self._decoding = decoding
        logger.info("HF model loaded")

    def generate_batch(self, messages_list: list[list[dict]]) -> list[str]:
        logger.info(f"Generating {len(messages_list)} answers with HF pipeline...")
        dec = self._decoding
        answers = []
        for i, messages in enumerate(messages_list):
            out = self._pipe(
                messages,
                max_new_tokens=dec.max_new_tokens if dec else 512,
                temperature=dec.temperature if dec else 0.0,
                top_p=dec.top_p if dec else 1.0,
                do_sample=(dec.temperature > 0) if dec else False,
                return_full_text=False,
            )
            # pipeline with chat messages returns [{generated_text: [{role, content}]}]
            generated = out[0]["generated_text"]
            if isinstance(generated, list):
                text = generated[-1].get("content", "")
            else:
                text = str(generated)
            answers.append(text.strip())
            if (i + 1) % 10 == 0:
                logger.info(f"  {i + 1}/{len(messages_list)} done")
        logger.info("Generation complete")
        return answers

    def unload(self) -> None:
        logger.info("Unloading HF model and freeing GPU memory...")
        del self._model
        del self._pipe
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("GPU memory freed")


def build_generator(config) -> Generator:
    """Factory: build the right Generator from a GeneratorConfig."""
    if config.backend == "vllm":
        return VLLMGenerator(
            model=config.model,
            revision=config.revision,
            dtype=config.dtype,
            quantization=config.quantization,
            decoding=config.decoding,
            max_model_len=config.max_model_len,
        )
    elif config.backend == "hf":
        return HFGenerator(
            model=config.model,
            revision=config.revision,
            dtype=config.dtype,
            quantization=config.quantization,
            decoding=config.decoding,
        )
    else:
        raise ValueError(
            f"Unknown generator backend: {config.backend!r}. Must be 'vllm' or 'hf'."
        )
