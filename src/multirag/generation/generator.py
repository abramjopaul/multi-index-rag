"""Generator abstraction for Track C local GPU inference.

Both implementations expose the same interface:
  generate_batch(messages_list) -> list[str]
  generate_batch_with_metadata(messages_list) -> list[GenerationMeta]
  unload() -> None

messages_list is a list of conversations, each a list of chat message dicts
as returned by PromptTemplate.render().

VLLMGenerator uses vLLM's LLM.chat() for fast offline batched inference.
HFGenerator uses HuggingFace transformers as a fallback.

generate_batch_with_metadata() is additive -- generate_batch() is untouched
so existing callers are unaffected -- and returns per-sample finish_reason
and raw token counts, needed to audit truncation/refusal behavior.
"""

from __future__ import annotations

import gc
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GenerationMeta:
    text: str
    finish_reason: str  # "stop" | "length" | "unknown"
    prompt_tokens: int
    completion_tokens: int


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
    def generate_batch_with_metadata(
        self, messages_list: list[list[dict]]
    ) -> list[GenerationMeta]:
        """Same as generate_batch(), plus finish_reason and raw token counts."""

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

    def generate_batch_with_metadata(
        self, messages_list: list[list[dict]]
    ) -> list[GenerationMeta]:
        logger.info(f"Generating {len(messages_list)} answers with vLLM (with metadata)...")
        outputs = self._llm.chat(
            messages=messages_list,
            sampling_params=self._sampling_params,
            use_tqdm=True,
        )
        results = []
        for o in outputs:
            completion = o.outputs[0]
            results.append(
                GenerationMeta(
                    text=completion.text.strip(),
                    finish_reason=completion.finish_reason or "unknown",
                    prompt_tokens=len(o.prompt_token_ids or []),
                    completion_tokens=len(completion.token_ids or []),
                )
            )
        logger.info("Generation complete")
        return results

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

    def generate_batch_with_metadata(
        self, messages_list: list[list[dict]]
    ) -> list[GenerationMeta]:
        """Bypasses the pipeline() wrapper (which doesn't expose token ids)
        to call model.generate() directly, so finish_reason/token counts can
        be computed from the real generated token ids rather than the
        text -- pipeline()'s output has no ids to inspect.
        """
        logger.info(f"Generating {len(messages_list)} answers with HF (with metadata)...")
        dec = self._decoding
        max_new_tokens = dec.max_new_tokens if dec else 512

        eos_ids = self._tokenizer.eos_token_id
        if eos_ids is None:
            eos_ids = []
        elif not isinstance(eos_ids, list):
            eos_ids = [eos_ids]

        results: list[GenerationMeta] = []
        for i, messages in enumerate(messages_list):
            prompt_ids = self._tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(self._model.device)
            prompt_len = prompt_ids.shape[-1]

            gen_ids = self._model.generate(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                temperature=dec.temperature if dec else 0.0,
                top_p=dec.top_p if dec else 1.0,
                do_sample=(dec.temperature > 0) if dec else False,
                pad_token_id=eos_ids[0] if eos_ids else None,
            )
            completion_ids = gen_ids[0][prompt_len:]
            completion_tokens = int(completion_ids.shape[-1])
            text = self._tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

            last_token_id = int(completion_ids[-1].item()) if completion_tokens > 0 else None
            if completion_tokens >= max_new_tokens:
                finish_reason = "length"
            elif last_token_id in eos_ids:
                finish_reason = "stop"
            else:
                finish_reason = "unknown"

            results.append(
                GenerationMeta(
                    text=text,
                    finish_reason=finish_reason,
                    prompt_tokens=int(prompt_len),
                    completion_tokens=completion_tokens,
                )
            )
            if (i + 1) % 10 == 0:
                logger.info(f"  {i + 1}/{len(messages_list)} done")
        logger.info("Generation complete")
        return results

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
