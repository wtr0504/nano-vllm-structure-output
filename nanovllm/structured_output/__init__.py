



from concurrent.futures import Future, ThreadPoolExecutor
import os
from pydantic import Field
import multiprocessing

import torch
from transformers import AutoTokenizer
from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.structured_output.backend_types import StructuredOutputBackend, StructuredOutputGrammar
from nanovllm.structured_output.backend_xgrammar import XgrammarBackend
from nanovllm.utils.tokenizer import init_tokenizer_from_configs

import numpy as np
import numpy.typing as npt

from vllm.config import ModelConfig

class StructuredOutputManager:
    """Engine-level manager for structured output requests."""

    def __init__(self, config:Config):
        self.backend: StructuredOutputBackend | None = None
        self.max_batch_size = config.max_num_seqs
        self.config = config
        max_workers = max(1, (multiprocessing.cpu_count() + 1) // 2)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._full_mask = torch.tensor(-1, dtype=torch.int32)
        self._grammar_bitmask: torch.Tensor | None = None

        model_config = ModelConfig()
        model_config.model = config.model
        
        self.tokenizer = init_tokenizer_from_configs(
            model_config=model_config
        )
        # path = os.path.expanduser("/data/taoran/models/Qwen3-0.6B/")
        # self.tokenizer = AutoTokenizer.from_pretrained(path)
        
        
        self.fill_bitmask_parallel_threshold = 128
        if self.fill_bitmask_parallel_threshold < self.max_batch_size:
            self.fill_bitmask_parallel_batch_size = 16
            # Use:
            # - at least 1 CPU
            # - at most half the number of CPUs or 8, whichever is less
            max_workers = max(1, min(multiprocessing.cpu_count() // 2, 8))
            self.executor_for_fillmask = ThreadPoolExecutor(max_workers=max_workers)

    def grammar_init(self, request: Sequence) -> None:
        if request.structured_output_request is None:
            return
        
        if self.backend is None:
            assert request.sampling_params is not None
            backend = request.sampling_params.structured_outputs._backend
            vocab_size = self.config.vocab_size
            if backend == "xgrammar":
                self.backend = XgrammarBackend(
                    self.config,
                    tokenizer=self.tokenizer,
                    vocab_size=vocab_size,
                )
        # grammar = self.executor.submit(self._async_create_grammar, request)
        # grammar.result()
        grammar = self._async_create_grammar(request)
        request.structured_output_request.grammar = grammar  # type: ignore[assignment]
        
        
    def _async_create_grammar(
        self,
        request: Sequence,
    ) -> StructuredOutputGrammar:
        key = request.structured_output_request.structured_output_key  # type: ignore[union-attr]

        request_type, grammar_spec = key

        assert self.backend is not None
        return self.backend.compile_grammar(request_type, grammar_spec)

    def _fill_bitmasks(
        self,
        batch: list[tuple[StructuredOutputGrammar, int, bool]],
    ) -> None:
        assert self._grammar_bitmask is not None
        for grammar, index, apply_bitmask in batch:
            if apply_bitmask and not grammar.is_terminated():
                grammar.fill_bitmask(self._grammar_bitmask, index)
            else:
                # Note that for thinking support, we will need to
                # reset the relevant part of the bitmask for consequent
                # requests here.
                self._grammar_bitmask[index].fill_(self._full_mask)

    def _async_submit_fill_bitmask(
        self,
        batch: list[tuple[StructuredOutputGrammar, int, bool]],
    ) -> Future:
        return self.executor_for_fillmask.submit(self._fill_bitmasks, batch)
    
    
    def grammar_bitmask(
        self,
        requests: dict[int, Sequence],
        structured_output_request_ids: list[int],
    ) -> "npt.NDArray[np.int32] | None":
        # Prepare the structured output bitmask for this batch.
        if not structured_output_request_ids:
            return None
        # breakpoint()
        if self._grammar_bitmask is None:
            assert self.backend is not None
            max_batch_size = self.max_batch_size

            # Allocate a bitmask for each token needing to be checked:
            # one for each speculative position, and one more for the
            # bonus token / non-speculative token.
            self._grammar_bitmask = self.backend.allocate_token_bitmask(max_batch_size)
        
        cumulative_index = 0
        
        if (
            len(structured_output_request_ids) > self.fill_bitmask_parallel_threshold
        ):
            promises = []
            batch = []
            for req_id in structured_output_request_ids:
                request = requests[req_id]
                structured_output_request = request.structured_output_request

                apply_bitmask = self.should_fill_bitmask(request)
                batch.append(
                    (structured_output_request.grammar, cumulative_index, apply_bitmask)
                )
                if len(batch) == self.fill_bitmask_parallel_batch_size:
                    promises.append(self._async_submit_fill_bitmask(batch))
                    batch = []

                cumulative_index += 1
            if batch:
                promises.append(self._async_submit_fill_bitmask(batch))

            # Wait for all bitmask filling tasks to complete.
            for promise in promises:
                promise.result()
        else:
            # Fallback to serial filling of bitmasks for small-batch-size cases
            for req_id in structured_output_request_ids:
                request = requests[req_id]
                structured_output_request = request.structured_output_request

                apply_bitmask = self.should_fill_bitmask(request)
                self._fill_bitmasks(
                    [
                        (
                            structured_output_request.grammar,
                            cumulative_index,
                            apply_bitmask,
                        )
                    ]
                )

                cumulative_index += 1
                

        bitmask_tensor = self._grammar_bitmask
        if cumulative_index < bitmask_tensor.shape[0]:
            bitmask_tensor = bitmask_tensor[:cumulative_index]

        # After finishing with the xgrammar operations, we convert to
        # np.ndarray, because that is much more efficient for serialization
        # and deserialization when sending this to the GPU workers.
        return bitmask_tensor.numpy()
                


    def should_fill_bitmask(self, request: Sequence) -> bool:
        return True