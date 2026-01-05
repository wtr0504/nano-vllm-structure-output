from collections import deque
from dataclasses import dataclass

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager
import numpy as np
import numpy.typing as npt

from nanovllm.structured_output import StructuredOutputManager

@dataclass
class SchedulerOutput:
    sequences: list[Sequence]
    # finished_req_ids: list[int]
    structured_output_request_ids: list[int]
    grammar_bitmask: "npt.NDArray[np.int32] | None"


class Scheduler:

    def __init__(self, config: Config, structured_output_manager: StructuredOutputManager):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.structured_output_manager = structured_output_manager

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[SchedulerOutput, bool]:
        # prefill
        scheduled_seqs: list[Sequence] = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        
        structured_output_request_ids, grammar_bitmask = self.get_grammar_bitmask(scheduled_seqs)
        scheduler_output = SchedulerOutput(
            sequences=scheduled_seqs,
            structured_output_request_ids=structured_output_request_ids,
            grammar_bitmask=grammar_bitmask
        )
        if scheduled_seqs:
            # breakpoint()
            return scheduler_output, True

        # decode
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        
        structured_output_request_ids, grammar_bitmask = self.get_grammar_bitmask(scheduled_seqs)
        
        scheduler_output = SchedulerOutput(
            sequences=scheduled_seqs,
            structured_output_request_ids=structured_output_request_ids,
            grammar_bitmask=grammar_bitmask
        )
        
        # breakpoint()
        return scheduler_output, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, scheduler_seqs: SchedulerOutput, token_ids: list[int]) -> list[bool]:
        seqs = scheduler_seqs.sequences
        
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            advanced = True
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)

            if seq.structured_output_request:
                seq.structured_output_request.grammar.accept_tokens(seq.seq_id, [token_id])
                if not advanced or seq.structured_output_request.grammar.is_terminated():
                    if seq.status == SequenceStatus.FINISHED:
                        continue
                    seq.status = SequenceStatus.FINISHED
                    self.block_manager.deallocate(seq)
                    self.running.remove(seq)


    def get_grammar_bitmask(
        self,
        scheduled_requests: list[Sequence],
    ) -> tuple[list[str], "npt.NDArray[np.int32] | None"]:
        # Collect list of scheduled request ids that use structured output.
        # The corresponding rows of the bitmask will be in this order.
        # PERF: in case of chunked prefill,
        # request might not include any new tokens.
        # Therefore, we might introduce some additional
        # cycle to fill in the bitmask, which could be a big no-op.
        structured_output_request_ids = [
            seq.seq_id
            for seq in scheduled_requests
            if seq.structured_output_request
        ]
        if not structured_output_request_ids:
            return structured_output_request_ids, None

        requests = {
            seq.seq_id: seq
            for seq in scheduled_requests
        }
        
        bitmask = self.structured_output_manager.grammar_bitmask(
            requests,
            structured_output_request_ids,
        )
        return structured_output_request_ids, bitmask
