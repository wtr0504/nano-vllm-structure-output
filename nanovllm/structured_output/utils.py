

import numpy as np
import torch
from nanovllm.engine.scheduler import SchedulerOutput
import xgrammar as xgr

def apply_grammar_bitmask(
    scheduler_output: SchedulerOutput,
    logits: torch.Tensor,
) -> None:
    """
    Apply grammar bitmask to output logits of the model with xgrammar function.

    Args:
        scheduler_output (SchedulerOutput): The result of engine scheduling.
        input_batch (InputBatch): The input of model runner.
        logits (torch.Tensor): The output logits of model forward.
    """
    grammar_bitmask = scheduler_output.grammar_bitmask
    if grammar_bitmask is None:
        return

    # We receive the structured output bitmask from the scheduler,
    # compacted to contain bitmasks only for structured output requests.
    # The order of the requests in the bitmask is not guaranteed to be the
    # same as the order of the requests in the gpu runner's batch. We need
    # to sort the bitmask to match the order of the requests used here.

    # Get the batch indices of the structured output requests.
    # Keep track of the number of speculative tokens scheduled for every
    # request in the batch, as the logit indices are offset by this amount.
    struct_out_req_batch_indices: dict[int, int] = {}
    cumulative_offset = 0
    batch_size = len(scheduler_output.sequences)
    seq = [(scheduler_output.sequences[i].seq_id, i) for i in range(batch_size)]
    # seq = sorted(input_batch.req_id_to_index.items(), key=lambda x: x[1])
    for req_id, batch_index in seq:
        logit_index = batch_index
        if req_id in scheduler_output.structured_output_request_ids:
            struct_out_req_batch_indices[req_id] = logit_index

    out_indices = []

    # Reorder the bitmask to match the order of the requests in the batch.
    sorted_bitmask = np.full(
        shape=(logits.shape[0], grammar_bitmask.shape[1]),
        fill_value=-1,
        dtype=grammar_bitmask.dtype,
    )
    cumulative_index = 0
    for req_id in scheduler_output.structured_output_request_ids:

        if req_id in struct_out_req_batch_indices:
            logit_index = struct_out_req_batch_indices[req_id]
            sorted_bitmask[logit_index] = grammar_bitmask[cumulative_index]
            out_indices.append(logit_index)
        cumulative_index += 1
    grammar_bitmask = sorted_bitmask

    # If the length of out indices and the logits have the same shape
    # we don't need to pass indices to the kernel,
    # since the bitmask is already aligned with the logits.
    skip_out_indices = len(out_indices) == logits.shape[0]

    # Serialization of np.ndarray is much more efficient than a tensor,
    # so we receive it in that format.
    grammar_bitmask = torch.from_numpy(grammar_bitmask).contiguous()

    xgr.apply_token_bitmask_inplace(
        logits,
        grammar_bitmask.to(logits.device, non_blocking=True),
        indices=out_indices if not skip_out_indices else None,
    )