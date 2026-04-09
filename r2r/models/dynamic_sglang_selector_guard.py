import multiprocessing as mp
from functools import wraps
import sys
import uuid
from typing import List, Union

import torch
import torch.distributed as dist

from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.server_args import PortArgs, ServerArgs

from r2r.models.dynamic_sglang_selector import DynamicSimpleSGLangSelector


class DynamicSimpleSGLangSelectorGuard(DynamicSimpleSGLangSelector):
    """Guard-specific selector with TP-safe reference worker path.

    This avoids the custom KV cache write path used by the base selector and
    uses scheduler-native batching for reference generation, which is more
    stable for Guard-family models with tp_size > 1.
    """

    @staticmethod
    def reference_model_worker(
        rank,
        world_size: int,
        server_args: ServerArgs,
        input_queues: List[mp.Queue],
        output_queue: mp.Queue,
        ack_queue: mp.Queue,
    ):
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

        input_queue = input_queues[rank]

        def _patch_guard_output_to_tuple() -> None:
            for module in list(sys.modules.values()):
                if module is None:
                    continue
                module_name = getattr(module, "__name__", "")
                if "modeling_qwen3_guard" not in module_name:
                    continue
                try:
                    guard_output_cls = getattr(module, "GuardLogitsOutputWithPast", None)
                except Exception:
                    continue
                if guard_output_cls is None:
                    continue
                if not isinstance(guard_output_cls, type):
                    continue
                if "to_tuple" not in vars(guard_output_cls):
                    def _to_tuple(self):
                        hidden_states = getattr(self, "hidden_states", None)
                        last_hidden_state = getattr(self, "last_hidden_state", None)

                        if last_hidden_state is None:
                            if isinstance(hidden_states, (tuple, list)) and len(hidden_states) > 0:
                                last_hidden_state = hidden_states[-1]
                            elif torch.is_tensor(hidden_states):
                                last_hidden_state = hidden_states

                        values = [last_hidden_state]
                        past_key_values = getattr(self, "past_key_values", None)
                        attentions = getattr(self, "attentions", None)
                        if past_key_values is not None:
                            values.append(past_key_values)
                        if hidden_states is not None:
                            values.append(hidden_states)
                        if attentions is not None:
                            values.append(attentions)

                        return tuple(v for v in values if v is not None)

                    setattr(guard_output_cls, "to_tuple", _to_tuple)

                guard_model_cls = getattr(module, "Qwen3ForGuardModel", None)
                if isinstance(guard_model_cls, type):
                    original_forward = getattr(guard_model_cls, "forward", None)
                    if callable(original_forward) and not getattr(original_forward, "_r2r_guard_patched", False):
                        @wraps(original_forward)
                        def _forward_with_hidden(self, *args, **kwargs):
                            kwargs["output_hidden_states"] = True
                            output = original_forward(self, *args, **kwargs)

                            hidden_states = getattr(output, "hidden_states", None)
                            if getattr(output, "last_hidden_state", None) is None:
                                if isinstance(hidden_states, (tuple, list)) and len(hidden_states) > 0:
                                    output.last_hidden_state = hidden_states[-1]
                                elif torch.is_tensor(hidden_states):
                                    output.last_hidden_state = hidden_states
                            return output

                        _forward_with_hidden._r2r_guard_patched = True
                        setattr(guard_model_cls, "forward", _forward_with_hidden)

        def _build_scheduler() -> Scheduler:
            port_args = PortArgs.init_new(server_args)
            return Scheduler(
                server_args=server_args,
                port_args=port_args,
                gpu_id=rank,
                tp_rank=rank,
                dp_rank=0,
                moe_ep_rank=0,
                pp_rank=0,
            )

        scheduler = _build_scheduler()
        _patch_guard_output_to_tuple()

        while True:
            reqs: Union[List[Req], int, str] = input_queue.get()

            if isinstance(reqs, int):
                break

            if isinstance(reqs, str):
                if reqs == "RESET_CACHE":
                    scheduler.waiting_queue.clear()
                    scheduler.last_batch = None
                    ack_queue.put(0)
                continue

            for req in reqs:
                scheduler.waiting_queue.append(req)

            batch = scheduler.get_next_batch_to_run()
            if batch is None:
                next_token_ids_list = []
            else:
                _patch_guard_output_to_tuple()
                result = scheduler.run_batch(batch)
                next_token_ids_list = result.next_token_ids.tolist()

                for req, next_token_id in zip(batch.reqs, result.next_token_ids):
                    if next_token_id.item() in scheduler.model_config.hf_eos_token_id:
                        scheduler.abort_request(AbortReq(req.rid))
                    req.output_ids.append(next_token_id.item())
                    req.check_finished()
                    if req.finished():
                        scheduler.tree_cache.cache_finished_req(req)
                scheduler.last_batch = batch

            if rank == 0:
                output_queue.put(next_token_ids_list)

    def extend_step(self, input_ids: List[List[int]], input_indices: List[int], sampling_params) -> List[int]:
        """Guard-safe reference extend step.

        Avoids base selector's prefix-index / manual cache reuse path and sends
        cache-independent one-step requests to the reference workers.
        """
        input_texts = self.tokenizer.batch_decode(input_ids)
        reqs = []
        for input_text, input_id in zip(input_texts, input_ids):
            req = Req(
                rid=str(uuid.uuid4()),
                origin_input_text=input_text,
                origin_input_ids=input_id,
                sampling_params=sampling_params,
                eos_token_ids=self.quick_scheduler.model_config.hf_eos_token_id,
                return_hidden_states=False,
                vocab_size=self.quick_scheduler.model_config.vocab_size,
            )
            req.sampling_params.normalize(None)
            reqs.append(req)

        for q in self.reference_model_input_queues:
            q.put_nowait(reqs)

        next_token_ids = self.reference_model_output_queue.get()
        return next_token_ids
