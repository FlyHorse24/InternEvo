#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import queue
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch.optim.optimizer import Optimizer
from internlm.core.naive_amp import NaiveAMPModel
from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.core.engine import Engine
from internlm.core.scheduler import comm
from internlm.utils.common import SchedulerHook, get_current_device
from internlm.utils.logger import get_logger
from internlm.utils.parallel import is_using_isp

from .pipeline_scheduler_1f1b import (
    InterleavedPipelineScheduler,
    PipelineScheduler,
    pack_return_tensors,
)

from .pipeline_scheduler_zb import WeightGradStore,ZeroBubblePipelineVShapeScheduler
from enum import Enum
import fcntl
import queue
import logging
import random
import time
import json
import os
import grpc
import rank_pb2
import rank_pb2_grpc
from internlm.utils.timeout import llm_timeout

logger = get_logger(__file__)

def write_json(jsonpath, content):
    with open(jsonpath, 'a',encoding='utf-8') as f:
        json.dump(content, f, indent=4)

def _get_chunk_by_stage(stage_id: int,stage_alignment:list) -> int:
    for device_stage in stage_alignment:
        for chunk_id in range(len(device_stage)):
            if device_stage[chunk_id] == stage_id:
                return chunk_id
def _get_deviceid_by_alignment(stage_id: int, stage_alignment:list) -> int:
    for device_id in range(len(stage_alignment)):
        for stage_in_device in stage_alignment[device_id]:
            if stage_in_device == stage_id:
                return device_id
 
def build_stage_to_device_map(stage_placement):
    stage_to_device = {}
    for device_id in range(len(stage_placement)):
        for stage_id in stage_placement[device_id]:
            stage_to_device[stage_id] = device_id
    return stage_to_device
def do_compute():
    x = torch.rand(100, 100).cuda()
    for i in range(random.randint(1,10000)):
        result = torch.bmm(x.unsqueeze(0), x.unsqueeze(0))
    return result
def judge_Vshape_like(stage_placement):
    mutex = False
    for stages in stage_placement:
        for s in stages:
            if s+1 in stages:
                mutex = True
                break
    return mutex

def write_json(jsonpath, content):
    with open(jsonpath, 'a',encoding='utf-8') as f:
        json.dump(content, f, indent=4)
class Stage(Enum):
    FORWARD = 'f'
    BACKWARD = 'b'
    WEIGHT = 'w'

LOG_RANKS = [0,1,2,3,4,5,6,7]
def debug_print(input_rank, msg: str) -> None:
    rank = gpc.get_global_rank()
    if rank not in LOG_RANKS:
        return
    if gpc.get_local_rank(ParallelMode.DATA) == 0 and gpc.get_local_rank(ParallelMode.PIPELINE) in (input_rank):#gpc.get_local_rank(ParallelMode.TENSOR) == 0 :
        print(f"# rank {rank}: {msg}, flush=True")


class UnifiedSingleChunkPipelineScheduler(PipelineScheduler):
    """
    A helper schedule class for pipeline parallelism running environment.
    It uses non-interleaved 1F1B strategy. Other properties are similar as
    :class:`NonPipelineSchedule`.

    Args:
        num_microbatches (int): The number of microbatches.
        dtype (torch.dtype): Type of data. torch.float by default.
        data_process_func (Callable, optional):
            The post processing function which receives a micro batch of data, and it will be executed
            in `load_micro_batch`.
        tensor_shape (torch.Size, optional): Specified shape in pipeline communication.
        scatter_gather_tensors (bool, optional):
            If set to `True`, communication will be reduced over pipeline when using 1D tensor parallelization.
        scheduler_hooks (Optional[List[SchedulerHook]], optional): List of scheduler hooks.
    """

    def __init__(
        self,
        num_microbatches: int,
        dtype: torch.dtype = torch.float,
        data_process_func: Callable = None,
        tensor_shape: Union[torch.Size, List[int], Tuple[int]] = None,
        scatter_gather_tensors: bool = False,
        scheduler_hooks: Optional[List[SchedulerHook]] = None,
        optimizer: Optimizer = None,
        unified_scheduler: List[tuple] = None,
        comm_graph: List[List[tuple]] = None,
    ):
        super().__init__(
            num_microbatches,
            dtype=dtype,
            data_process_func=data_process_func,
            tensor_shape=tensor_shape,
            scatter_gather_tensors=scatter_gather_tensors,
            scheduler_hooks=scheduler_hooks,
        )
        assert len(unified_scheduler) == gpc.pipeline_parallel_size
        if unified_scheduler[0][-1][0] == Stage.WEIGHT.value:
            WeightGradStore.set_pp_mode("ZBH1")
            WeightGradStore.set_optim(optimizer)
        else:
            WeightGradStore.set_pp_mode("1F1B")

        gpc._config['jsonpath'] = f"./jsonResult/{WeightGradStore.get_pp_mode()}_pp{gpc.pipeline_parallel_size}_mb{self.num_microbatches}"
        self.unified_scheduler = unified_scheduler
        self.comm_graph = comm_graph

    def _forward_backward_step(self, engine, return_loss=True, return_output_label=True, batch_count = 0):
        """
        This function schedules the forward and backward computation of microbatches in the pipeline in a 1F1B manner.
        It consists of three stages: warmup, 1F1B, and cooldown.

        1. Warmup Stage:
        The warmup stage performs num_warmup forward microsteps. The calculation of num_warmup is the pipeline length
        minus the rank of the current pipeline minus 1. For each microstep, it receives data as input from the previous
        stage, performs the forward computation, and then sends the result to the next stage.

        2. 1F1B Stage:
        The 1F1B stage consists of pairs of forward and backward microsteps. It performs num_1f1b_micropairs iterations,
        where num_1f1b_micropairs is calculated as the total number of microbatches minus the number of microbatches in
        the warmup stage. In each iteration, it first performs a forward computation, sends the result to the next
        stage, receives input for the backward computation, performs the backward computation, and finally sends the
        result to the previous stage to receive input for the next forward computation.

        3. Cooldown Stage:
        The cooldown stage performs the same number of iterations as the warmup stage. In each iteration, it receives
        input for the backward computation, performs the backward computation, and finally sends the result to the
        previous stage.

        There are two special cases to consider:
        1. The first stage of the pipeline does not need to receive forward input or send backward output. The last
        stage does not need to send forward output or receive backward input.
        2. Pay attention to the communication between stages and use additional communication to bridge the gap.

        Args:
            engine (Engine): The engine used for computation.
            return_loss (bool, optional): Whether to return the accumulated loss.
            return_output_label (bool, optional): Whether to return outputs and labels.

        Returns:
            Tuple[Union[torch.Tensor, None], Union[torch.Tensor, None], Union[torch.Tensor, None]]:
            The output, label, and accumulated loss.
        """

        # Input, output tensors only need to be saved when doing backward passes
        async_communicator_recv_forward_queue = queue.Queue()
        async_communicator_recv_backward_queue = queue.Queue()
        
        input_objs = queue.Queue()
        output_objs = queue.Queue()
        moe_losses = queue.Queue()
        return_tensors = queue.Queue()
        accum_loss = (
            torch.zeros(1, device=get_current_device())
            if return_loss and gpc.is_pipeline_last_stage(ignore_virtual=True)
            else None
        )
        accum_moe_loss = torch.zeros(1, device=get_current_device())

        # Used for tensor meta information communication
        forward_recv_shapes = self.tensor_shape
        backward_recv_shapes = None
        need_forward_meta = self.tensor_shape is None

        #rank_info
        local_rank = gpc.get_local_rank(ParallelMode.PIPELINE)
        stage_id = local_rank
        last_stage = len(self.unified_scheduler)-1
        steps = self.unified_scheduler[local_rank]
        comm_list = self.comm_graph[local_rank]
        num_steps = len(steps)
        jsonpath = gpc._config['jsonpath']+f"/iter_{batch_count}_opeartion_list.json"
        os.makedirs(gpc._config['jsonpath'], exist_ok=True)
        torch.distributed.barrier()
        for s in range(num_steps):
            step_type, microbatch_id, stage_id, _, _, _ = steps[s]
            before_recv_list = comm_list[s]['B']
            after_recv_list = comm_list[s]['A']
            for before_ops in before_recv_list:
                op=before_ops[0]
                if op == Stage.FORWARD.value:
                    async_communicator_recv_forward_queue.put(
                        comm.recv_forward(
                        forward_recv_shapes,
                        dtype=self.dtype,
                        scatter_gather_tensors=self.scatter_gather_tensors,
                        )
                    )
                elif op == Stage.BACKWARD.value:
                    async_communicator_recv_backward_queue.put(
                        comm.recv_backward(
                        backward_recv_shapes,
                        dtype=self.dtype,
                        scatter_gather_tensors=self.scatter_gather_tensors,
                        )
                    )
            if step_type == Stage.FORWARD.value:# Forward pass
                # Receive the input from the previous stage
                ##debugprint(f"step:{step_type}, microbatch_id:{microbatch_id}, stage_id:{stage_id}, _forward_step_begin") 
                if stage_id>0:
                    if async_communicator_recv_forward_queue.qsize()>0:
                        input_obj = async_communicator_recv_forward_queue.get()
                else:
                    input_obj = None

                # Perform forward computation
                start_time = time.perf_counter()
                output_obj, moe_loss = self._forward_step(
                    engine,
                    input_obj,
                    return_tensors,
                    return_output_label=return_output_label,
                    accum_loss=accum_loss,
                    accum_moe_loss=accum_moe_loss,
                )
                end_time = time.perf_counter()
                json_content = {"local_rank":local_rank, "chunk_id":0, "microbatch_id":microbatch_id, "step_type":step_type, "operation":"compute", "start_time":start_time,  "timespan":(end_time - start_time)}
                write_json(jsonpath, json_content)
                ##debugprint(f"step:{step_type}, microbatch_id:{microbatch_id}, stage_id:{stage_id} forward_step, output_obj_shape:{output_obj.shape}")
                if stage_id < last_stage:
                    if isinstance(output_obj, torch.Tensor):
                        backward_recv_shapes = output_obj.shape
                    else:
                        backward_recv_shapes = [out_tensor.shape for out_tensor in output_obj]    
                    if need_forward_meta:
                        comm.send_obj_meta(output_obj)
                        need_forward_meta = False  # send only once.
                # Send the output of forward computation of this pipeline stage to the next pipeline stage as input for
                # forward computation

                if stage_id < last_stage :
                    assert output_obj.dtype == self.dtype

                send_forward_once = True
                for after_ops  in  after_recv_list:
                    op, mutex = after_ops[0],after_ops[-1]
                    if op == Stage.FORWARD.value:
                        async_communicator_recv_forward_queue.put(
                            comm.recv_forward(
                            forward_recv_shapes,
                            dtype=self.dtype,
                            scatter_gather_tensors=self.scatter_gather_tensors,
                            )
                        )
                    elif op == Stage.BACKWARD.value:
                        if send_forward_once and mutex == 1:
                            async_communicator_recv_backward_queue.put(
                                comm.send_forward_recv_backward(
                                    output_obj,
                                    backward_recv_shapes,
                                    dtype=self.dtype,
                                    scatter_gather_tensors=self.scatter_gather_tensors,
                                )
                            )
                            send_forward_once = False
                        else:
                            async_communicator_recv_backward_queue.put(
                                comm.recv_backward(
                                backward_recv_shapes,
                                dtype=self.dtype,
                                scatter_gather_tensors=self.scatter_gather_tensors,
                                )
                            )
                if send_forward_once and stage_id < last_stage :
                    comm.send_forward(
                            output_obj,
                            scatter_gather_tensors=self.scatter_gather_tensors,
                        )

                input_objs.put(input_obj)
                output_objs.put(output_obj)
                moe_losses.put(moe_loss)

            elif step_type == Stage.BACKWARD.value:# Backward pass
                ##debugprint(f"step:{step_type}, microbatch_id:{microbatch_id}, stage_id:{stage_id}, _backward_step_begin")
                input_obj = input_objs.get()
                output_obj = output_objs.get()
                moe_loss = moe_losses.get()
                
                if stage_id<last_stage:
                    if async_communicator_recv_backward_queue.qsize()>0:
                        output_obj_grad = async_communicator_recv_backward_queue.get()
                else:
                    output_obj_grad = None
            
                start_time = time.perf_counter()
                input_obj_grad = self._backward_step(
                    engine, microbatch_id, input_obj, output_obj, output_obj_grad, moe_loss
                )
                end_time =time.perf_counter()
                json_content = {"local_rank":local_rank, "chunk_id":0, "microbatch_id":microbatch_id, "step_type":step_type, "operation":"compute", "start_time":start_time,  "timespan":(end_time - start_time)}
                write_json(jsonpath, json_content)

                send_backward_once = True
                for after_ops  in  after_recv_list:
                    op, mutex = after_ops[0],after_ops[-1]
                    if op == Stage.FORWARD.value:
                        if send_backward_once and mutex == 1:
                            async_communicator_recv_forward_queue.put(
                                comm.send_backward_recv_forward(
                                    input_obj_grad,
                                    forward_recv_shapes,
                                    dtype=self.dtype,
                                    scatter_gather_tensors=self.scatter_gather_tensors,
                                )
                            )
                            send_backward_once = False
                        else:
                            async_communicator_recv_forward_queue.put(
                                comm.recv_forward(
                                forward_recv_shapes,
                                dtype=self.dtype,
                                scatter_gather_tensors=self.scatter_gather_tensors,
                                )
                            )
                    elif op == Stage.BACKWARD.value:
                        async_communicator_recv_backward_queue.put(
                            comm.recv_backward(
                            backward_recv_shapes,
                            dtype=self.dtype,
                            scatter_gather_tensors=self.scatter_gather_tensors,
                            )
                        )
                if send_backward_once and stage_id > 0:
                    comm.send_backward(
                        input_obj_grad,
                        scatter_gather_tensors=self.scatter_gather_tensors,
                    )

                WeightGradStore.flush()

            elif step_type == Stage.WEIGHT.value: # Weight update
                start_time = time.perf_counter()
                WeightGradStore.pop()
                end_time = time.perf_counter()
                json_content = {"local_rank":local_rank, "chunk_id":0, "microbatch_id":microbatch_id, "step_type":step_type, "operation":"compute", "start_time":start_time,  "timespan":(end_time - start_time)}
                write_json(jsonpath, json_content)
                for after_ops  in  after_recv_list:
                    op = after_ops[0]
                    if op == Stage.FORWARD.value:
                        async_communicator_recv_forward_queue.put(
                            comm.recv_forward(
                            forward_recv_shapes,
                            dtype=self.dtype,
                            scatter_gather_tensors=self.scatter_gather_tensors,
                            )
                        )
                    elif op == Stage.BACKWARD.value:
                        async_communicator_recv_backward_queue.put(
                            comm.recv_backward(
                            backward_recv_shapes,
                            dtype=self.dtype,
                            scatter_gather_tensors=self.scatter_gather_tensors,
                            )
                        )
            
        torch.distributed.barrier()

        output, label = pack_return_tensors(return_tensors) if return_tensors.qsize() > 0 else (None, None)

        if hasattr(gpc.config.model, "num_experts") and gpc.config.model.num_experts > 1:
            dist.all_reduce(accum_moe_loss, group=gpc.get_group(ParallelMode.PIPELINE))

        if accum_loss is not None:
            accum_loss += accum_moe_loss

        return output, label, accum_loss, accum_moe_loss

class UnifiedMultipleChunksPipelineScheduler(ZeroBubblePipelineVShapeScheduler):
    def __init__(
        self,
        num_microbatches: int,
        num_chunks: int,
        dtype: torch.dtype = torch.float,
        data_process_func: Callable = None,
        tensor_shape: Union[torch.Size, List[int], Tuple[int]] = None,
        scatter_gather_tensors: bool = False,
        scheduler_hooks: Optional[List[SchedulerHook]] = None,
        optimizer: Optimizer = None,
        unified_scheduler: List[tuple] = None,
        stage_placement: List[List[int]] = None,
        comm_graph: List[List[tuple]] = None,
    ):
        super().__init__(
            num_microbatches,
            num_chunks=num_chunks,
            dtype=dtype,
            data_process_func=data_process_func,
            tensor_shape=tensor_shape,
            scatter_gather_tensors=scatter_gather_tensors,
            scheduler_hooks=scheduler_hooks,
        )
        assert len(unified_scheduler) == gpc.pipeline_parallel_size
        self.unified_scheduler = unified_scheduler
        self.stage_placement = stage_placement
        self.comm_graph = comm_graph
        self.last_stage = max([stage_id for _, stage_id in stage_placement])
        if unified_scheduler[0][-1][0] != Stage.WEIGHT.value:
            WeightGradStore.set_pp_mode('Interleaved')
        if judge_Vshape_like(stage_placement):
            gpc.v_shape = True
        else:
            gpc.v_shape = False
        gpc._config['jsonpath'] = f"./jsonResult/{WeightGradStore.get_pp_mode()}_pp{gpc.pipeline_parallel_size}_mb{self.num_microbatches}"
        os.makedirs(gpc._config['jsonpath'], exist_ok=True)
    def recv_all(self,recv_forward_queue_list,recv_backward_queue_list,recvlist):
        for ops in recvlist:
            recv_op_type, recv_end_time, recv_device_id, recv_stage_id, recv_chunk_id, recv_microbatch_id, index, _= ops
            if recv_op_type == Stage.FORWARD.value:
                recv_forward_tensor = comm.recv_forward(
                            input_tensor_shape = self._input_obj_shapes[recv_chunk_id],
                            prev_rank=recv_device_id,
                            dtype=self.dtype,
                            scatter_gather_tensors=self.scatter_gather_tensors,
                )
                if not gpc.v_shape and recv_stage_id in self.stage_placement[-1]:
                    recv_forward_queue_list[recv_chunk_id+1].put(recv_forward_tensor)
                else:
                    recv_forward_queue_list[recv_chunk_id].put(recv_forward_tensor)
            elif recv_op_type == Stage.BACKWARD.value:
                recv_backward_tensor = comm.recv_backward(
                        output_grad_shape = self._output_obj_shapes[recv_chunk_id],
                        next_rank=recv_device_id,
                        dtype=self.dtype,
                        scatter_gather_tensors=self.scatter_gather_tensors,
                    )
                if not gpc.v_shape and recv_stage_id in self.stage_placement[0]:
                    recv_backward_queue_list[recv_chunk_id-1].put(recv_backward_tensor)
                else:
                    recv_backward_queue_list[recv_chunk_id].put(recv_backward_tensor)
        return recv_forward_queue_list, recv_backward_queue_list

    def _forward_backward_step(self, engine, return_loss=True, return_output_label=True, batch_count=0):
        """
        This function schedules the forward and backward computation of microbatches in the pipeline in a 1F1B manner.
        It consists of three stages: warmup, 1F1B, and cooldown.

        1. Warmup Stage:
        The warmup stage performs num_warmup forward microsteps. The calculation of num_warmup is the pipeline length
        minus the rank of the current pipeline minus 1. For each microstep, it receives data as input from the previous
        stage, performs the forward computation, and then sends the result to the next stage.

        2. 1F1B Stage:
        The 1F1B stage consists of pairs of forward and backward microsteps. It performs num_1f1b_micropairs iterations,
        where num_1f1b_micropairs is calculated as the total number of microbatches minus the number of microbatches in
        the warmup stage. In each iteration, it first performs a forward computation, sends the result to the next
        stage, receives input for the backward computation, performs the backward computation, and finally sends the
        result to the previous stage to receive input for the next forward computation.

        3. Cooldown Stage:
        The cooldown stage performs the same number of iterations as the warmup stage. In each iteration, it receives
        input for the backward computation, performs the backward computation, and finally sends the result to the
        previous stage.

        There are two special cases to consider:
        1. The first stage of the pipeline does not need to receive forward input or send backward output. The last
        stage does not need to send forward output or receive backward input.
        2. Pay attention to the communication between stages and use additional communication to bridge the gap.

        Args:
            engine (Engine): The engine used for computation.
            return_loss (bool, optional): Whether to return the accumulated loss.
            return_output_label (bool, optional): Whether to return outputs and labels.

        Returns:
            Tuple[Union[torch.Tensor, None], Union[torch.Tensor, None], Union[torch.Tensor, None]]:
            The output, label, and accumulated loss.
        """

        # Input, output tensors only need to be saved when doing backward passes

        # Used for tensor meta information communication
        local_rank = gpc.get_local_rank(ParallelMode.PIPELINE)
        global_rank = gpc.get_global_rank()
        stage_placement=self.stage_placement
        last_stage = max([stage_id for _, stage_id in stage_placement])
        steps = self.unified_scheduler[local_rank]
        stages_in_this_device = stage_placement[local_rank]
        comm_list = self.comm_graph[local_rank]
        chunks = len(stages_in_this_device)
        chunk_to_prev_stage_id = [-1 for _ in range(chunks)]
        chunk_to_next_stage_id = [last_stage+1 for _ in range(chunks)]
        chunk_to_prev_global_rank = [-1 for _ in range(chunks)]
        chunk_to_next_global_rank = [last_stage+1 for _ in range(chunks)]
        recv_backward_queue_list = [queue.Queue() for _ in range(chunks)]
        recv_forward_queue_list = [queue.Queue() for _ in range(chunks)]
        input_obj_grad_queue_list = [queue.Queue() for _ in range(chunks)]
        jsonpath = gpc._config['jsonpath'] + f"/iter_{batch_count}_opeartion_list.json"

        for i in range(chunks):
            chunk_id = i
            stage_id=stages_in_this_device[chunk_id]
            prev_stage = stage_id - 1
            next_stage = stage_id + 1
            chunk_to_prev_stage_id[chunk_id] = prev_stage
            chunk_to_next_stage_id[chunk_id] = next_stage
            if prev_stage >= 0:
                chunk_to_prev_global_rank[chunk_id] = _get_deviceid_by_alignment(prev_stage,stage_placement)
            if next_stage <= last_stage:
                chunk_to_next_global_rank[chunk_id] = _get_deviceid_by_alignment(next_stage,stage_placement)
        
        for s in range(len(steps)):
            step_type, microbatch_id, stage_id, chunk_id, startTime, end_time = steps[s]
            prev_stage = chunk_to_prev_stage_id[chunk_id]
            next_stage = chunk_to_next_stage_id[chunk_id]
            prev_global_rank = chunk_to_prev_global_rank[chunk_id]
            next_global_rank = chunk_to_next_global_rank[chunk_id]
            before_recv_list = comm_list[s]['B']
            after_recv_list = comm_list[s]['A']

            recv_forward_queue_list, recv_backward_queue_list = \
                self.recv_all(recv_forward_queue_list,recv_backward_queue_list,before_recv_list)
            if step_type == Stage.FORWARD.value:# Forward pass
                input_obj = None
                if stage_id>0:
                    if recv_forward_queue_list[chunk_id].qsize()>0:
                        input_obj = recv_forward_queue_list[chunk_id].get()
                        self._input_objs[chunk_id].append(input_obj)

                # Perform forward computation
                start_time = time.perf_counter()
                output_obj = self._forward_step(engine, chunk_id, input_obj)  
                end_time = time.perf_counter()
                json_content = {"local_rank":local_rank, "chunk_id":chunk_id, "microbatch_id":microbatch_id, "step_type":step_type, "operation":"compute", "start_time":start_time,  "timespan":(end_time - start_time)}
                write_json(jsonpath,json_content)
                
                if stage_id < last_stage:
                    if isinstance(output_obj, torch.Tensor):
                        self._output_obj_shapes[chunk_id] = output_obj.shape
                    else:
                        self._output_obj_shapes[chunk_id] = [out_tensor.shape for out_tensor in output_obj]

                    assert self._output_obj_shapes[chunk_id] == self._input_obj_shapes[chunk_id]

                    if self._send_tensor_shape_flags[chunk_id]:
                        comm.send_obj_meta(output_obj,next_global_rank)
                        self._send_tensor_shape_flags[chunk_id] = False  # send only once for each chunk.
                if not gpc.v_shape and stage_id in self.stage_placement[-1] and stage_id < last_stage:
                    if isinstance(output_obj, torch.Tensor):
                        self._output_obj_shapes[chunk_id+1] = output_obj.shape
                    else:
                        self._output_obj_shapes[chunk_id+1] = [out_tensor.shape for out_tensor in output_obj]

                if stage_id > 0 and self._input_obj_shapes[chunk_id] is None:
                    self._input_obj_shapes[chunk_id] = comm.recv_obj_meta(prev_rank=prev_global_rank)
                send_forward_once = True
                for after_ops in after_recv_list:
                    recv_op_type, recv_end_time, recv_device_id, recv_stage_id, recv_chunk_id, recv_microbatch_id, index, mutex = after_ops      
                    if recv_op_type == Stage.FORWARD.value:
                        if mutex == 1:
                            assert recv_device_id == next_global_rank
                            tensor_recv_prev,_ = comm.fused_send_recv_tensor(
                                    object_send_next=output_obj,
                                    recv_prev_shape=self._input_obj_shapes[recv_chunk_id],
                                    prev_rank=recv_device_id,
                                    next_rank=next_global_rank,
                                    dtype=self.dtype,
                                    scatter_gather_tensors=self.scatter_gather_tensors,
                                )
                            send_forward_once = False
                        else:
                            tensor_recv_prev = comm.recv_forward(
                                input_tensor_shape=self._input_obj_shapes[recv_chunk_id],
                                prev_rank=recv_device_id,
                                dtype=self.dtype,
                                scatter_gather_tensors=self.scatter_gather_tensors
                            )
                        if not gpc.v_shape and recv_stage_id in self.stage_placement[-1]:
                            recv_forward_queue_list[recv_chunk_id+1].put(tensor_recv_prev)
                        else:
                            recv_forward_queue_list[recv_chunk_id].put(tensor_recv_prev)
                    elif recv_op_type == Stage.BACKWARD.value:
                        if mutex == 1:
                            assert recv_device_id == next_global_rank
                            _, tensor_recv_next = comm.fused_send_recv_tensor(
                                    object_send_next=output_obj,
                                    recv_next_shape=self._output_obj_shapes[recv_chunk_id],
                                    next_rank=next_global_rank,
                                    dtype=self.dtype,
                                    scatter_gather_tensors=self.scatter_gather_tensors,
                                )
                            send_forward_once = False
                        else:
                            tensor_recv_next = comm.recv_backward(
                                    output_grad_shape=self._output_obj_shapes[recv_chunk_id],
                                    next_rank=recv_device_id,
                                    dtype=self.dtype,
                                    scatter_gather_tensors=self.scatter_gather_tensors
                                )
                        if not gpc.v_shape and recv_stage_id in self.stage_placement[0]:
                            recv_backward_queue_list[recv_chunk_id-1].put(tensor_recv_next)
                        else:
                            recv_backward_queue_list[recv_chunk_id].put(tensor_recv_next)

                if global_rank == next_global_rank:
                    recv_forward_queue_list[chunk_id+1].put(output_obj.clone().detach().requires_grad_())
                    send_forward_once = False
                elif stage_id<last_stage and send_forward_once:
                    comm.send_forward(
                        output_tensor=output_obj,
                        next_rank=next_global_rank,
                        scatter_gather_tensors=self.scatter_gather_tensors
                    )

            elif step_type == Stage.BACKWARD.value:# Backward pass
                output_obj_grad = None
                if stage_id<last_stage:
                    if recv_backward_queue_list[chunk_id].qsize()>0:
                        output_obj_grad = recv_backward_queue_list[chunk_id].get()
                        self._output_obj_grads[chunk_id].append(output_obj_grad)

                start_time = time.perf_counter()

                if gpc.v_shape:
                    origin_skip = engine.optimizer.skip_grad_reduce
                    input_obj_grad = self._schedule_backward(engine, chunk_id)
                else:
                    input_obj_grad = InterleavedPipelineScheduler._backward_step(self, engine, chunk_id, microbatch_id)

                end_time =time.perf_counter()
                json_content = {"local_rank":local_rank, "chunk_id":chunk_id, "microbatch_id":microbatch_id, "step_type":step_type, "operation":"compute", "start_time":start_time,  "timespan":(end_time - start_time)}
                write_json(jsonpath,json_content)
                input_obj_grad_queue_list[chunk_id].put(input_obj_grad)
                send_backward_once = True
                for after_ops in after_recv_list:
                    recv_op_type, recv_end_time, recv_device_id, recv_stage_id, recv_chunk_id, recv_microbatch_id, index, mutex = after_ops      
                    if recv_op_type == Stage.FORWARD.value:
                        if mutex == 1:
                            assert recv_device_id == prev_global_rank
                            tensor_recv_prev = comm.send_backward_recv_forward(
                                    input_tensor_grad=input_obj_grad,
                                    input_tensor_shape=self._input_obj_shapes[recv_chunk_id],
                                    prev_rank=recv_device_id,
                                    dtype=self.dtype,
                                    scatter_gather_tensors=self.scatter_gather_tensors
                                )
                            
                            send_backward_once = False
                        else:
                            tensor_recv_prev = comm.recv_forward(
                                    input_tensor_shape=self._input_obj_shapes[recv_chunk_id],
                                    prev_rank=recv_device_id,
                                    dtype=self.dtype,
                                    scatter_gather_tensors=self.scatter_gather_tensors
                                )
                        if not gpc.v_shape and recv_stage_id in self.stage_placement[-1]:
                            recv_forward_queue_list[recv_chunk_id+1].put(tensor_recv_prev)
                        else:
                            recv_forward_queue_list[recv_chunk_id].put(tensor_recv_prev)
                    elif recv_op_type == Stage.BACKWARD.value:
                        if mutex == 1:
                            assert recv_device_id == prev_global_rank
                            tensor_recv_next = comm.send_backward_recv_backward(
                                    input_tensor_grad=input_obj_grad,
                                    output_grad_shape=self._output_obj_shapes[recv_chunk_id],
                                    prev_rank=prev_global_rank,
                                    next_rank=recv_device_id,
                                    dtype=self.dtype,
                                    scatter_gather_tensors=self.scatter_gather_tensors
                                )
                            send_backward_once = False
                        else:
                            tensor_recv_next = (
                                comm.recv_backward(
                                    output_grad_shape=self._output_obj_shapes[recv_chunk_id],
                                    next_rank=recv_device_id,
                                    dtype=self.dtype,
                                    scatter_gather_tensors=self.scatter_gather_tensors
                                )
                            )
                        if not gpc.v_shape and recv_stage_id in self.stage_placement[0]:
                            recv_backward_queue_list[recv_chunk_id-1].put(tensor_recv_next)
                        else:
                            recv_backward_queue_list[recv_chunk_id].put(tensor_recv_next)

                if global_rank == prev_global_rank:
                    recv_backward_queue_list[chunk_id-1].put(input_obj_grad)
                    send_forward_once = False
                elif stage_id > 0 and send_backward_once:
                    comm.send_backward(
                        input_tensor_grad=input_obj_grad,
                        prev_rank=prev_global_rank,
                        scatter_gather_tensors=self.scatter_gather_tensors
                    )

            elif step_type == Stage.WEIGHT.value: # Weight update
                start_time = time.perf_counter()
                WeightGradStore.pop()
                self._call_hooks("after_backward",input_obj_grad_queue_list[chunk_id].get())
                engine.optimizer.skip_grad_reduce = origin_skip
                end_time = time.perf_counter()
                json_content = {"local_rank":local_rank, "chunk_id":0, "microbatch_id":microbatch_id, "step_type":step_type, "operation":"compute", "start_time":start_time,  "timespan":(end_time - start_time)}
                write_json(jsonpath, json_content)
                recv_forward_queue_list, recv_backward_queue_list = \
                    self.recv_all(recv_forward_queue_list,recv_backward_queue_list,after_recv_list)

