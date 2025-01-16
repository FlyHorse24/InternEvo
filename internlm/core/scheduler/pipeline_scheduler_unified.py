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
from .pipeline_scheduler_zb import WeightGradStore
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

logger = get_logger(__file__)

def do_compute():
    x = torch.rand(100, 100).cuda()
    for i in range(random.randint(1,10000)):
        result = torch.bmm(x.unsqueeze(0), x.unsqueeze(0))
    return result

def create_rank(stub, local_rank):
    create_response = stub.CreateRank(rank_pb2.CreateRankRequest(
        local_rank=local_rank,
        sendforwardtimes=0,
        sendbackwardtimes=0
    ))
    #print(f"Created Rank: {create_response.rank}")
    return create_response.rank.id
    
def get_infor_by_local_rank(stub, local_rank):
    #print(f"get_infor_by_local_rank: {local_rank}")
    get_by_local_rank_response = stub.GetRankByLocalRank(rank_pb2.GetRankByLocalRankRequest(local_rank=local_rank))
    return get_by_local_rank_response.rank

def update_rank(stub, rank_id, local_rank, sendforwardtimes, sendbackwardtimes,index):
    update_response = stub.UpdateRank(rank_pb2.UpdateRankRequest(
            id=rank_id,
            local_rank=local_rank,
            sendforwardtimes=sendforwardtimes,
            sendbackwardtimes=sendbackwardtimes,
            index=index
        ))
    return update_response.rank.timestamp.ToDatetime() >= update_response.rank.last_getrank_timestamp.ToDatetime()
    #print(f"Updated Rank: {update_response.rank}")

def update_rankSendFT(stub, rank_id, sendforwardtimes):
    update_response = stub.UpdateSendForwardTimes(rank_pb2.UpdateSendForwardTimesRequest(
            id=rank_id,
            sendforwardtimes=sendforwardtimes,
        ))
    # print(f"sendforwardtimes: {sendforwardtimes}")
    # print(f"Updated Rank: {update_response.rank}")

def update_rankSendBT(stub, rank_id, sendbackwardtimes):
    update_response = stub.UpdateSendBackwardTimes(rank_pb2.UpdateSendBackwardTimesRequest(
            id=rank_id,
            sendbackwardtimes=sendbackwardtimes,
        ))
            # update_sendforward_response = stub.UpdateSendForwardTimes(rank_pb2.UpdateSendForwardTimesRequest(
        #     id=rank_id,
        #     sendforwardtimes=20
        # ))
    # print(f"sendbackwardtimes: {sendbackwardtimes}")
    # print(f"Updated Rank: {update_response.rank}")


def delete_rank(stub, rank_id):
    delete_response = stub.DeleteRank(rank_pb2.DeleteRankRequest(id=rank_id))
    #print(f"Delete Rank Success: {delete_response.success}")

def write_json(jsonpath, content):
    with open(jsonpath, 'a',encoding='utf-8') as f:
        json.dump(content, f, indent=4)

def get_prevRankBeforeIndexFtimes(index, rank_list):
    Ftimes = 0
    for s in range(index-1, -1, -1):
        step_type, microbatch_id, _= rank_list[s]
        if step_type == Stage.FORWARD.value:
            if microbatch_id>Ftimes:
                Ftimes = microbatch_id
            else:
                break
        else:
            continue
    return Ftimes

def get_nextRankbeforeIndexInfoBtimes(index, rank_list):
    Btimes = 0
    for s in range(index-1, -1, -1):
        step_type, microbatch_id, _ = rank_list[s]
        if step_type == Stage.BACKWARD.value:
            if microbatch_id>Btimes:
                Btimes = microbatch_id
            else: 
                break
        else:
            continue
    return Btimes

class Stage(Enum):
    FORWARD = 'f'
    BACKWARD = 'b'
    WEIGHT = 'w'

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
    ):
        super().__init__(
            num_microbatches,
            dtype=dtype,
            data_process_func=data_process_func,
            tensor_shape=tensor_shape,
            scatter_gather_tensors=scatter_gather_tensors,
            scheduler_hooks=scheduler_hooks,
        )
        WeightGradStore.set_pp_mode("ZBH1")
        WeightGradStore.set_optim(optimizer)
        self.unified_scheduler = unified_scheduler

    def PreRecvAll(self, last_stage, step_type, microbatch_id, stage_id, jsonpath, stub,
                prev_rank, recv_forward_start_time, forward_recv_shapes, async_communicator_recv_forward_queue,
                next_rank,  recv_backward_start_time, backward_recv_shapes, async_communicator_recv_backward_queue):
        if prev_rank > -1:
            sendforwardtimes = get_prevRankBeforeIndexFtimes(get_infor_by_local_rank(stub, prev_rank).index, self.unified_scheduler[prev_rank])
            Current_NeedRecvForwardStartTimes = sendforwardtimes-recv_forward_start_time
            #print(f"step:{step_type}, microbatch_id:{microbatch_id}, stage_id:{stage_id} ,Current_NeedRecvForwardStartTimes:{Current_NeedRecvForwardStartTimes}, sendforwardtimes:{sendforwardtimes}, recv_forward_start_time:{recv_forward_start_time}")
            if Current_NeedRecvForwardStartTimes>0:
                for i in range(Current_NeedRecvForwardStartTimes):
                    async_communicator_recv_forward = comm.AsynCommunicator(
                        recv_prev_shape = forward_recv_shapes,
                        dtype=self.dtype,
                        scatter_gather_tensors=self.scatter_gather_tensors,
                        stage_id=stage_id,
                        microbatch_id=recv_forward_start_time,#mirobatch的id是从0开始，recv_forward_start_time从1开始
                        step_type=step_type,
                    )
                    commOperation_info = async_communicator_recv_forward.start()
                    recv_forward_start_time += 1
                    write_json(jsonpath, commOperation_info)
                    async_communicator_recv_forward_queue.put(async_communicator_recv_forward)
                    ##debugprint(f"step:{step_type}, microbatch_id:{microbatch_id}, stage_id:{stage_id} recv_forward_queue.put, recv_forward_start:{recv_forward_start_time}")
            
        if next_rank < last_stage+1:
            sendbackwardtimes = get_nextRankbeforeIndexInfoBtimes(get_infor_by_local_rank(stub, next_rank).index, self.unified_scheduler[next_rank])
            Current_NeedRecvBackwardStartTimes = sendbackwardtimes-recv_backward_start_time
            #print(f"step:{step_type}, microbatch_id:{microbatch_id}, stage_id:{stage_id} ,Current_NeedRecvBackwardStartTimes:{Current_NeedRecvBackwardStartTimes}, sendbackwardtimes:{sendbackwardtimes}, recv_backward_start:{recv_backward_start_time}")
            if Current_NeedRecvBackwardStartTimes>0:
                for i in range(Current_NeedRecvBackwardStartTimes):
                    async_communicator_recv_backward = comm.AsynCommunicator(
                        recv_next_shape=backward_recv_shapes,
                        dtype=self.dtype,
                        scatter_gather_tensors=self.scatter_gather_tensors,
                        stage_id=stage_id,
                        microbatch_id=recv_backward_start_time,
                        step_type=step_type,
                    )
                    commOperation_info = async_communicator_recv_backward.start()
                    recv_backward_start_time += 1
                    write_json(jsonpath, commOperation_info)
                    async_communicator_recv_backward_queue.put(async_communicator_recv_backward)
                    ##debugprint(f"step:{step_type}, microbatch_id:{microbatch_id}, stage_id:{stage_id} recv_backward_queue.put, recv_backward_start:{recv_backward_start_time}")
        return recv_forward_start_time, recv_backward_start_time, async_communicator_recv_forward_queue, async_communicator_recv_backward_queue

    def recvAll(self, last_stage, step_type, microbatch_id, stage_id, jsonpath, stub,
                prev_rank, recv_forward_start_time, forward_recv_shapes, async_communicator_recv_forward_queue,
                next_rank, recv_backward_start_time, backward_recv_shapes, async_communicator_recv_backward_queue):
        if prev_rank > -1:
            sendforwardtimes = get_infor_by_local_rank(stub, prev_rank).sendforwardtimes
            Current_NeedRecvForwardStartTimes = sendforwardtimes-recv_forward_start_time
            print(f"recvAll, local_rank:{gpc.get_local_rank(ParallelMode.PIPELINE)} step_type:{step_type} microbatch_id:{microbatch_id}, sendforwardtimes:{sendforwardtimes}, Current_NeedRecvForwardStartTimes:{Current_NeedRecvForwardStartTimes}")
            #print(f"step:{step_type}, microbatch_id:{microbatch_id}, stage_id:{stage_id} ,Current_NeedRecvForwardStartTimes:{Current_NeedRecvForwardStartTimes}, sendforwardtimes:{sendforwardtimes}, recv_forward_start_time:{recv_forward_start_time}")
            if Current_NeedRecvForwardStartTimes>0:
                for i in range(Current_NeedRecvForwardStartTimes):
                    async_communicator_recv_forward = comm.AsynCommunicator(
                        recv_prev_shape = forward_recv_shapes,
                        dtype=self.dtype,
                        scatter_gather_tensors=self.scatter_gather_tensors,
                        stage_id=stage_id,
                        microbatch_id=recv_forward_start_time,#mirobatch的id是从0开始，recv_forward_start_time从1开始
                        step_type=step_type,
                    )
                    commOperation_info = async_communicator_recv_forward.start()
                    recv_forward_start_time += 1
                    write_json(jsonpath, commOperation_info)
                    async_communicator_recv_forward_queue.put(async_communicator_recv_forward)
                    ##debugprint(f"step:{step_type}, microbatch_id:{microbatch_id}, stage_id:{stage_id} recv_forward_queue.put, recv_forward_start:{recv_forward_start_time}")
            
        if next_rank < last_stage+1:
            sendbackwardtimes = get_infor_by_local_rank(stub, next_rank).sendbackwardtimes
            #sendbackwardtimes = read_from_file(readsendbackwardpath)
            Current_NeedRecvBackwardStartTimes = sendbackwardtimes-recv_backward_start_time
            print(f"recvAll, local_rank:{gpc.get_local_rank(ParallelMode.PIPELINE)} step_type:{step_type} microbatch_id:{microbatch_id}, sendbackwardtimes:{sendbackwardtimes}, Current_NeedRecvBackwardStartTimes:{Current_NeedRecvBackwardStartTimes}")
            #print(f"step:{step_type}, microbatch_id:{microbatch_id}, stage_id:{stage_id} ,Current_NeedRecvBackwardStartTimes:{Current_NeedRecvBackwardStartTimes}, sendbackwardtimes:{sendbackwardtimes}, recv_backward_start:{recv_backward_start_time}")
            if Current_NeedRecvBackwardStartTimes>0:
                for i in range(Current_NeedRecvBackwardStartTimes):
                    async_communicator_recv_backward = comm.AsynCommunicator(
                        recv_next_shape=backward_recv_shapes,
                        dtype=self.dtype,
                        scatter_gather_tensors=self.scatter_gather_tensors,
                        stage_id=stage_id,
                        microbatch_id=recv_backward_start_time,
                        step_type=step_type,
                    )
                    commOperation_info = async_communicator_recv_backward.start()
                    recv_backward_start_time += 1
                    write_json(jsonpath, commOperation_info)
                    async_communicator_recv_backward_queue.put(async_communicator_recv_backward)
                    ##debugprint(f"step:{step_type}, microbatch_id:{microbatch_id}, stage_id:{stage_id} recv_backward_queue.put, recv_backward_start:{recv_backward_start_time}")
        return recv_forward_start_time, recv_backward_start_time, async_communicator_recv_forward_queue, async_communicator_recv_backward_queue
    
    def recv_prevrank_allsendforward(self, step_type, microbatch_id, stage_id, jsonpath, stub,
                 prev_rank, recv_forward_start_time, forward_recv_shapes, async_communicator_recv_forward_queue):
        prevRankInfo = get_infor_by_local_rank(stub, prev_rank)
        sendforwardtimes = prevRankInfo.sendforwardtimes
        time_get_prevrank_sendforward = prevRankInfo.last_getrank_timestamp.ToDatetime()
        Current_NeedRecvForwardStartTimes = sendforwardtimes-recv_forward_start_time
        #print(f"step:{step_type}, microbatch_id:{microbatch_id}, stage_id:{stage_id} ,Current_NeedRecvForwardStartTimes:{Current_NeedRecvForwardStartTimes}, sendforwardtimes:{sendforwardtimes}, recv_forward_start_time:{recv_forward_start_time}")
        if Current_NeedRecvForwardStartTimes>0:
            for i in range(Current_NeedRecvForwardStartTimes):
                async_communicator_recv_forward = comm.AsynCommunicator(
                    recv_prev_shape = forward_recv_shapes,
                    dtype=self.dtype,
                    scatter_gather_tensors=self.scatter_gather_tensors,
                    stage_id=stage_id,
                    microbatch_id=recv_forward_start_time,#mirobatch的id是从0开始，recv_forward_start_time从1开始
                    step_type=step_type,
                )
                commOperation_info = async_communicator_recv_forward.start()
                recv_forward_start_time += 1
                write_json(jsonpath, commOperation_info)
                async_communicator_recv_forward_queue.put(async_communicator_recv_forward)
                ##debugprint(f"step:{step_type}, microbatch_id:{microbatch_id}, stage_id:{stage_id} recv_forward_queue.put, recv_forward_start:{recv_forward_start_time}")
        return recv_forward_start_time, async_communicator_recv_forward_queue, time_get_prevrank_sendforward
    
    def recv_nextrank_allsendbackward(self, step_type, microbatch_id, stage_id, jsonpath, stub,
                next_rank, recv_backward_start_time, backward_recv_shapes, async_communicator_recv_backward_queue):
        nextRankInfo = get_infor_by_local_rank(stub, next_rank)
        sendbackwardtimes = nextRankInfo.sendforwardtimes
        time_get_nextrank_sendbackward = nextRankInfo.last_getrank_timestamp.ToDatetime()
        Current_NeedRecvBackwardStartTimes = sendbackwardtimes-recv_backward_start_time
        #print(f"step:{step_type}, microbatch_id:{microbatch_id}, stage_id:{stage_id} ,Current_NeedRecvBackwardStartTimes:{Current_NeedRecvBackwardStartTimes}, sendbackwardtimes:{sendbackwardtimes}, recv_backward_start:{recv_backward_start_time}")
        if Current_NeedRecvBackwardStartTimes>0:
            for i in range(Current_NeedRecvBackwardStartTimes):
                async_communicator_recv_backward = comm.AsynCommunicator(
                    recv_next_shape=backward_recv_shapes,
                    dtype=self.dtype,
                    scatter_gather_tensors=self.scatter_gather_tensors,
                    stage_id=stage_id,
                    microbatch_id=recv_backward_start_time,
                    step_type=step_type,
                )
                commOperation_info = async_communicator_recv_backward.start()
                recv_backward_start_time += 1
                write_json(jsonpath, commOperation_info)
                async_communicator_recv_backward_queue.put(async_communicator_recv_backward)
                ##debugprint(f"step:{step_type}, microbatch_id:{microbatch_id}, stage_id:{stage_id} recv_backward_queue.put, recv_backward_start:{recv_backward_start_time}")
        return recv_backward_start_time, async_communicator_recv_backward_queue, time_get_nextrank_sendbackward
    
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
        #dist._DEFAULT_FIRST_BUCKET_BYTES =512 * 1024 * 1024  #FIX,25MB
        async_communicator_recv_backward_queue = queue.Queue()
        async_communicator_recv_forward_queue = queue.Queue()
        input_objs = queue.Queue()
        output_objs = queue.Queue()
        moe_losses = queue.Queue()
        return_tensors = queue.Queue()
        recv_forward_queue = queue.Queue()
        recv_backward_queue = queue.Queue()
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
        local_rank = gpc.get_local_rank(ParallelMode.PIPELINE)
        golbal_rank = gpc.get_global_rank()
        prev_rank = local_rank-1
        next_rank = local_rank+1
        recv_forward_start_time = 0
        send_forward_start_time = 0
        recv_backward_start_time = 0
        send_backward_start_time = 0
        last_stage = len(self.unified_scheduler)-1
        steps = self.unified_scheduler[local_rank]

        jsonpath = f"./jsonResult/iter_{batch_count}_opeartion_list.json"
        # Configure logging to only log to a file
        # logging.basicConfig(
        #     level=logging.INFO,  # Set the logging level
        #     format='%(message)s',  # Set the log message format
        #     handlers=[
        #         logging.FileHandler(f'./jsonResult/test_8p16m.log', mode='w')  # Log output to 'test.log' in write mode
        #     ]
        # )
        # # Get the logger instance
        # logger = logging.getLogger()
        # #Use logger.info as ##debugprint
        # debugprint = logger.info

        # 禁用代理
        os.environ["http_proxy"] = ""
        os.environ["https_proxy"] = ""
        os.environ["no_proxy"] = "10.140.62.208,127.0.0.1"

        with grpc.insecure_channel(
            '10.140.62.208:50051',
            options=[
                ('grpc.enable_http_proxy', 0),  # 禁用 HTTP 代理
                ('grpc.enable_https_proxy', 0),  # 禁用 HTTPS 代理
            ]
        ) as channel:
            stub = rank_pb2_grpc.RankServiceStub(channel)
            #this_rank_id = create_rank(stub, local_rank)
            this_rank_id = get_infor_by_local_rank(stub, local_rank).id
            #FIX,必须重新初始化
            update_rank(stub=stub, rank_id=this_rank_id, local_rank=local_rank, sendforwardtimes=0, sendbackwardtimes=0, index=0)
            torch.distributed.barrier()
            for s in range(len(steps)):
                step_type, microbatch_id, stage_id = steps[s]
                # update_rank(stub=stub, rank_id=this_rank_id, local_rank=local_rank, sendforwardtimes=send_forward_start_time,sendbackwardtimes=recv_backward_start_time,index=s)
                # recv_forward_start_time,recv_backward_start_time, \
                #         async_communicator_recv_forward_queue, async_communicator_recv_backward_queue =  self.PreRecvAll(
                #             last_stage, step_type, microbatch_id, stage_id, jsonpath, stub,
                #             prev_rank, recv_forward_start_time, forward_recv_shapes, async_communicator_recv_forward_queue,
                #             next_rank, recv_backward_start_time, backward_recv_shapes, async_communicator_recv_backward_queue)

                recv_forward_start_time,recv_backward_start_time, \
                    async_communicator_recv_forward_queue, async_communicator_recv_backward_queue = self.recvAll(
                            last_stage=last_stage,
                            step_type=step_type, 
                            microbatch_id=microbatch_id, #这个参数是指本操作的microbatch_id
                            stage_id=stage_id,
                            jsonpath=jsonpath,
                            stub=stub,
                            prev_rank=prev_rank,
                            recv_forward_start_time=recv_forward_start_time, 
                            forward_recv_shapes=forward_recv_shapes, 
                            async_communicator_recv_forward_queue=async_communicator_recv_forward_queue,
                            next_rank=next_rank, 
                            recv_backward_start_time=recv_backward_start_time, 
                            backward_recv_shapes=backward_recv_shapes, 
                            async_communicator_recv_backward_queue=async_communicator_recv_backward_queue
                    )
                if step_type == Stage.FORWARD.value:# Forward pass
                    # Receive the input from the previous stage
                    ##debugprint(f"step:{step_type}, microbatch_id:{microbatch_id}, stage_id:{stage_id}, _forward_step_begin") 
                    if stage_id>0:
                        if async_communicator_recv_forward_queue.qsize()>0:
                            commOperation_info,(input_obj,_) = async_communicator_recv_forward_queue.get().wait_and_receive()
                            write_json(jsonpath, commOperation_info)
                            ##debugprint(f"step:{step_type}, microbatch_id:{microbatch_id}, stage_id:{stage_id}, forward_queue.get().wait_and_receive()")
                        else:
                            if forward_recv_shapes is None:
                                forward_recv_shapes = comm.recv_obj_meta()
                            async_communicator_recv_forward = comm.AsynCommunicator(
                                recv_prev_shape = forward_recv_shapes,
                                dtype=self.dtype,
                                scatter_gather_tensors=self.scatter_gather_tensors,
                                stage_id=stage_id,
                                microbatch_id=microbatch_id,#TODO这里可以使用recv_forward_start_time
                                step_type=step_type,
                            )
                            commOperation_info = async_communicator_recv_forward.start()
                            recv_forward_start_time += 1
                            write_json(jsonpath, commOperation_info)
                            #debugprint(f"step:{step_type}, microbatch_id:{microbatch_id}, stage_id:{stage_id}, recv_forward.start():{recv_forward_start_time}")
                            commOperation_info,(input_obj,_)  = async_communicator_recv_forward.wait_and_receive()
                            write_json(jsonpath, commOperation_info)
                        ##debugprint(f"step:{step_type}, microbatch_id:{microbatch_id}, stage_id:{stage_id}, input_obj_shape:{input_obj.shape}")
                    else:
                        input_obj = None
                    recv_forward_start_time,recv_backward_start_time, \
                        async_communicator_recv_forward_queue, async_communicator_recv_backward_queue = self.recvAll(
                            last_stage=last_stage,
                            step_type=step_type, 
                            microbatch_id=microbatch_id, #这个参数是指本操作的microbatch_id
                            stage_id=stage_id,
                            jsonpath=jsonpath,
                            stub=stub,
                            prev_rank=prev_rank,
                            recv_forward_start_time=recv_forward_start_time, 
                            forward_recv_shapes=forward_recv_shapes, 
                            async_communicator_recv_forward_queue=async_communicator_recv_forward_queue,
                            next_rank=next_rank, 
                            recv_backward_start_time=recv_backward_start_time, 
                            backward_recv_shapes=backward_recv_shapes, 
                            async_communicator_recv_backward_queue=async_communicator_recv_backward_queue
                        )
                    if stage_id < last_stage:
                        send_forward_start_time += 1
                        update_rankSendFT(
                            stub=stub, 
                            rank_id=this_rank_id, 
                            sendforwardtimes=send_forward_start_time,
                            )
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
                    json_content = {"stage_id":stage_id, "microbatch_id":microbatch_id, "step_type":step_type, "operation":"compute", "start_time":start_time,  "timespan":(end_time - start_time)}
                    write_json(jsonpath, json_content)
                    ##debugprint(f"step:{step_type}, microbatch_id:{microbatch_id}, stage_id:{stage_id} forward_step, output_obj_shape:{output_obj.shape}")

                    recv_forward_start_time,recv_backward_start_time, \
                        async_communicator_recv_forward_queue, async_communicator_recv_backward_queue = self.recvAll(
                            last_stage=last_stage,
                            step_type=step_type, 
                            microbatch_id=microbatch_id, #这个参数是指本操作的microbatch_id
                            stage_id=stage_id,
                            jsonpath=jsonpath,
                            stub=stub,
                            prev_rank=prev_rank,
                            recv_forward_start_time=recv_forward_start_time, 
                            forward_recv_shapes=forward_recv_shapes, 
                            async_communicator_recv_forward_queue=async_communicator_recv_forward_queue,
                            next_rank=next_rank, 
                            recv_backward_start_time=recv_backward_start_time, 
                            backward_recv_shapes=backward_recv_shapes, 
                            async_communicator_recv_backward_queue=async_communicator_recv_backward_queue
                        )

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
                    if stage_id < last_stage:
                        assert output_obj.dtype == self.dtype
                        async_communicator_send_forward = comm.AsynCommunicator(
                            object_send_next=output_obj,
                            dtype=self.dtype,
                            scatter_gather_tensors=self.scatter_gather_tensors,
                            stage_id=stage_id,
                            microbatch_id=microbatch_id,
                            step_type=step_type,
                        )
                        #print(f"this_rank: {local_rank},sendforwardtimes:{send_forward_start_time}")
                        commOperation_info = async_communicator_send_forward.start()
                        write_json(jsonpath, commOperation_info)
                        ##debugprint(f"step:{step_type}, microbatch_id:{microbatch_id}, stage_id:{stage_id}, send_forward.start():{send_forward_start_time}")
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
                            commOperation_info, (_, output_obj_grad) = async_communicator_recv_backward_queue.get().wait_and_receive()
                            write_json(jsonpath, commOperation_info)
                            ##debugprint(f"step:{step_type}, microbatch_id:{microbatch_id}, stage_id:{stage_id}, backward_queue.get().wait_and_receive()")
                        else:
                            async_communicator_recv_backward = comm.AsynCommunicator(
                                recv_next_shape = backward_recv_shapes,
                                dtype=self.dtype,
                                scatter_gather_tensors=self.scatter_gather_tensors,
                                stage_id=stage_id,
                                microbatch_id=microbatch_id,
                                step_type=step_type,
                            )
                            commOperation_info = async_communicator_recv_backward.start()
                            recv_backward_start_time += 1
                            write_json(jsonpath, commOperation_info)
                            ##debugprint(f"step:{step_type}, microbatch_id:{microbatch_id}, stage_id:{stage_id}, recv_backward.start():{recv_backward_start_time}")
                            commOperation_info, (_, output_obj_grad) = async_communicator_recv_backward.wait_and_receive()
                            write_json(jsonpath, commOperation_info)
                        ##debugprint(f"step:{step_type}, microbatch_id:{microbatch_id}, stage_id:{stage_id}, output_obj_grad:{output_obj_grad}")
                    else:
                        output_obj_grad = None
             
                    #debugprint(f"stage_id: {stage_id}, microbatch_id: {microbatch_id}, step_type: {step_type}, input_obj: {input_obj}, output_obj: {output_obj}, output_obj_grad: {output_obj_grad}, moe_loss: {moe_loss}")
                    recv_forward_start_time,recv_backward_start_time, \
                        async_communicator_recv_forward_queue, async_communicator_recv_backward_queue = self.recvAll(
                            last_stage=last_stage,
                            step_type=step_type, 
                            microbatch_id=microbatch_id, #这个参数是指本操作的microbatch_id
                            stage_id=stage_id,
                            jsonpath=jsonpath,
                            stub=stub,
                            prev_rank=prev_rank,
                            recv_forward_start_time=recv_forward_start_time, 
                            forward_recv_shapes=forward_recv_shapes, 
                            async_communicator_recv_forward_queue=async_communicator_recv_forward_queue,
                            next_rank=next_rank, 
                            recv_backward_start_time=recv_backward_start_time, 
                            backward_recv_shapes=backward_recv_shapes, 
                            async_communicator_recv_backward_queue=async_communicator_recv_backward_queue
                        )
                    if stage_id>0:
                        send_backward_start_time += 1
                        update_rankSendBT(
                            stub=stub, 
                            rank_id=this_rank_id, 
                            sendbackwardtimes=send_backward_start_time,
                            )
                    start_time = time.perf_counter()
                    input_obj_grad = self._backward_step(
                        engine, microbatch_id, input_obj, output_obj, output_obj_grad, moe_loss
                    )
                    end_time =time.perf_counter()
                    json_content = {"stage_id":stage_id, "microbatch_id":microbatch_id, "step_type":step_type, "operation":"compute", "start_time":start_time,  "timespan":(end_time - start_time)}
                    write_json(jsonpath, json_content)
                    ##debugprint(f"step:{step_type}, microbatch_id:{microbatch_id}, stage_id:{stage_id}, _backward_step")
                    recv_forward_start_time,recv_backward_start_time, \
                        async_communicator_recv_forward_queue, async_communicator_recv_backward_queue = self.recvAll(
                            last_stage=last_stage,
                            step_type=step_type, 
                            microbatch_id=microbatch_id, #这个参数是指本操作的microbatch_id
                            stage_id=stage_id,
                            jsonpath=jsonpath,
                            stub=stub,
                            prev_rank=prev_rank,
                            recv_forward_start_time=recv_forward_start_time, 
                            forward_recv_shapes=forward_recv_shapes, 
                            async_communicator_recv_forward_queue=async_communicator_recv_forward_queue,
                            next_rank=next_rank, 
                            recv_backward_start_time=recv_backward_start_time, 
                            backward_recv_shapes=backward_recv_shapes, 
                            async_communicator_recv_backward_queue=async_communicator_recv_backward_queue
                        )

                    if stage_id>0:
                        async_communicator_send_backward = comm.AsynCommunicator(
                                object_send_prev=input_obj_grad,
                                dtype=self.dtype,
                                scatter_gather_tensors=self.scatter_gather_tensors,
                                stage_id=stage_id,
                                microbatch_id=microbatch_id,
                                step_type=step_type,
                            )
                        #print(f"this_rank: {local_rank}, sendbackwardtimes:{send_backward_start_time}")
                        commOperation_info = async_communicator_send_backward.start()
                        write_json(jsonpath, commOperation_info)
                        ##debugprint(f"step:{step_type}, microbatch_id:{microbatch_id}, stage_id:{stage_id}, send_backward_start:{send_backward_start_time}")
                    WeightGradStore.flush()
                
                elif step_type == Stage.WEIGHT.value: # Weight update
                    recv_forward_start_time,recv_backward_start_time, \
                        async_communicator_recv_forward_queue, async_communicator_recv_backward_queue = self.recvAll(
                            last_stage=last_stage,
                            step_type=step_type, 
                            microbatch_id=microbatch_id, #这个参数是指本操作的microbatch_id
                            stage_id=stage_id,
                            jsonpath=jsonpath,
                            stub=stub,
                            prev_rank=prev_rank,
                            recv_forward_start_time=recv_forward_start_time, 
                            forward_recv_shapes=forward_recv_shapes, 
                            async_communicator_recv_forward_queue=async_communicator_recv_forward_queue,
                            next_rank=next_rank, 
                            recv_backward_start_time=recv_backward_start_time, 
                            backward_recv_shapes=backward_recv_shapes, 
                            async_communicator_recv_backward_queue=async_communicator_recv_backward_queue
                        )
                    start_time = time.perf_counter()
                    WeightGradStore.pop()
                    end_time = time.perf_counter()
                    json_content = {"stage_id":stage_id, "microbatch_id":microbatch_id, "step_type":step_type, "operation":"compute", "start_time":start_time,  "timespan":(end_time - start_time)}
                    write_json(jsonpath, json_content)
                recv_forward_start_time,recv_backward_start_time, \
                        async_communicator_recv_forward_queue, async_communicator_recv_backward_queue = self.recvAll(
                            last_stage=last_stage,
                            step_type=step_type, 
                            microbatch_id=microbatch_id, #这个参数是指本操作的microbatch_id
                            stage_id=stage_id,
                            jsonpath=jsonpath,
                            stub=stub,
                            prev_rank=prev_rank,
                            recv_forward_start_time=recv_forward_start_time, 
                            forward_recv_shapes=forward_recv_shapes, 
                            async_communicator_recv_forward_queue=async_communicator_recv_forward_queue,
                            next_rank=next_rank, 
                            recv_backward_start_time=recv_backward_start_time, 
                            backward_recv_shapes=backward_recv_shapes, 
                            async_communicator_recv_backward_queue=async_communicator_recv_backward_queue
                        )    
            update_rank(stub=stub, rank_id=this_rank_id, local_rank=local_rank, sendforwardtimes=0, sendbackwardtimes=0, index=0)
            torch.distributed.barrier()

        output, label = pack_return_tensors(return_tensors) if return_tensors.qsize() > 0 else (None, None)

        if hasattr(gpc.config.model, "num_experts") and gpc.config.model.num_experts > 1:
            dist.all_reduce(accum_moe_loss, group=gpc.get_group(ParallelMode.PIPELINE))

        if accum_loss is not None:
            accum_loss += accum_moe_loss

        return output, label, accum_loss, accum_moe_loss

class UnifiedMultipleChunksPipelineScheduler(InterleavedPipelineScheduler):
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
        self.unified_scheduler = unified_scheduler
        self.stage_placement = stage_placement
        
    def create_rank(self,stub, local_rank, global_rank, stage_id, chunk_id):
        create_response = stub.CreateRank(rank_pb2.CreateRankRequest(
            local_rank=local_rank,
            global_rank=global_rank,
            stage_id=stage_id,
            chunk_id=chunk_id,
            sendforwardtimes=0,
            sendbackwardtimes=0
        ))
        return create_response.rank.id

    def get_rank_by_stageID(self,stub, stage_id):
        get_by_stage_id_response = stub.GetRanksByStageId(rank_pb2.GetRanksByStageIdRequest(stage_id=stage_id))
        return get_by_stage_id_response.ranks

    def update_rank(self, stub, rank_id, local_rank, global_rank, stage_id, chunk_id, sendforwardtimes, sendbackwardtimes):
        update_response = stub.UpdateRank(rank_pb2.UpdateRankRequest(
                id=rank_id,
                local_rank=local_rank,
                global_rank=global_rank,
                stage_id=stage_id,
                chunk_id=chunk_id,
                sendforwardtimes=sendforwardtimes,
                sendbackwardtimes=sendbackwardtimes
            ))

    def _get_chunk_by_stage(self, stage_id: int, stage_placement: List[List[int]]) -> int:
        for device_stage in stage_placement:
            for chunk_id in range(device_stage):
                if device_stage[chunk_id] == stage_id:
                    return chunk_id

    def build_stage_to_device_map(self, stage_placement):
        stage_to_device = {}
        for device_id in range(len(stage_placement)):
            for stage_id in stage_placement[device_id]:
                stage_to_device[stage_id] = device_id
        return stage_to_device
    
    # def _forward_step(self, engine, chunk_id, input_obj=None):
    #     """Forward step for passed-in model. If it is the first stage, the input tensor
    #     is obtained from data_iterator, otherwise the passed-in input_obj is used.
    #     Returns output tensor. This is a helper function and can be ignored by users.

    #     Args:
    #         engine (colossalai.engine.Engine): Colossalai engine for training and inference.
    #         chunk_id (int): The id of model chunks.
    #     Returns:
    #         Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]: output or the loss value of the current
    #             pipeline stage.
    #     """
    #     #gpc.set_virtual_pipeline_parallel_rank(chunk_id)

    #     if gpc.is_pipeline_first_stage() and len(self._input_objs[chunk_id]) == len(self._output_objs[chunk_id]):
    #         self._input_objs[chunk_id].append(None)

    #     if input_obj is None:
    #         input_obj = self._input_objs[chunk_id][-1]

    #     if not gpc.is_pipeline_first_stage():
    #         assert input_obj is not None, f"{gpc.get_global_rank()} input is None"
    #     micro_batch_data = self.load_micro_batch(chunk_id)
    #     data, label = self._get_data_label_for_current_step(input_obj, micro_batch_data)

    #     self._call_hooks("before_forward", data)
    #     if hasattr(gpc.config.model, "num_experts"):
    #         output_obj, moe_losses = self._call_engine(engine.model[chunk_id], data)
    #     else:
    #         output_obj = self._call_engine(engine.model[chunk_id], data)
    #     # Convert output_obj to fp32 when last model chunk of last stage
    #     if gpc.is_pipeline_last_stage(ignore_virtual=False) and isinstance(engine.model[chunk_id], NaiveAMPModel):
    #         output_obj = engine.model[chunk_id].convert_to_fp32(output_obj)
    #     self._call_hooks("after_forward", output_obj)

    #     if gpc.is_pipeline_last_stage():
    #         self._call_hooks("post_helper_func", output_obj, label)

    #         if self._return_tensors is not None:
    #             self._return_tensors.append((output_obj, label))
    #         if self._accum_loss is not None:
    #             self._call_hooks("before_criterion", output_obj, label)
    #             loss = self._call_engine_criterion(engine, output_obj, label)
    #             self._call_hooks("after_criterion", loss)

    #             loss_reduced = loss / self.num_microbatches
    #             self._accum_loss.add_(loss_reduced.detach())
    #             output_obj = loss_reduced

    #     moe_loss = (
    #         sum(moe_losses) * gpc.config.loss.moe_loss_coeff  # pylint: disable=E0606
    #         if hasattr(gpc.config.model, "num_experts") and gpc.config.model.num_experts > 1
    #         else torch.tensor(0.0, device=get_current_device(), dtype=gpc.config.model.get("dtype"))
    #     )
    #     # the moe_loss is computed among the "tensor" group if sequence parallel is enabled, so we need to do allreduce
    #     if gpc.config.parallel.sequence_parallel or gpc.config.parallel.expert.no_tp:
    #         dist.all_reduce(moe_loss, op=dist.ReduceOp.AVG, group=gpc.get_group(ParallelMode.TENSOR))
    #     moe_loss /= self.num_microbatches

    #     if self._accum_moe_loss is not None:
    #         self._accum_moe_loss.add_(moe_loss.detach())

    #     self._output_objs[chunk_id].append(output_obj)
    #     self._moe_losses[chunk_id].append(moe_loss)

    #     assert output_obj is not None, f"{gpc.get_global_rank()} chunk{chunk_id} output is None"

    #     return output_obj

    # def _backward_step(self, engine, chunk_id, step_id):
    #     """
    #     Backward step for passed-in model. If it is the last stage, the input tensor
    #     is obtained from the previous forward step, otherwise the passed-in input_obj is used.
    #     Returns input tensor gradient. This is a helper function and can be ignored by users.

    #     Args:
    #         engine (colossalai.engine.Engine): Colossalai engine for training and inference.
    #         chunk_id (int): The id of model chunks.
    #         step_id (int): The current step id.

    #     Returns:
    #         Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]: input tensor gradient.
    #     """
    #     gpc.set_virtual_pipeline_parallel_rank(chunk_id)

    #     if gpc.is_pipeline_last_stage() and len(self._output_obj_grads[chunk_id]) == 0:
    #         self._output_obj_grads[chunk_id].append(None)

    #     input_obj = self._input_objs[chunk_id].pop(0)
    #     output_obj = self._output_objs[chunk_id].pop(0)
    #     output_obj_grad = self._output_obj_grads[chunk_id].pop(0)
    #     moe_loss = self._moe_losses[chunk_id].pop(0)

    #     input_obj_grad = PipelineScheduler._backward_step(engine, step_id, input_obj, output_obj, output_obj_grad, moe_loss)

    #     return input_obj_grad

    def recvAll(self, last_stage, step_type, microbatch_id, stage_id, local_rank, chunk_id, jsonpath, stub,
                prev_stage, prev_global_rank, recv_forward_start_time, forward_recv_shapes, recv_forward_queue,
                next_stage, next_global_rank, recv_backward_start_time, backward_recv_shapes, recv_backward_queue):
        if prev_stage > -1:
            sendforwardtimes = self.get_rank_by_stageID(stub, prev_stage)[0].sendforwardtimes
            Current_NeedRecvForwardStartTimes = sendforwardtimes-recv_forward_start_time
            #print(f"step:{step_type}, microbatch_id:{microbatch_id}, stage_id:{stage_id} ,Current_NeedRecvForwardStartTimes:{Current_NeedRecvForwardStartTimes}, sendforwardtimes:{sendforwardtimes}, recv_forward_start_time:{recv_forward_start_time}")
            if Current_NeedRecvForwardStartTimes>0:
                for i in range(Current_NeedRecvForwardStartTimes):
                    async_communicator_recv_forward = comm.AsynCommunicator(
                        recv_prev_shape = forward_recv_shapes,
                        prev_rank=prev_global_rank,
                        dtype=self.dtype,
                        scatter_gather_tensors=self.scatter_gather_tensors,
                        stage_id=stage_id,
                        microbatch_id=recv_forward_start_time,#mirobatch的id是从0开始，recv_forward_start_time从1开始
                        step_type=step_type,
                    )
                    commOperation_info = async_communicator_recv_forward.start()
                    recv_forward_start_time += 1
                    write_json(jsonpath, commOperation_info)
                    recv_forward_queue.put(async_communicator_recv_forward)
                    ##debugprint(f"step:{step_type}, microbatch_id:{microbatch_id}, stage_id:{stage_id} recv_forward_queue.put, recv_forward_start:{recv_forward_start_time}")
            
        if next_stage < last_stage+1:
            sendbackwardtimes = self.get_rank_by_stageID(stub, next_stage)[0].sendbackwardtimes
            #sendbackwardtimes = read_from_file(readsendbackwardpath)
            Current_NeedRecvBackwardStartTimes = sendbackwardtimes-recv_backward_start_time
            #print(f"step:{step_type}, microbatch_id:{microbatch_id}, stage_id:{stage_id} ,Current_NeedRecvBackwardStartTimes:{Current_NeedRecvBackwardStartTimes}, sendbackwardtimes:{sendbackwardtimes}, recv_backward_start:{recv_backward_start_time}")
            if Current_NeedRecvBackwardStartTimes>0:
                for i in range(Current_NeedRecvBackwardStartTimes):
                    async_communicator_recv_backward = comm.AsynCommunicator(
                        recv_next_shape=backward_recv_shapes,
                        next_rank=next_global_rank,
                        dtype=self.dtype,
                        scatter_gather_tensors=self.scatter_gather_tensors,
                        stage_id=stage_id,
                        microbatch_id=recv_backward_start_time,
                        step_type=step_type,
                    )
                    commOperation_info = async_communicator_recv_backward.start()
                    recv_backward_start_time += 1
                    write_json(jsonpath, commOperation_info)
                    recv_backward_queue.put(async_communicator_recv_backward)
                    ##debugprint(f"step:{step_type}, microbatch_id:{microbatch_id}, stage_id:{stage_id} recv_backward_queue.put, recv_backward_start:{recv_backward_start_time}")
        return recv_forward_start_time, recv_backward_start_time, recv_forward_queue, recv_backward_queue
    

    
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

        # Used for tensor meta information communication
        forward_recv_shapes = self.tensor_shape
        backward_recv_shapes = None
        need_forward_meta = self.tensor_shape is None

        local_rank = gpc.get_local_rank(ParallelMode.PIPELINE)
        global_rank = gpc.get_global_rank()
        stage_placement=self.stage_placement
        last_stage = 7
        stage_to_device = self.build_stage_to_device_map(stage_placement)
        steps = self.unified_scheduler[local_rank]
        stages_in_this_device = stage_placement[local_rank]

        chunks = len(stages_in_this_device)
        chunk_to_rankId = ["-1" for _ in range(chunks)]
        chunk_to_prev_stage_id = [-1 for _ in range(chunks)]
        chunk_to_next_stage_id = [last_stage+1 for _ in range(chunks)]
        chunk_to_prev_global_rank = [-1 for _ in range(chunks)]
        chunk_to_next_global_rank = [last_stage+1 for _ in range(chunks)]#FIX
        recv_backward_queue_list = [queue.Queue() for _ in range(chunks)]
        recv_forward_queue_list = [queue.Queue() for _ in range(chunks)]
        recv_forward_from_same_rank = queue.Queue()
        recv_backawrd_from_same_rank = queue.Queue()
        recv_forward_start_time_list = [0 for _ in range(chunks)]
        send_forward_start_time_list = [0 for _ in range(chunks)]
        recv_backward_start_time_list = [0 for _ in range(chunks)]
        send_backward_start_time_list = [0 for _ in range(chunks)]
        
        jsonpath = f"./jsonResult/Interleaved/iter_{batch_count}_opeartion_list.json"
        # Configure logging to only log to a file
        # logging.basicConfig(
        #     level=logging.INFO,  # Set the logging level
        #     format='%(message)s',  # Set the log message format
        #     handlers=[
        #         logging.FileHandler(f'./jsonResult/Interleaved/test_Interleaved.log', mode='w')  # Log output to 'test.log' in write mode
        #     ]
        # )
        # # Get the logger instance
        # logger = logging.getLogger()
        # #Use logger.info as #debugprint
        # #debugprint = logger.info
        os.environ["http_proxy"] = ""
        os.environ["https_proxy"] = ""
        os.environ["no_proxy"] = "10.140.62.208,127.0.0.1"

        with grpc.insecure_channel(
        '10.140.62.208:50051',
        options=[
            ('grpc.enable_http_proxy', 0),  # 禁用 HTTP 代理
            ('grpc.enable_https_proxy', 0),  # 禁用 HTTPS 代理
            ]
            ) as channel:
            stub = rank_pb2_grpc.RankServiceStub(channel)
            for i in range(chunks):
                chunk_id = i
                stage_id=stages_in_this_device[chunk_id]
                if len(self.get_rank_by_stageID(stub=stub,stage_id=stage_id)) == 0:
                        print(f"stage: {stage_id} not found")
                else:
                    chunk_to_rankId[chunk_id] = self.get_rank_by_stageID(stub=stub,stage_id=stage_id)[0].id
                # if len(self.get_rank_by_stageID(stub, stage_id)) == 0:
                #     chunk_to_rankId[chunk_id] = self.create_rank(
                #         stub=stub, 
                #         local_rank=local_rank, 
                #         global_rank=global_rank,
                #         stage_id=stage_id,
                #         chunk_id=chunk_id,
                #     )
                # else:
                #     print(self.get_rank_by_stageID(stub, stage_id)[0].global_rank)
                #     print(global_rank)
                #     chunk_to_rankId[chunk_id] = self.get_rank_by_stageID(stub, stage_id)[0].id
                #     update_rank(stub=stub, 
                #                 rank_id=chunk_to_rankId[chunk_id], 
                #                 local_rank=local_rank, 
                #                 global_rank=global_rank, 
                #                 stage_id=stage_id, 
                #                 chunk_id=chunk_id, 
                #                 sendforwardtimes=0, 
                #                 sendbackwardtimes=0)
                
                prev_stage = stage_id - 1
                next_stage = stage_id + 1
                chunk_to_prev_stage_id[chunk_id] = prev_stage
                chunk_to_next_stage_id[chunk_id] = next_stage
                if prev_stage >= 0:
                    if len(self.get_rank_by_stageID(stub=stub,stage_id=prev_stage)) == 0:
                        print(f"stage: {prev_stage} not found")
                    else:
                        chunk_to_prev_global_rank[chunk_id] = self.get_rank_by_stageID(stub=stub,stage_id=prev_stage)[0].global_rank
                if next_stage <= last_stage:
                    if len(self.get_rank_by_stageID(stub=stub,stage_id=next_stage)) == 0:
                        print(f"stage: {next_stage} not found")
                    else:
                        chunk_to_next_global_rank[chunk_id] = self.get_rank_by_stageID(stub=stub,stage_id=next_stage)[0].global_rank

            for s in range(len(steps)):
                step_type, microbatch_id, stage_id, chunk_id, startTime = steps[s]
                step_type = step_type[0]#FIX demo
                prev_stage = chunk_to_prev_stage_id[chunk_id]
                next_stage = chunk_to_next_stage_id[chunk_id]
                prev_global_rank = chunk_to_prev_global_rank[chunk_id]
                next_global_rank = chunk_to_next_global_rank[chunk_id]
                this_rank_id = chunk_to_rankId[chunk_id]

                if step_type == Stage.FORWARD.value:# Forward pass
                    # Receive the input from the previous stage
                    #debugprint(f"step:{step_type}, microbatch_id:{microbatch_id}, stage_id:{stage_id}, _forward_step_begin")
                    if stage_id>0:
                        if global_rank == next_global_rank:
                            input_obj = recv_forward_from_same_rank.get()
                        elif recv_forward_queue_list[chunk_id].qsize()>0:
                            commOperation_info,(input_obj,_) = recv_forward_queue_list[chunk_id].get().wait_and_receive()
                            write_json(jsonpath, commOperation_info)
                            #debugprint(f"step:{step_type}, microbatch_id:{microbatch_id}, stage_id:{stage_id}, forward_queue.get().wait_and_receive()")
                        else:
                            if forward_recv_shapes is None:
                                forward_recv_shapes = comm.recv_obj_meta(prev_global_rank)
                            async_communicator_recv_forward = comm.AsynCommunicator(
                                recv_prev_shape = forward_recv_shapes,
                                prev_rank=prev_global_rank,
                                dtype=self.dtype,
                                scatter_gather_tensors=self.scatter_gather_tensors,
                                stage_id=stage_id,
                                microbatch_id=microbatch_id,#TODO这里可以使用recv_forward_start_time_list[chunk_id]
                                step_type=step_type,
                                local_rank=local_rank,
                                chunk_id=chunk_id,
                            )
                            commOperation_info = async_communicator_recv_forward.start()
                            recv_forward_start_time_list[chunk_id] += 1
                            write_json(jsonpath, commOperation_info)
                            #debugprint(f"step:{step_type}, microbatch_id:{microbatch_id}, stage_id:{stage_id}, recv_forward.start():{recv_forward_start_time_list[chunk_id]}")
                            commOperation_info,(input_obj,_)  = async_communicator_recv_forward.wait_and_receive()
                            write_json(jsonpath, commOperation_info)
                        #debugprint(f"step:{step_type}, microbatch_id:{microbatch_id}, stage_id:{stage_id}, input_obj_shape:{input_obj.shape}")
                    else:
                        input_obj = None
                    self._input_objs[chunk_id].append(input_obj)

                    recv_forward_start_time_list[chunk_id] , recv_backward_start_time_list[chunk_id], \
                        recv_forward_queue_list[chunk_id] ,recv_backward_queue_list[chunk_id]=self.recvAll(
                                last_stage=last_stage,
                                step_type=step_type, 
                                microbatch_id=microbatch_id, #这个参数是指本操作的microbatch_id
                                stage_id=stage_id,
                                local_rank=local_rank,
                                chunk_id=chunk_id,
                                jsonpath=jsonpath,
                                stub=stub,
                                #debugprint=#debugprint,
                                prev_stage=prev_stage,
                                prev_global_rank=prev_global_rank,
                                recv_forward_start_time=recv_forward_start_time_list[chunk_id], 
                                forward_recv_shapes=forward_recv_shapes, 
                                recv_forward_queue=recv_forward_queue_list[chunk_id],
                                next_stage=next_stage,
                                next_global_rank=next_global_rank, 
                                recv_backward_start_time=recv_backward_start_time_list[chunk_id], 
                                backward_recv_shapes=backward_recv_shapes, 
                                recv_backward_queue=recv_backward_queue_list[chunk_id])

                    # Perform forward computation
                    start_time = time.perf_counter()
                    output_obj = self._forward_step(engine, chunk_id)  
                    end_time = time.perf_counter()
                    json_content = {"local_rank":local_rank, "chunk_id":chunk_id, "microbatch_id":microbatch_id, "step_type":step_type, "operation":"compute", "start_time":start_time,  "timespan":(end_time - start_time)}
                    write_json(jsonpath,json_content)
                    #debugprint(f"step:{step_type}, microbatch_id:{microbatch_id}, stage_id:{stage_id} forward_step, output_obj_shape:{output_obj.shape}")
                    if stage_id < last_stage:
                        if global_rank == next_global_rank:
                            recv_forward_from_same_rank.put(output_obj)
                            continue
                        if isinstance(output_obj, torch.Tensor):
                            backward_recv_shapes = output_obj.shape
                        else:
                            backward_recv_shapes = [out_tensor.shape for out_tensor in output_obj]    
                        if need_forward_meta:
                            comm.send_obj_meta(output_obj,next_rank=next_global_rank)
                            need_forward_meta = False  # send only once.

                    # Send the output of forward computation of this pipeline stage to the next pipeline stage as input for
                    # forward computation
                    if stage_id < last_stage:
                        assert output_obj.dtype == self.dtype
                        async_communicator_send_forward = comm.AsynCommunicator(
                            object_send_next=output_obj,
                            next_rank=next_global_rank,
                            dtype=self.dtype,
                            scatter_gather_tensors=self.scatter_gather_tensors,
                            stage_id=stage_id,
                            microbatch_id=microbatch_id,
                            step_type=step_type,
                        )
                        send_forward_start_time_list[chunk_id] += 1
                        update_rank(stub=stub, 
                            rank_id=this_rank_id, 
                            local_rank=local_rank, 
                            global_rank=global_rank, 
                            stage_id=stage_id, 
                            chunk_id=chunk_id, 
                            sendforwardtimes=send_forward_start_time_list[chunk_id], 
                            sendbackwardtimes=send_backward_start_time_list[chunk_id]
                            )
                        commOperation_info = async_communicator_send_forward.start()
                        write_json(jsonpath, commOperation_info)
                        #debugprint(f"step:{step_type}, microbatch_id:{microbatch_id}, stage_id:{stage_id}, send_forward.start():{send_forward_start_time_list[chunk_id]}")


                elif step_type == Stage.BACKWARD.value:# Backward pass
                    #debugprint(f"step:{step_type}, microbatch_id:{microbatch_id}, stage_id:{stage_id}, _backward_step_begin")

                    if stage_id<last_stage:
                        if prev_global_rank == global_rank:
                            output_obj_grad = recv_backawrd_from_same_rank.get()
                        elif recv_backward_queue_list[chunk_id].qsize()>0:
                            commOperation_info, (_, output_obj_grad) = recv_backward_queue_list[chunk_id].get().wait_and_receive()
                            write_json(jsonpath, commOperation_info)
                            #debugprint(f"step:{step_type}, microbatch_id:{microbatch_id}, stage_id:{stage_id}, backward_queue.get().wait_and_receive()")
                        else:
                            async_communicator_recv_backward = comm.AsynCommunicator(
                                recv_next_shape = backward_recv_shapes,
                                next_rank=next_global_rank,
                                dtype=self.dtype,
                                scatter_gather_tensors=self.scatter_gather_tensors,
                                stage_id=stage_id,
                                microbatch_id=microbatch_id,
                                step_type=step_type,
                                local_rank=local_rank,
                                chunk_id=chunk_id,
                            )
                            commOperation_info = async_communicator_recv_backward.start()
                            recv_backward_start_time_list[chunk_id] += 1
                            write_json(jsonpath, commOperation_info)
                            #debugprint(f"step:{step_type}, microbatch_id:{microbatch_id}, stage_id:{stage_id}, recv_backward.start():{recv_backward_start_time_list[chunk_id]}")
                            commOperation_info, (_, output_obj_grad) = async_communicator_recv_backward.wait_and_receive()
                            write_json(jsonpath, commOperation_info)
                        #debugprint(f"step:{step_type}, microbatch_id:{microbatch_id}, stage_id:{stage_id}, output_obj_grad:{output_obj_grad}")
                    else:
                        output_obj_grad = None
                    self._output_obj_grads[chunk_id].append(output_obj_grad)

                    recv_forward_start_time_list[chunk_id] , recv_backward_start_time_list[chunk_id], \
                        recv_forward_queue_list[chunk_id] ,recv_backward_queue_list[chunk_id]=self.recvAll(
                                last_stage=last_stage,
                                step_type=step_type, 
                                microbatch_id=microbatch_id, #这个参数是指本操作的microbatch_id
                                stage_id=stage_id,
                                local_rank=local_rank,
                                chunk_id=chunk_id,
                                jsonpath=jsonpath,
                                stub=stub,
                                #debugprint=#debugprint,
                                prev_stage=prev_stage,
                                prev_global_rank=prev_global_rank,
                                recv_forward_start_time=recv_forward_start_time_list[chunk_id], 
                                forward_recv_shapes=forward_recv_shapes, 
                                recv_forward_queue=recv_forward_queue_list[chunk_id],
                                next_stage=next_stage,
                                next_global_rank=next_global_rank, 
                                recv_backward_start_time=recv_backward_start_time_list[chunk_id], 
                                backward_recv_shapes=backward_recv_shapes, 
                                recv_backward_queue=recv_backward_queue_list[chunk_id])
                    start_time = time.perf_counter()
                    input_obj_grad = self._backward_step(engine, chunk_id, microbatch_id)
                    end_time =time.perf_counter()
                    json_content = {"local_rank":local_rank, "chunk_id":chunk_id, "microbatch_id":microbatch_id, "step_type":step_type, "operation":"compute", "start_time":start_time,  "timespan":(end_time - start_time)}
                    write_json(jsonpath,json_content)
                    #debugprint(f"step:{step_type}, microbatch_id:{microbatch_id}, stage_id:{stage_id}, _backward_step")
                    #do_compute()
                    if stage_id>0:
                        if prev_global_rank == global_rank:
                            recv_backawrd_from_same_rank.put(input_obj_grad)
                            continue
                        async_communicator_send_backward = comm.AsynCommunicator(
                                object_send_prev=input_obj_grad,
                                prev_rank=prev_global_rank,
                                dtype=self.dtype,
                                scatter_gather_tensors=self.scatter_gather_tensors,
                                stage_id=stage_id,
                                microbatch_id=microbatch_id,
                                step_type=step_type,
                                local_rank=local_rank,
                                chunk_id=chunk_id,
                            )
                        send_backward_start_time_list[chunk_id] += 1

                        commOperation_info = async_communicator_send_backward.start()
                        write_json(jsonpath, commOperation_info)
                        #debugprint(f"step:{step_type}, microbatch_id:{microbatch_id}, stage_id:{stage_id}, send_backward_start:{send_backward_start_time_list[chunk_id]}")

class UnifiedMultipleChunksZeroBubblePipelineScheduler(InterleavedPipelineScheduler):
    """
    ZB-V Scheduler.

    Args:
        num_microbatches (int): The number of microbatches.
        num_chunks (int): The number of model chunks.
        dtype (torch.dtype, optional): The data type of the tensors. Default is torch.float.
        data_process_func (Callable, optional):
            The preprocessing function which receives a batch of data, and it will be executed in `load_batch`.
        tensor_shape (torch.Size, optional): Specified shape in pipeline communication.
        scatter_gather_tensors (bool, optional):
            If set to `True`, communication will be reduced over pipeline when using 1D tensor parallelization.
        scheduler_hooks (List[SchedulerHook], optional): List of scheduler hooks. Default is None.
        optimizer (Optimizer): The optimizer to do param update.
    """

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
    ):
        """A helper schedule class for pipeline parallelism running environment.
        It uses ZB-V strategy. Other properties are similar as
        :class:`NonPipelineSchedule`.

        Args:
            num_microbatches (int): The number of microbatches.
            num_chunks (int): The number of model chunks.
            dtype (torch.dtype, optional): The data type of the tensors. Default is torch.float.
            data_process_func (Callable, optional):
                The preprocessing function which receives a batch of data, and it will be executed in `load_batch`.
            tensor_shape (torch.Size, optional): Specified shape in pipeline communication.
            scatter_gather_tensors (bool, optional):
                If set to `True`, communication will be reduced over pipeline when using 1D tensor parallelization.
            scheduler_hooks (List[SchedulerHook], optional): List of scheduler hooks. Default is None.
        """

        assert (
            isinstance(num_chunks, int) and num_chunks == 2
        ), f"expect num_chunks to be an integer and equal to 2 for ZBV, but got {num_chunks}."

        assert num_microbatches >= 2 * gpc.get_world_size(
            ParallelMode.PIPELINE
        ), "For ZBV, num_microbatches must be greater than or equal to twice pp size."

        assert gpc.v_shape

        super().__init__(
            num_microbatches,
            num_chunks=num_chunks,
            dtype=dtype,
            data_process_func=data_process_func,
            tensor_shape=tensor_shape,
            scatter_gather_tensors=scatter_gather_tensors,
            scheduler_hooks=scheduler_hooks,
        )

        del self._run_1f1b_loop

        WeightGradStore.set_pp_mode("ZBV")
        WeightGradStore.set_optim(optimizer)

        self._special_chunk0_forward = True
        self._chunk1_need_recv_prev_chunk1_grad = True
        self._backward_step_num = [0, 0]
        self._num_microbatches = num_microbatches

    def _clear_state(self) -> None:
        super()._clear_state()
        self._special_chunk0_forward = True
        self._chunk1_need_recv_prev_chunk1_grad = True
        self._backward_step_num = [0, 0]

    def _backward_step(self, engine, input_obj, output_obj, output_obj_grad, skip_grad_sync=True, moe_loss=None):
        """
        Backward step through the passed-in output tensor. If it is the last stage, the
        output_obj_grad is None, otherwise it is the gradients with respect to stage's output tensor.
        Returns the gradients with respect to the input tensor (None if first stage).
        This is a helper function and can be ignored by users.

        Args:
            engine (colossalai.engine.Engine): Colossalai engine for training and inference.
            input_obj (Union[torch.Tensor, List[torch.Tensor]]): Input tensor for this stage.
            output_obj (Union[torch.Tensor, List[torch.Tensor]]): Output tensor for this stage.
            output_obj_grad (Union[torch.Tensor, List[torch.Tensor]]): Gradient of output tensor for this stage.
            skip_grad_sync (bool): Whether skip grad sync or not.

        Returns:
            Union[torch.Tensor, List[torch.Tensor]]: Gradient of input tensor.
        """

        # Retain the grad on the input_obj.
        if input_obj is not None:
            assert input_obj.requires_grad
            if isinstance(input_obj, torch.Tensor):
                input_obj.retain_grad()
            else:
                for in_tensor in input_obj:
                    if in_tensor is not None:
                        in_tensor.retain_grad()

        # Only the last microbatch does syncing grad.
        engine.optimizer.skip_grad_reduce = skip_grad_sync
        self._call_hooks("before_backward", output_obj, output_obj_grad)
        # with switch_optimizer_grad_sync_skip_mode(engine.optimizer, skip_grad_sync):
        if moe_loss is None or moe_loss.item() == 0.0:
            if output_obj_grad is None:
                engine.backward(output_obj)
            else:
                engine.backward_by_grad(output_obj, output_obj_grad)
        else:
            if output_obj_grad is None:
                engine.backward(output_obj + moe_loss)
            else:
                # scale the latent loss
                moe_loss = moe_loss * engine.optimizer.loss_scale
                # we perform chain rule here by projecting the grad to the direction of
                # [output_obj_grad, 1], Because moe_loss have no relation with subsequent
                # layer, we set it to None (will be ragarded as 1).
                engine.backward_by_grad([output_obj, moe_loss], [output_obj_grad, None])

        # Collect the grad of the input_obj.
        input_obj_grad = None
        if input_obj is not None:
            assert input_obj.grad is not None
            if isinstance(input_obj, torch.Tensor):
                input_obj_grad = input_obj.grad
            else:
                input_obj_grad = []
                for in_tensor in input_obj:
                    input_obj_grad.append(in_tensor.grad)

        return input_obj_grad

    def _schedule_backward(self, engine, chunk_id):
        """
        Backward step for passed-in model. If it is the last stage, the input tensor
        is obtained from the previous forward step, otherwise the passed-in input_obj is used.
        Returns input tensor gradient. This is a helper function and can be ignored by users.

        Args:
            engine (colossalai.engine.Engine): Colossalai engine for training and inference.
            chunk_id (int): The id of model chunks.
            step_id (int): The current step id.

        Returns:
            Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]: input tensor gradient.
        """
        gpc.set_virtual_pipeline_parallel_rank(chunk_id)

        self._backward_step_num[chunk_id] += 1
        if self._backward_step_num[chunk_id] == self._num_microbatches:
            skip_grad_sync = False
        else:
            skip_grad_sync = True

        if gpc.is_pipeline_last_stage() and len(self._output_obj_grads[chunk_id]) == 0:
            self._output_obj_grads[chunk_id].append(None)

        input_obj = self._input_objs[chunk_id].pop(0)
        output_obj = self._output_objs[chunk_id].pop(0)
        output_obj_grad = self._output_obj_grads[chunk_id].pop(0)
        moe_loss = self._moe_losses[chunk_id].pop(0)

        if not gpc.is_pipeline_last_stage():
            assert output_obj_grad is not None
        if not gpc.is_pipeline_first_stage():
            assert input_obj is not None

        input_obj_grad = self._backward_step(engine, input_obj, output_obj, output_obj_grad, skip_grad_sync, moe_loss)

        WeightGradStore.flush()

        return input_obj_grad

    def _schedule_1f1b_F(self, engine, chunk_id):
        output_obj = self._forward_step(engine, chunk_id)

        object_send_next = None
        object_send_prev = None
        recv_next_shape = None
        recv_prev_shape = None

        if chunk_id == 1:
            if not gpc.is_first_rank(ParallelMode.PIPELINE):
                object_send_prev = output_obj
                if self._chunk1_need_recv_prev_chunk1_grad:
                    recv_prev_shape = self._output_obj_shapes[chunk_id]
        else:
            self._chunk1_need_recv_prev_chunk1_grad = False
            if gpc.is_last_rank(ParallelMode.PIPELINE):
                # For last rank, chunk0 output does not need to be sent but is directly used for chunk1;
                input_obj = output_obj.clone().detach()
                input_obj.requires_grad_()
                self._input_objs[1].append(input_obj)
            else:
                object_send_next = output_obj
                recv_next_shape = self._output_obj_shapes[chunk_id]

        # chunk1 send output prev, recv output_grad prev
        # chunk0 send output next, recv output_grad next
        tensor_recv_prev, tensor_recv_next = comm.fused_send_recv_tensor(
            object_send_next=object_send_next,
            object_send_prev=object_send_prev,
            recv_next_shape=recv_next_shape,
            recv_prev_shape=recv_prev_shape,
            dtype=self.dtype,
            scatter_gather_tensors=self.scatter_gather_tensors,
        )

        if chunk_id == 1 and not self._chunk1_need_recv_prev_chunk1_grad:
            assert tensor_recv_prev is None

        if tensor_recv_prev is not None:
            self._output_obj_grads[1].append(tensor_recv_prev)

        if tensor_recv_next is not None:
            self._output_obj_grads[0].append(tensor_recv_next)

    def _schedule_1f1b_B_W(self, engine, chunk_id, next_unit_chunk_id, need_recv_chunk0_output=True):

        # 1B
        input_obj_grad = self._schedule_backward(engine, chunk_id)

        object_send_next = None
        object_send_prev = None
        recv_next_shape = None
        recv_prev_shape = []
        chunk0_B_need_recv_prev_chunk0_output = need_recv_chunk0_output

        if chunk_id == 1:
            if gpc.is_last_rank(ParallelMode.PIPELINE):
                # For last rank, chunk1 input_grad does not need to be sent but is directly used for chunk0.
                self._output_obj_grads[0].append(input_obj_grad)
            else:
                object_send_next = input_obj_grad

            if next_unit_chunk_id == 1:
                if gpc.is_last_rank(ParallelMode.PIPELINE):
                    assert False, "The last pp rank can never have two consecutive unit1 of the same chunk."
                recv_next_shape = self._input_obj_shapes[next_unit_chunk_id]
        else:
            assert next_unit_chunk_id != 0, "There will never be two consecutive chunk0 unit1."

            if not gpc.is_first_rank(ParallelMode.PIPELINE):
                object_send_prev = input_obj_grad
                # pre receive chunk1 grad
                recv_prev_shape.append(self._output_obj_shapes[1])
                # pre receive chunk0 input
                if chunk0_B_need_recv_prev_chunk0_output:
                    recv_prev_shape.append(self._input_obj_shapes[0])

            if not gpc.is_last_rank(ParallelMode.PIPELINE):
                recv_next_shape = self._input_obj_shapes[next_unit_chunk_id]

        if len(recv_prev_shape) == 0:
            recv_prev_shape = None

        # chunk1 send input_grad next, chunk0 send input_grad prev
        # if chunk_id == 1 and next_unit_chunk_id == 1, recv chunk1 input next
        # if chunk_id == 0 and next_unit_chunk_id == 1, pre-recv chunk1 grad recv;
        # pre-recv chunk0 input prev and recv chunk1 input next
        async_communicator = comm.AsynCommunicator(
            object_send_prev=object_send_prev,
            object_send_next=object_send_next,
            recv_prev_shape=recv_prev_shape,
            recv_next_shape=recv_next_shape,
            dtype=self.dtype,
            scatter_gather_tensors=self.scatter_gather_tensors,
        )
        async_communicator.start()

        # 1W
        WeightGradStore.pop()
        self._call_hooks("after_backward", input_obj_grad)

        tensor_recv_prev, tensor_recv_next = async_communicator.wait_and_receive()

        # for the special case, input_obj has already been received and appended at the end of warmup.
        if next_unit_chunk_id == 0 and self._special_chunk0_forward:
            self._special_chunk0_forward = False
        else:
            if chunk_id == 0:
                # For chunk0, it's necessary to pre-fetch the output_grad of the next chunk1
                # to prevent the sender from being blocked due to the absence of a receiving op.
                # Except for the stage1 last chunk0 or stage2, the chunk0 BW also needs to pre-fetch
                # the input of the next chunk0 unit to prevent the sender from being blocked.

                if gpc.is_first_rank(ParallelMode.PIPELINE):
                    # first_rank only receive chunk1 input from next rank
                    self._input_objs[1].append(tensor_recv_next)
                elif gpc.is_last_rank(ParallelMode.PIPELINE):
                    # For last rank, chunk1 input does not need to be received
                    self._output_obj_grads[1].append(tensor_recv_prev[0])
                    if chunk0_B_need_recv_prev_chunk0_output:
                        self._input_objs[0].append(tensor_recv_prev[1])
                else:
                    self._output_obj_grads[1].append(tensor_recv_prev[0])
                    if chunk0_B_need_recv_prev_chunk0_output:
                        self._input_objs[0].append(tensor_recv_prev[1])
                    self._input_objs[1].append(tensor_recv_next)
            else:
                if next_unit_chunk_id == 1:
                    self._input_objs[1].append(tensor_recv_next)

    def _1f1b_unit_1(self, engine, chunk_id, next_unit_chunk_id, need_recv_chunk0_output):
        """
        unit1 consists of: 1F + 1B + 1W, all are chunk0 or chunk1
        """
        # 1F
        self._schedule_1f1b_F(engine, chunk_id)

        # 1B + 1W
        self._schedule_1f1b_B_W(engine, chunk_id, next_unit_chunk_id, need_recv_chunk0_output)

    def _1f1b_unit_2(self, engine, chunk_id):
        """
        unit2 consists of: chunk1 (1F + 1B + 1W) + chunk0 (1B + 1W)
        """
        assert not gpc.is_last_rank(ParallelMode.PIPELINE)

        # 1F (chunk1)
        self._schedule_1f1b_F(engine, chunk_id)

        # 1B + 1W (chunk1)
        input_obj_grad = self._schedule_backward(engine, chunk_id)

        # chunk1 send input_grad next, chunk0 recv output_grad next
        async_communicator = comm.AsynCommunicator(
            object_send_next=input_obj_grad,
            recv_next_shape=self._output_obj_shapes[1 - chunk_id],
            dtype=self.dtype,
            scatter_gather_tensors=self.scatter_gather_tensors,
        )
        async_communicator.start()

        WeightGradStore.pop()
        self._call_hooks("after_backward", input_obj_grad)

        _, output_obj_grad = async_communicator.wait_and_receive()
        self._output_obj_grads[1 - chunk_id].append(output_obj_grad)

        # 1B + 1W (chunk0)
        self._schedule_1f1b_B_W(engine, 1 - chunk_id, chunk_id, need_recv_chunk0_output=False)

    def _schedule_warmup_F(self, engine, chunk_id, input_obj=None, forward_only=False):
        output_obj = self._forward_step(engine, chunk_id, input_obj)

        if forward_only:
            # when forward-only, no need to save tensors for a backward pass
            self._input_objs[chunk_id].pop()
            self._output_objs[chunk_id].pop()
            self._moe_losses[chunk_id].pop()

        if not gpc.is_pipeline_last_stage():
            if isinstance(output_obj, torch.Tensor):
                self._output_obj_shapes[chunk_id] = output_obj.shape
            else:
                self._output_obj_shapes[chunk_id] = [out_tensor.shape for out_tensor in output_obj]

            assert self._output_obj_shapes[chunk_id] == self._input_obj_shapes[chunk_id]

            if self._send_tensor_shape_flags[chunk_id]:
                comm.send_obj_meta(output_obj)
                self._send_tensor_shape_flags[chunk_id] = False  # send only once for each chunk.

        if not gpc.is_pipeline_first_stage() and self._input_obj_shapes[chunk_id] is None:
            self._input_obj_shapes[chunk_id] = comm.recv_obj_meta()

        return output_obj

    def _run_warmup_loop(
        self,
        engine: Engine,
        num_warmup_microsteps: int,
        forward_only: bool = False,
    ) -> None:
        """
        Run the warm-up loop and prepare data for the steady stage.

        Args:
            engine (Engine): The engine to run the warm-up loop.
            num_warmup_microsteps (int): The number of warm-up microsteps.
            forward_only (bool, optional): Whether to only perform forward pass. Default is False.
        """

        # For each rank, the warmup stage will be divided into two sub-phases for scheduling.
        num_warmup_microsteps_phase_1 = min(self.num_microbatches, (self._pp_size - self._pp_rank) * 2 - 1)
        num_warmup_microsteps_phase_2 = num_warmup_microsteps - num_warmup_microsteps_phase_1

        if gpc.is_first_rank(ParallelMode.PIPELINE):
            assert num_warmup_microsteps_phase_2 == 0
        if gpc.is_last_rank(ParallelMode.PIPELINE):
            assert num_warmup_microsteps_phase_1 == 1

        # get first forward input
        chunk_id = 0
        if not gpc.is_pipeline_first_stage():
            if self._input_obj_shapes[chunk_id] is None:
                self._input_obj_shapes[chunk_id] = comm.recv_obj_meta()
            self._input_objs[chunk_id].append(
                comm.recv_forward(
                    self._input_obj_shapes[chunk_id],
                    dtype=self.dtype,
                    scatter_gather_tensors=self.scatter_gather_tensors,
                )
            )
        else:
            self._input_objs[chunk_id].append(None)

        # Phase1 will only do chunk0 forward
        for micro_step in range(num_warmup_microsteps_phase_1):
            # forward
            output_obj = self._schedule_warmup_F(engine, chunk_id, forward_only=forward_only)

            object_send_next = None
            recv_prev_shape = None
            recv_next_shape = None

            # For stage1, the last chunk0 unit needs to do recv op to prevent the sender from being blocked.
            if not gpc.is_first_rank(ParallelMode.PIPELINE):
                recv_prev_shape = self._input_obj_shapes[0]

            # For last rank, chunk0 output does not need to be sent but is directly used for chunk1.
            if not gpc.is_last_rank(ParallelMode.PIPELINE):
                object_send_next = output_obj
            else:
                input_obj = output_obj.clone().detach()
                input_obj.requires_grad_()
                self._input_objs[1].append(input_obj)

            if micro_step == num_warmup_microsteps_phase_1 - 1:
                if not gpc.is_last_rank(ParallelMode.PIPELINE):
                    recv_next_shape = self._input_obj_shapes[1]

            tensor_recv_prev, tensor_recv_next = comm.fused_send_recv_tensor(
                object_send_next=object_send_next,
                recv_prev_shape=recv_prev_shape,
                recv_next_shape=recv_next_shape,
                dtype=self.dtype,
                scatter_gather_tensors=self.scatter_gather_tensors,
            )

            self._input_objs[0].append(tensor_recv_prev)

            if micro_step == num_warmup_microsteps_phase_1 - 1:
                if not gpc.is_last_rank(ParallelMode.PIPELINE):
                    self._input_objs[1].append(tensor_recv_next)

        # Phase2 will execute chunk1 and chunk0 forward alternately
        for micro_step in range(num_warmup_microsteps_phase_2):
            chunk_id = 1 - chunk_id
            next_chunk_id = 1 - chunk_id

            if chunk_id == 0:
                input_obj = self._input_objs[chunk_id][-2]
            else:
                input_obj = self._input_objs[chunk_id][-1]

            output_obj = self._schedule_warmup_F(engine, chunk_id, input_obj=input_obj, forward_only=forward_only)

            object_send_next = None
            object_send_prev = None
            recv_next_shape = None
            recv_prev_shape = None

            if chunk_id == 1:
                assert micro_step < num_warmup_microsteps_phase_2 - 1
                object_send_prev = output_obj
                recv_prev_shape = self._input_obj_shapes[next_chunk_id]
            else:
                if not gpc.is_last_rank(ParallelMode.PIPELINE):
                    object_send_next = output_obj
                    recv_next_shape = self._input_obj_shapes[next_chunk_id]

            # chunk1 send output prev, chunk0 recv input prev
            # chunk0 send output next, chunk1 recv input next
            tensor_recv_prev, tensor_recv_next = comm.fused_send_recv_tensor(
                object_send_next=object_send_next,
                object_send_prev=object_send_prev,
                recv_next_shape=recv_next_shape,
                recv_prev_shape=recv_prev_shape,
                dtype=self.dtype,
                scatter_gather_tensors=self.scatter_gather_tensors,
            )

            # For last rank, chunk0 output does not need to be sent but is directly used for chunk1
            if chunk_id == 0 and gpc.is_last_rank(ParallelMode.PIPELINE):
                input_obj = output_obj.clone().detach()
                input_obj.requires_grad_()
            else:
                input_obj = tensor_recv_prev if tensor_recv_prev is not None else tensor_recv_next

            self._input_objs[next_chunk_id].append(input_obj)

    def _run_steady_loop(
        self,
        engine: Engine,
        num_1f1b_units: int,
    ) -> None:
        """
        1F1B unit schedule:
        stage1: (pp_size + 1 + pp_rank + 2 * (micro_num - 2 * pp_size)) * unit1
        stage2: (pp_size - 1 - pp_rank) * unit2
        stage3: 1 * special chunk1 unit1

        Args:
            engine (Engine): The engine to use for computation.
            num_1f1b_units (int): The number of 1F1B units.
        """
        # unit schedule
        num_units_stage1 = 2 * self.num_microbatches - 3 * self._pp_size + 1 + self._pp_rank
        num_units_stage2 = self._pp_size - 1 - self._pp_rank
        assert num_units_stage1 + num_units_stage2 + 1 == num_1f1b_units

        # chunk schedule: stage1 + stage2 + stage1
        # stage1: chunk1
        # stage2: chunk0 and chunk1 alternately
        stage1_length = self._pp_size - self._pp_rank
        stage2_length = 2 * self._pp_rank + 1 + 2 * (self.num_microbatches - 2 * self._pp_size)
        stage2_list = list(range(stage1_length, stage1_length + stage2_length))
        chunk0_units = [stage2_list[i] for i in range(len(stage2_list)) if i % 2 == 0]

        # unit stage1
        for unit_step in range(num_units_stage1):
            if unit_step in chunk0_units:
                chunk_id = 0
            else:
                chunk_id = 1

            if unit_step + 1 in chunk0_units:
                next_unit_chunk_id = 0
            else:
                next_unit_chunk_id = 1

            if unit_step == num_units_stage1 - 1:
                chunk0_B_need_recv_prev_chunk0_output = False
            else:
                chunk0_B_need_recv_prev_chunk0_output = True

            self._1f1b_unit_1(
                engine, chunk_id, next_unit_chunk_id, need_recv_chunk0_output=chunk0_B_need_recv_prev_chunk0_output
            )

        # unit stage2
        for unit_step in range(num_units_stage2):
            assert unit_step + num_units_stage1 not in chunk0_units
            self._1f1b_unit_2(engine, 1)

        # unit stage3
        assert num_1f1b_units - 1 not in chunk0_units
        self._schedule_1f1b_F(engine, 1)
        origin_skip = engine.optimizer.skip_grad_reduce
        input_obj_grad = self._schedule_backward(engine, 1)
        if gpc.is_last_rank(ParallelMode.PIPELINE):
            # For last rank, chunk1 input_grad does not need to be sent but is directly used for chunk0.
            self._output_obj_grads[0].append(input_obj_grad)
            tensor_to_send = None
            recv_shape = None
        else:
            tensor_to_send = input_obj_grad
            recv_shape = self._output_obj_shapes[0]

        # chunk1 send input_grad next, chunk0 recv output_grad next
        async_communicator = comm.AsynCommunicator(
            object_send_next=tensor_to_send,
            recv_next_shape=recv_shape,
            dtype=self.dtype,
            scatter_gather_tensors=self.scatter_gather_tensors,
        )
        async_communicator.start()

        WeightGradStore.pop()
        self._call_hooks("after_backward", input_obj_grad)
        engine.optimizer.skip_grad_reduce = origin_skip

        _, output_obj_grad = async_communicator.wait_and_receive()
        if not gpc.is_last_rank(ParallelMode.PIPELINE):
            self._output_obj_grads[0].append(output_obj_grad)

    def _run_cooldown_loop(self, engine):
        """
        Cooldown unit schedule:
        Unit: 1B + 1W
        Schedule unit chunk0 and unit chunk1 alternatively
        Each pp rank has pp_size chunk0, but only pp_rank chunk1
        """
        chunk0_length = self._pp_size
        chunk1_length = self._pp_rank
        num_cooldown_units = chunk0_length + chunk1_length
        total_list = list(range(chunk1_length * 2))
        chunk1_units = [total_list[i] for i in range(chunk1_length * 2) if i % 2 != 0]

        cool_down = [0, 0]

        for unit_step in range(num_cooldown_units):
            if unit_step in chunk1_units:
                chunk_id = 1
            else:
                chunk_id = 0

            cool_down[chunk_id] += 1

            if unit_step + 1 in chunk1_units:
                next_unit_chunk_id = 1
            else:
                next_unit_chunk_id = 0

            origin_skip = engine.optimizer.skip_grad_reduce
            input_obj_grad = self._schedule_backward(engine, chunk_id)

            object_send_next = None
            object_send_prev = None
            recv_next_shape = None
            recv_prev_shape = None

            if chunk_id == 1:
                assert not gpc.is_first_rank(ParallelMode.PIPELINE)
                if gpc.is_last_rank(ParallelMode.PIPELINE):
                    # For last rank, chunk1 input_grad does not need to be sent but is directly used for chunk0.
                    self._output_obj_grads[0].append(input_obj_grad)
                else:
                    object_send_next = input_obj_grad
                    # next unit should be chunk0
                    recv_next_shape = self._output_obj_shapes[0]
            else:
                if not gpc.is_first_rank(ParallelMode.PIPELINE):
                    object_send_prev = input_obj_grad

                if unit_step != num_cooldown_units - 1:
                    if next_unit_chunk_id == 1:
                        assert not gpc.is_first_rank(ParallelMode.PIPELINE)
                        recv_prev_shape = self._output_obj_shapes[next_unit_chunk_id]
                    else:
                        assert not gpc.is_last_rank(ParallelMode.PIPELINE)
                        recv_next_shape = self._output_obj_shapes[next_unit_chunk_id]

            # chunk1 send input_grad next, chunk0 send input_grad prev
            # if next_unit_chunk_id == 1, recv output_grad prev
            # if next_unit_chunk_id == 0, recv output_grad next
            async_communicator = comm.AsynCommunicator(
                object_send_prev=object_send_prev,
                object_send_next=object_send_next,
                recv_prev_shape=recv_prev_shape,
                recv_next_shape=recv_next_shape,
                dtype=self.dtype,
                scatter_gather_tensors=self.scatter_gather_tensors,
            )
            async_communicator.start()

            # 1W
            WeightGradStore.pop()
            self._call_hooks("after_backward", input_obj_grad)
            engine.optimizer.skip_grad_reduce = origin_skip

            tensor_recv_prev, tensor_recv_next = async_communicator.wait_and_receive()
            output_obj_grad = tensor_recv_prev if tensor_recv_prev is not None else tensor_recv_next

            if output_obj_grad is not None:
                self._output_obj_grads[next_unit_chunk_id].append(output_obj_grad)

    def _forward_only_step(self, engine: Engine):
        num_warmup_steps = self.num_microbatches * self._num_chunks

        self._run_warmup_loop(
            engine,
            num_warmup_steps,
            forward_only=True,
        )

    def _forward_backward_step(self, engine: Engine):
        assert self.num_microbatches > self._pp_size

        # Compute number of warmup microbatches.
        num_warmup_steps = self._pp_size * 2 - 1

        # Compute number of 1F1B unit.
        num_1f1b_units = 2 * self.num_microbatches - num_warmup_steps

        # 1. Warmup
        self._run_warmup_loop(
            engine,
            num_warmup_steps,
        )

        # 2. 1F1B
        self._run_steady_loop(
            engine,
            num_1f1b_units,
        )

        # 3. cooldown
        self._run_cooldown_loop(engine)
