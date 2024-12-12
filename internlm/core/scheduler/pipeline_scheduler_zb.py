#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import queue
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch.optim.optimizer import Optimizer

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
from enum import Enum

logger = get_logger(__file__)

class Stage(Enum):
    FORWARD = 'f'
    BACKWARD = 'b'
    WEIGHT = 'w'

class WeightGradStore:
    """
    When using zero bubble pp, WeightGradStore is used to store the args and func for computating weight grad.
    """

    _cache = []
    _weight_grad_queue = queue.Queue()
    _hooks = {}
    pp_mode = None
    optim = None
    temp = []

    @classmethod
    def set_pp_mode(cls, mode):
        cls.pp_mode = mode

    @classmethod
    def set_optim(cls, optim):
        cls.optim = optim

    @classmethod
    def size(cls):
        return cls._weight_grad_queue.qsize()

    @classmethod
    def put(cls, weight, bias, input_tensor, grad_output, has_d_bias, grad_compute_func, *args):
        if cls.pp_mode == "ZBH1":
            assert not gpc.is_first_rank(ParallelMode.PIPELINE), "pp rank 0 should not arrive here"
        # Store the weight gradient computation of linear layers.
        cls._cache.append((weight, bias, input_tensor, grad_output, has_d_bias, grad_compute_func, *args))

    @classmethod
    def flush(cls):
        if cls.pp_mode == "ZBH1" and gpc.is_first_rank(ParallelMode.PIPELINE):
            return
        # Collect all stored computations during backward as a W for each micro batch.
        cls._weight_grad_queue.put(cls._cache)
        cls._cache = []

    @classmethod
    def pop(cls):
        if cls.pp_mode == "ZBH1" and gpc.is_first_rank(ParallelMode.PIPELINE):
            return
        assert cls._weight_grad_queue.qsize() > 0
        stored_w_grad_computation = cls._weight_grad_queue.get()
        # Run computation for a single W.
        for weight, bias, input_tensor, grad_output, has_d_bias, grad_compute_func, *args in stored_w_grad_computation:
            assert weight.requires_grad
            grad_weight, grad_bias = grad_compute_func(input_tensor, grad_output, has_d_bias)

            if is_using_isp():
                isp_grad_hook = args[0]
                module = args[1]
                grad_weight, handle_weight = isp_grad_hook(grad_weight, async_op=True, is_bias=False, module=module)
                handle_weight.wait()
                if grad_bias is not None:
                    grad_bias, handle_bias = isp_grad_hook(grad_bias, async_op=True, is_bias=True, module=module)
                    handle_bias.wait()

            # Gradient Accumulation
            weight.grad = weight.grad.data + grad_weight if weight.grad is not None else grad_weight
            if has_d_bias:
                bias.grad = bias.grad.data + grad_bias if bias.grad is not None else grad_bias

            # overlap hook
            if weight in cls._hooks:
                for hook in cls._hooks[weight]:
                    hook()
                if has_d_bias:
                    for hook in cls._hooks[bias]:
                        hook()

    @classmethod
    def register_hook(cls, param, hooks):
        cls._hooks[param] = hooks


class ZeroBubblePipelineScheduler(PipelineScheduler):
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
    # print(f'device:{get_current_device()}')
    # print(f'local_rank:{gpc.get_local_rank(ParallelMode.PIPELINE)}')
    # print(f'ranks_in_group:{gpc.get_ranks_in_group(ParallelMode.PIPELINE)}')
    # print(f'global_rank:{gpc.get_global_rank()}')
    # print(f'prev_rank:{gpc.get_prev_global_rank(ParallelMode.PIPELINE)}')
    # print(f'next_rank:{gpc.get_next_global_rank(ParallelMode.PIPELINE)}')
    # print(f'ops:{len(ops)}')
    # print(f'p2p_time_local_rank:{gpc.get_local_rank(ParallelMode.PIPELINE)}')

        self.device_steps = [
        [('f', 0, 0, 0), ('f', 1, 0, 4), ('f', 2, 0, 8), ('f', 3 ,0, 12), ('b', 0, 0, 34), ('w', 0, 0, 40), ('b', 1, 0, 44), ('w', 1, 0, 50), ('b', 2, 0, 54), ('w', 2, 0, 60), ('b', 3, 0, 64), ('w', 3, 0, 70)], 
        [('f', 0, 1, 4), ('f', 1, 1, 8), ('f', 2, 1, 12), ('b', 0, 1, 28), ('f', 3, 1, 34), ('b', 1, 1, 38), ('w', 0, 1, 44), ('b', 2, 1, 48), ('w', 1, 1, 54), ('b', 3, 1, 58), ('w', 2, 1, 66), ('w', 3, 1, 70)], 
        [('f', 0, 2, 8), ('f', 1, 2, 12), ('b', 0, 2, 22), ('f', 2, 2, 28), ('b', 1, 2, 32), ('f', 3, 2, 38), ('b', 2, 2, 42), ('b', 3, 2, 52), ('w', 0, 2, 58), ('w', 1, 2, 62), ('w', 2, 2, 66), ('w', 3, 2, 70)], 
        [('f', 0, 3, 12), ('b', 0, 3, 16), ('f', 1, 3, 22), ('b', 1, 3, 26), ('f', 2, 3, 32), ('b', 2, 3, 36), ('f', 3, 3, 42), ('b', 3, 3, 46), ('w', 0, 3, 52), ('w', 1, 3, 56), ('w', 2, 3, 60), ('w', 3, 3, 64)]]


        # self.device_steps = [[('f', 0, 0, 0), ('f', 1, 0, 6), ('f', 2, 0, 12), ('f', 3, 0, 18), ('f', 4, 0, 24), ('f', 5, 0, 30), ('f', 6, 0, 36), ('f', 7, 0, 42), ('f', 8, 0, 48), ('f', 9, 0, 54), ('f', 10, 0, 60), ('f', 11, 0, 66), ('f', 12, 0, 72), ('f', 13, 0, 78), ('f', 14, 0, 84), ('f', 15, 0, 90), ('b', 0, 0, 138), ('w', 0, 0, 148), ('b', 1, 0, 164), ('w', 1, 0, 174), ('b', 2, 0, 188), ('w', 2, 0, 200), ('b', 3, 0, 204), ('w', 3, 0, 214), ('b', 4, 0, 218), ('w', 4, 0, 228), ('b', 5, 0, 232), ('w', 5, 0, 242), ('b', 6, 0, 246), ('w', 6, 0, 256), ('b', 7, 0, 260), ('w', 7, 0, 270), ('b', 8, 0, 274), ('w', 8, 0, 284), ('b', 9, 0, 288), ('b', 10, 0, 298), ('w', 9, 0, 308), ('w', 10, 0, 312), ('b', 11, 0, 316), ('w', 11, 0, 326), ('b', 12, 0, 330), ('w', 12, 0, 340), ('b', 13, 0, 344), ('w', 13, 0, 354), ('b', 14, 0, 358), ('w', 14, 0, 368), ('b', 15, 0, 372), ('w', 15, 0, 382)], 
        #  [('f', 0, 1, 7), ('f', 1, 1, 13), ('f', 2, 1, 19), ('f', 3, 1, 25), ('f', 4, 1, 31), ('f', 5, 1, 37), ('f', 6, 1, 43), ('f', 7, 1, 49), ('f', 8, 1, 55), ('f', 9, 1, 61), ('f', 10, 1, 67), ('f', 11, 1, 73), ('f', 12, 1, 79), ('f', 13, 1, 85), ('f', 14, 1, 91), ('f', 15, 1, 97), ('b', 0, 1, 127), ('w', 0, 1, 146), ('b', 1, 1, 153), ('w', 1, 1, 163), ('b', 2, 1, 177), ('w', 2, 1, 187), ('b', 3, 1, 193), ('b', 4, 1, 203), ('w', 3, 1, 213), ('w', 4, 1, 217), ('b', 5, 1, 221), ('w', 5, 1, 231), ('b', 6, 1, 235), ('w', 6, 1, 245), ('b', 7, 1, 249), ('w', 7, 1, 259), ('b', 8, 1, 263), ('w', 8, 1, 273), ('b', 9, 1, 277), ('b', 10, 1, 287), ('w', 9, 1, 297), ('w', 10, 1, 301), ('b', 11, 1, 305), ('w', 11, 1, 315), ('b', 12, 1, 319), ('w', 12, 1, 329), ('b', 13, 1, 333), ('b', 14, 1, 347), ('b', 15, 1, 361), ('w', 13, 1, 371), ('w', 14, 1, 375), ('w', 15, 1, 382)], 
        #  [('f', 0, 2, 14), ('f', 1, 2, 20), ('f', 2, 2, 26), ('f', 3, 2, 32), ('f', 4, 2, 38), ('f', 5, 2, 44), ('f', 6, 2, 50), ('f', 7, 2, 56), ('f', 8, 2, 62), ('f', 9, 2, 68), ('f', 10, 2, 74), ('f', 11, 2, 80), ('f', 12, 2, 86), ('f', 13, 2, 92), ('f', 14, 2, 98), ('f', 15, 2, 104), ('b', 0, 2, 116), ('w', 0, 2, 126), ('b', 1, 2, 142), ('w', 1, 2, 152), ('b', 2, 2, 166), ('w', 2, 2, 176), ('b', 3, 2, 182), ('b', 4, 2, 192), ('b', 5, 2, 210), ('b', 6, 2, 224), ('w', 3, 2, 234), ('b', 7, 2, 238), ('w', 4, 2, 248), ('b', 8, 2, 252), ('w', 5, 2, 262), ('b', 9, 2, 266), ('b', 10, 2, 276), ('w', 6, 2, 289), ('b', 11, 2, 293), ('w', 7, 2, 304), ('b', 12, 2, 308), ('w', 8, 2, 318), ('b', 13, 2, 322), ('w', 9, 2, 332), ('b', 14, 2, 336), ('w', 10, 2, 346), ('b', 15, 2, 350), ('w', 11, 2, 360), ('w', 12, 2, 364), ('w', 13, 2, 368), ('w', 14, 2, 372), ('w', 15, 2, 382)], 
        #  [('f', 0, 3, 21), ('f', 1, 3, 27), ('f', 2, 3, 33), ('f', 3, 3, 39), ('f', 4, 3, 45), ('f', 5, 3, 51), ('f', 6, 3, 57), ('f', 7, 3, 63), ('f', 8, 3, 69), ('f', 9, 3, 75), ('f', 10, 3, 81), ('f', 11, 3, 87), ('f', 12, 3, 93), ('f', 13, 3, 99), ('b', 0, 3, 105), ('b', 1, 3, 131), ('w', 0, 3, 141), ('f', 14, 3, 145), ('b', 2, 3, 155), ('f', 15, 3, 165), ('b', 3, 3, 171), ('b', 4, 3, 181), ('w', 1, 3, 195), ('b', 5, 3, 199), ('w', 2, 3, 209), ('b', 6, 3, 213), ('w', 3, 3, 223), ('b', 7, 3, 227), ('b', 8, 3, 241), ('w', 4, 3, 251), ('b', 9, 3, 255), ('b', 10, 3, 265), ('b', 11, 3, 282), ('b', 12, 3, 297), ('w', 5, 3, 307), ('b', 13, 3, 311), ('w', 6, 3, 321), ('b', 14, 3, 325), ('b', 15, 3, 339), ('w', 7, 3, 350), ('w', 8, 3, 354), ('w', 9, 3, 358), ('w', 10, 3, 362), ('w', 11, 3, 366), ('w', 12, 3, 370), ('w', 13, 3, 374), ('w', 14, 3, 378), ('w', 15, 3, 382)], 
        #  [('f', 0, 4, 28), ('f', 1, 4, 34), ('f', 2, 4, 40), ('f', 3, 4, 46), ('f', 4, 4, 52), ('f', 5, 4, 58), ('f', 6, 4, 64), ('f', 7, 4, 70), ('f', 8, 4, 76), ('f', 9, 4, 82), ('f', 10, 4, 88), ('b', 0, 4, 94), ('f', 11, 4, 104), ('f', 12, 4, 110), ('b', 1, 4, 120), ('w', 0, 4, 130), ('w', 1, 4, 134), ('b', 2, 4, 144), ('f', 13, 4, 154), ('b', 3, 4, 160), ('b', 4, 4, 170), ('f', 14, 4, 182), ('b', 5, 4, 188), ('b', 6, 4, 198), ('f', 15, 4, 210), ('b', 7, 4, 216), ('b', 8, 4, 228), ('b', 9, 4, 244), ('b', 10, 4, 254), ('w', 2, 4, 267), ('b', 11, 4, 271), ('w', 3, 4, 282), ('b', 12, 4, 286), ('w', 4, 4, 296), ('b', 13, 4, 300), ('b', 14, 4, 314), ('b', 15, 4, 328), ('w', 5, 4, 342), ('w', 6, 4, 346), ('w', 7, 4, 350), ('w', 8, 4, 354), ('w', 9, 4, 358), ('w', 10, 4, 362), ('w', 11, 4, 366), ('w', 12, 4, 370), ('w', 13, 4, 374), ('w', 14, 4, 378), ('w', 15, 4, 382)], 
        #  [('f', 0, 5, 35), ('f', 1, 5, 41), ('f', 2, 5, 47), ('f', 3, 5, 53), ('f', 4, 5, 59), ('f', 5, 5, 65), ('f', 6, 5, 71), ('f', 7, 5, 77), ('b', 0, 5, 83), ('f', 8, 5, 93), ('f', 9, 5, 99), ('b', 1, 5, 109), ('f', 10, 5, 119), ('f', 11, 5, 125), ('b', 2, 5, 133), ('b', 3, 5, 143), ('b', 4, 5, 153), ('f', 12, 5, 163), ('b', 5, 5, 169), ('f', 13, 5, 179), ('b', 6, 5, 185), ('w', 0, 5, 195), ('f', 14, 5, 199), ('b', 7, 5, 205), ('b', 8, 5, 217), ('f', 15, 5, 227), ('b', 9, 5, 233), ('b', 10, 5, 243), ('w', 1, 5, 256), ('b', 11, 5, 260), ('w', 2, 5, 270), ('b', 12, 5, 274), ('b', 13, 5, 289), ('w', 3, 5, 299), ('b', 14, 5, 303), ('w', 4, 5, 313), ('b', 15, 5, 317), ('w', 5, 5, 342), ('w', 6, 5, 346), ('w', 7, 5, 350), ('w', 8, 5, 354), ('w', 9, 5, 358), ('w', 10, 5, 362), ('w', 11, 5, 366), ('w', 12, 5, 370), ('w', 13, 5, 374), ('w', 14, 5, 378), ('w', 15, 5, 382)], 
        #  [('f', 0, 6, 42), ('f', 1, 6, 48), ('f', 2, 6, 54), ('f', 3, 6, 60), ('b', 0, 6, 66), ('w', 0, 6, 76), ('f', 4, 6, 80), ('f', 5, 6, 86), ('f', 6, 6, 92), ('b', 1, 6, 98), ('f', 7, 6, 108), ('f', 8, 6, 114), ('b', 2, 6, 122), ('b', 3, 6, 132), ('b', 4, 6, 142), ('f', 9, 6, 152), ('b', 5, 6, 158), ('f', 10, 6, 168), ('b', 6, 6, 174), ('f', 11, 6, 184), ('w', 1, 6, 190), ('b', 7, 6, 194), ('b', 8, 6, 204), ('f', 12, 6, 214), ('b', 9, 6, 220), ('b', 10, 6, 230), ('f', 13, 6, 240), ('b', 11, 6, 246), ('f', 14, 6, 256), ('b', 12, 6, 262), ('f', 15, 6, 272), ('b', 13, 6, 278), ('w', 2, 6, 288), ('b', 14, 6, 292), ('b', 15, 6, 306), ('w', 3, 6, 328), ('w', 4, 6, 332), ('w', 5, 6, 336), ('w', 6, 6, 346), ('w', 7, 6, 350), ('w', 8, 6, 354), ('w', 9, 6, 358), ('w', 10, 6, 362), ('w', 11, 6, 366), ('w', 12, 6, 370), ('w', 13, 6, 374), ('w', 14, 6, 378), ('w', 15, 6, 382)], 
        #  [('f', 0, 7, 49), ('b', 0, 7, 55), ('f', 1, 7, 65), ('b', 1, 7, 71), ('f', 2, 7, 81), ('f', 3, 7, 87), ('f', 4, 7, 93), ('f', 5, 7, 99), ('f', 6, 7, 105), ('b', 2, 7, 111), ('b', 3, 7, 121), ('b', 4, 7, 131), ('f', 7, 7, 141), ('b', 5, 7, 147), ('f', 8, 7, 157), ('b', 6, 7, 163), ('f', 9, 7, 173), ('b', 7, 7, 179), ('b', 8, 7, 189), ('f', 10, 7, 199), ('b', 9, 7, 205), ('b', 10, 7, 215), ('f', 11, 7, 225), ('b', 11, 7, 231), ('f', 12, 7, 241), ('b', 12, 7, 247), ('f', 13, 7, 257), ('b', 13, 7, 263), ('f', 14, 7, 273), ('b', 14, 7, 279), ('f', 15, 7, 289), ('b', 15, 7, 295), ('w', 0, 7, 305), ('w', 1, 7, 309), ('w', 2, 7, 313), ('w', 3, 7, 317), ('w', 4, 7, 321), ('w', 5, 7, 325), ('w', 6, 7, 329), ('w', 7, 7, 333), ('w', 8, 7, 337), ('w', 9, 7, 341), ('w', 10, 7, 345), ('w', 11, 7, 349), ('w', 12, 7, 353), ('w', 13, 7, 362), ('w', 14, 7, 367), ('w', 15, 7, 382)]]
        
        # self.device_steps = [[('f', 0, 0, 0), ('f', 1, 0, 24), ('f', 2, 0, 48), ('f', 3, 0, 72), ('f', 4, 0, 96), ('b', 0, 0, 198), ('f', 5, 0, 230), ('b', 1, 0, 254), ('f', 6, 0, 286), ('w', 0, 0, 310), ('b', 2, 0, 326), ('f', 7, 0, 358), ('b', 3, 0, 382), ('w', 1, 0, 414), ('w', 2, 0, 430), ('w', 3, 0, 446), ('b', 4, 0, 462), ('w', 4, 0, 494), ('b', 5, 0, 510), ('w', 5, 0, 542), ('b', 6, 0, 558), ('b', 7, 0, 590), ('w', 6, 0, 622), ('w', 7, 0, 638)], 
        #                 [('f', 0, 1, 25), ('f', 1, 1, 59), ('f', 2, 1, 83), ('f', 3, 1, 107), ('f', 4, 1, 131), ('b', 0, 1, 165), ('w', 0, 1, 205), ('b', 1, 1, 221), ('f', 5, 1, 255), ('b', 2, 1, 293), ('f', 6, 1, 325), ('b', 3, 1, 349), ('f', 7, 1, 385), ('b', 4, 1, 409), ('w', 1, 1, 441), ('w', 2, 1, 457), ('b', 5, 1, 473), ('w', 3, 1, 505), ('b', 6, 1, 525), ('b', 7, 1, 557), ('w', 4, 1, 589), ('w', 5, 1, 605), ('w', 6, 1, 621), ('w', 7, 1, 638)], 
        #                 [('f', 0, 2, 50), ('f', 1, 2, 84), ('f', 2, 2, 108), ('b', 0, 2, 132), ('f', 3, 2, 164), ('b', 1, 2, 188), ('f', 4, 2, 220), ('b', 2, 2, 244), ('w', 0, 2, 276), ('f', 5, 2, 292), ('b', 3, 2, 316), ('f', 6, 2, 350), ('b', 4, 2, 376), ('f', 7, 2, 410), ('b', 5, 2, 434), ('w', 1, 2, 466), ('b', 6, 2, 492), ('b', 7, 2, 524), ('w', 2, 2, 556), ('w', 3, 2, 572), ('w', 4, 2, 588), ('w', 5, 2, 604), ('w', 6, 2, 620), ('w', 7, 2, 636)], 
        #                 [('f', 0, 3, 75), ('b', 0, 3, 99), ('f', 1, 3, 131), ('b', 1, 3, 155), ('f', 2, 3, 187), ('b', 2, 3, 211), ('f', 3, 3, 243), ('b', 3, 3, 267), ('f', 4, 3, 299), ('b', 4, 3, 323), ('f', 5, 3, 355), ('b', 5, 3, 379), ('f', 6, 3, 411), ('f', 7, 3, 435), ('b', 6, 3, 459), ('b', 7, 3, 491), ('w', 0, 3, 523), ('w', 1, 3, 539), ('w', 2, 3, 555), ('w', 3, 3, 571), ('w', 4, 3, 587), ('w', 5, 3, 603), ('w', 6, 3, 619), ('w', 7, 3, 635)]]


    def _unifiedPP2(self, engine, return_loss=True, return_output_label=True):
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
        input_objs = []
        output_objs = []
        moe_losses = []
        return_tensors = []
        output_obj_caches = []
        input_obj_caches = []
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

        steps = self.device_steps[gpc.get_local_rank(ParallelMode.PIPELINE)]
        for s in range(len(steps)):
            step_type, microbatch_id, stage_id, start_time = steps[s]
            if s<len(steps)-1:
                next_step_type = steps[s+1][0]
            else:
                next_step_type = ''
            if s>0:
                prev_step_type = steps[s-1][0]
            else:
                prev_step_type = ''
            if step_type == Stage.FORWARD.value:# Forward pass
                # Receive the input from the previous stage
                if not gpc.is_first_rank(ParallelMode.PIPELINE):
                    if prev_step_type == Stage.BACKWARD.value:
                        input_obj = input_obj_caches.pop(0)
                    else:
                        if forward_recv_shapes is None:
                            forward_recv_shapes = comm.recv_obj_meta()
                        input_obj = comm.recv_forward(
                            forward_recv_shapes,
                            dtype=self.dtype,
                            scatter_gather_tensors=self.scatter_gather_tensors,
                        )

                else:
                    input_obj = None

                # Perform forward computation
                output_obj, moe_loss = self._forward_step(
                    engine,
                    input_obj,
                    return_tensors,
                    return_output_label=return_output_label,
                    accum_loss=accum_loss,
                    accum_moe_loss=accum_moe_loss,
                )

                if not gpc.is_last_rank(ParallelMode.PIPELINE):
                    if isinstance(output_obj, torch.Tensor):
                        backward_recv_shapes = output_obj.shape
                    else:
                        backward_recv_shapes = [out_tensor.shape for out_tensor in output_obj]

                    if need_forward_meta:
                        comm.send_obj_meta(output_obj)
                        need_forward_meta = False  # send only once.

                # Send the output of forward computation of this pipeline stage to the next pipeline stage as input for
                # forward computation

                if not gpc.is_last_rank(ParallelMode.PIPELINE):
                    assert output_obj.dtype == self.dtype
                    if next_step_type == Stage.BACKWARD.value:
                        output_obj_cache = comm.send_forward_recv_backward(
                            output_obj,
                            backward_recv_shapes,
                            dtype=self.dtype,
                            scatter_gather_tensors=self.scatter_gather_tensors,
                        )
                        output_obj_caches.append(output_obj_cache)
                    else:
                        comm.send_forward(output_obj, scatter_gather_tensors=self.scatter_gather_tensors)

                input_objs.append(input_obj)
                output_objs.append(output_obj)
                moe_losses.append(moe_loss)

            elif step_type == Stage.BACKWARD.value:# Backward pass
                input_obj = input_objs.pop(0)
                output_obj = output_objs.pop(0)
                moe_loss = moe_losses.pop(0)

                if not gpc.is_last_rank(ParallelMode.PIPELINE):
                    if prev_step_type == Stage.FORWARD.value:
                        output_obj_grad = output_obj_caches.pop(0)
                    else:
                        output_obj_grad = comm.recv_backward(
                            backward_recv_shapes,
                            dtype=self.dtype,
                            scatter_gather_tensors=self.scatter_gather_tensors,
                        )

                else:
                    output_obj_grad = None
                
                input_obj_grad = self._backward_step(
                    engine, microbatch_id, input_obj, output_obj, output_obj_grad, moe_loss
                )

                if not gpc.is_first_rank(ParallelMode.PIPELINE):
                    if next_step_type == Stage.FORWARD.value:
                        input_obj_cache = comm.send_backward_recv_forward(
                            input_obj_grad,
                            forward_recv_shapes,
                            dtype=self.dtype,
                            scatter_gather_tensors=self.scatter_gather_tensors,
                        )
                        input_obj_caches.append(input_obj_cache)
                    else:
                        comm.send_backward(input_obj_grad, scatter_gather_tensors=self.scatter_gather_tensors)
                        
                WeightGradStore.flush()

            elif step_type == Stage.WEIGHT.value: # Weight update
                WeightGradStore.pop()

        output, label = pack_return_tensors(return_tensors) if len(return_tensors) > 0 else (None, None)

        if hasattr(gpc.config.model, "num_experts") and gpc.config.model.num_experts > 1:
            dist.all_reduce(accum_moe_loss, group=gpc.get_group(ParallelMode.PIPELINE))

        if accum_loss is not None:
            accum_loss += accum_moe_loss

        return output, label, accum_loss, accum_moe_loss

    def _unifiedPP2_async(self, engine, return_loss=True, return_output_label=True):
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
        input_objs = []
        output_objs = []
        moe_losses = []
        return_tensors = []
        output_obj_caches = []
        input_obj_caches = []
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

        steps = self.device_steps[gpc.get_local_rank(ParallelMode.PIPELINE)]
        for s in range(len(steps)):
            step_type, microbatch_id, stage_id, start_time = steps[s]
            if s<len(steps)-1:
                next_step_type = steps[s+1][0]
            else:
                next_step_type = ''
            if s>0:
                prev_step_type = steps[s-1][0]
            else:
                prev_step_type = ''
            if step_type == Stage.FORWARD.value:# Forward pass
                # Receive the input from the previous stage
                print(f"step:{step_type}, microbatch_id:{microbatch_id}, stage_id:{stage_id} _forward_step_begin")
                if not gpc.is_first_rank(ParallelMode.PIPELINE):
                    if prev_step_type == Stage.BACKWARD.value:
                        input_obj = input_obj_caches.pop(0)
                        print(f"step:{step_type}, microbatch_id:{microbatch_id}, stage_id:{stage_id} input_obj_caches.pop(0)")
                    else:
                        if forward_recv_shapes is None:
                            forward_recv_shapes = comm.recv_obj_meta()
                        async_communicator_recv_forward = comm.AsynCommunicator(
                            recv_prev_shape = forward_recv_shapes,
                            dtype=self.dtype,
                            scatter_gather_tensors=self.scatter_gather_tensors,
                        )
                        async_communicator_recv_forward.start()
                        input_obj,_  = async_communicator_recv_forward.wait_and_receive()
                        print(f"step:{step_type}, microbatch_id:{microbatch_id}, stage_id:{stage_id} recv_forward")

                else:
                    input_obj = None

                # Perform forward computation
                output_obj, moe_loss = self._forward_step(
                    engine,
                    input_obj,
                    return_tensors,
                    return_output_label=return_output_label,
                    accum_loss=accum_loss,
                    accum_moe_loss=accum_moe_loss,
                )
                print(f"step:{step_type}, microbatch_id:{microbatch_id}, stage_id:{stage_id} _forward_step")
                # if microbatch_id != 0 and microbatch_id+stage_id == 3:
                #     import pdb;pdb.set_trace()
                if not gpc.is_last_rank(ParallelMode.PIPELINE):
                    if isinstance(output_obj, torch.Tensor):
                        backward_recv_shapes = output_obj.shape
                    else:
                        backward_recv_shapes = [out_tensor.shape for out_tensor in output_obj]

                    if need_forward_meta:
                        comm.send_obj_meta(output_obj)
                        need_forward_meta = False  # send only once.

                # Send the output of forward computation of this pipeline stage to the next pipeline stage as input for
                # forward computation

                if not gpc.is_last_rank(ParallelMode.PIPELINE):
                    assert output_obj.dtype == self.dtype
                    if next_step_type == Stage.BACKWARD.value:
                        async_communicator = comm.AsynCommunicator(
                            object_send_prev=None,
                            object_send_next=output_obj,
                            recv_prev_shape=None,
                            recv_next_shape=backward_recv_shapes,
                            dtype=self.dtype,
                            scatter_gather_tensors=self.scatter_gather_tensors,
                        )
                        async_communicator.start()
                        _, output_obj_cache = async_communicator.wait_and_receive()
                        print(f"step:{step_type}, microbatch_id:{microbatch_id}, stage_id:{stage_id} send_forward_recv_backward")
                        output_obj_caches.append(output_obj_cache)
                    else:
                        comm.AsynCommunicator(object_send_next = output_obj, dtype=self.dtype, scatter_gather_tensors=self.scatter_gather_tensors).start()
                        print(f"step:{step_type}, microbatch_id:{microbatch_id}, stage_id:{stage_id} send_forward")

                input_objs.append(input_obj)
                output_objs.append(output_obj)
                moe_losses.append(moe_loss)

            elif step_type == Stage.BACKWARD.value:# Backward pass
                print(f"step:{step_type}, microbatch_id:{microbatch_id}, stage_id:{stage_id} _backward_step_begin")
                input_obj = input_objs.pop(0)
                output_obj = output_objs.pop(0)
                moe_loss = moe_losses.pop(0)

                if not gpc.is_last_rank(ParallelMode.PIPELINE):
                    if prev_step_type == Stage.FORWARD.value:
                        output_obj_grad = output_obj_caches.pop(0)
                        print(f"step:{step_type}, microbatch_id:{microbatch_id}, stage_id:{stage_id} output_obj_caches.pop(0)")
                    else:
                        async_communicator_recv_forward = comm.AsynCommunicator(
                            recv_next_shape = backward_recv_shapes,
                            dtype=self.dtype,
                            scatter_gather_tensors=self.scatter_gather_tensors,
                        )
                        async_communicator_recv_forward.start()
                        _, output_obj_grad = async_communicator_recv_forward.wait_and_receive()
                        print(f"step:{step_type}, microbatch_id:{microbatch_id}, stage_id:{stage_id} recv_backward")
                else:
                    output_obj_grad = None
                
                
                input_obj_grad = self._backward_step(
                    engine, microbatch_id, input_obj, output_obj, output_obj_grad, moe_loss
                )
                print(f"step:{step_type}, microbatch_id:{microbatch_id}, stage_id:{stage_id} _backward_step")
                if not gpc.is_first_rank(ParallelMode.PIPELINE):
                    if next_step_type == Stage.FORWARD.value:
                        async_communicator = comm.AsynCommunicator(
                            object_send_prev=input_obj_grad,
                            object_send_next=None,
                            recv_prev_shape=forward_recv_shapes,
                            recv_next_shape=None,
                            dtype=self.dtype,
                            scatter_gather_tensors=self.scatter_gather_tensors,
                        )
                        async_communicator.start()
                        input_obj_cache, _ = async_communicator.wait_and_receive()
                        print(f"step:{step_type}, microbatch_id:{microbatch_id}, stage_id:{stage_id} send_backward_recv_forward")
                        input_obj_caches.append(input_obj_cache)
                    else:
                        comm.AsynCommunicator(object_send_prev = input_obj_grad, dtype=self.dtype,scatter_gather_tensors=self.scatter_gather_tensors).start()
                        print(f"step:{step_type}, microbatch_id:{microbatch_id}, stage_id:{stage_id} send_backward")
                WeightGradStore.flush()

            elif step_type == Stage.WEIGHT.value: # Weight update
                WeightGradStore.pop()

        output, label = pack_return_tensors(return_tensors) if len(return_tensors) > 0 else (None, None)

        if hasattr(gpc.config.model, "num_experts") and gpc.config.model.num_experts > 1:
            dist.all_reduce(accum_moe_loss, group=gpc.get_group(ParallelMode.PIPELINE))

        if accum_loss is not None:
            accum_loss += accum_moe_loss

        return output, label, accum_loss, accum_moe_loss

    def _forward_backward_step(self, engine, return_loss=True, return_output_label=True):
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

        num_warmup_microsteps = (
            gpc.get_world_size(ParallelMode.PIPELINE) - gpc.get_local_rank(ParallelMode.PIPELINE) - 1
        )
        num_warmup_microsteps = min(num_warmup_microsteps, self.num_microbatches)
        num_1f1b_micropairs = self.num_microbatches - num_warmup_microsteps

        # Input, output tensors only need to be saved when doing backward passes
        input_objs = []
        output_objs = []
        moe_losses = []
        return_tensors = []
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

        f_times = 0
        # Run warmup forward passes.
        for i in range(num_warmup_microsteps):
            # Receive the input from the previous stage
            if not gpc.is_first_rank(ParallelMode.PIPELINE):
                if forward_recv_shapes is None:
                    forward_recv_shapes = comm.recv_obj_meta()
                input_obj = comm.recv_forward(
                    forward_recv_shapes,
                    dtype=self.dtype,
                    scatter_gather_tensors=self.scatter_gather_tensors,
                )
            else:
                input_obj = None

            # Perform forward computation
            output_obj, moe_loss = self._forward_step(
                engine,
                input_obj,
                return_tensors,
                return_output_label=return_output_label,
                accum_loss=accum_loss,
                accum_moe_loss=accum_moe_loss,
            )
            f_times += 1

            if not gpc.is_last_rank(ParallelMode.PIPELINE):
                if isinstance(output_obj, torch.Tensor):
                    backward_recv_shapes = output_obj.shape
                else:
                    backward_recv_shapes = [out_tensor.shape for out_tensor in output_obj]

                if need_forward_meta:
                    comm.send_obj_meta(output_obj)
                    need_forward_meta = False  # send only once.

            # Send the output of forward computation of this pipeline stage to the next pipeline stage as input for
            # forward computation
            if not gpc.is_last_rank(ParallelMode.PIPELINE):
                assert output_obj.dtype == self.dtype
                comm.send_forward(output_obj, scatter_gather_tensors=self.scatter_gather_tensors)

            input_objs.append(input_obj)
            output_objs.append(output_obj)
            moe_losses.append(moe_loss)
        # Before running 1F1B, need to receive first forward tensor.
        # If all microbatches are run in warmup / cooldown phase, then no need to
        # receive this tensor here.
        if num_1f1b_micropairs > 0:
            if not gpc.is_first_rank(ParallelMode.PIPELINE):
                if forward_recv_shapes is None:
                    forward_recv_shapes = comm.recv_obj_meta()
                input_obj = comm.recv_forward(
                    forward_recv_shapes,
                    dtype=self.dtype,
                    scatter_gather_tensors=self.scatter_gather_tensors,
                )
            else:
                input_obj = None

        # Run 1F1B in steady state.
        for i in range(num_1f1b_micropairs):
            # Perform forward computation
            output_obj, moe_loss = self._forward_step(
                engine,
                input_obj,
                return_tensors,
                return_output_label=return_output_label,
                accum_loss=accum_loss,
                accum_moe_loss=accum_moe_loss,
            )
            f_times += 1

            if gpc.is_last_rank(ParallelMode.PIPELINE):
                output_obj_grad = None
            else:
                assert output_obj.dtype == self.dtype
                output_obj_grad = comm.send_forward_recv_backward(
                    output_obj,
                    backward_recv_shapes,
                    dtype=self.dtype,
                    scatter_gather_tensors=self.scatter_gather_tensors,
                )

            # Add input_obj and output_obj to end of list.
            input_objs.append(input_obj)
            output_objs.append(output_obj)
            moe_losses.append(moe_loss)

            # Pop output_obj and output_obj from the start of the list for
            # the backward pass.
            input_obj = input_objs.pop(0)
            output_obj = output_objs.pop(0)
            moe_loss = moe_losses.pop(0)

            input_obj_grad = self._backward_step(engine, i, input_obj, output_obj, output_obj_grad, moe_loss)

            if i == (num_1f1b_micropairs - 1):
                input_obj = None
                if not gpc.is_first_rank(ParallelMode.PIPELINE):
                    comm.send_backward(
                        input_obj_grad,
                        scatter_gather_tensors=self.scatter_gather_tensors,
                    )
            else:
                if gpc.is_first_rank(ParallelMode.PIPELINE):
                    input_obj = None
                else:
                    input_obj = comm.send_backward_recv_forward(
                        input_obj_grad,
                        forward_recv_shapes,
                        dtype=self.dtype,
                        scatter_gather_tensors=self.scatter_gather_tensors,
                    )

            WeightGradStore.flush()
            if i >= gpc.get_local_rank(ParallelMode.PIPELINE):
                WeightGradStore.pop()

        # Run cooldown backward passes.
        for i in range(num_warmup_microsteps):
            input_obj = input_objs.pop(0)
            output_obj = output_objs.pop(0)
            moe_loss = moe_losses.pop(0)

            if not gpc.is_last_rank(ParallelMode.PIPELINE):
                output_obj_grad = comm.recv_backward(
                    backward_recv_shapes,
                    dtype=self.dtype,
                    scatter_gather_tensors=self.scatter_gather_tensors,
                )
            else:
                output_obj_grad = None

            input_obj_grad = self._backward_step(
                engine, num_1f1b_micropairs + i, input_obj, output_obj, output_obj_grad, moe_loss
            )

            if not gpc.is_first_rank(ParallelMode.PIPELINE):
                comm.send_backward(input_obj_grad, scatter_gather_tensors=self.scatter_gather_tensors)

            WeightGradStore.flush()
            WeightGradStore.pop()

        while WeightGradStore.size() > 0:
            WeightGradStore.pop()

        output, label = pack_return_tensors(return_tensors) if len(return_tensors) > 0 else (None, None)

        if hasattr(gpc.config.model, "num_experts") and gpc.config.model.num_experts > 1:
            dist.all_reduce(accum_moe_loss, group=gpc.get_group(ParallelMode.PIPELINE))

        if accum_loss is not None:
            accum_loss += accum_moe_loss

        return output, label, accum_loss, accum_moe_loss


class ZeroBubblePipelineVShapeScheduler(InterleavedPipelineScheduler):
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
