import torch
import torch.distributed as dist
import os
import subprocess
import queue
from typing import Callable, List, Optional, Tuple, Union
import torch
import torch.distributed as dist
from internlm.core.context import ParallelMode
from internlm.core.scheduler import comm
import queue
from enum import Enum
import fcntl
import logging


# logging.basicConfig(level=logging.INFO, format='%(message)s')
# logger = logging.getLogger()
# logger.addHandler(logging.FileHandler('test.log', 'w'))
# debugeprint = logger.info

# Remove all handlers associated with the root logger (including the console handler set by basicConfig)
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Configure logging to only log to a file
logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format='%(message)s',  # Set the log message format
    handlers=[
        logging.FileHandler('test.log', mode='w')  # Log output to 'test.log' in write mode
    ]
)

# Get the logger instance
logger = logging.getLogger()

# Use logger.info as debugeprint
debugeprint = logger.info

class Stage(Enum):
    FORWARD = 'f'
    BACKWARD = 'b'
    WEIGHT = 'w'

def _get_deviceid_by_alignment(stage_id: int, stage_alignment:list) -> int:
    for device_id in range(len(stage_alignment)):
        for stage_in_device in stage_alignment[device_id]:
            if stage_in_device == stage_id:
                return device_id
                
def order_result_J(input: str, stage_alignment: list) -> None:
    input = input.replace("\n", "")
    input = input.replace(" ", "")
    device_steps = [[] for _ in range(len(stage_alignment))]
    all_step = input.split('.')
    for step in all_step:
        start_time = step.split(',')[-1]
        infor = step.split(',')[0]
        if infor is None or infor == '' or infor[0] not in ['b','w','f']:
            continue
        step_type, microbatch_id, stage_id = infor.split('_')
        microbatch_id = int(microbatch_id)
        stage_id = int(stage_id)
        start_time = int(start_time)
        device_id = _get_deviceid_by_alignment(stage_id, stage_alignment)
        device_steps[device_id].append((step_type, microbatch_id, stage_id, start_time))
    for d in range(len(stage_alignment)):
        device_steps[d].sort(key=lambda x: x[3])

    result = []
    for stage_idx, stage_list in enumerate(device_steps):
        new_stage_list = []
        for current_tuple in stage_list:
            op, mb_id, stage_id, time = current_tuple
            new_stage_list.append((op, mb_id, stage_id))
        result.append(new_stage_list)
    return result


def add_new_attribute_NeedRecvBackStart(device_steps):
    result = []
    
    for stage_idx, stage_list in enumerate(device_steps):
        new_stage_list = []
        for current_tuple in stage_list:
            op, mb_id, stage_id, time = current_tuple
            new_attr = False
            
            # 只检查f操作，且不是最后一行
            if op == 'f' and stage_idx < len(device_steps) - 1:
                # 获取下一行(stage)的操作列表
                next_stage = device_steps[stage_idx + 1]
                
                # 寻找下一行中相同micro_batch_id的操作位置
                for i, next_op in enumerate(next_stage):
                    if next_op[1] == mb_id and next_op[0] == 'f':  # 找到相同micro_batch_id的操作
                        #查看是否为b、f组合
                        #查看前一个操作是否为b，若为w则继续查看前一个，直到遇到b或f为止
                        s = 1
                        while (i-s)>0 :
                            judgestep = next_stage[i - s]
                            if judgestep[0] == 'b':
                                new_attr = True
                                break
                            elif judgestep[0] == 'f':
                                break
                            elif judgestep[0] == 'w':
                                s+=1

            new_stage_list.append((op, mb_id, stage_id, time, new_attr))
        result.append(new_stage_list)
    
    return result

#unuse
def add_new_attribute_NeedRecvForwardStart(device_steps):
    result = []
    
    for stage_idx, stage_list in enumerate(device_steps):
        new_stage_list = []
        for current_tuple in stage_list:
            op, mb_id, stage_id, time, NeedRecvBackStart = current_tuple
            new_attr = False
            
            # 只检查b操作，且不是第一行
            if op == 'b' and stage_idx > 0:
                # 获取上一行(stage)的操作列表
                prev_stage = device_steps[stage_idx - 1]
                
                # 寻找上一行中相同micro_batch_id的操作位置
                for i, prev_op in enumerate(prev_stage):
                    if prev_op[1] == mb_id and prev_op[0] == 'b' :  # 找到相同micro_batch_id的操作
                        #查看是否为f、b组合
                        #查看前一个操作是否为f，若为w则继续查看前一个，直到遇到b或f为止
                        s = 1
                        while (i-s)>0 :
                            judgestep = prev_stage[i - s]
                            if judgestep[0] == 'f':
                                new_attr = True
                                break
                            elif judgestep[0] == 'b':
                                break
                            elif judgestep[0] == 'w':
                                s+=1
 
            new_stage_list.append((op, mb_id, stage_id, time, NeedRecvBackStart, new_attr))
        result.append(new_stage_list)
    
    return result


def add_new_attribute_NeedRecvForwardStartTimes(device_steps):
    result = []
    
    for stage_idx, stage_list in enumerate(device_steps):
        new_stage_list = []
        for current_tuple in stage_list:
            op, mb_id, stage_id, time, NeedRecvBackStart = current_tuple
            new_attr = 0
            # 只检查b操作，且不是第一行
            if op == 'b' and stage_idx > 0:
                debugeprint(f'current_tuple, mb_id:{mb_id}, stage_idx:{stage_idx}')
                # 获取上一行(stage)的操作列表
                prev_stage = device_steps[stage_idx - 1]
                
                # 寻找上一行中相同micro_batch_id的操作位置
                for i, prev_op in enumerate(prev_stage):
                    if prev_op[1] == mb_id and prev_op[0] == 'b' :  # 找到相同micro_batch_id的操作
                        #查看是否为f、b组合
                        #查看前一个操作是否为f，若为w则继续查看前一个，直到遇到b或f为止
                        s = 1
                        while (i-s)>0 :
                            judgestep = prev_stage[i - s]
                            if judgestep[0] == 'f':
                                new_attr = judgestep[1]
                                break
                            s += 1
                        break
            new_stage_list.append((op, mb_id, stage_id, time, NeedRecvBackStart, new_attr))
        result.append(new_stage_list)
    return result

def drop_w(device_steps):
    result = []
    for stage_idx, stage_list in enumerate(device_steps):
        new_stage_list = []
        for current_tuple in stage_list:
            op, mb_id, stage_id, time = current_tuple
            if op == 'f' or op == 'b':
                new_stage_list.append(current_tuple)
        result.append(new_stage_list)
    return result

def setup_distributed(backend="nccl", port=None):
    """Initialize distributed training environment.
    support both slurm and torch.distributed.launch
    see torch.distributed.init_process_group() for more details
    """
    num_gpus = torch.cuda.device_count()

    if "SLURM_JOB_ID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_world_size = int(os.environ['SLURM_NTASKS_PER_NODE']) 
        # import pdb; pdb.set_trace() 
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        # specify master port
        if port is not None:
            os.environ["MASTER_PORT"] = str(port)
        elif "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29500"
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank % num_gpus)
        os.environ["RANK"] = str(rank)
        os.environ['LOCAL_WORLD_SIZE'] = str(local_world_size)
        os.environ['GROUP_RANK'] = str(rank // local_world_size)
        # debugeprint(addr, node_list, rank, os.environ["LOCAL_RANK"])
    else:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, init_method="env://", rank=rank, world_size=world_size)
    return rank % num_gpus

def do_compute():
    x = torch.rand(100, 100).cuda()
    for i in range(1000):
        result = torch.bmm(x.unsqueeze(0), x.unsqueeze(0))
    return result

def write_to_file(file_path, content):
    with open(file_path, 'w') as file:
        fcntl.flock(file, fcntl.LOCK_EX)  # 获取互斥锁
        file.write(str(content))
        fcntl.flock(file, fcntl.LOCK_UN)  # 释放锁

def read_from_file(file_path):
    with open(file_path, 'r') as file:
        fcntl.flock(file, fcntl.LOCK_SH)  # 获取共享锁
        content = file.read()
        content = content.replace('\x00','')
        if content == '':
            content = 0
        fcntl.flock(file, fcntl.LOCK_UN)  # 释放锁
        return int(content)

def recvAll(step_type, microbatch_id, stage_id, dtype, scatter_gather_tensors, last_stage,
            prev_rank,recv_forward_start_time, forward_recv_shapes, async_communicator_recv_forward_queue,
            next_rank, recv_backward_start_time, backward_recv_shapes, async_communicator_recv_backward_queue):
    if prev_rank > -1:
        sendforwardtimes = read_from_file(f'./shared_mem/rank{prev_rank}_sendforward')
        Current_NeedRecvForwardStartTimes = sendforwardtimes-recv_forward_start_time
        if Current_NeedRecvForwardStartTimes>0:
            for i in range(Current_NeedRecvForwardStartTimes):
                async_communicator_recv_forward = comm.AsynCommunicator(
                    recv_prev_shape = forward_recv_shapes,
                    dtype=dtype,
                    scatter_gather_tensors=scatter_gather_tensors,
                    prev_rank=prev_rank,
                    next_rank=next_rank,
                )
                async_communicator_recv_forward.start()
                recv_forward_start_time += 1
                async_communicator_recv_forward_queue.put(async_communicator_recv_forward)
                debugeprint(f"step:{step_type}, microbatch_id:{microbatch_id}, stage_id:{stage_id} recv_forward_queue.put, recv_forward_start:{recv_forward_start_time}")
    
    if next_rank < last_stage+1:
        sendbackwardtimes = read_from_file(f'./shared_mem/rank{next_rank}_sendbackward')
        Current_NeedRecvBackwardStartTimes = sendbackwardtimes-recv_backward_start_time
        if Current_NeedRecvBackwardStartTimes>0:
            for i in range(Current_NeedRecvBackwardStartTimes):
                async_communicator_recv_backward = comm.AsynCommunicator(
                    recv_next_shape=backward_recv_shapes,
                    dtype=dtype,
                    scatter_gather_tensors=scatter_gather_tensors,
                    prev_rank=prev_rank,
                    next_rank=next_rank,
                )
                async_communicator_recv_backward.start()
                recv_backward_start_time += 1
                async_communicator_recv_backward_queue.put(async_communicator_recv_backward)
                debugeprint(f"step:{step_type}, microbatch_id:{microbatch_id}, stage_id:{stage_id} recv_backward_queue.put, recv_backward_start:{recv_backward_start_time}")
    
    return recv_forward_start_time, recv_backward_start_time, async_communicator_recv_forward_queue, async_communicator_recv_backward_queue

def main():
    stage_alignment = [[0], [1], [2], [3], [4], [5], [6], [7]]
    inputstr = '''
f_0_0,0.0.
b_0_0,268.0.
w_0_0,288.0.
f_1_0,12.0.
b_1_0,316.0.
w_1_0,336.0.
f_2_0,24.0.
b_2_0,354.0.
w_2_0,394.0.
f_3_0,36.0.
b_3_0,412.0.
w_3_0,432.0.
f_4_0,48.0.
b_4_0,460.0.
w_4_0,480.0.
f_5_0,60.0.
b_5_0,498.0.
w_5_0,518.0.
f_6_0,72.0.
b_6_0,536.0.
w_6_0,556.0.
f_7_0,84.0.
b_7_0,576.0.
w_7_0,596.0.
f_8_0,96.0.
b_8_0,614.0.
w_8_0,634.0.
f_9_0,108.0.
b_9_0,652.0.
w_9_0,672.0.
f_10_0,120.0.
b_10_0,696.0.
w_10_0,716.0.
f_11_0,132.0.
b_11_0,734.0.
w_11_0,754.0.
f_12_0,144.0.
b_12_0,772.0.
w_12_0,792.0.
f_13_0,156.0.
b_13_0,810.0.
w_13_0,830.0.
f_14_0,168.0.
b_14_0,858.0.
w_14_0,878.0.
f_15_0,180.0.
b_15_0,896.0.
w_15_0,916.0.
f_16_0,294.0.
b_16_0,934.0.
w_16_0,1054.0.
f_17_0,342.0.
b_17_0,976.0.
w_17_0,1060.0.
f_18_0,400.0.
b_18_0,1014.0.
w_18_0,1108.0.
f_19_0,438.0.
b_19_0,1034.0.
w_19_0,1154.0.
f_20_0,486.0.
b_20_0,1088.0.
w_20_0,1180.0.
f_21_0,524.0.
b_21_0,1114.0.
w_21_0,1186.0.
f_22_0,562.0.
b_22_0,1134.0.
w_22_0,1212.0.
f_23_0,602.0.
b_23_0,1160.0.
w_23_0,1238.0.
f_24_0,640.0.
b_24_0,1192.0.
w_24_0,1264.0.
f_25_0,678.0.
b_25_0,1218.0.
w_25_0,1290.0.
f_26_0,722.0.
b_26_0,1244.0.
w_26_0,1296.0.
f_27_0,760.0.
b_27_0,1270.0.
w_27_0,1322.0.
f_28_0,798.0.
b_28_0,1302.0.
w_28_0,1348.0.
f_29_0,836.0.
b_29_0,1328.0.
w_29_0,1374.0.
f_30_0,884.0.
b_30_0,1354.0.
w_30_0,1380.0.
f_31_0,922.0.
b_31_0,1386.0.
w_31_0,1406.0.
f_0_1,13.0.
b_0_1,247.0.
w_0_1,267.0.
f_1_1,25.0.
b_1_1,295.0.
w_1_1,349.0.
f_2_1,37.0.
b_2_1,329.0.
w_2_1,367.0.
f_3_1,49.0.
b_3_1,391.0.
w_3_1,425.0.
f_4_1,61.0.
b_4_1,435.0.
w_4_1,487.0.
f_5_1,73.0.
b_5_1,467.0.
w_5_1,493.0.
f_6_1,85.0.
b_6_1,513.0.
w_6_1,549.0.
f_7_1,97.0.
b_7_1,555.0.
w_7_1,587.0.
f_8_1,109.0.
b_8_1,593.0.
w_8_1,651.0.
f_9_1,121.0.
b_9_1,631.0.
w_9_1,669.0.
f_10_1,133.0.
b_10_1,675.0.
w_10_1,729.0.
f_11_1,145.0.
b_11_1,707.0.
w_11_1,767.0.
f_12_1,157.0.
b_12_1,747.0.
w_12_1,773.0.
f_13_1,169.0.
b_13_1,787.0.
w_13_1,819.0.
f_14_1,181.0.
b_14_1,837.0.
w_14_1,907.0.
f_15_1,193.0.
b_15_1,875.0.
w_15_1,949.0.
f_16_1,315.0.
b_16_1,913.0.
w_16_1,987.0.
f_17_1,355.0.
b_17_1,955.0.
w_17_1,1033.0.
f_18_1,413.0.
b_18_1,993.0.
w_18_1,1073.0.
f_19_1,455.0.
b_19_1,1013.0.
w_19_1,1079.0.
f_20_1,499.0.
b_20_1,1053.0.
w_20_1,1107.0.
f_21_1,537.0.
b_21_1,1087.0.
w_21_1,1179.0.
f_22_1,575.0.
b_22_1,1113.0.
w_22_1,1205.0.
f_23_1,615.0.
b_23_1,1139.0.
w_23_1,1211.0.
f_24_1,657.0.
b_24_1,1159.0.
w_24_1,1217.0.
f_25_1,695.0.
b_25_1,1185.0.
w_25_1,1243.0.
f_26_1,735.0.
b_26_1,1223.0.
w_26_1,1269.0.
f_27_1,807.0.
b_27_1,1249.0.
w_27_1,1275.0.
f_28_1,825.0.
b_28_1,1281.0.
w_28_1,1385.0.
f_29_1,895.0.
b_29_1,1307.0.
w_29_1,1391.0.
f_30_1,937.0.
b_30_1,1333.0.
w_30_1,1397.0.
f_31_1,975.0.
b_31_1,1365.0.
w_31_1,1403.0.
f_0_2,26.0.
b_0_2,216.0.
w_0_2,258.0.
f_1_2,38.0.
b_1_2,274.0.
w_1_2,362.0.
f_2_2,50.0.
b_2_2,308.0.
w_2_2,392.0.
f_3_2,62.0.
b_3_2,342.0.
w_3_2,398.0.
f_4_2,74.0.
b_4_2,414.0.
w_4_2,480.0.
f_5_2,86.0.
b_5_2,446.0.
w_5_2,486.0.
f_6_2,98.0.
b_6_2,492.0.
w_6_2,566.0.
f_7_2,110.0.
b_7_2,534.0.
w_7_2,604.0.
f_8_2,122.0.
b_8_2,572.0.
w_8_2,642.0.
f_9_2,134.0.
b_9_2,610.0.
w_9_2,704.0.
f_10_2,146.0.
b_10_2,648.0.
w_10_2,742.0.
f_11_2,158.0.
b_11_2,682.0.
w_11_2,748.0.
f_12_2,170.0.
b_12_2,722.0.
w_12_2,806.0.
f_13_2,182.0.
b_13_2,766.0.
w_13_2,864.0.
f_14_2,194.0.
b_14_2,812.0.
w_14_2,944.0.
f_15_2,236.0.
b_15_2,832.0.
w_15_2,986.0.
f_16_2,330.0.
b_16_2,892.0.
w_16_2,1026.0.
f_17_2,368.0.
b_17_2,912.0.
w_17_2,1052.0.
f_18_2,434.0.
b_18_2,962.0.
w_18_2,1086.0.
f_19_2,468.0.
b_19_2,992.0.
w_19_2,1248.0.
f_20_2,520.0.
b_20_2,1032.0.
w_20_2,1254.0.
f_21_2,554.0.
b_21_2,1066.0.
w_21_2,1280.0.
f_22_2,592.0.
b_22_2,1092.0.
w_22_2,1306.0.
f_23_2,630.0.
b_23_2,1118.0.
w_23_2,1332.0.
f_24_2,670.0.
b_24_2,1138.0.
w_24_2,1364.0.
f_25_2,710.0.
b_25_2,1164.0.
w_25_2,1370.0.
f_26_2,754.0.
b_26_2,1202.0.
w_26_2,1376.0.
f_27_2,852.0.
b_27_2,1228.0.
w_27_2,1382.0.
f_28_2,870.0.
b_28_2,1260.0.
w_28_2,1388.0.
f_29_2,932.0.
b_29_2,1286.0.
w_29_2,1394.0.
f_30_2,950.0.
b_30_2,1312.0.
w_30_2,1400.0.
f_31_2,1014.0.
b_31_2,1344.0.
w_31_2,1406.0.
f_0_3,39.0.
b_0_3,195.0.
w_0_3,239.0.
f_1_3,51.0.
b_1_3,253.0.
w_1_3,313.0.
f_2_3,63.0.
b_2_3,287.0.
w_2_3,413.0.
f_3_3,75.0.
b_3_3,321.0.
w_3_3,463.0.
f_4_3,87.0.
b_4_3,393.0.
w_4_3,501.0.
f_5_3,99.0.
b_5_3,425.0.
w_5_3,545.0.
f_6_3,111.0.
b_6_3,469.0.
w_6_3,583.0.
f_7_3,123.0.
b_7_3,513.0.
w_7_3,641.0.
f_8_3,135.0.
b_8_3,551.0.
w_8_3,691.0.
f_9_3,147.0.
b_9_3,589.0.
w_9_3,749.0.
f_10_3,159.0.
b_10_3,609.0.
w_10_3,767.0.
f_11_3,171.0.
b_11_3,647.0.
w_11_3,793.0.
f_12_3,183.0.
b_12_3,697.0.
w_12_3,831.0.
f_13_3,215.0.
b_13_3,729.0.
w_13_3,837.0.
f_14_3,227.0.
b_14_3,773.0.
w_14_3,843.0.
f_15_3,273.0.
b_15_3,811.0.
w_15_3,869.0.
f_16_3,369.0.
b_16_3,849.0.
w_16_3,943.0.
f_17_3,381.0.
b_17_3,887.0.
w_17_3,981.0.
f_18_3,451.0.
b_18_3,919.0.
w_18_3,1039.0.
f_19_3,489.0.
b_19_3,949.0.
w_19_3,1065.0.
f_20_3,533.0.
b_20_3,999.0.
w_20_3,1175.0.
f_21_3,571.0.
b_21_3,1045.0.
w_21_3,1227.0.
f_22_3,629.0.
b_22_3,1071.0.
w_22_3,1279.0.
f_23_3,679.0.
b_23_3,1097.0.
w_23_3,1317.0.
f_24_3,717.0.
b_24_3,1117.0.
w_24_3,1343.0.
f_25_3,755.0.
b_25_3,1143.0.
w_25_3,1349.0.
f_26_3,799.0.
b_26_3,1181.0.
w_26_3,1355.0.
f_27_3,875.0.
b_27_3,1207.0.
w_27_3,1361.0.
f_28_3,907.0.
b_28_3,1239.0.
w_28_3,1367.0.
f_29_3,969.0.
b_29_3,1259.0.
w_29_3,1394.0.
f_30_3,987.0.
b_30_3,1291.0.
w_30_3,1400.0.
f_31_3,1027.0.
b_31_3,1323.0.
w_31_3,1406.0.
f_0_4,52.0.
b_0_4,172.0.
w_0_4,216.0.
f_1_4,64.0.
b_1_4,222.0.
w_1_4,342.0.
f_2_4,76.0.
b_2_4,266.0.
w_2_4,360.0.
f_3_4,88.0.
b_3_4,300.0.
w_3_4,468.0.
f_4_4,100.0.
b_4_4,366.0.
w_4_4,486.0.
f_5_4,112.0.
b_5_4,404.0.
w_5_4,558.0.
f_6_4,124.0.
b_6_4,448.0.
w_6_4,646.0.
f_7_4,136.0.
b_7_4,492.0.
w_7_4,684.0.
f_8_4,148.0.
b_8_4,524.0.
w_8_4,690.0.
f_9_4,160.0.
b_9_4,564.0.
w_9_4,780.0.
f_10_4,192.0.
b_10_4,584.0.
w_10_4,818.0.
f_11_4,204.0.
b_11_4,620.0.
w_11_4,824.0.
f_12_4,242.0.
b_12_4,652.0.
w_12_4,830.0.
f_13_4,254.0.
b_13_4,708.0.
w_13_4,836.0.
f_14_4,286.0.
b_14_4,728.0.
w_14_4,842.0.
f_15_4,348.0.
b_15_4,760.0.
w_15_4,944.0.
f_16_4,386.0.
b_16_4,798.0.
w_16_4,970.0.
f_17_4,426.0.
b_17_4,860.0.
w_17_4,976.0.
f_18_4,474.0.
b_18_4,880.0.
w_18_4,982.0.
f_19_4,512.0.
b_19_4,912.0.
w_19_4,1032.0.
f_20_4,546.0.
b_20_4,950.0.
w_20_4,1070.0.
f_21_4,604.0.
b_21_4,1012.0.
w_21_4,1212.0.
f_22_4,672.0.
b_22_4,1038.0.
w_22_4,1258.0.
f_23_4,696.0.
b_23_4,1076.0.
w_23_4,1264.0.
f_24_4,748.0.
b_24_4,1096.0.
w_24_4,1290.0.
f_25_4,786.0.
b_25_4,1122.0.
w_25_4,1296.0.
f_26_4,848.0.
b_26_4,1160.0.
w_26_4,1322.0.
f_27_4,900.0.
b_27_4,1186.0.
w_27_4,1328.0.
f_28_4,932.0.
b_28_4,1218.0.
w_28_4,1334.0.
f_29_4,988.0.
b_29_4,1238.0.
w_29_4,1340.0.
f_30_4,1000.0.
b_30_4,1270.0.
w_30_4,1346.0.
f_31_4,1058.0.
b_31_4,1302.0.
w_31_4,1406.0.
f_0_5,65.0.
b_0_5,149.0.
w_0_5,193.0.
f_1_5,77.0.
b_1_5,199.0.
w_1_5,295.0.
f_2_5,89.0.
b_2_5,243.0.
w_2_5,377.0.
f_3_5,101.0.
b_3_5,275.0.
w_3_5,415.0.
f_4_5,113.0.
b_4_5,325.0.
w_4_5,421.0.
f_5_5,125.0.
b_5_5,383.0.
w_5_5,511.0.
f_6_5,137.0.
b_6_5,427.0.
w_6_5,593.0.
f_7_5,169.0.
b_7_5,471.0.
w_7_5,683.0.
f_8_5,181.0.
b_8_5,491.0.
w_8_5,753.0.
f_9_5,219.0.
b_9_5,541.0.
w_9_5,803.0.
f_10_5,231.0.
b_10_5,561.0.
w_10_5,809.0.
f_11_5,263.0.
b_11_5,599.0.
w_11_5,815.0.
f_12_5,301.0.
b_12_5,631.0.
w_12_5,905.0.
f_13_5,313.0.
b_13_5,657.0.
w_13_5,911.0.
f_14_5,345.0.
b_14_5,689.0.
w_14_5,949.0.
f_15_5,403.0.
b_15_5,733.0.
w_15_5,955.0.
f_16_5,447.0.
b_16_5,759.0.
w_16_5,1013.0.
f_17_5,459.0.
b_17_5,833.0.
w_17_5,1083.0.
f_18_5,529.0.
b_18_5,853.0.
w_18_5,1121.0.
f_19_5,581.0.
b_19_5,885.0.
w_19_5,1127.0.
f_20_5,619.0.
b_20_5,929.0.
w_20_5,1159.0.
f_21_5,709.0.
b_21_5,973.0.
w_21_5,1185.0.
f_22_5,721.0.
b_22_5,993.0.
w_22_5,1191.0.
f_23_5,779.0.
b_23_5,1019.0.
w_23_5,1237.0.
f_24_5,791.0.
b_24_5,1063.0.
w_24_5,1243.0.
f_25_5,821.0.
b_25_5,1101.0.
w_25_5,1316.0.
f_26_5,873.0.
b_26_5,1133.0.
w_26_5,1322.0.
f_27_5,917.0.
b_27_5,1165.0.
w_27_5,1328.0.
f_28_5,961.0.
b_28_5,1197.0.
w_28_5,1334.0.
f_29_5,1039.0.
b_29_5,1217.0.
w_29_5,1340.0.
f_30_5,1051.0.
b_30_5,1249.0.
w_30_5,1346.0.
f_31_5,1089.0.
b_31_5,1281.0.
w_31_5,1352.0.
f_0_6,78.0.
b_0_6,126.0.
w_0_6,282.0.
f_1_6,90.0.
b_1_6,174.0.
w_1_6,288.0.
f_2_6,102.0.
b_2_6,218.0.
w_2_6,294.0.
f_3_6,114.0.
b_3_6,250.0.
w_3_6,356.0.
f_4_6,150.0.
b_4_6,300.0.
w_4_6,426.0.
f_5_6,162.0.
b_5_6,362.0.
w_5_6,452.0.
f_6_6,194.0.
b_6_6,394.0.
w_6_6,534.0.
f_7_6,206.0.
b_7_6,432.0.
w_7_6,560.0.
f_8_6,238.0.
b_8_6,458.0.
w_8_6,566.0.
f_9_6,270.0.
b_9_6,514.0.
w_9_6,720.0.
f_10_6,320.0.
b_10_6,540.0.
w_10_6,758.0.
f_11_6,332.0.
b_11_6,572.0.
w_11_6,808.0.
f_12_6,344.0.
b_12_6,604.0.
w_12_6,846.0.
f_13_6,382.0.
b_13_6,636.0.
w_13_6,884.0.
f_14_6,414.0.
b_14_6,668.0.
w_14_6,924.0.
f_15_6,478.0.
b_15_6,700.0.
w_15_6,950.0.
f_16_6,490.0.
b_16_6,738.0.
w_16_6,1020.0.
f_17_6,502.0.
b_17_6,776.0.
w_17_6,1026.0.
f_18_6,592.0.
b_18_6,814.0.
w_18_6,1088.0.
f_19_6,624.0.
b_19_6,864.0.
w_19_6,1094.0.
f_20_6,688.0.
b_20_6,902.0.
w_20_6,1164.0.
f_21_6,726.0.
b_21_6,930.0.
w_21_6,1248.0.
f_22_6,764.0.
b_22_6,956.0.
w_22_6,1294.0000678573188.
f_23_6,796.0.
b_23_6,988.0.
w_23_6,1300.00006785732.
f_24_6,834.0.
b_24_6,1032.0.
w_24_6,1306.0000678573213.
f_25_6,852.0.
b_25_6,1064.0.
w_25_6,1312.0000678573222.
f_26_6,890.0.
b_26_6,1112.0.
w_26_6,1318.0000678573233.
f_27_6,976.0.
b_27_6,1144.0.
w_27_6,1324.0000678573242.
f_28_6,1008.0.
b_28_6,1170.0.
w_28_6,1330.000067857325.
f_29_6,1052.0.
b_29_6,1190.0.
w_29_6,1340.0.
f_30_6,1100.0.
b_30_6,1228.0.
w_30_6,1346.0.
f_31_6,1132.0.
b_31_6,1260.0.
w_31_6,1406.0.
f_0_7,91.0.
b_0_7,103.0.
w_0_7,179.0.
f_1_7,123.0.
b_1_7,147.0.
w_1_7,281.0.
f_2_7,135.0.
b_2_7,197.0.
w_2_7,287.0.
f_3_7,167.0.
b_3_7,229.0.
w_3_7,337.0.
f_4_7,185.0.
b_4_7,249.0.
w_4_7,451.0.
f_5_7,217.0.
b_5_7,305.0.
w_5_7,469.0.
f_6_7,269.0.
b_6_7,355.0.
w_6_7,591.0.
f_7_7,293.0.
b_7_7,399.0.
w_7_7,597.0.
f_8_7,325.0.
b_8_7,419.0.
w_8_7,603.0.
f_9_7,343.0.
b_9_7,487.0.
w_9_7,609.0.
f_10_7,375.0.
b_10_7,507.0.
w_10_7,699.0.
f_11_7,387.0.
b_11_7,551.0.
w_11_7,845.0.
f_12_7,439.0.
b_12_7,571.0.
w_12_7,883.0.
f_13_7,457.0.
b_13_7,615.0.
w_13_7,973.0.
f_14_7,475.0.
b_14_7,647.0.
w_14_7,999.0.
f_15_7,527.0.
b_15_7,667.0.
w_15_7,1005.0.
f_16_7,539.0.
b_16_7,705.0.
w_16_7,1075.0.
f_17_7,635.0.
b_17_7,737.0.
w_17_7,1081.0.
f_18_7,687.0.
b_18_7,793.0.
w_18_7,1119.0.
f_19_7,725.0.
b_19_7,825.0.
w_19_7,1125.0.
f_20_7,757.0.
b_20_7,851.0.
w_20_7,1151.0.
f_21_7,769.0.
b_21_7,889.0.
w_21_7,1201.0.
f_22_7,781.0.
b_22_7,921.0.
w_22_7,1227.0.
f_23_7,813.0.
b_23_7,941.0.
w_23_7,1233.0.
f_24_7,871.0.
b_24_7,979.0.
w_24_7,1273.0000678573197.
f_25_7,909.0.
b_25_7,1023.0.
w_25_7,1288.0.
f_26_7,961.0.
b_26_7,1043.0.
w_26_7,1294.0.
f_27_7,1011.0.
b_27_7,1099.0.
w_27_7,1300.0.
f_28_7,1063.0.
b_28_7,1131.0.
w_28_7,1306.0.
f_29_7,1087.0.
b_29_7,1157.0.
w_29_7,1312.0.
f_30_7,1177.0.
b_30_7,1207.0.
w_30_7,1338.0.
f_31_7,1189.0.
b_31_7,1239.0.
w_31_7,1406.0.
MinExeTime:1412.0.'''
    device_steps = order_result_J(inputstr,stage_alignment)
    #device_steps = drop_w(device_steps)
    debugeprint(device_steps)
    for s in device_steps:
        debugeprint(len(s))
    local_rank = setup_distributed()
    async_communicator_recv_backward_queue  = queue.Queue()
    async_communicator_recv_forward_queue  = queue.Queue()
    
    input_objs = queue.Queue()
    output_objs = queue.Queue()

    tensor_shape = torch.Size([1, 100, 100])
    dtype = torch.float16
    forward_recv_shapes = tensor_shape
    scatter_gather_tensors = False

    backward_recv_shapes = None
    need_forward_meta = tensor_shape is None
    
    steps = device_steps[local_rank]
    last_stage = len(device_steps)-1
    prev_rank = local_rank - 1
    next_rank = local_rank + 1
    send_forward_start_time = 0
    recv_forward_start_time = 0
    recv_backward_start_time = 0
    send_backward_start_time = 0
    write_to_file(f'./shared_mem/rank{local_rank}_sendforward', 0)
    write_to_file(f'./shared_mem/rank{local_rank}_sendbackward', 0)
    for s in range(len(steps)):
        step_type, microbatch_id, stage_id = steps[s]
        if step_type == Stage.FORWARD.value:# Forward pass
            # Receive the input from the previous stage
            debugeprint(f"step:{step_type}, microbatch_id:{microbatch_id}, stage_id:{stage_id} _forward_step_begin")
            if stage_id>0:
                if async_communicator_recv_forward_queue.qsize()>0:
                    input_obj,_ = async_communicator_recv_forward_queue.get().wait_and_receive()
                    debugeprint(f"step:{step_type}, microbatch_id:{microbatch_id}, stage_id:{stage_id} async_communicator_recv_forward_queue.get().wait_and_receive()")
                else:
                    if forward_recv_shapes is None:
                        forward_recv_shapes = comm.recv_obj_meta(prev_rank)
                        debugeprint(f'step:{step_type}, microbatch_id:{microbatch_id}, stage_id:{stage_id} forward_recv_shapes:{forward_recv_shapes}')
                    async_communicator_recv_forward = comm.AsynCommunicator(
                        recv_prev_shape = forward_recv_shapes,
                        dtype=dtype,
                        scatter_gather_tensors=scatter_gather_tensors,
                        prev_rank=prev_rank,
                        next_rank=next_rank,
                    )
                    async_communicator_recv_forward.start()
                    recv_forward_start_time += 1
                    debugeprint(f"step:{step_type}, microbatch_id:{microbatch_id}, stage_id:{stage_id} recv_forward.start() {recv_forward_start_time}")
                    input_obj,_  = async_communicator_recv_forward.wait_and_receive()
                debugeprint(f"step:{step_type}, microbatch_id:{microbatch_id}, stage_id:{stage_id} input_obj_shape:{input_obj.shape}")
            else:
                input_obj = None

            recv_forward_start_time, recv_backward_start_time, \
                async_communicator_recv_forward_queue, async_communicator_recv_backward_queue = recvAll(
                    step_type, microbatch_id, stage_id, dtype, scatter_gather_tensors, last_stage,
                    prev_rank,recv_forward_start_time, forward_recv_shapes, async_communicator_recv_forward_queue,
                    next_rank, recv_backward_start_time, backward_recv_shapes, async_communicator_recv_backward_queue)
            
            # Perform forward computation
            output_obj = do_compute()
            debugeprint(f"step:{step_type}, microbatch_id:{microbatch_id}, stage_id:{stage_id} forward_step output_obj_shape:{output_obj.shape}")
            
            if stage_id < last_stage:
                if isinstance(output_obj, torch.Tensor):
                    backward_recv_shapes = output_obj.shape
                else:
                    backward_recv_shapes = [out_tensor.shape for out_tensor in output_obj]    
                if need_forward_meta:
                    comm.send_obj_meta(output_obj, next_rank)
                    need_forward_meta = False  # send only once.
            debugeprint(f"step:{step_type}, microbatch_id:{microbatch_id}, stage_id:{stage_id} backward_recv_shapes:{backward_recv_shapes}")
            
            # Send the output of forward computation of this pipeline stage to the next pipeline stage as input for
            # forward computation
            if stage_id < last_stage:
                async_communicator_send_forward = comm.AsynCommunicator(
                    object_send_next=output_obj,
                    dtype=dtype,
                    scatter_gather_tensors=scatter_gather_tensors,
                    prev_rank=prev_rank,
                    next_rank=next_rank,
                )
                send_forward_start_time += 1
                write_to_file(f'./shared_mem/rank{local_rank}_sendforward', send_forward_start_time)
                async_communicator_send_forward.start()
                debugeprint(f"step:{step_type}, microbatch_id:{microbatch_id}, stage_id:{stage_id} send_forward.start() {send_forward_start_time}")

            input_objs.put(input_obj)
            output_objs.put(output_obj)

        elif step_type == Stage.BACKWARD.value:# Backward pass
            debugeprint(f"step:{step_type}, microbatch_id:{microbatch_id}, stage_id:{stage_id} _backward_step_begin")
            input_obj = input_objs.get()
            output_obj = output_objs.get()
            
            if stage_id<last_stage:
                if async_communicator_recv_backward_queue.qsize()>0:
                    _, output_obj_grad = async_communicator_recv_backward_queue.get().wait_and_receive()

                    debugeprint(f"step:{step_type}, microbatch_id:{microbatch_id}, stage_id:{stage_id} async_communicator_recv_backward_queue.get().wait_and_receive()")
                else:
                    async_communicator_recv_backward = comm.AsynCommunicator(
                        recv_next_shape = backward_recv_shapes,
                        dtype=dtype,
                        scatter_gather_tensors=scatter_gather_tensors,
                        prev_rank=prev_rank,
                        next_rank=next_rank,
                    )
                    async_communicator_recv_backward.start()
                    recv_backward_start_time += 1
                    _, output_obj_grad = async_communicator_recv_backward.wait_and_receive()
                    debugeprint(f"step:{step_type}, microbatch_id:{microbatch_id}, stage_id:{stage_id} async_communicator_recv_backward")
                debugeprint(f"step:{step_type}, microbatch_id:{microbatch_id}, stage_id:{stage_id} output_obj_grad_shape:{output_obj_grad.shape}")
            else:
                output_obj_grad = None

            recv_forward_start_time, recv_backward_start_time, \
                async_communicator_recv_forward_queue, async_communicator_recv_backward_queue = recvAll(
                    step_type, microbatch_id, stage_id, dtype, scatter_gather_tensors, last_stage,
                    prev_rank,recv_forward_start_time, forward_recv_shapes, async_communicator_recv_forward_queue,
                    next_rank, recv_backward_start_time, backward_recv_shapes, async_communicator_recv_backward_queue)
            
            input_obj_grad = do_compute()
            debugeprint(f"step:{step_type}, microbatch_id:{microbatch_id}, stage_id:{stage_id} _backward_step ")

            if stage_id>0:
                async_communicator_send_backward = comm.AsynCommunicator(
                        object_send_prev=input_obj_grad,
                        dtype=dtype,
                        scatter_gather_tensors=scatter_gather_tensors,
                        prev_rank=prev_rank,
                        next_rank=next_rank,
                    )
                send_backward_start_time += 1
                write_to_file(f'./shared_mem/rank{local_rank}_sendbackward', send_backward_start_time)
                async_communicator_send_backward.start()
                debugeprint(f"step:{step_type}, microbatch_id:{microbatch_id}, stage_id:{stage_id} async_communicator_send_backward")
    write_to_file(f'./shared_mem/rank{local_rank}_sendforward', 0)
    write_to_file(f'./shared_mem/rank{local_rank}_sendbackward', 0)        
    dist.barrier() 
if __name__ == "__main__": 
    main()