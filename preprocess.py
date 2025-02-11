def distence(point,begin,end):
    if point < begin:
        return begin - point
    elif point > end:
        return point - end
    else:
        return 0
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
def recvFromSameDevice(stage_alignment):
    recvFfromSameDevice = []
    recvBfromSameDevice = []
    for stages in stage_alignment:
        for stage in stages:
            if stage-1 in stages:
                recvFfromSameDevice.append(stage)
            if stage+1 in stages:
                recvBfromSameDevice.append(stage)
    return recvFfromSameDevice,recvBfromSameDevice
def SendToSameDevice(stage_alignment):
    sendFtoSameDevice = []
    sendBtoSameDevice = []
    for stages in stage_alignment:
        for stage in stages:
            if stage+1 in stages:
                sendFtoSameDevice.append(stage)
            if stage-1 in stages:
                sendBtoSameDevice.append(stage)
    return sendFtoSameDevice,sendBtoSameDevice
def order_result_mutichunk(input: str, stage_alignment: list) -> None:
    # input = input.replace("", "")
    # input = input.replace(" ", "")
    device_steps = [[] for _ in range(len(stage_alignment))]
    all_step = input.split('\n')
    #print(all_step)
    for step in all_step:
        if step == '':
            continue
        start_time = int(step.split(',')[-2])
        end_time = int(step.split(',')[-1])
        infor = step.split(',')[0]
        if infor is None or infor == '' or infor[0] not in ['b','w','f']:
            continue
        step_type, microbatch_id, stage_id = infor.split('_')
        microbatch_id = int(microbatch_id)
        stage_id = int(stage_id)
        start_time = int(start_time)
        device_id = _get_deviceid_by_alignment(stage_id, stage_alignment)
        chunk_id = _get_chunk_by_stage(stage_id, stage_alignment)
        device_steps[device_id].append((step_type, microbatch_id, stage_id, chunk_id, start_time, end_time))
    print('[')
    for d in range(len(stage_alignment)):
        device_steps[d].sort(key=lambda x: x[-2])
        print(f'{device_steps[d]},')
    print(']')
    #result = []
    # for stage_idx, stage_list in enumerate(device_steps):
    #     new_stage_list = []
    #     for current_tuple in stage_list:
    #         op, mb_id, stage_id, chunk_id, start_time = current_tuple
    #         new_stage_list.append((op, mb_id, stage_id))
    #     result.append(new_stage_list)
    return device_steps
def comm_graph_muti_chunk(grouped_data, stage_alignment):   # 假设 grouped_data 是之前生成的计算图
    # 初始化通信图
    communication_graph = []
    max_stage_id = max([stage_id for _, stage_id in stage_alignment])

    # import pdb;pdb.set_trace()
    # 遍历每个 stage
    for device_id, stage_ops in enumerate(grouped_data):

        stages = stage_alignment[device_id]
        needrecv = {}
        needrecv['F_stage'] = [s-1 for s in stages if s > 0 and s-1 not in stages]
        needrecv['F_device'] = [_get_deviceid_by_alignment(s,stage_alignment) for s in needrecv['F_stage']]
        needrecv['B_stage'] = [s+1 for s in stages if s < max_stage_id and s+1 not in stages]
        needrecv['B_device'] = [_get_deviceid_by_alignment(s,stage_alignment) for s in needrecv['B_stage']]
        print(needrecv)
        communication_stage = []
        # 标记已经接收的操作
        received_prev_stage = set()  # 记录每个 stage 中已经接收的 f 操作
        received_next_stage = set()  # 记录每个 stage 中已经接收的 b 操作
        
        for m, current_op in enumerate(stage_ops):
            op, microbatch_id, stage_id, chunk_id, start_time, end_time = current_op
            comm_op = {}
            comm_op['Infor'] = (op, stage_id, microbatch_id)
            comm_op['B'] = []
            comm_op['A'] = []
            # if op != 'f' and op !='b':
            #     communication_stage.append(comm_op)
            #     continue
            
            # 处理上一个 stage (stage-1)
            for i in range(len(needrecv['F_stage'])):
                recvFstage_id = needrecv['F_stage'][i]
                recvFdevice_id = needrecv['F_device'][i]
                prev_stage_ops = grouped_data[recvFdevice_id]
                for n, prev_op in enumerate(prev_stage_ops):
                    prev_op_name, prev_microbatch_id, prev_stage_id, prev_chunk_id, prev_start_time, prev_end_time = prev_op
                    # if prev_start_time > end_time:
                    #     break
                    # 跳过已经接收的 f 操作
                    if prev_op in received_prev_stage:
                        continue
                    if prev_op_name != 'f' or prev_stage_id != recvFstage_id:
                        continue
                    # 如果本次操作为f 且microbatch_id 相同
                    if op == 'f' and prev_microbatch_id == microbatch_id and prev_stage_id == stage_id-1:           
                        comm_op['B'].append(('f', prev_end_time, recvFdevice_id, prev_stage_id,prev_chunk_id, prev_microbatch_id, n, 0))
                        received_prev_stage.add(prev_op)  # 标记为已接收
                        continue

                    # 计算时间区间
                    interval_start = prev_end_time
                    interval_end = prev_stage_ops[n + 1][-2] if n + 1 < len(prev_stage_ops) else prev_end_time

                    # 计算四个个时间点与区间的距离
                    current_start_dist = distence(start_time, interval_start, interval_end)
                    current_end_dist = distence(end_time, interval_start, interval_end)
                    next_start_dist = distence((stage_ops[m + 1][-2] if m + 1 < len(stage_ops) else end_time), interval_start, interval_end)
                    next_end_dist = distence((stage_ops[m + 1][-1] if m + 1 < len(stage_ops) else end_time), interval_start, interval_end)
                    # 找到最小距离
                    min_dist = min(current_start_dist, current_end_dist, next_start_dist,next_end_dist)
                    #这里将这个判断提前，为了防止出现两个操作各自的结束和开始在同一个时间点的情况，尽量让交给下一个操作前接收

                    if min_dist == current_start_dist:
                        comm_op['B'].append(('f', prev_end_time, recvFdevice_id, prev_stage_id,prev_chunk_id, prev_microbatch_id, n, 0))
                        received_prev_stage.add(prev_op)  # 标记为已接收
                        continue
                    elif min_dist == current_end_dist:
                        comm_op['A'].append(('f',prev_end_time, recvFdevice_id, prev_stage_id,prev_chunk_id, prev_microbatch_id, n, 0))
                        received_prev_stage.add(prev_op)  # 标记为已接收
                        continue
                    elif min_dist == next_start_dist:
                        break
                    elif min_dist == next_end_dist:
                        break

            # 处理下一个 stage (stage+1)
            for j in range(len(needrecv['B_stage'])):
                recvBstage_id = needrecv['B_stage'][j]
                recvBdevice_id = needrecv['B_device'][j]
                next_stage_ops = grouped_data[recvBdevice_id]

                for n, next_op in enumerate(next_stage_ops):
                    next_op_name, next_microbatch_id, next_stage_id, next_chunk_id, next_start_time, next_end_time = next_op
                    # if next_start_time > end_time:
                    #     break
                    if next_op_name != 'b' or recvBstage_id != next_stage_id:
                        continue
                    # 跳过已经接收的 f 操作
                    if next_op in received_next_stage :
                        continue
                    # 如果本次操作为b 且 microbatch_id 相同
                    if op == 'b' and next_microbatch_id == microbatch_id and next_stage_id == stage_id+1:
                        comm_op['B'].append(('b', next_end_time, recvBdevice_id, next_stage_id, next_chunk_id, next_microbatch_id,n,0))
                        received_next_stage.add(next_op)  # 标记为已接收
                        continue

                    # 计算时间区间
                    interval_start = next_end_time
                    interval_end = next_stage_ops[n + 1][-2] if n + 1 < len(next_stage_ops) else next_end_time

                    # 计算四个时间点与区间的距离
                    current_start_dist = distence(start_time, interval_start, interval_end)
                    current_end_dist = distence(end_time, interval_start, interval_end)
                    next_start_dist = distence((stage_ops[m + 1][-2] if m + 1 < len(stage_ops) else end_time), interval_start, interval_end)
                    next_end_dist = distence((stage_ops[m + 1][-1] if m + 1 < len(stage_ops) else end_time), interval_start, interval_end)
                    # 找到最小距离
                    min_dist = min(current_start_dist, current_end_dist, next_start_dist, next_end_dist)
                    #这里将这个判断提前，为了防止出现两个操作各自的结束和开始在同一个时间点的情况，尽量让交给下一个操作前接收
                    
                    if min_dist == current_start_dist:
                        comm_op['B'].append(('b', next_end_time, recvBdevice_id, next_stage_id, next_chunk_id, next_microbatch_id,n,0))
                        received_next_stage.add(next_op)
                        continue
                    elif min_dist == current_end_dist:
                        comm_op['A'].append(('b', next_end_time, recvBdevice_id, next_stage_id, next_chunk_id, next_microbatch_id,n,0))
                        received_next_stage.add(next_op)
                        continue
                    elif min_dist == next_start_dist:
                        break
                    elif min_dist == next_end_dist:
                        break
            comm_op['B'].sort(key=lambda x:x[1])
            comm_op['A'].sort(key=lambda x:x[1])

            # 将通信元组添加到当前 stage 的通信图中
            communication_stage.append(comm_op)
        # 将当前 stage 的通信图添加到总的通信图中
        communication_graph.append(communication_stage)
        #print(communication_stage)
    communication_graph = detect_deadlock_mutichunk(communication_graph,stage_alignment)
    print('[')
    # # 输出通信图
    for rank_id, comm_stage in enumerate(communication_graph):
        recvF = 0
        recvB = 0
        for comm_op in comm_stage:
            for recvlistB in comm_op['B']:
                if recvlistB[0] == 'b':
                    recvB += 1
                elif recvlistB[0] == 'f':
                    recvF += 1
            for recvlistA in comm_op['A']:
                if recvlistA[0] == 'b':
                    recvB += 1
                elif recvlistA[0] == 'f':
                    recvF += 1
        #print(f"rank_id {rank_id}: recvF {recvF}, recvB {recvB}")
        print(f'{comm_stage},')
    print(']')
    #print(communication_graph)
def detect_deadlock_mutichunk(communication_graph, stage_alignment):
    sendFtoSameDevice,sendBtoSameDevice = SendToSameDevice(stage_alignment)
    max_stage_id = max([stage_id for _, stage_id in stage_alignment])
    for rank_id, rank_ops in enumerate(communication_graph):
        len_rank_ops = len(rank_ops)
        for current_index, op in enumerate(rank_ops):
            op_type, stage_id, microbatch_id = op['Infor']
            #没有发送需求就不会有死锁
            if op_type == 'f':
                if stage_id in sendFtoSameDevice or stage_id == max_stage_id:
                    continue
                dst_rank_id = _get_deviceid_by_alignment(stage_id+1,stage_alignment)
            elif op_type == 'b':
                if stage_id in sendBtoSameDevice or stage_id == 0:
                    continue
                dst_rank_id = _get_deviceid_by_alignment(stage_id-1,stage_alignment)
            else:
                continue
            #检查要发送的操作
            if current_index < len_rank_ops - 1:
                next_op = rank_ops[current_index + 1]
                for next_op_before_recv_index,next_op_before_recv in enumerate(next_op['B']):
                    next_op_before_recv_deviceid = next_op_before_recv[2]
                    if next_op_before_recv_deviceid == dst_rank_id:
                        next_op_before_recv_indexInRecvDevice = next_op_before_recv[-2]
                        if next_op_before_recv_indexInRecvDevice+1<len_rank_ops:
                            dst_rank_id_next_op = communication_graph[dst_rank_id][next_op_before_recv_indexInRecvDevice+1]
                            for dst_rank_id_next_op_before_recv_index,dst_rank_id_next_op_before_recv in enumerate(dst_rank_id_next_op['B']):
                                if (dst_rank_id_next_op_before_recv[0],dst_rank_id_next_op_before_recv[3],dst_rank_id_next_op_before_recv[-3])== op['Infor']:
                                    op['A'].append(next_op_before_recv)
                                    next_op['B'].pop(next_op_before_recv_index)
                                    communication_graph[dst_rank_id][next_op_before_recv_indexInRecvDevice]['A'].append(dst_rank_id_next_op_before_recv)
                                    dst_rank_id_next_op['B'].pop(dst_rank_id_next_op_before_recv_index)
            # 检查计算后接收操作,只判断f即可？
            for i in range(len(op['A'])):
                recv_op_type, recv_end_time, recv_device_id, recv_stage_id, recv_chunk_id, recv_microbatch_id, index,_ = op['A'][i]
                if rank_id == recv_device_id or recv_device_id != dst_rank_id:
                    continue
                next_op = communication_graph[recv_device_id][index]
                for j in range(len(next_op['A'])):
                    next_recv_op_type, next_recv_end_time, next_recv_device_id, next_recv_stage_id, next_recv_chunk_id, next_recv_microbatch_id, next_index,_ = next_op['A'][j]
                    if (next_recv_op_type,next_recv_stage_id,next_recv_microbatch_id) == op['Infor']:
                        op['A'][i] = (recv_op_type, recv_end_time,recv_device_id, recv_stage_id, recv_chunk_id, recv_microbatch_id, index, 1)
                        next_op['A'][j] = (next_recv_op_type, next_recv_end_time, next_recv_device_id, next_recv_stage_id, next_recv_chunk_id, next_recv_microbatch_id, next_index, 1)
                for x in range(len(next_op['B'])):
                    next_recv_op_type, next_recv_end_time, next_recv_device_id, next_recv_stage_id, next_recv_chunk_id, next_recv_microbatch_id, next_index,_ = next_op['B'][x]
                    if (next_recv_op_type,next_recv_stage_id,next_recv_microbatch_id) == op['Infor']:
                         if current_index < len_rank_ops - 1:
                            rank_ops[current_index + 1]['B'].append(op['A'].pop(i))


    return communication_graph

def order_result(input_str):
    data = input_str.replace(" ","").split('\n')
    #print(data)

    # 创建一个字典来存储合并后的元组
    merged_data = {}

    # 解析数据并合并
    for entry in data:
        if entry == '':
            continue
        parts = entry.split(',')
        print(parts)
        # time = float(parts[1])
        # op_info = parts[0].split('_')
        # op = op_info[0]
        # microbatch_id = int(op_info[1])
        # stage_id = int(op_info[2])
        
        # if 'e' in parts[0]:
        #     key = (op, microbatch_id, stage_id)
        #     if key in merged_data:
        #         merged_data[key] = (merged_data[key][0], merged_data[key][1], merged_data[key][2], merged_data[key][3], time)
        # else:
        #     key = (op, microbatch_id, stage_id)
        #     merged_data[key] = (op, microbatch_id, stage_id, time, None)
        start_time = float(parts[1])
        end_time = float(parts[2])
        op_info = parts[0].split('_')
        op = op_info[0]
        microbatch_id = int(op_info[1])
        stage_id = int(op_info[2])
        key = (op, microbatch_id, stage_id)
        merged_data[key] = (op, microbatch_id, stage_id, start_time, end_time)
    # 按照stage_id分组
    grouped_data = {}
    for key, value in merged_data.items():
        stage_id = key[2]
        if stage_id not in grouped_data:
            grouped_data[stage_id] = []
        grouped_data[stage_id].append(value)

    # 对每个分组按照开始时间进行排序
    for stage_id in grouped_data:
        grouped_data[stage_id].sort(key=lambda x: x[3])

    # 将结果转换为二维列表
    result = [grouped_data[stage_id] for stage_id in sorted(grouped_data.keys())]
    return result
def detect_deadlock(communication_graph):
    for rank_id, rank_ops in enumerate(communication_graph):
        for op in rank_ops:
            op_type, microbatch_id = op['Infor']
            if op_type == 'f':
                for i in range(len(op['A'])):
                    recv_op_type, recv_end_time, recv_microbatch_id,_ = op['A'][i]
                    if recv_op_type == 'b':
                        # 检查下一个 rank 中对应的 b 操作
                        next_rank_id = rank_id + 1
                        if next_rank_id < len(communication_graph):
                            for next_op in communication_graph[next_rank_id]:
                                next_op_type, next_microbatch_id = next_op['Infor']
                                if next_op_type == 'b' and next_microbatch_id == recv_microbatch_id:
                                    for j in range(len(next_op['A'])):
                                        next_recv_op_type, next_recv_end_time, next_recv_microbatch_id,_ = next_op['A'][j]
                                        if next_recv_op_type == 'f' and next_recv_microbatch_id == microbatch_id:
                                            # 发现死锁
                                            print(f"Deadlock detected: Rank {rank_id} operation {op['Infor']} and Rank {next_rank_id} operation {next_op['Infor']} form a cycle.")
                                            # 在原有的 A 列表中增加一个判断标志
                                            op['A'][i] = (recv_op_type, recv_end_time, recv_microbatch_id, 1)
                                            next_op['A'][j] = (next_recv_op_type, next_recv_end_time, next_recv_microbatch_id, 1)
    return communication_graph
def comm_graph(grouped_data):   # 假设 grouped_data 是之前生成的计算图
    # 初始化通信图
    communication_graph = []

    # 标记已经接收的操作
    received_prev_stage = [set() for _ in grouped_data]  # 记录每个 stage 中已经接收的 f 操作
    received_next_stage = [set() for _ in grouped_data]  # 记录每个 stage 中已经接收的 b 操作
    # import pdb;pdb.set_trace()
    # 遍历每个 stage
    for stage_id, stage_ops in enumerate(grouped_data):

        communication_stage = []

        for m, current_op in enumerate(stage_ops):
            op, microbatch_id, current_stage_id, start_time, end_time = current_op
            comm_op = {}
            comm_op['Infor'] = (op, microbatch_id)
            comm_op['B'] = []
            comm_op['A'] = []
            # if op != 'f' and op !='b':
            #     communication_stage.append(comm_op)
            #     continue

            # 初始化通信元组
            
            # 处理上一个 stage (stage-1)
            if stage_id > 0:
                prev_stage_ops = grouped_data[stage_id - 1]
                for n, prev_op in enumerate(prev_stage_ops):
                    prev_op_name, prev_microbatch_id, prev_stage_id, prev_start_time, prev_end_time = prev_op
                    if prev_start_time > end_time:
                        break
                    if prev_op_name != 'f':
                        continue
                    # 跳过已经接收的 f 操作
                    if n in received_prev_stage[stage_id - 1]:
                        continue
                    # 如果本次操作为f 且microbatch_id 相同
                    if op == 'f' and prev_microbatch_id == microbatch_id:           
                        comm_op['B'].append(('f',prev_end_time,prev_microbatch_id, 0))
                        received_prev_stage[stage_id - 1].add(n)  # 标记为已接收
                        continue

                    # 计算时间区间
                    interval_start = prev_end_time
                    interval_end = prev_stage_ops[n + 1][3] if n + 1 < len(prev_stage_ops) else prev_end_time

                    # 计算三个时间点与区间的距离
                    current_start_dist = distence(start_time, interval_start, interval_end)
                    current_end_dist = distence(end_time, interval_start, interval_end)
                    next_start_dist = distence((stage_ops[m + 1][3] if m + 1 < len(stage_ops) else end_time), interval_start, interval_end)
                    next_end_dist = distence((stage_ops[m + 1][4] if m + 1 < len(stage_ops) else end_time), interval_start, interval_end)
                    # 找到最小距离
                    min_dist = min(current_start_dist, current_end_dist, next_start_dist,next_end_dist)

                    if min_dist == current_start_dist:
                        comm_op['B'].append(('f',prev_end_time, prev_microbatch_id,0))
                        received_prev_stage[stage_id - 1].add(n)  # 标记为已接收
                        continue
                    elif min_dist == current_end_dist:
                        comm_op['A'].append(('f',prev_end_time, prev_microbatch_id,0))
                        received_prev_stage[stage_id - 1].add(n)  # 标记为已接收
                        continue
                    elif min_dist == next_start_dist:
                        break
                    elif min_dist == next_end_dist:
                        break

            # 处理下一个 stage (stage+1)
            if stage_id < len(grouped_data) - 1:
                next_stage_ops = grouped_data[stage_id + 1]
                for n, next_op in enumerate(next_stage_ops):
                    next_op_name, next_microbatch_id, next_stage_id, next_start_time, next_end_time = next_op
                    if next_start_time > end_time:
                        break
                    if next_op_name != 'b':
                        continue
                    # 跳过已经接收的 f 操作
                    if n in received_next_stage :
                        continue

                    # 如果本次操作为b 且 microbatch_id 相同
                    if op == 'b' and next_microbatch_id == microbatch_id:
                        comm_op['B'].append(('b', next_end_time, next_microbatch_id,0))
                        received_next_stage.add(next_op)  # 标记为已接收
                        continue

                    # 计算时间区间
                    interval_start = next_end_time
                    interval_end = next_stage_ops[n + 1][3] if n + 1 < len(next_stage_ops) else next_end_time

                    # 计算三个时间点与区间的距离
                    current_start_dist = distence(start_time, interval_start, interval_end)
                    current_end_dist = distence(end_time, interval_start, interval_end)
                    next_start_dist = distence((stage_ops[m + 1][3] if m + 1 < len(stage_ops) else end_time), interval_start, interval_end)
                    next_end_dist = distence((stage_ops[m + 1][4] if m + 1 < len(stage_ops) else end_time), interval_start, interval_end)
                    # 找到最小距离
                    min_dist = min(current_start_dist, current_end_dist, next_start_dist, next_end_dist)

                    if min_dist == current_start_dist:
                        comm_op['B'].append(('b', next_end_time,next_microbatch_id,0))
                        received_next_stage.add(next_op)
                        continue
                    elif min_dist == current_end_dist:
                        comm_op['A'].append(('b', next_end_time, next_microbatch_id,0))
                        received_next_stage.add(next_op)
                        continue
                    elif min_dist == next_start_dist:
                        break
                    elif min_dist == next_end_dist:
                        break
            comm_op['B'].sort(key=lambda x:x[1])
            comm_op['A'].sort(key=lambda x:x[1])

            # 将通信元组添加到当前 stage 的通信图中
            communication_stage.append(comm_op)

        # 将当前 stage 的通信图添加到总的通信图中
        communication_graph.append(communication_stage)
        #print(communication_stage)
    communication_graph = detect_deadlock(communication_graph)
    print(communication_graph)
    # # 输出通信图
    # for stage_id, comm_stage in enumerate(communication_graph):
    #     print(f"Stage {stage_id}: {comm_stage}")

if __name__ == '__main__':
    stage_alignment = [[0,7],[1,6],[2,5],[3,4]]#[[0,4],[1,5],[2,6],[3,7]]#
    # 输入字符串
    input_str = """f_0_0,0,13
f_1_0,13,26
f_0_1,13,25
f_0_2,25,37
f_2_0,26,39
f_1_1,26,38
f_0_3,37,49
f_1_2,38,50
f_3_0,39,52
f_2_1,39,51
f_0_4,49,61
f_2_2,51,63
f_4_0,52,65
f_3_1,52,64
f_1_3,61,73
f_0_5,63,75
f_5_0,65,78
f_4_1,65,77
f_1_4,73,85
f_3_2,75,87
f_0_6,77,89
f_6_0,78,91
f_2_3,85,97
f_1_5,87,99
f_5_1,89,101
f_0_7,91,112
f_2_4,97,109
f_4_2,99,111
f_1_6,101,113
f_3_3,109,121
f_2_5,111,123
b_0_7,112,147
f_6_1,113,125
f_3_4,121,133
f_5_2,123,135
f_2_6,125,137
f_4_3,133,145
f_3_5,135,147
f_4_4,145,157
w_0_7,147,158
b_0_6,147,167
f_6_2,147,159
f_5_3,157,169
f_1_7,158,179
f_4_5,159,171
w_0_6,167,175
f_5_4,169,181
b_0_5,171,191
f_3_6,175,187
b_1_7,179,214
f_6_3,181,193
f_4_6,187,199
w_0_5,191,199
b_0_4,193,213
f_5_5,199,211
f_5_6,211,223
w_0_4,213,221
w_1_7,214,225
f_6_4,221,233
b_1_6,223,243
f_2_7,225,246
f_6_5,233,245
b_0_3,233,253
w_1_6,243,251
b_1_5,245,265
b_2_7,246,281
f_6_6,251,263
w_0_3,253,261
w_1_5,265,273
b_1_4,265,285
b_0_2,273,293
w_2_7,281,292
b_2_6,281,301
w_1_4,285,293
f_3_7,292,313
w_0_2,293,301
b_1_3,293,313
w_2_6,301,309
b_2_5,301,321
b_0_1,309,329
b_3_7,313,348
w_1_3,313,321
w_2_5,321,329
b_2_4,321,341
w_0_1,329,337
b_1_2,329,349
w_2_4,341,349
w_3_7,348,359
b_3_6,348,368
w_1_2,349,357
b_2_3,349,369
f_4_7,359,380
w_3_6,368,376
b_3_5,368,388
w_2_3,369,377
b_1_1,376,396
b_0_0,380,401
w_3_5,388,396
b_3_4,388,408
w_1_1,396,404
b_2_2,396,416
w_0_0,400,409
f_5_7,408,429
w_3_4,408,416
b_2_1,416,436
w_2_2,416,424
b_3_3,416,436
b_1_0,429,450
w_2_1,436,444
b_3_2,436,456
w_3_3,436,444
w_1_0,449,458
b_3_1,456,476
w_3_2,456,464
f_6_7,457,478
w_3_1,476,484
b_2_0,478,499
w_2_0,498,507
f_7_0,506,519
b_3_0,519,540
f_7_1,519,531
f_7_2,531,543
w_3_0,539,548
f_7_3,543,555
f_8_0,547,560
f_7_4,555,567
b_4_7,560,595
f_8_1,560,572
f_7_5,567,579
f_7_6,579,591
f_8_2,579,591
f_8_3,591,603
w_4_7,595,606
b_4_6,595,615
f_8_4,603,615
f_7_7,606,627
w_4_6,615,623
b_4_5,615,635
b_5_7,627,662
w_4_5,635,643
b_4_4,635,655
f_8_5,643,655
f_8_6,655,667
w_4_4,655,663
w_5_7,662,673
b_4_3,663,683
b_5_6,667,687
f_8_7,673,694
b_4_2,683,703
w_4_3,683,691
w_5_6,687,695
b_6_7,694,729
b_4_1,703,723
w_4_2,703,711
b_5_5,711,731
w_4_1,723,731
w_6_7,729,740
b_6_6,731,751
w_5_5,731,739
b_5_4,731,751
f_9_0,740,753
w_6_6,751,759
b_6_5,751,771
w_5_4,751,759
b_4_0,753,774
f_9_1,759,771
b_5_3,759,779
w_6_5,771,779
w_4_0,773,782
f_9_2,779,791
w_5_3,779,787
f_10_0,781,794
b_6_4,787,807
b_5_2,791,811
b_7_7,794,829
f_10_1,794,806
w_6_4,807,815
b_5_1,811,831
w_5_2,811,819
f_9_3,815,827
f_10_2,819,831
b_6_3,827,847
w_7_7,829,840
w_5_1,831,839
b_7_6,839,859
f_11_0,840,853
b_6_2,847,867
w_6_3,847,855
b_5_0,853,874
f_9_4,855,867
w_7_6,859,867
f_11_1,867,879
w_6_2,867,875
f_10_3,867,879
w_5_0,873,882
f_9_5,875,887
b_6_1,879,899
f_10_4,879,891
f_12_0,881,894
b_7_5,887,907
b_8_7,894,929
w_6_1,899,907
f_9_6,907,919
w_7_5,907,915
b_7_4,907,927
f_10_5,915,927
f_12_1,919,931
f_11_2,927,939
w_7_4,927,935
w_8_7,929,940
b_8_6,931,951
b_7_3,935,955
f_12_2,939,951
f_9_7,940,961
w_8_6,951,959
b_8_5,951,971
w_7_3,955,963
f_10_6,959,971
b_6_0,961,982
f_11_3,963,975
w_8_5,971,979
b_8_4,975,995
b_7_2,979,999
w_6_0,981,990
f_10_7,989,1010
w_8_4,995,1003
b_7_1,999,1019
w_7_2,999,1007
f_11_4,1003,1015
b_9_7,1010,1045
f_11_5,1015,1027
b_8_3,1015,1035
w_7_1,1019,1027
f_11_6,1027,1039
b_8_2,1035,1055
w_8_3,1035,1043
f_12_3,1043,1055
w_9_7,1045,1056
b_9_6,1045,1065
w_8_2,1055,1063
f_12_4,1055,1067
f_11_7,1056,1077
w_9_6,1065,1073
b_9_5,1065,1085
b_8_1,1073,1093
b_7_0,1077,1098
w_9_5,1085,1093
b_9_4,1085,1105
w_8_1,1093,1101
f_12_5,1093,1105
w_7_0,1097,1106
f_13_0,1105,1118
f_12_6,1105,1117
w_9_4,1105,1113
b_9_3,1113,1133
b_8_0,1118,1139
f_13_1,1118,1130
f_13_2,1130,1142
w_9_3,1133,1141
w_8_0,1138,1147
b_9_2,1142,1162
f_13_3,1142,1154
f_12_7,1146,1167
f_13_4,1154,1166
b_9_1,1162,1182
w_9_2,1162,1170
b_10_7,1167,1202
f_13_5,1170,1182
w_9_1,1182,1190
f_13_6,1190,1202
w_10_7,1202,1213
b_10_6,1202,1222
f_13_7,1213,1234
w_10_6,1222,1230
b_10_5,1222,1242
b_9_0,1234,1255
w_10_5,1242,1250
b_10_4,1242,1262
w_9_0,1254,1263
f_14_0,1262,1275
w_10_4,1262,1270
b_10_3,1270,1290
b_11_7,1275,1310
f_14_1,1275,1287
f_14_2,1287,1299
w_10_3,1290,1298
b_10_2,1299,1319
f_14_3,1299,1311
w_11_7,1310,1321
b_11_6,1310,1330
f_14_4,1311,1323
w_10_2,1319,1327
f_15_0,1321,1334
f_14_5,1327,1339
w_11_6,1330,1338
b_12_7,1334,1369
f_15_1,1338,1350
b_11_5,1339,1359
b_10_1,1350,1370
w_11_5,1359,1367
b_11_4,1359,1379
f_15_2,1367,1379
w_12_7,1369,1380
w_10_1,1370,1378
f_14_6,1378,1390
w_11_4,1379,1387
b_10_0,1380,1401
f_15_3,1387,1399
b_12_6,1390,1410
b_11_3,1399,1419
w_10_0,1400,1409
f_14_7,1408,1429
w_12_6,1410,1418
b_12_5,1410,1430
w_11_3,1419,1427
f_15_4,1427,1439
b_13_7,1429,1464
w_12_5,1430,1438
b_11_2,1438,1458
b_12_4,1439,1459
b_11_1,1458,1478
w_11_2,1458,1466
b_12_3,1459,1479
w_13_7,1464,1475
f_15_5,1466,1478
b_14_7,1475,1510
w_11_1,1478,1486
b_12_2,1479,1499
w_12_3,1479,1487
f_15_6,1486,1498
w_12_4,1487,1495
b_13_6,1498,1518
w_12_2,1499,1507
w_14_7,1510,1521
b_12_1,1518,1538
b_13_5,1518,1538
f_15_7,1521,1542
b_14_6,1538,1558
w_13_5,1538,1546
b_13_4,1538,1558
b_11_0,1542,1563
w_12_1,1558,1566
b_14_5,1558,1578
b_13_3,1558,1578
b_12_0,1562,1583
w_13_6,1566,1574
w_14_6,1574,1582
b_13_2,1578,1598
b_14_4,1578,1598
b_15_7,1582,1617
b_13_1,1598,1618
w_13_2,1598,1606
b_14_3,1598,1618
w_14_5,1606,1614
w_11_0,1617,1626
b_15_6,1618,1638
b_14_2,1618,1638
w_13_3,1618,1626
b_13_0,1625,1646
w_13_4,1626,1634
w_14_3,1634,1642
b_14_1,1638,1658
b_15_5,1638,1658
w_14_4,1642,1650
w_12_0,1645,1654
w_13_0,1653,1662
w_13_1,1658,1666
w_14_2,1658,1666
b_15_4,1658,1678
b_14_0,1661,1682
w_14_1,1666,1674
w_15_5,1666,1674
w_15_6,1674,1682
b_15_3,1678,1698
w_14_0,1681,1690
w_15_7,1689,1700
b_15_2,1698,1718
w_15_3,1698,1706
w_15_4,1706,1714
b_15_1,1718,1738
w_15_2,1718,1726
b_15_0,1738,1759
w_15_1,1738,1746
w_15_0,1758,1767
"""
    result = order_result_mutichunk(input_str,stage_alignment)
    comm_graph_muti_chunk(result,stage_alignment)



