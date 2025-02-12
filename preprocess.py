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



def dfs(op, rec_stack,visited,communication_graph):
    key = op['Infor']
    if key in rec_stack:
        return True, -1  # 发现环，返回 True 和无效的 index
    if key in visited:
        return False, -1  # 已经访问过，无需继续

    visited.add(key)
    rec_stack.add(key)

    for idx, a_task in enumerate(op['A']):
        _, _, recv_device_id, recv_stage_id, _, recv_microbatch_id, recv_index, _ = a_task
        # 找到对应的通信任务
        has_cycle, cycle_index = dfs(communication_graph[recv_device_id][recv_index],rec_stack, visited, communication_graph,)
        if has_cycle:
            # 如果发现环，返回 True 和当前任务的 index
            return True, idx
    rec_stack.remove(key)
    return False, -1  # 不存在环




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
            
            #判断环
            visited = set()  # 用于记录已经访问过的任务
            rec_stack = set()  # 用于记录当前递归栈中的任务
            has_cycle, cycle_index = dfs(op,rec_stack, visited,communication_graph,)
            if has_cycle and cycle_index != -1 and current_index < len_rank_ops - 1:
                rank_ops[current_index + 1]['B'].append(op['A'].pop(cycle_index))
                print(f"cycle dead lock:{op['Infor']}")

            #死锁场景1
            #判断本次操作op计算后需要接收op['A']，如果本次op要接收的 和本次op要发往的 在同一个设备上，则需要下一步判断
            for n in range(len(op['A'])):
                recv_op_type, recv_end_time, recv_device_id, recv_stage_id, recv_chunk_id, recv_microbatch_id, index,_ = op['A'][n]
                if rank_id == recv_device_id or recv_device_id != dst_rank_id:
                    continue
                next_op = communication_graph[recv_device_id][index]

                #判断要接收的这个操作 其计算前是否要接收 本次op的发送，如果需要则会死锁，因为所有step的执行逻辑均为 收-算-收-发，
                #解决方法：让本次op的下一个操作的计算前去接收
                for x in range(len(next_op['B'])):
                    next_recv_op_type, next_recv_end_time, next_recv_device_id, next_recv_stage_id, next_recv_chunk_id, next_recv_microbatch_id, next_index,_ = next_op['B'][x]
                    if (next_recv_op_type,next_recv_stage_id,next_recv_microbatch_id) == op['Infor']:
                         if current_index < len_rank_ops - 1:
                            rank_ops[current_index + 1]['B'].append(op['A'].pop(n))

            #死锁场景2
            #都要向对方发送，但双方都是下一个step计算前去收，需要将recv提前到本次计算后
            if current_index < len_rank_ops - 1:
                next_op = rank_ops[current_index + 1]
                for next_op_before_recv_index,next_op_before_recv in enumerate(next_op['B']):
                    next_op_before_recv_deviceid = next_op_before_recv[2]
                    if next_op_before_recv_deviceid == dst_rank_id:
                        next_op_before_recv_indexInRecvDevice = next_op_before_recv[-2]
                        if next_op_before_recv_indexInRecvDevice+1<len_rank_ops:
                            dst_rank_id_next_op = communication_graph[dst_rank_id][next_op_before_recv_indexInRecvDevice+1]
                            for dst_rank_id_next_op_before_recv_index,dst_rank_id_next_op_before_recv in enumerate(dst_rank_id_next_op['B']):
                                if (dst_rank_id_next_op_before_recv[0],dst_rank_id_next_op_before_recv[3],dst_rank_id_next_op_before_recv[-3]) == op['Infor']:
                                    op['A'].append(next_op['B'].pop(next_op_before_recv_index))
                                    communication_graph[dst_rank_id][next_op_before_recv_indexInRecvDevice]['A'].append(dst_rank_id_next_op['B'].pop(dst_rank_id_next_op_before_recv_index))
            #死锁场景3
            #都要向对方发，且都要从对方收，加标志位，用batch_isend_irecv
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
#     stage_alignment = [[0,15],[1,14],[2,13],[3,12],[4,11],[5,10],[6,9],[7,8]]#[[0,8],[1,9],[2,10],[3,11],[4,12],[5,13],[6,14],[7,15]]#[[0,4],[1,5],[2,6],[3,7]]##[[0,7],[1,6],[2,5],[3,4]]#[[0,4],[1,5],[2,6],[3,7]]
#     # 输入字符串
#     input_str = """f_0_0,0,13
# f_1_0,13,26
# f_0_1,13,25
# f_0_2,25,37
# f_2_0,26,39
# f_1_1,26,38
# f_0_3,37,49
# f_1_2,38,50
# f_3_0,39,52
# f_2_1,39,51
# f_0_4,49,61
# f_1_3,50,62
# f_2_2,51,63
# f_4_0,52,65
# f_3_1,52,64
# f_0_5,61,73
# f_1_4,62,74
# f_2_3,63,75
# f_3_2,64,76
# f_5_0,65,78
# f_4_1,65,77
# f_0_6,73,85
# f_1_5,74,86
# f_2_4,75,87
# f_3_3,76,88
# f_4_2,77,89
# f_6_0,78,91
# f_5_1,78,90
# f_0_7,85,97
# f_1_6,86,98
# f_2_5,87,99
# f_3_4,88,100
# f_4_3,89,101
# f_5_2,90,102
# f_7_0,91,104
# f_6_1,91,103
# f_0_8,97,109
# f_2_6,99,111
# f_3_5,100,112
# f_4_4,101,113
# f_5_3,102,114
# f_6_2,103,115
# f_8_0,104,117
# f_7_1,104,116
# f_1_7,109,121
# f_0_9,111,123
# f_4_5,113,125
# f_5_4,114,126
# f_6_3,115,127
# f_7_2,116,128
# f_9_0,117,130
# f_8_1,117,129
# f_1_8,121,133
# f_3_6,123,135
# f_0_10,125,137
# f_6_4,127,139
# f_7_3,128,140
# f_8_2,129,141
# f_10_0,130,143
# f_9_1,130,142
# f_2_7,133,145
# f_1_9,135,147
# f_5_5,137,149
# f_0_11,139,151
# f_8_3,141,153
# f_9_2,142,154
# f_11_0,143,156
# f_10_1,143,155
# f_2_8,145,157
# f_4_6,147,159
# f_1_10,149,161
# f_7_4,151,163
# f_0_12,153,165
# f_10_2,155,167
# f_12_0,156,169
# f_11_1,156,168
# f_3_7,157,169
# f_2_9,159,171
# f_6_5,161,173
# f_1_11,163,175
# f_9_3,165,177
# f_0_13,167,179
# f_13_0,169,182
# f_12_1,169,181
# f_3_8,169,181
# f_5_6,171,183
# f_2_10,173,185
# f_8_4,175,187
# f_1_12,177,189
# f_11_2,179,191
# f_0_14,181,193
# f_4_7,181,193
# f_14_0,182,195
# f_3_9,183,195
# f_7_5,185,197
# f_2_11,187,199
# f_10_3,189,201
# f_1_13,191,203
# f_13_1,193,205
# f_4_8,193,205
# f_0_15,195,218
# f_6_6,195,207
# f_3_10,197,209
# f_9_4,199,211
# f_2_12,201,213
# f_12_2,203,215
# f_1_14,205,217
# f_5_7,205,217
# f_4_9,207,219
# f_8_5,209,221
# f_3_11,211,223
# f_11_3,213,225
# f_2_13,215,227
# f_14_1,217,229
# f_5_8,217,229
# b_0_15,218,261
# f_7_6,219,231
# f_4_10,221,233
# f_10_4,223,235
# f_3_12,225,237
# f_13_2,227,239
# f_2_14,229,241
# f_6_7,229,241
# f_5_9,231,243
# f_9_5,233,245
# f_4_11,235,247
# f_12_3,237,249
# f_3_13,239,251
# f_6_8,241,253
# f_8_6,243,255
# f_5_10,245,257
# f_11_4,247,259
# f_4_12,249,261
# f_3_14,251,263
# f_14_2,251,263
# f_7_7,253,265
# f_6_9,255,267
# f_10_5,257,269
# f_5_11,259,271
# w_0_15,261,276
# f_13_3,261,273
# b_0_14,263,287
# f_4_13,263,275
# f_7_8,265,277
# f_9_6,267,279
# f_6_10,269,281
# f_12_4,271,283
# f_5_12,273,285
# f_1_15,276,299
# f_8_7,277,289
# f_7_9,279,291
# f_11_5,281,293
# f_6_11,283,295
# f_5_13,285,297
# f_14_3,285,297
# w_0_14,287,299
# f_8_8,289,301
# f_10_6,291,303
# f_7_10,293,305
# f_13_4,295,307
# b_0_13,297,321
# f_6_12,297,309
# b_1_15,299,342
# f_4_14,299,311
# f_9_7,301,313
# f_8_9,303,315
# f_12_5,305,317
# f_7_11,307,319
# f_5_14,311,323
# f_9_8,313,325
# f_11_6,315,327
# f_8_10,317,329
# f_7_12,319,331
# f_14_4,319,331
# w_0_13,321,333
# f_10_7,325,337
# f_9_9,327,339
# f_13_5,329,341
# b_0_12,331,355
# f_8_11,331,343
# f_6_13,333,345
# f_10_8,337,349
# f_12_6,339,351
# f_9_10,341,353
# w_1_15,342,357
# b_1_14,342,366
# f_7_13,345,357
# f_11_7,349,361
# f_10_9,351,363
# f_9_11,353,365
# f_14_5,353,365
# w_0_12,355,367
# f_2_15,357,380
# f_11_8,361,373
# f_13_6,363,375
# b_0_11,365,389
# f_10_10,365,377
# w_1_14,366,378
# b_1_13,366,390
# f_8_12,367,379
# f_12_7,373,385
# f_11_9,375,387
# f_6_14,378,390
# f_9_12,379,391
# b_2_15,380,423
# f_12_8,385,397
# f_11_10,387,399
# f_14_6,387,399
# w_0_11,389,401
# f_7_14,390,402
# w_1_13,390,402
# b_1_12,391,415
# f_13_7,397,409
# b_0_10,399,423
# f_12_9,399,411
# f_10_11,401,413
# f_8_13,402,414
# f_13_8,409,421
# f_11_11,413,425
# f_8_14,414,426
# f_9_13,414,426
# w_1_12,415,427
# f_13_9,421,433
# f_14_7,421,433
# w_2_15,423,438
# w_0_10,423,435
# b_1_11,425,449
# b_2_14,426,450
# f_10_12,427,439
# b_0_9,433,457
# f_14_8,433,445
# f_12_10,435,447
# f_3_15,438,461
# f_10_13,439,451
# f_11_12,439,451
# f_13_10,447,459
# w_1_11,449,461
# w_2_14,450,462
# b_2_13,451,475
# w_0_9,457,469
# b_0_8,457,481
# b_1_10,459,483
# b_3_15,461,504
# f_12_11,461,473
# f_9_14,462,474
# f_14_9,469,481
# f_12_12,473,485
# f_13_11,473,485
# f_10_14,474,486
# w_2_13,475,487
# w_0_8,481,493
# w_1_10,483,495
# b_1_9,483,507
# b_2_12,485,509
# f_11_13,487,499
# b_0_7,493,517
# f_14_10,495,507
# f_11_14,499,511
# f_12_13,499,511
# w_3_15,504,519
# f_14_11,507,519
# w_1_9,507,519
# w_2_12,509,521
# b_3_14,511,535
# w_0_7,517,529
# f_4_15,519,542
# b_2_11,519,543
# b_0_6,519,543
# f_13_12,521,533
# b_1_8,529,553
# f_13_13,533,545
# f_14_12,533,545
# w_3_14,535,547
# b_4_15,542,585
# w_2_11,543,555
# b_0_5,543,567
# w_0_6,543,555
# b_3_13,545,569
# f_12_14,547,559
# w_1_8,553,565
# f_13_14,559,571
# b_1_7,565,589
# b_0_4,567,591
# w_0_5,567,579
# w_3_13,569,581
# b_3_12,569,593
# b_2_10,579,603
# f_14_13,581,593
# w_4_15,585,600
# b_4_14,585,609
# b_1_6,589,613
# w_1_7,589,601
# w_0_4,591,603
# w_3_12,593,605
# f_5_15,600,623
# b_3_11,603,627
# w_2_10,603,615
# b_0_3,605,629
# w_4_14,609,621
# b_4_13,609,633
# w_1_6,613,625
# b_1_5,615,639
# f_14_14,621,633
# b_5_15,623,666
# b_2_9,625,649
# w_3_11,627,639
# w_0_3,629,641
# w_4_13,633,645
# b_1_4,639,663
# w_1_5,639,651
# b_4_12,641,665
# b_0_2,645,669
# w_2_9,649,661
# b_2_8,649,673
# b_3_10,651,675
# w_1_4,663,675
# w_4_12,665,677
# w_5_15,666,681
# b_5_14,666,690
# w_0_2,669,681
# w_2_8,673,685
# b_4_11,675,699
# w_3_10,675,687
# b_3_9,675,699
# b_1_3,677,701
# f_6_15,681,704
# b_2_7,685,709
# w_5_14,690,702
# b_5_13,690,714
# w_4_11,699,711
# b_4_10,699,723
# w_3_9,699,711
# w_1_3,701,713
# b_0_1,702,726
# b_6_15,704,747
# w_2_7,709,721
# b_2_6,711,735
# w_5_13,714,726
# b_5_12,714,738
# b_3_8,721,745
# w_4_10,723,735
# w_0_1,726,738
# b_1_2,726,750
# b_2_5,735,759
# w_2_6,735,747
# w_5_12,738,750
# b_5_11,738,762
# w_3_8,745,757
# w_6_15,747,762
# b_6_14,747,771
# b_4_9,747,771
# w_1_2,750,762
# b_3_7,757,781
# w_2_5,759,771
# f_7_15,762,785
# w_5_11,762,774
# w_6_14,771,783
# b_6_13,771,795
# b_5_10,771,795
# w_4_9,771,783
# b_2_4,774,798
# w_3_7,781,793
# b_1_1,783,807
# b_3_6,783,807
# b_0_0,785,809
# b_4_8,793,817
# w_6_13,795,807
# b_6_12,795,819
# w_5_10,795,807
# w_2_4,798,810
# w_1_1,807,819
# b_3_5,807,831
# w_3_6,807,819
# w_0_0,809,821
# w_4_8,817,829
# w_6_12,819,831
# b_6_11,819,843
# b_5_9,819,843
# f_8_15,821,844
# b_4_7,829,853
# b_2_3,831,855
# w_3_5,831,843
# w_6_11,843,855
# b_6_10,843,867
# w_5_9,843,855
# b_1_0,844,868
# w_4_7,853,865
# b_2_2,855,879
# w_2_3,855,867
# b_3_4,855,879
# b_4_6,855,879
# b_5_8,865,889
# w_6_10,867,879
# w_1_0,868,880
# b_2_1,879,903
# w_2_2,879,891
# b_3_3,879,903
# w_3_4,879,891
# b_4_5,879,903
# w_4_6,879,891
# f_9_15,880,903
# w_5_8,889,901
# b_6_9,891,915
# b_5_7,901,925
# b_2_0,903,927
# w_2_1,903,915
# b_3_2,903,927
# w_3_3,903,915
# b_4_4,903,927
# w_4_5,903,915
# w_6_9,915,927
# w_5_7,925,937
# w_2_0,927,939
# b_3_1,927,951
# w_3_2,927,939
# b_4_3,927,951
# w_4_4,927,939
# b_5_6,927,951
# b_6_8,937,961
# f_10_15,939,962
# w_3_1,951,963
# b_4_2,951,975
# w_4_3,951,963
# b_5_5,951,975
# w_5_6,951,963
# w_6_8,961,973
# b_3_0,962,986
# b_6_7,973,997
# b_4_1,975,999
# w_4_2,975,987
# b_5_4,975,999
# w_5_5,975,987
# w_3_0,986,998
# b_6_6,997,1021
# w_6_7,997,1009
# f_11_15,998,1021
# w_4_1,999,1011
# b_5_3,999,1023
# w_5_4,999,1011
# b_4_0,1021,1045
# b_6_5,1021,1045
# w_6_6,1021,1033
# b_5_2,1023,1047
# w_5_3,1023,1035
# w_4_0,1045,1057
# b_6_4,1045,1069
# w_6_5,1045,1057
# b_5_1,1047,1071
# w_5_2,1047,1059
# f_12_15,1057,1080
# b_6_3,1069,1093
# w_6_4,1069,1081
# w_5_1,1071,1083
# b_5_0,1080,1104
# b_6_2,1093,1117
# w_6_3,1093,1105
# w_5_0,1104,1116
# f_13_15,1116,1139
# b_6_1,1117,1141
# w_6_2,1117,1129
# b_7_15,1139,1182
# w_6_1,1141,1153
# w_7_15,1182,1197
# b_7_14,1182,1206
# f_14_15,1197,1220
# w_7_14,1206,1218
# b_7_13,1206,1230
# b_6_0,1220,1244
# w_7_13,1230,1242
# b_7_12,1230,1254
# w_6_0,1244,1256
# w_7_12,1254,1266
# b_7_11,1254,1278
# f_15_0,1256,1269
# b_8_15,1269,1312
# f_15_1,1269,1281
# w_7_11,1278,1290
# b_7_10,1278,1302
# f_15_2,1281,1293
# f_15_3,1293,1305
# w_7_10,1302,1314
# b_7_9,1302,1326
# f_15_4,1305,1317
# w_8_15,1312,1327
# b_8_14,1312,1336
# f_15_5,1317,1329
# w_7_9,1326,1338
# b_7_8,1326,1350
# b_9_15,1327,1370
# w_8_14,1336,1348
# b_8_13,1336,1360
# f_15_6,1338,1350
# w_7_8,1350,1362
# w_8_13,1360,1372
# b_8_12,1360,1384
# f_15_7,1362,1374
# w_9_15,1370,1385
# b_9_14,1370,1394
# b_7_7,1374,1398
# w_8_12,1384,1396
# b_8_11,1384,1408
# b_10_15,1385,1428
# w_9_14,1394,1406
# b_9_13,1394,1418
# b_7_6,1398,1422
# w_7_7,1398,1410
# w_8_11,1408,1420
# b_8_10,1408,1432
# f_15_8,1410,1422
# w_9_13,1418,1430
# b_9_12,1418,1442
# w_7_6,1422,1434
# w_10_15,1428,1443
# b_10_14,1428,1452
# w_8_10,1432,1444
# f_15_9,1434,1446
# w_9_12,1442,1454
# b_9_11,1442,1466
# b_11_15,1443,1486
# b_7_5,1444,1468
# b_8_9,1446,1470
# w_10_14,1452,1464
# b_10_13,1452,1476
# w_9_11,1466,1478
# w_7_5,1468,1480
# w_8_9,1470,1482
# b_8_8,1470,1494
# w_10_13,1476,1488
# b_10_12,1476,1500
# b_7_4,1478,1502
# f_15_10,1480,1492
# w_11_15,1486,1501
# b_11_14,1486,1510
# b_9_10,1492,1516
# b_8_7,1494,1518
# w_10_12,1500,1512
# b_12_15,1501,1544
# w_7_4,1502,1514
# w_11_14,1510,1522
# b_11_13,1510,1534
# b_7_3,1512,1536
# f_15_11,1514,1526
# w_9_10,1516,1528
# b_9_9,1516,1540
# w_8_7,1518,1530
# b_10_11,1526,1550
# w_8_8,1530,1542
# w_11_13,1534,1546
# w_7_3,1536,1548
# b_8_6,1540,1564
# b_9_8,1542,1566
# w_12_15,1544,1559
# b_12_14,1544,1568
# b_7_2,1546,1570
# f_15_12,1548,1560
# w_10_11,1550,1562
# b_10_10,1550,1574
# b_13_15,1559,1602
# b_11_12,1560,1584
# w_8_6,1564,1576
# b_9_7,1566,1590
# w_12_14,1568,1580
# w_7_2,1570,1582
# b_8_5,1574,1598
# b_10_9,1576,1600
# b_7_1,1580,1604
# f_15_13,1582,1594
# w_11_12,1584,1596
# b_11_11,1584,1608
# w_9_7,1590,1602
# b_12_13,1594,1618
# w_8_5,1598,1610
# b_9_6,1600,1624
# w_13_15,1602,1617
# b_10_8,1602,1626
# w_7_1,1604,1616
# b_8_4,1608,1632
# b_11_10,1610,1634
# f_15_14,1616,1628
# b_7_0,1617,1641
# w_12_13,1618,1630
# b_12_12,1618,1642
# w_9_6,1624,1636
# b_10_7,1626,1650
# b_13_14,1628,1652
# w_8_4,1632,1644
# b_9_5,1634,1658
# b_11_9,1636,1660
# w_7_0,1641,1653
# b_8_3,1642,1666
# b_12_11,1644,1668
# w_9_8,1650,1662
# w_13_14,1652,1664
# b_13_13,1652,1676
# f_15_15,1653,1676
# w_9_5,1658,1670
# b_10_6,1660,1684
# b_11_8,1662,1686
# w_8_3,1666,1678
# b_9_4,1668,1692
# b_12_10,1670,1694
# b_14_15,1676,1719
# b_8_2,1676,1700
# b_13_12,1678,1702
# w_9_9,1684,1696
# b_11_7,1686,1710
# w_9_4,1692,1704
# b_10_5,1694,1718
# b_12_9,1696,1720
# b_8_1,1700,1724
# w_8_2,1700,1712
# b_9_3,1702,1726
# b_13_11,1704,1728
# w_10_7,1710,1722
# w_13_13,1712,1724
# w_10_5,1718,1730
# b_15_15,1719,1762
# b_11_6,1720,1744
# b_12_8,1722,1746
# b_14_14,1724,1748
# b_9_2,1726,1750
# w_9_3,1726,1738
# b_10_4,1728,1752
# b_13_10,1730,1754
# w_12_12,1738,1750
# w_10_6,1744,1756
# b_12_7,1746,1770
# w_8_1,1748,1760
# b_14_13,1750,1774
# w_13_12,1750,1762
# w_10_4,1752,1764
# b_11_5,1754,1778
# b_13_9,1756,1780
# b_9_1,1760,1784
# b_8_0,1762,1786
# b_10_3,1762,1786
# w_11_11,1764,1776
# w_10_8,1770,1782
# w_9_2,1774,1786
# w_12_11,1776,1788
# w_10_10,1778,1790
# b_12_6,1780,1804
# b_13_8,1782,1806
# b_15_14,1784,1808
# b_9_0,1786,1810
# b_10_2,1786,1810
# b_14_12,1786,1810
# b_11_4,1788,1812
# w_11_5,1790,1802
# w_11_10,1802,1814
# w_10_9,1804,1816
# b_13_7,1806,1830
# w_9_1,1808,1820
# w_8_0,1810,1822
# b_15_13,1810,1834
# w_10_3,1810,1822
# b_14_11,1812,1836
# b_12_5,1814,1838
# w_11_6,1816,1828
# b_10_1,1820,1844
# w_9_0,1822,1834
# b_11_3,1822,1846
# w_11_9,1828,1840
# w_11_7,1830,1842
# w_14_15,1834,1849
# w_10_2,1834,1846
# w_11_4,1836,1848
# b_14_10,1838,1862
# b_13_6,1840,1864
# w_11_8,1842,1854
# w_10_1,1844,1856
# b_11_2,1846,1870
# b_15_12,1846,1870
# b_12_4,1848,1872
# b_10_0,1849,1873
# w_12_7,1854,1866
# w_14_14,1856,1868
# w_12_5,1862,1874
# b_14_9,1864,1888
# w_12_8,1866,1878
# w_15_14,1868,1880
# w_11_2,1870,1882
# w_11_3,1870,1882
# b_15_11,1872,1896
# w_10_0,1873,1885
# b_13_5,1874,1898
# w_13_7,1878,1890
# b_11_1,1880,1904
# w_14_13,1882,1894
# b_12_3,1882,1906
# w_15_15,1885,1900
# w_12_6,1888,1900
# b_14_8,1890,1914
# w_15_13,1894,1906
# w_12_4,1896,1908
# b_15_10,1898,1922
# w_12_9,1900,1912
# b_11_0,1904,1928
# w_11_1,1904,1916
# b_12_2,1906,1930
# w_12_3,1906,1918
# b_13_4,1908,1932
# w_13_6,1912,1924
# b_14_7,1914,1938
# w_14_12,1918,1930
# w_12_10,1922,1934
# b_15_9,1924,1948
# w_11_0,1928,1940
# b_12_1,1930,1954
# w_12_2,1930,1942
# w_15_12,1930,1942
# w_13_4,1932,1944
# w_13_5,1934,1946
# w_13_8,1938,1950
# b_13_3,1942,1966
# w_13_11,1944,1956
# w_13_10,1946,1958
# b_14_6,1948,1972
# b_15_8,1950,1974
# b_12_0,1954,1978
# w_12_1,1954,1966
# w_14_11,1956,1968
# w_14_10,1958,1970
# b_13_2,1966,1990
# w_13_3,1966,1978
# w_15_11,1968,1980
# w_15_10,1970,1982
# w_13_9,1972,1984
# b_15_7,1974,1998
# w_12_0,1978,1990
# b_14_5,1982,2006
# w_14_6,1984,1996
# b_13_1,1990,2014
# w_13_2,1990,2002
# w_14_9,1996,2008
# w_14_7,1998,2010
# b_14_4,2006,2030
# w_14_5,2006,2018
# b_15_6,2008,2032
# w_14_8,2010,2022
# b_13_0,2014,2038
# w_13_1,2014,2026
# w_15_7,2022,2034
# b_14_3,2030,2054
# w_14_4,2030,2042
# b_15_5,2032,2056
# w_15_6,2032,2044
# w_15_8,2034,2046
# w_13_0,2038,2050
# w_15_9,2044,2056
# b_14_2,2054,2078
# w_14_3,2054,2066
# b_15_4,2056,2080
# w_15_5,2056,2068
# b_14_1,2078,2102
# w_14_2,2078,2090
# b_15_3,2080,2104
# w_15_4,2080,2092
# b_14_0,2102,2126
# w_14_1,2102,2114
# b_15_2,2104,2128
# w_15_3,2104,2116
# w_14_0,2126,2138
# b_15_1,2128,2152
# w_15_2,2128,2140
# b_15_0,2152,2176
# w_15_1,2152,2164
# w_15_0,2176,2188
# """
#     result = order_result_mutichunk(input_str,stage_alignment)
#     comm_graph_muti_chunk(result,stage_alignment)

    input_str=''


