import grpc
from concurrent import futures
from google.protobuf.timestamp_pb2 import Timestamp  # 引入 Timestamp
import rank_pb2
import rank_pb2_grpc

# 模拟数据库
ranks_db = {}

class RankService(rank_pb2_grpc.RankServiceServicer):
    def CreateRank(self, request, context):
        rank_id = str(len(ranks_db) + 1)  # 生成一个简单的 ID
        timestamp = Timestamp()  # 创建时间戳
        timestamp.GetCurrentTime()  # 设置为当前时间
        rank = rank_pb2.Rank(
            id=rank_id,
            local_rank=request.local_rank,
            sendforwardtimes=request.sendforwardtimes,
            sendbackwardtimes=request.sendbackwardtimes,
            timestamp=timestamp,  # 设置操作时间戳
            last_getrank_timestamp=Timestamp(),  # 初始化 last_getrank_timestamp
            index=request.index  # 使用客户端传入的 index
        )
        ranks_db[rank_id] = rank
        return rank_pb2.RankResponse(rank=rank)

    def GetRank(self, request, context):
        rank = ranks_db.get(request.id)
        if rank:
            # 更新 last_getrank_timestamp
            last_getrank_timestamp = Timestamp()
            last_getrank_timestamp.GetCurrentTime()
            rank.last_getrank_timestamp.CopyFrom(last_getrank_timestamp)
            return rank_pb2.RankResponse(rank=rank)
        context.set_code(grpc.StatusCode.NOT_FOUND)
        context.set_details("Rank not found")
        return rank_pb2.RankResponse()

    def UpdateRank(self, request, context):
        rank = ranks_db.get(request.id)
        if rank:
            timestamp = Timestamp()  # 创建时间戳
            timestamp.GetCurrentTime()  # 设置为当前时间
            rank.local_rank = request.local_rank
            rank.sendforwardtimes = request.sendforwardtimes
            rank.sendbackwardtimes = request.sendbackwardtimes
            rank.timestamp.CopyFrom(timestamp)  # 更新操作时间戳
            rank.index = request.index  # 更新 index
            return rank_pb2.RankResponse(rank=rank)
        context.set_code(grpc.StatusCode.NOT_FOUND)
        context.set_details("Rank not found")
        return rank_pb2.RankResponse()

    def UpdateSendForwardTimes(self, request, context):  # 新增方法
        rank = ranks_db.get(request.id)
        if rank:
            timestamp = Timestamp()  # 创建时间戳
            timestamp.GetCurrentTime()  # 设置为当前时间
            rank.sendforwardtimes = request.sendforwardtimes  # 只更新 sendforwardtimes
            rank.timestamp.CopyFrom(timestamp)  # 更新操作时间戳
            return rank_pb2.RankResponse(rank=rank)
        context.set_code(grpc.StatusCode.NOT_FOUND)
        context.set_details("Rank not found")
        return rank_pb2.RankResponse()

    def UpdateSendBackwardTimes(self, request, context):  # 新增方法
        rank = ranks_db.get(request.id)
        if rank:
            timestamp = Timestamp()  # 创建时间戳
            timestamp.GetCurrentTime()  # 设置为当前时间
            rank.sendbackwardtimes = request.sendbackwardtimes  # 只更新 sendbackwardtimes
            rank.timestamp.CopyFrom(timestamp)  # 更新操作时间戳
            return rank_pb2.RankResponse(rank=rank)
        context.set_code(grpc.StatusCode.NOT_FOUND)
        context.set_details("Rank not found")
        return rank_pb2.RankResponse()

    def DeleteRank(self, request, context):
        if request.id in ranks_db:
            del ranks_db[request.id]
            return rank_pb2.DeleteRankResponse(success=True)
        context.set_code(grpc.StatusCode.NOT_FOUND)
        context.set_details("Rank not found")
        return rank_pb2.DeleteRankResponse(success=False)

    def GetRankByLocalRank(self, request, context):
        for rank in ranks_db.values():
            if rank.local_rank == request.local_rank:
                return rank_pb2.RankResponse(rank=rank)
        context.set_code(grpc.StatusCode.NOT_FOUND)
        context.set_details("Rank not found for the given local_rank")
        return rank_pb2.RankResponse()

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    rank_pb2_grpc.add_RankServiceServicer_to_server(RankService(), server)
    server.add_insecure_port('[::]:50051')
    print("Server started, listening on port 50051...")
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()