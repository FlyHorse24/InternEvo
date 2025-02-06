import grpc
from concurrent import futures
import rank_pb2
import rank_pb2_grpc

# 模拟数据库
ranks_db = {}

class RankService(rank_pb2_grpc.RankServiceServicer):
    def CreateRank(self, request, context):
        rank_id = str(len(ranks_db) + 1)  # 生成一个简单的 ID
        rank = rank_pb2.Rank(
            id=rank_id,
            stage_id=request.stage_id,
            chunk_id=request.chunk_id,
            local_rank=request.local_rank,
            global_rank=request.global_rank,
            sendforwardtimes=request.sendforwardtimes,
            sendbackwardtimes=request.sendbackwardtimes,
            latest_getrank_info=request.latest_getrank_info  # 初始化 latest_getrank_info
        )
        ranks_db[rank_id] = rank
        return rank_pb2.RankResponse(rank=rank)

    def GetRank(self, request, context):
        rank = ranks_db.get(request.id)
        if rank:
            return rank_pb2.RankResponse(rank=rank)
        return rank_pb2.RankResponse()

    def UpdateRank(self, request, context):
        rank = ranks_db.get(request.id)
        if rank:
            rank.stage_id = request.stage_id
            rank.chunk_id = request.chunk_id
            rank.local_rank = request.local_rank
            rank.global_rank = request.global_rank
            rank.sendforwardtimes = request.sendforwardtimes
            rank.sendbackwardtimes = request.sendbackwardtimes
            rank.latest_getrank_info['sendforwardtimes'] = request.latest_getrank_info['sendforwardtimes']
            rank.latest_getrank_info['sendbackwardtimes'] = request.latest_getrank_info['sendbackwardtimes']
            return rank_pb2.RankResponse(rank=rank)
        context.set_code(grpc.StatusCode.NOT_FOUND)
        context.set_details("Rank not found")
        return rank_pb2.RankResponse()

    def UpdateSendForwardTimes(self, request, context):
        rank = ranks_db.get(request.id)
        if rank:
            rank.sendforwardtimes = request.sendforwardtimes
            return rank_pb2.RankResponse(rank=rank)
        context.set_code(grpc.StatusCode.NOT_FOUND)
        context.set_details("Rank not found")
        return rank_pb2.RankResponse()

    def UpdateSendBackwardTimes(self, request, context):
        rank = ranks_db.get(request.id)
        if rank:
            rank.sendbackwardtimes = request.sendbackwardtimes
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

    def GetRankByStageId(self, request, context):
        for rank in ranks_db.values():
            if rank.stage_id == request.stage_id:
                return rank_pb2.RankResponse(rank=rank)
        return rank_pb2.RankResponse()

    def GetSendForwardTimesByStageId(self, request, context):
        for rank in ranks_db.values():
            if rank.stage_id == request.stage_id:
                # 更新 latest_getrank_info 中的 sendforwardtimes
                rank.latest_getrank_info['sendforwardtimes'] = rank.sendforwardtimes
                return rank_pb2.GetSendForwardTimesResponse(sendforwardtimes=rank.sendforwardtimes)
        # 未找到 Rank，返回错误
        return rank_pb2.GetSendForwardTimesResponse()

    def GetSendBackwardTimesByStageId(self, request, context):
        for rank in ranks_db.values():
            if rank.stage_id == request.stage_id:
                # 更新 latest_getrank_info 中的 sendbackwardtimes
                rank.latest_getrank_info['sendbackwardtimes'] = rank.sendbackwardtimes
                return rank_pb2.GetSendBackwardTimesResponse(sendbackwardtimes=rank.sendbackwardtimes)
        return rank_pb2.GetSendBackwardTimesResponse()



def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    rank_pb2_grpc.add_RankServiceServicer_to_server(RankService(), server)
    server.add_insecure_port('[::]:50051')
    print("Server started, listening on port 50051...")
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()