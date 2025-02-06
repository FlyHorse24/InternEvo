# client.py
import os
import grpc
import rank_pb2
import rank_pb2_grpc

# 禁用代理
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""
os.environ["no_proxy"] = "10.140.62.208,127.0.0.1"

def run():
    try:
        print("Connecting to server...")
        # 创建 gRPC 通道，禁用代理
        channel = grpc.insecure_channel(
            '10.140.62.208:50051',
            options=[
                ('grpc.enable_http_proxy', 0),  # 禁用 HTTP 代理
                ('grpc.enable_https_proxy', 0),  # 禁用 HTTPS 代理
            ]
        )
        print("Connected to server.")
        stub = rank_pb2_grpc.RankServiceStub(channel)

        # 创建 Rank
        print("Creating Rank...")
        create_response = stub.CreateRank(rank_pb2.CreateRankRequest(
            stage_id=1,
            chunk_id=0,
            local_rank=2,
            global_rank=2,
            sendforwardtimes=10,
            sendbackwardtimes=5,
            latest_getrank_info={'sendforwardtimes': 0, 'sendbackwardtimes': 0}  # 初始化 latest_getrank_info
        ))
        print(f"Created Rank: {create_response.rank}")
        rank_id = create_response.rank.id

        # 获取 Rank
        print("Getting Rank...")
        get_response = stub.GetRank(rank_pb2.GetRankRequest(id=rank_id))
        print(f"Get Rank: {get_response.rank}")
        print(f"Latest GetRank Info: {get_response.rank.latest_getrank_info}")

        print("Updating ...")
        update_response = stub.UpdateRank(rank_pb2.UpdateRankRequest(
            id=rank_id,
            stage_id=2,
            chunk_id=0,
            local_rank=0,
            global_rank=0,
            sendforwardtimes=0,
            sendbackwardtimes=0,
            latest_getrank_info={'sendforwardtimes': 3, 'sendbackwardtimes': 1}
            )
        )
        print(f"Updated Rank : {update_response.rank}")



        # 更新 sendforwardtimes
        print("Updating sendforwardtimes...")
        update_sendforward_response = stub.UpdateSendForwardTimes(rank_pb2.UpdateSendForwardTimesRequest(
            id=rank_id,
            sendforwardtimes=20
        ))
        print(f"Updated Rank (sendforwardtimes): {update_sendforward_response.rank}")
        print(f"Latest GetRank Info: {update_sendforward_response.rank.latest_getrank_info}")

        # 根据 stage_id 查询 Rank
        print("Getting Rank by stage_id...")
        get_by_stage_id_response = stub.GetRankByStageId(rank_pb2.GetRankByStageIdRequest(stage_id=1))
        print(f"Get Rank by stage_id: {get_by_stage_id_response.rank}")
        print(f"Latest GetRank Info: {get_by_stage_id_response.rank.latest_getrank_info['sendforwardtimes']}")

    except grpc.RpcError as e:
        print(f"gRPC error: {e.code()}: {e.details()}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == '__main__':
    run()