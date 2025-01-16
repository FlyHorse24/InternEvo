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

        for i in range(4):
            # 创建 Rank
            print("Creating Rank...")
            create_response = stub.CreateRank(rank_pb2.CreateRankRequest(
                local_rank=i,
                sendforwardtimes=0,
                sendbackwardtimes=0
            ))
            print(f"Created Rank: {create_response.rank}")

        # create_response = stub.CreateRank(rank_pb2.CreateRankRequest(
        #     local_rank=2,
        #     sendforwardtimes=7,
        #     sendbackwardtimes=0
        # ))
        # print(f"Created Rank: {create_response.rank}")
        # rank_id = create_response.rank.id


        # # 更新 Rank
        # print("Updating Rank...")
        # update_response = stub.UpdateRank(rank_pb2.UpdateRankRequest(
        #     id="2",
        #     local_rank=2,
        #     sendforwardtimes=20,
        #     sendbackwardtimes=1
        # ))
        # print(f"Updated Rank: {update_response.rank}")


        # # 获取 Rank
        # print("Getting Rank by ID...")
        # get_response = stub.GetRank(rank_pb2.GetRankRequest(id="2"))
        # print(f"Get Rank sendforwardtimes: {get_response.rank.sendforwardtimes}")
        
        # 根据 local_rank 查询 Rank
        for i in range(4):
            print("Getting Rank by local_rank...")
            get_by_local_rank_response = stub.GetRankByLocalRank(rank_pb2.GetRankByLocalRankRequest(local_rank=i))
            print(f"Get Rank by local_rank : {get_by_local_rank_response.rank}")

        # # 删除 Rank
        # print("Deleting Rank...")
        # delete_response = stub.DeleteRank(rank_pb2.DeleteRankRequest(id=rank_id))
        # print(f"Delete Rank Success: {delete_response.success}")

    except grpc.RpcError as e:
        print(f"gRPC error: {e.code()}: {e.details()}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == '__main__':
    run()