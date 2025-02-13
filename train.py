#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from internlm.core.context import global_context as gpc
from internlm.core.trainer_builder import TrainerBuilder
from internlm.data import (
    build_train_loader_with_data_type,
    build_valid_loader_with_data_type,
)
from internlm.initialize import initialize_distributed_env
from internlm.model.builder import create_model
from internlm.monitor import internevo_monitor
from internlm.utils.common import parse_args
from internlm.core.context import ParallelMode
import os
from internlm.accelerator import get_accelerator

internlm_accelerator=get_accelerator()
@internevo_monitor(feishu_alert=True, clean_run=True)
def main(args):
    # initialize model
    model = create_model(model_type=gpc.config.model_type)

    # initialize train dataloader
    train_dl, dataset_types = build_train_loader_with_data_type()

    # initialize validation dataloader
    val_dls = build_valid_loader_with_data_type()

    # build trainer
    merged_args = {**vars(args), "dataset_types": dataset_types}
    trainer = TrainerBuilder(model, train_dl, val_dls, **merged_args)

    if args.profiling and gpc.get_local_rank(ParallelMode.DATA) == 0 and (gpc.is_pipeline_first_stage() or gpc.is_pipeline_last_stage() or gpc.get_local_rank(ParallelMode.PIPELINE) == 1):
        internlm_accelerator.memory._record_memory_history()
    # training
    trainer.fit()

    if args.profiling and gpc.get_local_rank(ParallelMode.DATA) == 0 and (gpc.is_pipeline_first_stage() or gpc.is_pipeline_last_stage() or gpc.get_local_rank(ParallelMode.PIPELINE) == 1):
        slurm_job_id = os.getenv('SLURM_JOB_ID')
        snapshot_dir_path = f"/mnt/petrelfs/matenghui/InternEvo/ci_scripts/train/output/{slurm_job_id}_{gpc._config['jsonpath'].split('/')[-1]}/pp_rank{gpc.get_local_rank(ParallelMode.PIPELINE)}"
        snapshot_file_name = (
            f"snapshot{gpc.get_global_rank()}_recomp{gpc._config['model']['checkpoint']}_mb{gpc.micro_num}_"
            + f"tp{gpc.expert_tensor_parallel_size}_pp{gpc.pipeline_parallel_size}_{gpc._config['parallel']['pipeline']['mode']}_chunks{gpc._config['model']['num_chunks']}_"
            + f"seq{gpc._config['data']['seq_len']}_hidden{gpc._config['model']['hidden_size']}.pickle"
        )
        os.makedirs(snapshot_dir_path, exist_ok=True)
        internlm_accelerator.memory._dump_snapshot(os.path.join(snapshot_dir_path, snapshot_file_name))

if __name__ == "__main__":
    args = parse_args()

    # Initialize distributed environment
    initialize_distributed_env(config=args.config, launcher=args.launcher, master_port=args.port, seed=args.seed)
    assert hasattr(gpc, "config") and gpc.config is not None

    # Run the main function with parsed arguments
    main(args)
