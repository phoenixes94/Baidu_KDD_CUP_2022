# -*-Encoding: utf-8 -*-
################################################################################
#
# Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Description: Prepare the experimental settings
"""


def prep_env():
    # type: () -> dict
    """
    Desc:
        Prepare the experimental settings
    Returns:
        The initialized arguments
    """
    settings = {
        "checkpoints": "./models/baseline",
        "pred_file": "predict.py",
        "framework": "paddlepaddle",
        "start_col": 3,
        "path_to_test_x": "./data/sdwpf_baidukddcup2022_test_toy/test_x/0001in.csv",
        "path_to_test_y": "./data/sdwpf_baidukddcup2022_test_toy/test_y/0001out.csv",
        "data_path": "./data",
        "filename": "wtbdata_245days.csv",
        # "task": "MS",
        # "target": "Patv",
        "input_len": 144,
        "output_len": 288,
        "var_len": 10,
        "out_var": 1,
        # "day_len": 144,
        "train_days": 153,
        "val_days": 16,
        "test_days": 15,
        "total_days": 245,
        "num_workers": 12,
        "epoch": 40,
        "batch_size": 32,
        "patient": 6,
        "log_per_steps": 100,
        "lr": 0.00005,
        # "lr_adjust": "type1",
        # "gpu": 0,
        "capacity": 134,
        # "turbine_id": 0,
        # "is_debug": True
        "model": {
            "hidden_dims": 128,
            "nhead": 8,
            "dropout": 0.5,
            "encoder_layers": 2,
            "decoder_layers": 1
        },
        "loss": {
            "name": "FilterMSELoss"
        }
    }
    ###
    return settings
