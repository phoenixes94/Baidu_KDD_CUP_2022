# -*- coding: utf-8 -*-
import numpy as np
import paddle
import paddle.nn.functional as F
from prepare import prep_env
from wpf_dataset import PGL4WPFDataset, TestPGL4WPFDataset
from wpf_model import WPFModel
from utils import load_model


def forecast(_settings: dict):
    """
    预测\n
    :param _settings: 配置
    :return: 预测值
    """
    print(_settings)
    size = [_settings["input_len"], _settings["output_len"]]
    train_data = PGL4WPFDataset(
        _settings["data_path"],
        filename=_settings["filename"],
        size=size,
        flag='train',
        total_days=_settings["total_days"],
        train_days=_settings["train_days"],
        val_days=_settings["val_days"],
        test_days=_settings["test_days"]
    )

    data_mean = paddle.to_tensor(train_data.data_mean, dtype="float32")
    data_scale = paddle.to_tensor(train_data.data_scale, dtype="float32")

    graph = train_data.graph
    graph = graph.tensor()

    model = WPFModel(config=_settings)

    global_step = load_model(_settings["checkpoints"], model)
    model.eval()

    test_x = _settings["path_to_test_x"]
    test_x_ds = TestPGL4WPFDataset(filename=test_x)
    test_x = paddle.to_tensor(test_x_ds.get_data()[:, :, -_settings["input_len"]:, :], dtype="float32")

    pred_y = model(test_x, paddle.zeros((test_x.shape[0], _settings["capacity"], _settings["output_len"], test_x.shape[-1])), data_mean, data_scale, graph)
    pred_y = F.relu(pred_y * data_scale[:, :, :, -1] + data_mean[:, :, :, -1])

    pred_y = np.expand_dims(pred_y.numpy(), -1)

    pred_y = np.transpose(pred_y, [
        1,
        0,
        2,
        3,
    ])

    return pred_y.reshape((_settings["capacity"], _settings["output_len"], _settings["out_var"]))


if __name__ == '__main__':
    settings = prep_env()
    result = forecast(settings)
    print(result.shape)