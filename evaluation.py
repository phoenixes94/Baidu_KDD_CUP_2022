# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import paddle
import paddle.nn.functional as F
import numpy as np
from prepare import prep_env
from wpf_dataset import PGL4WPFDataset, TestPGL4WPFDataset
from wpf_model import WPFModel
from metrics import regressor_detailed_scores
from utils import load_model


@paddle.no_grad()
def predict(config, train_data):  # , valid_data, test_data):
    data_mean = paddle.to_tensor(train_data.data_mean, dtype="float32")
    data_scale = paddle.to_tensor(train_data.data_scale, dtype="float32")

    graph = train_data.graph
    graph = graph.tensor()

    model = WPFModel(config=config)

    global_step = load_model(config["checkpoints"], model)
    model.eval()

    test_x = config["path_to_test_x"]
    test_y = config["path_to_test_y"]

    maes, rmses = [], []

    test_x_ds = TestPGL4WPFDataset(filename=test_x)
    test_y_ds = TestPGL4WPFDataset(filename=test_y)

    test_x = paddle.to_tensor(test_x_ds.get_data()[:, :, -config["input_len"]:, :], dtype="float32")
    test_y = paddle.to_tensor(test_y_ds.get_data()[:, :, :config["output_len"], :], dtype="float32")

    pred_y = model(test_x, test_y, data_mean, data_scale, graph)
    pred_y = F.relu(pred_y * data_scale[:, :, :, -1] + data_mean[:, :, :, -1])
    pred_y = np.expand_dims(pred_y.numpy(), -1)
    test_y = test_y[:, :, :, -1:].numpy()

    pred_y = np.transpose(pred_y, [
        1,
        0,
        2,
        3,
    ])
    test_y = np.transpose(test_y, [
        1,
        0,
        2,
        3,
    ])
    test_y_df = test_y_ds.get_raw_df()

    _mae, _rmse = regressor_detailed_scores(pred_y, test_y, test_y_df, config["capacity"], config["output_len"])
    print('\n\tThe prediction for File -- \n RMSE: {}, MAE: {}, Score: {}'.format(_rmse, _mae, (_rmse + _mae) / 2))
    maes.append(_mae)
    rmses.append(_rmse)

    avg_mae = np.array(maes).mean()
    avg_rmse = np.array(rmses).mean()
    total_score = (avg_mae + avg_rmse) / 2

    print('\n --- Final MAE: {}, RMSE: {} ---'.format(avg_mae, avg_rmse))
    print('--- Final Score --- \n\t{}'.format(total_score))


if __name__ == "__main__":
    config = prep_env()
    print(config)

    size = [config["input_len"], config["output_len"]]
    train_data = PGL4WPFDataset(
        config["data_path"],
        filename=config["filename"],
        size=size,
        flag='train',
        total_days=config["total_days"],
        train_days=config["train_days"],
        val_days=config["val_days"],
        test_days=config["test_days"])

    predict(config, train_data)  # , valid_data, test_data)
