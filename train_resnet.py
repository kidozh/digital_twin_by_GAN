from keras.callbacks import TensorBoard, ModelCheckpoint
from resnet_model import build_multi_input_main_residual_network

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

plt.rcParams['font.sans-serif'] = ['Hiragino Sans GB']  # 用来正常显示中文标签
# plt.rcParams['font.sans-serif'] = ['YaHei Consolas Hybrid']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

from phm_dataset import PHMToolWearDataset

data = PHMToolWearDataset()
signal, tool_wear = data.get_all_data
# down-sample for better generation
signal = signal[:, ::10, :]

print(signal.shape, tool_wear.shape)

# y = data.gen_y_dat()

# y = data.get_rul_dat()

import random

index = [i for i in range(tool_wear.shape[0])]
random.shuffle(index)
y = tool_wear[index]
x = signal[index]
y = np.max(y,axis=1)

# y = data.get_rul_dat()

# reshape y


for i in [20, 15, 10, 5, 35]:
    DEPTH = i

    log_dir = 'Resnet_down_sample_logs/'

    train_name = 'Resnet_block_%s_down_sample_1' % (DEPTH)
    MODEL_CHECK_PT = "%s.kerascheckpts" % (train_name)


    model_name = '%s.kerasmodel' % (train_name)

    predict = True
    model = build_multi_input_main_residual_network(32, 500, 7, 1, loop_depth=DEPTH)
    if not predict:
        tb_cb = TensorBoard(log_dir=log_dir + train_name)
        ckp_cb = ModelCheckpoint(MODEL_CHECK_PT, monitor='val_loss', save_weights_only=True, verbose=1,
                                 save_best_only=True, period=5)



        # model = build_simple_rnn_model(5000,7,3)
        import os.path
        if os.path.exists(MODEL_CHECK_PT):
            model.load_weights(MODEL_CHECK_PT)

        print('Model has been established.')

        model.fit(x, y, batch_size=16, epochs=1000, callbacks=[tb_cb, ckp_cb], validation_split=0.2)

        model.save(model_name)

    else:

        PRED_PATH = 'Y_PRED'
        y = data.get_tool_wear_data
        x = data.get_signal_data
        y = np.max(y, axis=1)
        # signal, tool_wear = data.get_all_data
        # down-sample for better generation
        #signal = signal[:, ::10, :]
        # x,y = signal,tool_wear
        # x = signal
        # y = tool_wear
        x = x[:,::10,:]
        print("Tool shape ",x.shape)
        # MODEL_CHECK_PT = "Resnet_block_20_downSample_toolMax.kerascheckpts"

        # model = build_multi_input_main_residual_network(32, 50, 7, 1, loop_depth=DEPTH)

        model.load_weights(MODEL_CHECK_PT)

        # print(model.evaluate(x, y))
        y_pred = model.predict(x)
        print(model.metrics_names)

        import matplotlib.pyplot as plt
        import matplotlib as  mpl
        import matplotlib
        import matplotlib.mlab as mlab

        for j in range(3):
            plt.plot(y[j * 315:(j + 1) * 315], label=u'刀具磨损真实值')
            plt.plot(y_pred[j * 315:(j + 1) * 315,0], label=u'深度学习预测值')
            plt.ylabel('磨损量 ($\mu m$)')
            plt.xlabel('行程')
            plt.legend()
            plt.show()
    break
