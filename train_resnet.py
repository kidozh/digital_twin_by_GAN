from keras.callbacks import TensorBoard,ModelCheckpoint
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

print(signal.shape,tool_wear.shape)

# y = data.gen_y_dat()

# y = data.get_rul_dat()

import random

index = [i for i in range(tool_wear.shape[0])]
random.shuffle(index)
y = tool_wear[index]
x = signal[index]


# y = data.get_rul_dat()

# reshape y


for i in [20,15,10,5,35]:
    DEPTH = i

    log_dir = 'Resnet_logs/'

    train_name = 'Resnet_block_%s_shuffle_manually' % (DEPTH)
    MODEL_CHECK_PT = "%s.kerascheckpts" % (train_name)

    model_name = '%s.kerasmodel' % (train_name)

    predict = True

    if not predict:
        tb_cb = TensorBoard(log_dir=log_dir + train_name)
        ckp_cb = ModelCheckpoint(MODEL_CHECK_PT, monitor='val_loss', save_weights_only=True, verbose=1,
                                 save_best_only=True, period=5)


        model = build_multi_input_main_residual_network(32,5000, 7, 3,loop_depth=DEPTH)

        # model = build_simple_rnn_model(5000,7,3)

        print('Model has been established.')

        model.fit(x, y, batch_size=16, epochs=1000, callbacks=[tb_cb,ckp_cb], validation_split=0.2)

        model.save(model_name)

    else:

        PRED_PATH = 'Y_PRED'
        y = data.get_tool_wear_data
        x = data.get_signal_data
        # x,y = signal,tool_wear

        model = build_multi_input_main_residual_network(32, 5000, 7, 3, loop_depth=DEPTH)

        model.load_weights(MODEL_CHECK_PT)

        print(model.evaluate(x,y))
        y_pred = model.predict(x)
        print(model.metrics_names)



        import matplotlib.pyplot as plt
        import matplotlib as  mpl
        import matplotlib
        import matplotlib.mlab as mlab



        for i in range(3):
            # plt.title('No.%s tool wear' % (i + 1))
            for j in range(3):
                # plt.subplot('31%s'%(j+1))
                fig = plt.figure()
                print("%s tool @ %s teeth" %(i,j))

                plt.plot(y[j * 315:(j + 1) * 315, i], label=u'刀具磨损真实值')
                plt.plot(y_pred[j * 315:(j + 1) * 315, i], label=u'深度学习预测值')
                plt.ylabel('磨损量 ($\mu m$)')
                plt.xlabel('行程')
                # plt.show()

                # plt.plot(y[j*315:(j+1)*315,i],label='Real')
                # plt.plot(y_pred[j*315:(j+1)*315,i],label='Predicted')
                # plt.ylabel('Tool wear ($\mu m$)')
                # plt.xlabel('Run')
                plt.legend()
                plt.savefig("High_Res_%s_tool_@_%s_teeth_ZH.svg"%(i,j))
                plt.show()
        break
    break