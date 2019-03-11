import pandas as pd
import numpy as np
import csv


class PHMToolWearDataset():
    signal_cache_path = ".cache/all_sample_dat_num_5000.npy"
    tool_wear_cache_path = ".cache/all_res_dat_num_5000.npy"
    window_number = 10

    @property
    def get_signal_data(self):
        return np.load(self.signal_cache_path)

    @property
    def get_tool_wear_data(self):
        return np.load(self.tool_wear_cache_path)
    @property
    def get_extended_data(self):
        extended_sample_array = []
        extended_sample_label = []

        all_data = self.get_signal_data
        tool_wear_data = self.get_tool_wear_data
        for tool_num in range(3):
            signal_data = all_data[tool_num*315:(tool_num+1)*315]
            label_data = tool_wear_data[tool_num*315:(tool_num+1)*315]
            for i in range(signal_data.shape[0] - 1):
                prev_array = signal_data[i]
                post_array = signal_data[i + 1]
                # only one dimension
                prev_value = label_data[i]
                post_value = label_data[i + 1]

                print(prev_array.shape)

                for part_num in range(1, self.window_number):
                    signal_length = prev_array.shape[0]
                    extended_array = np.concatenate((prev_array[part_num * signal_length // self.window_number:],
                                                     post_array[:part_num * signal_length // self.window_number]),
                                                    axis=0)
                    extended_value = (part_num * np.array(prev_value))/10 + ((10 - part_num)  * np.array(post_value)) /10
                    # extended_value = (part_num * post_value) / 10 + ((10 - part_num) * prev_value) / 10
                    extended_sample_array.append(extended_array)
                    extended_sample_label.append(extended_value)

        return np.array(extended_sample_array), np.array(extended_sample_label)

    @property
    def get_all_data(self):
        all_data = self.get_signal_data
        tool_wear_data = self.get_tool_wear_data

        extend_signal,extend_tool_wear = self.get_extended_data

        return np.concatenate((all_data,extend_signal),axis=0),np.concatenate((tool_wear_data,extend_tool_wear),axis=0)

    def get_recoginition_data(self):
        x,y = self.get_extended_data
        y = y.max(axis=1)
        tool_wear_peroid = np.zeros(y.shape,dtype=np.int)
        i = 30
        cnt = 0
        STEP = 2
        # 31 - 234
        class_num = 105
        # 0-102
        while i < 240:
            class_index = np.logical_and(y > i,y <= i+STEP)
            tool_wear_peroid[class_index] = cnt
            i += STEP
            cnt += 1
        print(y.shape,tool_wear_peroid.shape)
        return x,tool_wear_peroid

    def get_native_recoginition_data(self):
        x = self.get_signal_data
        y = self.get_tool_wear_data
        # x,y = self.get_extended_data
        y = y.max(axis=1)
        tool_wear_peroid = np.zeros(y.shape,dtype=np.int)
        i = 30
        cnt = 0
        STEP = 2
        # 31 - 234
        class_num = 105
        # 0-102
        while i < 240:
            class_index = np.logical_and(y > i,y <= i+STEP)
            tool_wear_peroid[class_index] = cnt
            i += STEP
            cnt += 1
        print(y.shape,tool_wear_peroid.shape)
        return x,tool_wear_peroid

    def get_native_recoginition_data_in_class_num(self,class_num=10):
        x = self.get_signal_data
        y = self.get_tool_wear_data
        # x,y = self.get_extended_data
        y = y.max(axis=1)
        tool_wear_peroid = np.zeros(y.shape,dtype=np.int)
        i = 30
        cnt = 0
        STEP = (240-30)/ class_num
        # 31 - 234

        # 0-102
        while i < 240:
            class_index = np.logical_and(y > i,y <= i+STEP)
            tool_wear_peroid[class_index] = cnt
            i += STEP
            cnt += 1
        assert cnt == class_num
        print(y.shape,tool_wear_peroid.shape)
        return x,tool_wear_peroid

    def get_recoginition_data_in_class_num(self,class_num=10):
        x, y = self.get_extended_data
        # x,y = self.get_extended_data
        y = y.max(axis=1)
        tool_wear_peroid = np.zeros(y.shape,dtype=np.int)
        i = 30
        cnt = 0
        STEP = (240-30) / class_num
        # 31 - 234

        # 0-102
        while i < 240:
            class_index = np.logical_and(y > i,y <= i+STEP)
            tool_wear_peroid[class_index] = cnt
            i += STEP
            cnt += 1
        print("Set apart... ",cnt,class_num,STEP,tool_wear_peroid.max(),tool_wear_peroid.min())
        # assert cnt - 1 == class_num
        print(y.shape,tool_wear_peroid.shape)
        return x,tool_wear_peroid

    def get_reinforce_short_extend_data(self):
        x, y = self.get_extended_data
        print(x.shape,y.shape)
        index = np.array([i for i in range(0,5000,10)])
        # print(index)
        reinforce_signal_data = []
        reinforce_wear_data = []
        MAX_SAMPLE_NUM = x.shape[0]
        for data_idx in range(x.shape[0]):
            tool_max_wear = y[data_idx].max()
            cur_signal_data = x[data_idx]
            cur_wear_data = y[data_idx]
            if data_idx // (MAX_SAMPLE_NUM // 3) == 0:
                # first wear filter
                if tool_max_wear > 110 and tool_max_wear < 120:
                    reinforce_signal_data.append(cur_signal_data[::10,:])
                    reinforce_wear_data.append(cur_wear_data)
                else:
                    # get reinforce
                    for stride_index in range(10):
                        reinforce_signal_data.append(cur_signal_data[index+stride_index])
                        reinforce_wear_data.append(cur_wear_data)
            elif data_idx // (MAX_SAMPLE_NUM // 3) == 1:
                # first wear filter
                if tool_max_wear > 110 and tool_max_wear < 120:
                    reinforce_signal_data.append(cur_signal_data[::10,:])
                    reinforce_wear_data.append(cur_wear_data)
                else:
                    # get reinforce
                    for stride_index in range(10):
                        reinforce_signal_data.append(cur_signal_data[index+stride_index])
                        reinforce_wear_data.append(cur_wear_data)

            else:
                # first wear filter
                if tool_max_wear > 110 and tool_max_wear < 120:
                    reinforce_signal_data.append(cur_signal_data[::10,:])
                    reinforce_wear_data.append(cur_wear_data)
                else:
                    # get reinforce
                    for stride_index in range(10):
                        reinforce_signal_data.append(cur_signal_data[index+stride_index])
                        reinforce_wear_data.append(cur_wear_data)
        return np.array(reinforce_signal_data), np.array(reinforce_wear_data)

    def get_reinforce_recoginition_data_in_class_num(self,class_num=10):
        x, y = self.get_reinforce_short_extend_data()
        # x,y = self.get_extended_data
        y = y.max(axis=1)
        tool_wear_peroid = np.zeros(y.shape,dtype=np.int)
        i = 30
        cnt = 0
        STEP = (240-30) / class_num
        # 31 - 234

        # 0-102
        while i < 240:
            class_index = np.logical_and(y > i,y <= i+STEP)
            tool_wear_peroid[class_index] = cnt
            i += STEP
            cnt += 1
        print("Set apart... ",cnt,class_num,STEP,tool_wear_peroid.max(),tool_wear_peroid.min())
        # assert cnt - 1 == class_num
        print(y.shape,tool_wear_peroid.shape)
        return x,tool_wear_peroid

if __name__ == "__main__":
    tool_wear_dataset = PHMToolWearDataset()
    x,y = tool_wear_dataset.get_reinforce_short_extend_data()
    print(x.shape,y.shape)
    # tool_wear_dataset.get_recoginition_data()
    # x = tool_wear_dataset.get_signal_data
    # print(x.shape)
    # print(x.max(axis=1).max(axis=0),"\n",x.min(axis=1).min(axis=0))

    # tool_wear_data = tool_wear_data.get_tool_wear_data
    #print(tool_wear_dataset.get_extended_data[0].shape,tool_wear_dataset.get_extended_data[1].shape)
    #print(tool_wear_dataset.get_all_data[0].shape, tool_wear_dataset.get_all_data[1].shape)


    # a = RNNSeriesDataSet(2,5)
    # dat_x,dat_y = a.get_rnn_data()
    # print(dat_x.shape,dat_y.shape)