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


if __name__ == "__main__":
    tool_wear_dataset = PHMToolWearDataset()
    # tool_wear_data = tool_wear_data.get_tool_wear_data
    print(tool_wear_dataset.get_extended_data[0].shape,tool_wear_dataset.get_extended_data[1].shape)
    print(tool_wear_dataset.get_all_data[0].shape, tool_wear_dataset.get_all_data[1].shape)


    # a = RNNSeriesDataSet(2,5)
    # dat_x,dat_y = a.get_rnn_data()
    # print(dat_x.shape,dat_y.shape)