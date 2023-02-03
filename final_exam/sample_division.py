import numpy as np
import time


class SampleDivision:
    def __init__(self):
        self.time = 0

    def shuffle_based(self, data_size):
        """

        :param data_size: size of the data
        :return: index
        """
        if data_size <= 0 or type(data_size) != int:
            raise "数据大小必须是大于1的整数"
        start = time.time()
        index = np.arange(0, data_size, 1)
        np.random.shuffle(index)
        end = time.time()
        print(f"打乱顺序索引的时间为{end - start}s")
        return index

    def random_int(self, data_size):
        """

        :param data_size: size of the data
        :return: index
        """
        if data_size <= 0 or type(data_size) != int:
            raise "数据大小必须是大于1的整数"
        start = time.time()
        index = np.random.choice(data_size, data_size, replace=False)
        end = time.time()
        print(f"打乱顺序索引的时间为{end - start}s")
        return index


if __name__ == '__main__':
    SD = SampleDivision()
    print(SD.shuffle_based(100000000))
    print(SD.random_int(100000000))
