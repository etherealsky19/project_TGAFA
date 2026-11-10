import numpy as np
import scipy.io
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset


class MultiViewDataset(Dataset):
    def __init__(self, data_name:str, data_X:np.ndarray, data_Y:np.ndarray):
        super(MultiViewDataset, self).__init__()
        self.data_name = data_name

        self.X = dict()
        self.num_views = data_X.shape[0]
        for v in range(self.num_views):
            self.X[v] = self.normalize(data_X[v])
            # self.X[v] = data_X[v]

        self.Y = data_Y
        self.Y = np.squeeze(self.Y)
        if np.min(self.Y) == 1:
            self.Y = self.Y - 1
        self.Y = self.Y.astype(dtype=np.int64)
        self.num_classes = len(np.unique(self.Y))
        self.dims = self.get_dims()

    def __getitem__(self, index):
        data = dict()
        for v_num in range(len(self.X)):
            data[v_num] = (self.X[v_num][index]).astype(np.float32)
        target = self.Y[index]
        return data, target, index

    def __len__(self):
        """
        统计数据集样本个数
        Returns:

        """
        return len(self.X[0])

    def get_dims(self):
        """
        统计数据集各视图维度
        Returns:
            dims：各视图维度列表
        """
        dims = []
        for view in range(self.num_views):
            dims.append(self.X[view].shape[1])
        return dims

    @staticmethod
    def normalize(x, min=0):
        if min == 0:
            scaler = MinMaxScaler((0, 1))
        else:  # min=-1
            scaler = MinMaxScaler((-1, 1))
        norm_x = scaler.fit_transform(x)
        return norm_x

def MATKind(dataset_name:str, path:str):
    data_path = f"{path}/{dataset_name}.mat"
    data = scipy.io.loadmat(data_path)
    data_X = data['X'][0]
    data_Y = data['Y']
    return MultiViewDataset(f"{dataset_name}", data_X, data_Y)