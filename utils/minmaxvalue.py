# 函数：获取数据集中的最小值和最大值
import torch
# from torch.utils.data import DataLoader
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dataset_min_max(dataloader):
    min_val = float('inf')
    max_val = float('-inf')

    for data, _ in dataloader:
        data = data.to(device)
        batch_min = torch.min(data)
        batch_max = torch.max(data)
        if batch_min < min_val:
            min_val = batch_min.item()
        if batch_max > max_val:
            max_val = batch_max.item()

    return min_val, max_val
