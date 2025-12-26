import torch

# 假设 x 是您的张量
x = torch.randn(5, 64, 100, 352)
record_len = [3, 2]

# 计算累积和
cum_sum_len = torch.cumsum(torch.tensor(record_len), dim=0)

# 分割张量 x
split_x = torch.tensor_split(x, cum_sum_len[:-1])
for tensor in split_x:
    print(tensor.size())
