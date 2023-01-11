import torch

def train_test_split(x, y, train_size=0.8) -> tuple(torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
  idx = torch.randperm(len(x))
  flag = int(len(idx) * train_size)
  train_idx = idx[:flag]
  test_idx = idx[flag:]
  x_train = x[train_idx]
  y_train = y[train_idx]
  x_test = x[test_idx]
  y_test = y[test_idx]
  return x_train, y_train, x_test, y_test