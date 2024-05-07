from datasets import Dataset
import torch
from torch import nn
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset as TorchDataset

from tqdm import tqdm

class BaseTrainer:
  def __init__(
    self,
    dataset: Dataset,
    batch_size: int = 8,
    num_workers: int = 4
  ):
    self.dataset = dataset
    self.batch_size = batch_size
    self.num_workers = num_workers

  def model(self) -> nn.Module:
    raise NotImplementedError

  def get_optimizer(self, model: nn.Module):
    return torch.optim.Adam(model.parameters(), lr=1e-3)

  def get_scheduler(self, optimizer: torch.optim.Optimizer):
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)

  def train(self, epochs: int):
    accelerator = Accelerator()

    model = self.model()
    optimizer = self.get_optimizer(model)
    scheduler = self.get_scheduler(optimizer)
    torch_dataset: TorchDataset = self.dataset.with_format("torch")
    dataloader = DataLoader(
      dataset=torch_dataset,
      batch_size=self.batch_size,
      num_workers=self.num_workers
    )

    model, optimizer, dataloader, scheduler = accelerator.prepare(
      model, optimizer, dataloader, scheduler
    )

    for epoch in range(epochs):
      batches = tqdm(dataloader)
      for batch in batches:
        optimizer.zero_grad()
        loss = self.train_batch(batch)
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()
        batches.set_postfix(epoch=epoch, loss=loss.item())

  def train_batch(self, batch) -> torch.Tensor:
    raise NotImplementedError
