from datasets import Dataset

dataset = Dataset.from_dict({
  "audio_path": [
    "audio/1.wav",
    "audio/2.wav",
    "audio/3.wav",
  ],
  "start": [0, 0, 0],
  "duration": [10, 10, 10],
})

import os

import functools
import torch

def assign_rank(rank):
  os.environ["CUDA_VISIBLE_DEVICES"] = str(rank % torch.cuda.device_count())

class Foo:
  def __init__(self, x):
    self.x = x

  @functools.cache
  def setup(self, device: str):
    print(f"create stft: {device}")
    self.stft = torch.zeros(10, 10).to(device)

  def __call__(self, x, rank):
    print(f"call: {rank}")
    device = f"cuda:{(rank or 0) % torch.cuda.device_count()}"
    print(f"device: {device}")
    self.setup(device)

    self.stft

    return {
      "start": [start + self.x for start in x["start"]]
    }

results = dataset.map(
  Foo(3),
  batched=True,
  with_rank=True,
  batch_size=1,
  num_proc=2,
)

print(results)
