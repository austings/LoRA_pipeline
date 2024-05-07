import torch
from typing import TypedDict

from neets_tts.trainers.base import BaseTrainer
from neets_tts.datasets.speech import SpeechDataset
from neets_tts.models.tortoise.diffuser import DiffusionTtsFlat
from neets_tts.models.tortoise.gpt import GPT

class DiffuserTrainingExample(TypedDict):
  mel_spectrogram: torch.Tensor
  gpt_latent: torch.Tensor

class GPTLatentInjector:
  def __init__(self, gpt_dataset_path: str):
    super().__init__()
    self.gpt = GPT().cuda()

  def __call__(self, item: dict) -> dict:
    text = item["text"]
    gpt_latent = self.gpt(text)
    item["gpt_latent"] = gpt_latent
    return item

class DiffuserDataset:
  def __init__(self, dataset_path: str, gpt_dataset_path: str, speech_dataset: SpeechDataset):
    self.dataset_path = dataset_path
    self.gpt_dataset_path = gpt_dataset_path
    self.speech_dataset = speech_dataset

  def build_cache(self):
    self.speech_dataset.map(

    )

class DiffuserTrainer(BaseTrainer):
  def __init__(self):
    self.diffuser = DiffusionTtsFlat().cuda()

  def train_batch(self, batch: DiffuserTrainingExample):
    target = batch["mel_spectrogram"].cuda()
    gpt_latent = batch["gpt_latent"].cuda()
    timesteps = torch.randn(9)

    _, mel = self.diffuser(
      x=gpt_latent,
      timesteps=timesteps,
      aligned_conditioning=None,
      conditioning_input=None,
      precomputed_aligned_embeddings=None,
      conditioning_free=False,
      return_code_pred=False
    )

    # l2 loss
    loss = torch.nn.functional.mse_loss(mel, target)
    return loss
