import os
import random
import functools
from typing import Optional
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np

import librosa
import soundfile as sf
from datasets import Dataset
import torchaudio

from neets_tts.models.tortoise.tacotron_stft import TacotronSTFT
from neets_tts.models.tortoise.tokenizer import VoiceBpeTokenizer
from neets_tts.models.tortoise.dvae import DiscreteVAE
from neets_tts.models.tortoise.tortoise import Tortoise, TortoiseConfig
# from neets_tts.trainers.tortoise.gpt import stack_and_pad_tensors
from huggingface_hub import hf_hub_download

hop_length = 256

def build_mel_spectrogram_generator(sample_rate: int, device: str):
  mel_norms_path = hf_hub_download(
    "jbetker/tortoise-tts-v2",
    "tortoise/data/mel_norms.pth",
  )

  mel_norms = torch.load(mel_norms_path).unsqueeze(0).unsqueeze(-1).to(device)

  stft = torchaudio.transforms.MelSpectrogram(
    n_fft=1024,
    hop_length=hop_length,
    win_length=1024,
    power=2,
    normalized=False,
    sample_rate=sample_rate,
    f_min=0,
    f_max=8000,
    n_mels=80,
    norm="slaney"
  )

  stft.to(device)

  def generate_mel_spectrogram(waveform: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
      mel = stft(waveform)

    # Perform dynamic range compression
    mel = torch.log(torch.clamp(mel, min=1e-5))

    # Normalize mel
    mel = mel / mel_norms

    return mel

  return generate_mel_spectrogram

def build_stft():
  return TacotronSTFT(
    filter_length=1024,
    hop_length=hop_length,
    win_length=1024,
    n_mel_channels=80,
    sampling_rate=22050,
    mel_fmin=0,
    mel_fmax=11050
  )

def pad_or_truncate(data: torch.Tensor, target_size: int, data_size: Optional[torch.Tensor] = None):
  # convert 1 dim to 2
  data_batch = data
  if len(data.shape) == 1:
    data_batch = data.unsqueeze(0)

  #if data_size is None:
  #  data_size = torch.full((data.shape[0],), target_size)

  rows = []

  for i, row in enumerate(data_batch):
    row_size = row.shape[-1] if data_size is None else int(data_size[i])

    if target_size > row_size:
      padding = torch.zeros(*row.shape[:-1], target_size - row_size, device=row.device)
      row = torch.cat([row[..., :row_size], padding], dim=-1)
    elif row_size > target_size:
      start = 0 # random.randint(0, row_size - target_size)
      row = row[..., start:(start + target_size)]
    else:
      row = row[..., :row_size]

    rows.append(row)

  if len(data.shape) == 1:
    return rows[0]

  return torch.stack(rows)

  # data_length = data.shape[-1]
  # if data_length < length:
  #   result = torch.full((length,), 0, dtype=data.dtype, device=data.device)
  #   result[:data_length] = data
  #   return result
  # else:
  #   start = random.randint(0, data_length - length)
  #   return data[..., start:(start + length)]

@functools.lru_cache(maxsize=16)
def load_audio(audio_file_path: str):
  return sf.read(audio_file_path)

def load_waveforms(batch: dict, root_path: str, target_sample_rate: int) -> dict:
  waveforms = []
  for (audio_path, start, duration) in zip(batch["audio_path"], batch["start"], batch["duration"]):
    resolved_audio_path = os.path.join(root_path, audio_path)
    audio_file, sr = load_audio(resolved_audio_path)
    waveform = audio_file[int(start * sr):int((start + duration) * sr)]
    if sr != target_sample_rate:
      waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sample_rate)
    # waveform = torch.from_numpy(waveform).to(torch.float32)
    waveform = waveform.astype(np.float16)
    waveforms.append(waveform)

  batch["waveform"] = waveforms
  batch["waveform_size"] = [len(waveform) for waveform in waveforms]
  return batch

def build_tokenizer():
  tokenizer = VoiceBpeTokenizer()

  def tokenize(batch: dict) -> dict:
    texts = [
      tokenizer.preprocess_text(text).replace(" ", "[SPACE]")
      for text in batch["text"]
    ]

    text_token_batch = [b.ids for b in tokenizer.tokenizer.encode_batch(texts)]

    batch["text_tokens"] = text_token_batch
    batch["text_tokens_size"] = [len(tokens) for tokens in text_token_batch]

    return batch

  return tokenize

def create_vectorizer():
  models = {}

  def get_models(rank: int, device: str):
    if rank not in models:

      stft = build_mel_spectrogram_generator(22050, device)
      dvae = DiscreteVAE(
        channels=80,
        codebook_dim=512,
        hidden_dim=512,
        kernel_size=3,
        num_layers=2,
        num_resnet_blocks=3,
        num_tokens=8192,
        positional_dims=1,
        use_transposed_convs=False
      )

      model_path = hf_hub_download(
        "jbetker/tortoise-tts-v2",
        ".models/dvae.pth",
        revision="hf"
      )

      dvae.load_state_dict(torch.load(model_path))
      dvae.to(device)
      models[rank] = (stft, dvae)

    return models[rank]

  def vectorize(batch: dict, rank: int = 0) -> dict:
    device = f"cuda:{(rank or 0) % torch.cuda.device_count()}"

    stft, dvae = get_models(rank, device)

    sizes = batch["waveform_size"]

    waveform_batch = pad_sequence([
      torch.from_numpy(arr).to(device) for arr in batch["waveform"]
    ], batch_first=True)

    with torch.no_grad():
      mel_batch = stft(waveform_batch)
      vq_code_batch = dvae.get_codebook_indices(mel_batch)

    vq_codes_size = [
      int(size / hop_length / 4)
      for size in sizes
    ]

    vq_codes = [
      vq_codes[:size].cpu()
      for vq_codes, size in zip(torch.unbind(vq_code_batch), vq_codes_size)
    ]

    return {
      "vq_codes": vq_codes,
      "vq_codes_size": vq_codes_size
    }

  return vectorize


def build_conditioning_extractor(
  dataset: Dataset,
  conditioning_samples: int,
  min_clips: int,
  max_clips: int
):
  device = "cuda"
  indexes_by_speaker_id = {}
  stft = build_mel_spectrogram_generator(22050, device)
  # stft = build_stft().to(device)

  for i, speaker_id in enumerate(dataset["speaker_id"]):
    indexes_by_speaker_id.setdefault(speaker_id, []).append(i)

  def extract_conditioning(batch: dict):
    clip_count = random.randint(min_clips, max_clips)

    speakers = []
    for speaker_id in batch["speaker_id"]:
      speaker_waveforms = []

      for idx in random.choices(indexes_by_speaker_id[speaker_id], k=clip_count):
        waveform = pad_or_truncate(dataset[idx]["waveform"].to(device), conditioning_samples)
        speaker_waveforms.append(waveform)

      assert len(speaker_waveforms) == clip_count
      speaker_mels = stft(torch.stack(speaker_waveforms))

      speakers.append(speaker_mels)

    return torch.stack(speakers)

    # return torch.stack([
    #   torch.stack([
    #     pad_or_truncate(torch.FloatTensor(dataset[idx]["waveform"]).to(device), conditioning_samples)
    #     for idx in random.choices(indexes_by_speaker_id[speaker_id], k=clip_count)
    #   ])
    #   for speaker_id in batch["speaker_id"]
    # ])

  return extract_conditioning

def infer_gpt(
  gpt: Tortoise,
  batch: dict,
  return_latent: bool = False
):
  print("sizes", {
    "gpt_conditioning": batch["gpt_conditioning"].shape,
    "text_tokens": batch["text_tokens"].shape,
    "text_lengths": batch["text_tokens_size"].shape,
    "mel_codes": batch["vq_codes"].shape,
    "wav_lengths": batch["waveform_size"].shape
  })
  return gpt(
    speech_conditioning_input=batch["gpt_conditioning"].cuda(),
    text_inputs=batch["text_tokens"].cuda(),
    text_lengths=batch["text_tokens_size"].cuda(),
    mel_codes=batch["vq_codes"].cuda(),
    # tortoise edits this tensor in place so we have to clone.
    wav_lengths=batch["waveform_size"].clone().cuda(),
    return_latent=return_latent
  )

def build_gpt_latent_extractor():
  tortoise_config = TortoiseConfig()
  gpt = Tortoise(tortoise_config)

  # from safetensors.torch import load_model
  # load_model(gpt, "/home/ubuntu/jsdir/neets-tts/checkpoint-4.pt/model.safetensors")

  model_path = hf_hub_download(
    "jbetker/tortoise-tts-v2",
    ".models/autoregressive.pth",
  )
  gpt.load_state_dict(torch.load(model_path))
  gpt.cuda()

  def extract_gpt_latent(batch: dict):
    gpt_conditioning = batch["gpt_conditioning"]
    with torch.no_grad():
      gpt_latent = infer_gpt(gpt, batch, return_latent=True)

      gpt_embedding = gpt._encode_conditioning(gpt_conditioning)

    return gpt_latent, gpt_embedding

  return extract_gpt_latent
