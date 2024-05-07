# Referencing https://github.com/huggingface/accelerate/blob/main/examples/by_feature/multi_process_metrics.py

import os
import random
import numpy as np
import functools
import soundfile as sf
import multiprocessing
from accelerate import Accelerator
import librosa
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import concatenate_datasets

from neets_tts.datasets.libriheavy import LibriHeavy
from neets_tts.models.tortoise.dvae import DiscreteVAE
from neets_tts.models.tortoise.hifigan_generator import HifiganGenerator
from neets_tts.models.tortoise.hifigan_discriminator import HifiganDiscriminator
from neets_tts.models.tortoise.tacotron_stft import TacotronSTFT
from neets_tts.models.tortoise.tokenizer import VoiceBpeTokenizer
from neets_tts.models.tortoise.tortoise import Tortoise, TortoiseConfig

hop_length = 256
SPEECH_EOS_TOKEN = 8193

@functools.cache
def build_tokenizer():
  return VoiceBpeTokenizer()

@functools.cache
def build_dvae():
  return DiscreteVAE(
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

@functools.cache
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

def stack_and_pad_tensors(tensors, value = 0):
  max_length = max(tensor.shape[-1] for tensor in tensors)
  return torch.stack([
    F.pad(tensor, (0, max_length - tensor.shape[-1]), "constant", value)
    for tensor in tensors
  ])

def load_speaker_id(batch: dict) -> dict:
  return {
    "speaker_id": [
      supervisions[0]["speaker"]
      for supervisions in batch["supervisions"]
    ]
  }

@functools.lru_cache(maxsize=16)
def load_audio(audio_file_path: str):
  return sf.read(audio_file_path)

def load_waveforms(batch: dict, dataset_path: str) -> dict:
  waveforms = []
  target_sr = 22050  # Target sample rate

  for (recording, start, duration) in zip(batch["recording"], batch["start"], batch["duration"]):
    audio_file_path = os.path.join(dataset_path, recording["sources"][0]["source"])
    audio_file, sr = load_audio(audio_file_path)
    waveform = audio_file[int(start * sr):int((start + duration) * sr)]
    if sr != target_sr:
      # Resample the audio if the sample rate is not 22050
      waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)
      sr = target_sr
    waveform = waveform.astype(np.float16)
    waveforms.append(waveform)

  batch["waveform"] = waveforms
  batch["waveform_size"] = [len(waveform) for waveform in waveforms]
  return batch

def tokenize(batch: dict) -> dict:
  tokenizer = build_tokenizer()

  text_batch = [
    supervisions[0]["custom"]["texts"][0]
    for supervisions in batch["supervisions"]
  ]

  texts = [
    tokenizer.preprocess_text(text).replace(" ", "[SPACE]")
    for text in text_batch
  ]

  text_token_batch = [b.ids for b in tokenizer.tokenizer.encode_batch(texts)]
  batch["text_tokens"] = text_token_batch
  batch["text_tokens_size"] = [
    len(tokens)
    for tokens in text_token_batch
  ]

  return batch

def vectorize(batch: dict, rank: int = 0) -> dict:
  stft = build_stft()
  dvae = build_dvae()

  device = f"cuda:{(rank or 0) % torch.cuda.device_count()}"
  stft.to(device)
  dvae.to(device)

  sizes = batch["waveform_size"]

  waveform_batch = stack_and_pad_tensors([
    torch.from_numpy(arr).to(device) for arr in batch["waveform"]
  ])

  with torch.no_grad():
    mel_batch = stft.mel_spectrogram(waveform_batch)
    vq_code_batch = dvae.get_codebook_indices(mel_batch)

  mels = [
    mel[:, :int(size/hop_length)]
    for mel, size in zip(torch.unbind(mel_batch), sizes)
  ]

  vq_codes_size = [
    int(size / hop_length / 4)
    for size in sizes
  ]

  vq_codes = [
    vq_codes[:size]
    for vq_codes, size in zip(torch.unbind(vq_code_batch), vq_codes_size)
  ]

  return {
    "vq_codes": vq_codes,
    "vq_codes_size": vq_codes_size,
    "mel": mels
  }

def collate_fn(batch):
  return {
    "waveform": [
      item["waveform"] for item in batch
    ],
    "speaker_id": [
      item["speaker_id"] for item in batch
    ],
    "vq_codes_size": torch.tensor([
      item["vq_codes_size"] for item in batch
    ]),
    "text_tokens_size": torch.tensor([
      item["text_tokens_size"] for item in batch
    ]),
    "waveform_size": torch.tensor([
      item["waveform_size"] for item in batch
    ]),
    # "mel_size": torch.tensor([
    #   item["mel"].shape[-1] for item in batch
    # ]),
    "vq_codes": stack_and_pad_tensors([
      item["vq_codes"] for item in batch
    ], value=SPEECH_EOS_TOKEN),
    "mel": stack_and_pad_tensors([
      item["mel"] for item in batch
    ]),
    "text_tokens": stack_and_pad_tensors([
      item["text_tokens"] for item in batch
    ]),
  }

def coerce_mel_to_length(mel, length):
  if mel.shape[-1] < length:
    return F.pad(mel, (0, length - mel.shape[-1]), "replicate", 0)
  else:
    # TODO change this to select random slice of length
    return mel[:, :length]

def get_dataset(tortoise_config: TortoiseConfig):
  from multiprocess import set_start_method # type: ignore
  set_start_method("spawn")

  dataset_path = "/var/neets/datasets/libriheavy"
  libriheavy = LibriHeavy(dataset_path)
  original = libriheavy.get_dataset("small")

  text_tokens = original.map(
    desc="Tokenizing",
    function=tokenize,
    batch_size=1000,
    batched=True,
    num_proc=multiprocessing.cpu_count(),
    writer_batch_size=100000,
    remove_columns=["id", "start", "duration", "channel", "supervisions", "recording", "custom", "type"]
  )

  waveforms = original.map(
    desc="Loading waveforms",
    function=load_waveforms,
    batch_size=64,
    batched=True,
    num_proc=multiprocessing.cpu_count(),
    writer_batch_size=100000,
    remove_columns=["id", "start", "duration", "channel", "supervisions", "recording", "custom", "type"],
    fn_kwargs={
      "dataset_path": dataset_path
    }
  )

  # with_format(type="torch") does not work with num_proc > 1
  waveforms_and_text_tokens = concatenate_datasets([waveforms, text_tokens], axis=1)

  vq_codes = waveforms_and_text_tokens.with_format("numpy").map( # with_format("torch").
    desc="generate vq codes",
    function=vectorize,
    batch_size=256,
    num_proc=torch.cuda.device_count(),
    batched=True,
    with_rank=True,
    writer_batch_size=100000,
    # cache_file_name=f"{dataset_path}/vq_codes.arrow",
    remove_columns=["waveform"]
  )

  speakers = original.map(
    desc="Loading speaker ids",
    function=load_speaker_id,
    batch_size=1000,
    batched=True,
    writer_batch_size=100000,
    # cache_file_name=f"{dataset_path}/speaker_ids.arrow",
    remove_columns=["id", "start", "duration", "channel", "supervisions", "recording", "custom", "type"]
  )

  dataset = concatenate_datasets([
    original, vq_codes, speakers
  ], axis=1)

  max_waveform_size = 16000 * 45

  print("getting indices")
  select_indices = [
    i for i, waveform_size in
    enumerate(dataset["waveform_size"])
    if waveform_size <= max_waveform_size
  ]
  dataset = dataset.select(select_indices)
  dataset = dataset.with_format("torch")

  speaker_ids = dataset["speaker_id"]
  indexes_by_speaker = {}
  for i, speaker_id in enumerate(speaker_ids):
    indexes_by_speaker.setdefault(speaker_id, []).append(i)

  return dataset, indexes_by_speaker

def train(
  epochs = 400,
  learning_rate = 0.0002, # 1e-4,
  # loss_text_weight = 0.5,
  # loss_mel_weight = 1.0,
  conditioning_length = 132300,
  min_conditioning_samples = 1,
  max_conditioning_samples = 5,
  lr_decay = 0.999
):
  print("new training")
  from accelerate import InitProcessGroupKwargs
  from datetime import timedelta
  accelerator = Accelerator(
    log_with=["wandb"],
    kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(hours=10))]
  )

  tortoise_config = TortoiseConfig()

  # Run this on the main process first so we don't have N processes
  # building and caching the dataset at once.
  with accelerator.main_process_first():
    dataset, indexes_by_speaker = get_dataset(tortoise_config)

  accelerator.wait_for_everyone()

  split_data = dataset.train_test_split(test_size=0.01)

  train_data_loader = DataLoader(
    split_data["train"],
    batch_size=8,
    collate_fn=collate_fn,
    shuffle=False, # True,
  )

  test_data_loader = DataLoader(
    split_data["test"],
    batch_size=8,
    collate_fn=collate_fn,
    shuffle=False,
  )

  generator = HifiganGenerator(
    in_channels=1024,
    out_channels = 1,
    resblock_type = "1",
    resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    resblock_kernel_sizes = [3, 7, 11],
    upsample_kernel_sizes = [16, 16, 4, 4],
    upsample_initial_channel = 512,
    upsample_factors = [8, 8, 2, 2],
    cond_channels=1024
  ).to(accelerator.device).eval()

  discriminator = HifiganDiscriminator().to(accelerator.device).eval()

  gpt = Tortoise(tortoise_config).to(accelerator.device)
  from safetensors.torch import load_model
  load_model(gpt, "/var/neets/checkpoints/tortoise-gpt-small-jsdir-4-19.pt/model.safetensors")

  optim_g = torch.optim.Adam(generator.parameters(), lr=learning_rate)
  optim_d = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)

  scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=lr_decay)
  scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=lr_decay)

  if accelerator.is_main_process:
    accelerator.init_trackers(
      project_name="tortoise-hifigan",
      config={"learning_rate": learning_rate},
    )

    # accelerator.register_for_checkpointing(model)

  (
    generator, discriminator,
    optim_g, optim_d,
    scheduler_g, scheduler_d,
    train_data_loader, test_data_loader
  ) = accelerator.prepare(
    generator, discriminator,
    optim_g, optim_d,
    scheduler_g, scheduler_d,
    train_data_loader, test_data_loader
  )

  global_step = 0

  stft = build_stft().to(accelerator.device)

  def build_model_inputs(batch):
    for key in batch:
      if isinstance(batch[key], torch.Tensor):
        batch[key] = batch[key].to(accelerator.device)
    hop_len = 256
    mel_length_compression = hop_len * 4

    conditioning_sample_count = random.randint(
      min_conditioning_samples, max_conditioning_samples)
    conditioning_mel_block = torch.stack([
      torch.stack([
        coerce_mel_to_length(dataset[idx]["mel"], conditioning_length // hop_len).to(accelerator.device)
        for idx in random.choices(indexes_by_speaker[speaker_id], k=conditioning_sample_count)
      ])
      for speaker_id in batch["speaker_id"]
    ])

    # Check model inputs
    assert batch["text_tokens"].max() < tortoise_config.number_text_tokens, \
      f"text token out of bounds: {batch['text_tokens'].max()} >= {tortoise_config.number_text_tokens}"
    assert batch["text_tokens_size"].max() <= tortoise_config.max_text_tokens, \
      f"text token too large: {batch['text_tokens_size'].max()} >= {tortoise_config.max_text_tokens}"
    assert batch["vq_codes"].max() < tortoise_config.number_mel_codes, \
      f"vq code out of bounds: {batch['vq_codes'].max()} >= {tortoise_config.number_mel_codes}"
    mel_tokens = batch["waveform_size"] // mel_length_compression
    assert mel_tokens.max() <= tortoise_config.max_mel_tokens, \
      f"too many vq codes: {mel_tokens.max()} >= {tortoise_config.max_mel_tokens}"

    return {
      "speech_conditioning_input": conditioning_mel_block, # torch.randn(len(batch["text_tokens"]), 80, 1).to(device),
      "text_inputs": batch["text_tokens"],
      "text_lengths": batch["text_tokens_size"],
      "mel_codes": batch["vq_codes"],
      "wav_lengths": batch["waveform_size"]
    }

  generator.train()
  discriminator.train()
  for epoch in range(epochs):
    batches = tqdm(train_data_loader)
    for batch in batches:
      # training logic pulled from:
      # https://github.com/jik876/hifi-gan/blob/master/train.py
      global_step += 1

      inputs = build_model_inputs(batch)
      with torch.no_grad():
        gpt_latent = gpt(**inputs, return_latent=True).permute(0, 2, 1)
        conds = gpt._encode_conditioning(inputs["speech_conditioning_input"])

      y_g_hat = generator(gpt_latent, conds.permute(0, 2, 1))
      waveforms = y_g_hat.squeeze(-2)
      y_g_hat_mel = stft.mel_spectrogram(waveforms)

      optim_d.zero_grad()

      discriminator(real_mel, real_waveform)

      target_mels = batch["mel"]
      res2 = F.interpolate(target_mels, scale_factor=0.25, mode="nearest")
      # pad res2 to match mels
      target_res2 = F.pad(res2, (0, mels.shape[-1] - res2.shape[-1]), "constant", 0)

      max_seq = mels.shape[-1]
      mask = torch.arange(max_seq).view(1, 1, -1).cuda() >= batch["vq_codes_size"].view(-1, 1, 1)
      mels = mels.masked_fill(mask, 0)
      target_res2 = target_res2.masked_fill(mask, 0)

      loss = F.mse_loss(mels, target_res2)

      if True: #  global_step % 100 == 0:
        img = mels[0]
        import matplotlib.pyplot as plt
        plt.imshow(img.squeeze().cpu().detach().numpy())
        plt.savefig("recon.png")
        plt.imshow(target_res2[0].squeeze().cpu().detach().numpy())
        plt.savefig("target.png")

      accelerator.backward(loss)


      optimizer.step()
      optimizer.zero_grad()

      log = {
        "epoch": epoch,
        "loss": loss.item(),
      }

      accelerator.log(log, step=global_step)
      batches.set_postfix(**log)

      break

    if accelerator.is_main_process:
      accelerator.save_state(f"checkpoint-{epoch}.pt")

  accelerator.end_training()

if __name__ == "__main__":
  train()


# waveform, text
# - tokenize
# - build conditioning
# - gpt_latents = gpt(text_tokens, conditioning)
# - trim waveform and gpt_latent to example size

# inputs: gpt_latent, conditioning
# outputs: waveform