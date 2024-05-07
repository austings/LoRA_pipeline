# Based on original HifiGAN training script:
# https://github.com/jik876/hifi-gan/blob/master/train.py

import itertools
from typing import Callable

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import LoggerType
from matplotlib import pyplot as plt

from neets_tts.trainers.tortoise.dataset import get_libriheavy_dataset, collate_fn
from neets_tts.models.hifigan.generator import HifiganGenerator
from neets_tts.models.hifigan.losses import (
  discriminator_loss, feature_loss, generator_loss
)
from neets_tts.models.hifigan.discriminator import (
  MultiPeriodDiscriminator, MultiScaleDiscriminator
)
from neets_tts.trainers.tortoise.injectors import (
  build_conditioning_extractor,
  build_gpt_latent_extractor,
  build_mel_spectrogram_generator,
)

from huggingface_hub import hf_hub_download
mhifi = hf_hub_download(
  "Manmay/tortoise-tts",
  "hifidecoder.pth",
)
def compose_injectors(injectors: list[Callable]):
  def inner(batch: dict, *args, **kwargs) -> dict:
    for injector in injectors:
      batch = injector(batch, *args, **kwargs)
    return batch

  return inner

import matplotlib.pylab as plt
import numpy as np

def save_figure_to_numpy(fig):
  # save it to a numpy array.
  data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
  data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  return data

def plot_spectrogram_to_numpy(spectrogram):
  spectrogram = spectrogram.astype(np.float32)
  fig, ax = plt.subplots(figsize=(12, 3))
  im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation='none')
  plt.colorbar(im, ax=ax)
  plt.xlabel("Frames")
  plt.ylabel("Channels")
  plt.tight_layout()

  fig.canvas.draw()
  data = save_figure_to_numpy(fig)
  plt.close()
  return data

def train(
  # Params copied from:
  # https://github.com/jik876/hifi-gan/blob/master/config_v3.json
  learning_rate = 1e-4,
  lr_decay = 0.999,
  sample_rate = 22050,
  adam_b1 = 0.8,
  adam_b2 = 0.99,
  epoch_count = 100,
):
  accelerator = Accelerator(
    log_with=[LoggerType.TENSORBOARD],
    project_dir="./outputs/tortoise-hifigan",
  )

  accelerator.init_trackers("tortoise-hifigan", config={
    "learning_rate": learning_rate,
    "lr_decay": lr_decay,
    "sample_rate": sample_rate,
    "adam_b1": adam_b1,
    "adam_b2": adam_b2,
    "epoch_count": epoch_count,
  })

  # Run this on the main process first so we don't have N processes
  # building and caching the dataset at once.
  with accelerator.main_process_first():
    dataset = get_libriheavy_dataset(sample_rate)
  accelerator.wait_for_everyone()

  stft = build_mel_spectrogram_generator(sample_rate, "cuda")

  dataloader = DataLoader(
    dataset, # type: ignore
    batch_size=32,
    collate_fn=collate_fn,
    shuffle=True,
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
  )

  # generator.load_state_dict(torch.load(mhifi))

  generator.cuda()

  mpd = MultiPeriodDiscriminator().cuda()
  msd = MultiScaleDiscriminator().cuda()

  optim_g = torch.optim.AdamW(generator.parameters(), learning_rate, betas=(adam_b1, adam_b2))
  optim_d = torch.optim.AdamW(itertools.chain(msd.parameters(), mpd.parameters()), learning_rate, betas=(adam_b1, adam_b2))

  scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=lr_decay)
  scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=lr_decay)

  (
    dataloader,
    generator,
    mpd, msd,
    optim_g, optim_d,
    scheduler_g, scheduler_d
  ) = accelerator.prepare(
    dataloader,
    generator, mpd, msd,
    optim_g, optim_d,
    scheduler_g, scheduler_d
  )

  generator.train()
  mpd.train()
  msd.train()

  current_step = 0

  extract_conditioning = build_conditioning_extractor(
    dataset=dataset,
    conditioning_samples=sample_rate * 5, # 5 seconds of conditioning
    min_clips=1,
    max_clips=5
  )

  extract_gpt_latent = build_gpt_latent_extractor()

  for epoch in range(epoch_count):
    batches = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in batches:
      current_step += 1

      gpt_conditioning = extract_conditioning(batch)
      batch["gpt_conditioning"] = gpt_conditioning

      # TODO temp: limit the length of the shape for integration with gpt2
      batch["text_tokens"] = batch["text_tokens"][..., :400]
      batch["vq_codes"] = batch["vq_codes"][..., :600]

      # batch["vq_codes"][..., 0 :9  ] = 1
      # batch["vq_codes"][..., 10:19] = 2
      # batch["vq_codes"][..., 20:29] = 1
      # batch["vq_codes"][..., 30:39] = 2
      # batch["vq_codes"][..., 40:49] = 1

      gpt_latent, gpt_embedding = extract_gpt_latent(batch)

      # Trim to chunk size
      hop_length = 256
      chunk_size = ((sample_rate * 2) // hop_length) * hop_length

      target_waveform = batch["waveform"][..., :chunk_size]

      plt.imshow(gpt_latent[0].cpu().float().numpy())
      plt.savefig(f"gpt_latent_unpermuted_{current_step}.png")
      plt.imshow(gpt_latent.permute(0, 2, 1)[0].cpu().float().numpy())
      plt.savefig(f"gpt_latent_unpermuted1_{current_step}.png")

      gpt_latent = gpt_latent.permute(0, 2, 1)[..., :(chunk_size // hop_length)]
      print(f"target_waveform_{current_step}", target_waveform.shape)
      print(f"gpt_latent_{current_step}", gpt_latent.shape)
      print(f"gpt_embedding_{current_step}", gpt_embedding.permute(0, 2, 1).shape)
      plt.plot(target_waveform[0].cpu().float().numpy())
      plt.savefig(f"target_waveform_{current_step}.png")
      torchaudio.save(f"target_waveform_{current_step}.wav", target_waveform[0].cpu().unsqueeze(0), sample_rate=sample_rate)
      plt.imshow(gpt_latent[0].cpu().float().numpy())
      plt.savefig(f"gpt_latent_{current_step}.png")
      plt.imshow(gpt_embedding[0].cpu().float().numpy())
      plt.savefig(f"gpt_embedding_{current_step}.png")
      # target_waveform = pad_or_truncate(batch["waveform"], chunk_size, batch["waveform_size"]).cuda()
      # gpt_latent = pad_or_truncate(gpt_latent.permute(0, 2, 1), chunk_size // hop_length, batch["vq_codes_size"]).cuda()

      predicted_waveform = generator(gpt_latent, gpt_embedding.permute(0, 2, 1)).squeeze(-2)
      print(f"predicted_waveform_{current_step}", predicted_waveform.shape)
      torchaudio.save(f"predicted_{current_step}.wav", predicted_waveform[0].cpu().unsqueeze(0), sample_rate=sample_rate)


      with torch.no_grad():
        predicted_mel = stft(predicted_waveform)
        target_mel = stft(target_waveform)

      plt.imshow(predicted_mel[0].cpu().float().numpy())
      plt.savefig(f"predicted_mel_{current_step}.png")
      plt.imshow(target_mel[0].cpu().float().numpy())
      plt.savefig(f"target_mel_{current_step}.png")


      y = target_waveform.unsqueeze(-2)
      y_mel = target_mel
      y_g_hat = predicted_waveform.unsqueeze(-2)
      y_g_hat_mel = predicted_mel

      optim_d.zero_grad()

      # MPD
      y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
      loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

      # MSD
      y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
      loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

      loss_disc_all = loss_disc_s + loss_disc_f
      ld = loss_disc_all.item()

      # loss_disc_all.backward()
      accelerator.backward(loss_disc_all)
      optim_d.step()

      # Generator
      optim_g.zero_grad()

      # L1 Mel-Spectrogram Loss
      loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45

      y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
      y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
      loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
      loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
      loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
      loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
      loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel

      accelerator.backward(loss_gen_all)
      # loss_gen_all.backward()
      optim_g.step()

      log = {
        "loss_mel": loss_mel.item(),
        "loss_discriminator": ld,
        "loss_generator": loss_gen_all.item(),
        "step": current_step,
        "epoch": epoch,
      }
      batches.set_postfix(log)
      accelerator.log(log, step=current_step)

      if accelerator.is_main_process:
        if current_step % 10 == 0:
          # draw target_mel with matplotlib and save to file
          plt.imshow(target_mel[0].cpu().float().numpy())
          plt.savefig(f"mel.png")
          plt.imshow(predicted_mel[0].cpu().float().numpy())
          plt.savefig(f"predict.png")
          plt.imshow(gpt_latent[0].cpu().float().numpy())
          plt.savefig(f"gpt_latent.png")
          plt.imshow(gpt_conditioning[0][0].cpu().float().numpy())
          plt.savefig(f"gpt_conditioning.png")

          # Log
          tensorboard_tracker = accelerator.get_tracker("tensorboard")

          print(predicted_mel[0].cpu().float().numpy().shape, "SHAPE")
          tensorboard_tracker.writer.add_image(
            "predicted_mel",
            predicted_mel[0].cpu().float().numpy(),
            current_step,
            dataformats="HW"
          )

          tensorboard_tracker.writer.add_image(
            "target_mel",
            target_mel[0].cpu().float().numpy(),
            current_step,
            dataformats="HW"
          )
          # tensorboard_tracker.log_images({
          #   "predicted_mel": plot_spectrogram_to_numpy(
          #     predicted_mel[0].cpu().float().numpy()
          #   ),
          #   "target_mel": plot_spectrogram_to_numpy(
          #     target_mel[0].cpu().float().numpy()
          #   ),
          # }, step=current_step)
          # accelerator.save_checkpoint(f"checkpoint_{current_step}")

    scheduler_g.step()
    scheduler_d.step()

    # TODO add validation logging with wandb

  accelerator.end_training()

if __name__ == "__main__":
  train()
