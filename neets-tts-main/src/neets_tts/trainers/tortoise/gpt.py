# Referencing https://github.com/huggingface/accelerate/blob/main/examples/by_feature/multi_process_metrics.py

import numpy as np
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import LoggerType
from tqdm import tqdm
from datetime import timedelta

import torch
from torch.utils.data import DataLoader

from neets_tts.models.tortoise.tortoise import Tortoise, TortoiseConfig
from neets_tts.trainers.tortoise.dataset import get_libriheavy_dataset, collate_fn
from neets_tts.trainers.tortoise.injectors import infer_gpt
from neets_tts.trainers.tortoise.injectors import build_conditioning_extractor

hop_length = 256

def train(
  epochs = 10,
  learning_rate = 1e-4,
  loss_text_weight = 0.5,
  loss_mel_weight = 1.0,
  sample_rate = 22050
):
  accelerator = Accelerator(
    log_with=[LoggerType.TENSORBOARD],
    project_dir="./outputs/tortoise-gpt",
    kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(hours=30))]
  )

  tortoise_config = TortoiseConfig()

  # Run this on the main process first so we don't have N processes
  # building and caching the dataset at once.
  with accelerator.main_process_first():
    dataset = get_libriheavy_dataset(22050)

  accelerator.wait_for_everyone()

  split_data = dataset.train_test_split(test_size=0.01)

  train_data_loader = DataLoader(
    split_data["train"],
    batch_size=32,
    collate_fn=collate_fn,
    shuffle=True,
  )

  test_data_loader = DataLoader(
    split_data["test"],
    batch_size=16,
    collate_fn=collate_fn,
    shuffle=False,
  )

  model = Tortoise(tortoise_config)
  print("accelerator.device", accelerator.device)
  model = model.to(accelerator.device)
  # model.gpt.gradient_checkpointing_enable()
  # model.inference_model.gradient_checkpointing_enable()

  optimizer = torch.optim.AdamW(model.parameters(), lr=4e-6, weight_decay=1e-2, betas=(0.9, 0.96))
  # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10000, 20000, 40000], gamma=0.2)

  if accelerator.is_main_process:
    print("initing trackers")
    accelerator.init_trackers(
      project_name="tortoise-gpt",
      config={"learning_rate": learning_rate},
    )

    accelerator.register_for_checkpointing(model)

  model, optimizer, train_data_loader, test_data_loader = accelerator.prepare(
    model, optimizer, train_data_loader, test_data_loader
  )

  current_step = 0

  extract_conditioning = build_conditioning_extractor(
    dataset=dataset,
    conditioning_samples=sample_rate * 5, # 5 seconds of conditioning
    min_clips=1,
    max_clips=5
  )

  for epoch in range(epochs):
    model.train()
    batches = tqdm(train_data_loader)
    for batch in batches:
      current_step += 1

      gpt_conditioning = extract_conditioning(batch)
      batch["gpt_conditioning"] = gpt_conditioning

      loss_text, loss_mel, mel_logits = infer_gpt(model, batch)
      loss = (loss_text_weight * loss_text) + (loss_mel_weight * loss_mel)

      accelerator.backward(loss)

      grad_norm = 0
      if accelerator.sync_gradients:
        clip_grad_eps = 4
        grad_norm = accelerator.clip_grad_norm_(model.parameters(), clip_grad_eps)

      optimizer.step()
      # scheduler.step()
      optimizer.zero_grad()

      log = {
        "epoch": epoch,
        "loss": loss.item(),
        "loss_text": loss_text.item(),
        "loss_mel": loss_mel.item(),
        "grad_norm": grad_norm.item(),
      }

      accelerator.log(log, step=current_step)
      batches.set_postfix(**log)

      if accelerator.is_main_process:
        if current_step % 100 == 0:
          tracker = accelerator.get_tracker("tensorboard", unwrap=True)
          tracker.add_images("mel_logits", mel_logits[0], global_step=current_step, dataformats="HW")

          with torch.no_grad():
            gpt_latent = infer_gpt(model, batch, return_latent=True)

          tracker.add_images("gpt_latent", gpt_latent[0], global_step=current_step, dataformats="HW")

    # Save model state for the epoch.
    if accelerator.is_main_process:
      accelerator.save_state(f"checkpoint-{epoch}.pt")

    model.eval()
    batches = tqdm(test_data_loader)
    losses = []
    for batch in batches:
      gpt_conditioning = extract_conditioning(batch)
      batch["gpt_conditioning"] = gpt_conditioning

      with torch.no_grad():
        loss_text, loss_mel, mel_logits = infer_gpt(model, batch)

      loss = (loss_text_weight * loss_text) + (loss_mel_weight * loss_mel)
      losses.append(loss.item())

    accelerator.log({
      "accuracy_loss": np.mean(losses),
    }, step=current_step)

  accelerator.end_training()

if __name__ == "__main__":
  train()
