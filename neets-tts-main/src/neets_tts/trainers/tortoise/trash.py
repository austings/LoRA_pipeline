
class TextTokenizer:
  def __init__(self):
    self.tokenizer = VoiceBpeTokenizer()

  def __call__(self, batch: dict):
    texts = [
      self.tokenizer.preprocess_text(text).replace(" ", "[SPACE]")
      for text in batch["text"]
    ]

    text_token_batch = [b.ids for b in self.tokenizer.tokenizer.encode_batch(texts)]

    batch["text_tokens"] = text_token_batch
    batch["text_tokens_size"] = [len(tokens) for tokens in text_token_batch]

    return batch

class WaveformLoader:
  def __init__(
    self,
    root_path: str,
    audio_path_key="audio_path",
    start_key="start",
    duration_key="duration",
    target_sample_rate=22050
  ):
    self.root_path = root_path
    self.audio_path_key = audio_path_key
    self.start_key = start_key
    self.duration_key = duration_key
    self.target_sample_rate = target_sample_rate

  def __call__(self, batch: dict):
    waveforms = []

    for (audio_path, start, duration) in zip(batch["audio_path"], batch["start"], batch["duration"]):
      resolved_audio_path = os.path.join(self.root_path, audio_path)
      audio_file, sr = load_audio(resolved_audio_path)
      waveform = audio_file[int(start * sr):int((start + duration) * sr)]
      if sr != self.target_sample_rate:
        # Resample the audio if the sample rate is not 22050
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.target_sample_rate)
      # waveform = torch.from_numpy(waveform).to(torch.float32)
      waveform = waveform.astype(np.float16)
      waveforms.append(waveform)

    batch["waveform"] = waveforms
    batch["waveform_size"] = [len(waveform) for waveform in waveforms]

    return batch

class VQCodeExtractor:
  @functools.cache
  def setup(self, rank: int):
    device = device_from_rank(rank)
    self.stft = build_stft().to(device)
    self.dvae = DiscreteVAE(
      channels=80,
      codebook_dim=512,
      hidden_dim=512,
      kernel_size=3,
      num_layers=2,
      num_resnet_blocks=3,
      num_tokens=8192,
      positional_dims=1,
      use_transposed_convs=False
    ).to(device)

  def __call__(self, batch: dict, rank: int):
    self.setup(rank)
    device = device_from_rank(rank)

    waveform_batch = pad_sequence([
      a.to(device)
      for a in batch["waveform"]
    ], batch_first=True)
    sizes = batch["waveform_size"]
    vq_codes_size = [
      int(size / hop_length / 4) for size in sizes
    ]

    with torch.no_grad():
      mel = self.stft.mel_spectrogram(waveform_batch)
      waveform_batch.cpu()
      vq_code_batch = self.dvae.get_codebook_indices(mel)

    batch["vq_codes"] = vq_code_batch
    batch["vq_codes_size"] = [
      vq_codes[:size]
      for vq_codes, size in zip(torch.unbind(vq_code_batch), vq_codes_size)
    ]

    return batch

class GPTConditioningExtractor:
  def __init__(
    self,
    dataset: Dataset,
    conditioning_samples: int,
    min_clips: int,
    max_clips: int
  ):
    self.dataset = dataset
    self.indexes_by_speaker_id = {}

    for i, speaker_id in enumerate(dataset["speaker_id"]):
      self.indexes_by_speaker_id.setdefault(speaker_id, []).append(i)

    self.conditioning_samples = conditioning_samples
    self.min_clips = min_clips
    self.max_clips = max_clips

  @functools.cache
  def setup(self, rank: int):
    device = device_from_rank(rank)
    self.stft = build_stft().to(device)

  def __call__(self, batch: dict, rank: int):
    self.setup(rank)
    device = device_from_rank(rank)

    # hop_len = 256
    clip_count = random.randint(self.min_clips, self.max_clips)

    # # [B, clip, S]
    import time
    ss = time.time()
    gpt_conditioning = []
    with torch.no_grad():
      for speaker_id in batch["speaker_id"]:
        waveforms = []
        for idx in random.choices(self.indexes_by_speaker_id[speaker_id], k=clip_count):
          waveform = self.dataset[idx]["waveform"].to(device)
          if len(waveform) > self.conditioning_samples:
            start = random.randint(0, len(waveform) - self.conditioning_samples)
            waveform = waveform[start:(start + self.conditioning_samples)]
          else:
            waveform = waveform[0:self.conditioning_samples]
          waveforms.append(waveform)

        wavs = pad_sequence(waveforms).permute(1, 0)
        mel = self.stft.mel_spectrogram(wavs)
        gpt_conditioning.append(mel)

    end = time.time()
    print(f"Time taken: {end - ss}")

    # wavs = pad_sequence([
    #   torch.FloatTensor(
    #     self.dataset[idx]["waveform"][0:self.conditioning_samples]
    #   ).to(device)
    #   for speaker_id in batch["speaker_id"]
    #   for idx in random.choices(self.indexes_by_speaker_id[speaker_id], k=clip_count)
    # ]).permute(1,0)

    # print("generate mel specs")
    # with torch.no_grad():
    #   mels = self.stft.mel_spectrogram(wavs)

    # end = time.time()
    # print(f"Time taken: {end - start}")

    # conditioning_mel_block = torch.stack(torch.split(mels, clip_count))


    # conditioning_mel_block = torch.stack([
    #   self.stft.mel_spectrogram(
    #     torch.stack([
    #       pad_or_truncate(torch.FloatTensor(self.dataset[idx]["waveform"]).to(device), self.conditioning_samples)
    #       for idx in random.choices(self.indexes_by_speaker_id[speaker_id], k=clip_count)
    #     ])
    #   )
    #   for speaker_id in batch["speaker_id"]
    # ])
    # conditioning_mel_block = torch.stack([
    #   torch.stack([
    #     pad_or_truncate(torch.FloatTensor(self.dataset[idx]["waveform"]).to(device), self.conditioning_samples)
    #     for idx in random.choices(self.indexes_by_speaker_id[speaker_id], k=clip_count)
    #   ])
    #   for speaker_id in batch["speaker_id"]
    # ]); print(1)

    # [B, clip, F, S]
    batch["gpt_conditioning"] = gpt_conditioning

    return batch

class GPTLatentExtractor:
  @functools.cache
  def setup(self, rank: int):
    device = device_from_rank(rank)
    tortoise_config = TortoiseConfig()
    self.gpt = Tortoise(tortoise_config).to(device)

  def __call__(self, batch: dict, rank: int):
    self.setup(rank)
    device = device_from_rank(rank)

    text_tokens = stack_and_pad_tensors(batch["text_tokens"]).to(device)
    gpt_conditioning = stack_and_pad_tensors(batch["gpt_conditioning"]).to(device)

    with torch.no_grad():
      gpt_latent = self.gpt(
        speech_conditioning_input=gpt_conditioning,
        text_inputs=text_tokens,
        text_lengths=batch["text_tokens_size"],
        mel_codes=batch["vq_codes"],
        wav_lengths=batch["waveform_size"],
        return_latent=True
      )
      gpt_embedding = self.gpt._encode_conditioning(gpt_conditioning).squeeze(-2)


    batch["gpt_latent"] = [
      tn[:, :size]
      for tn, size in zip(torch.unbind(gpt_latent), batch["text_tokens_size"])
    ]
    batch["gpt_embedding"] = gpt_embedding # list(torch.unbind(gpt_embedding))


    return batch

class MelExtractor:
  @functools.cache
  def setup(self, rank: int):
    device = device_from_rank(rank)
    self.stft = build_stft().to(device)

  def __call__(self, batch: dict, rank: int):
    self.setup(rank)
    device = device_from_rank(rank)

    print("pad_sequcne")
    waveform_batch = pad_sequence([
      torch.FloatTensor(a).to(device)
      for a in batch["waveform"]
    ], batch_first=True)
    print("done pad")

    with torch.no_grad():
      mel_batch = self.stft.mel_spectrogram(waveform_batch)
    print("foio")

    return {
      "mel": mel_batch,
      "mel_size": [len(mel) for mel in mel_batch]
    }
    #batch["mel"] = mel_batch
    #batch["mel_size"] = [len(mel) for mel in mel_batch]

    #return batch

def device_from_rank(rank: int):
  return f"cuda:{(rank or 0) % torch.cuda.device_count()}"
