import torchaudio

from neets_tts.models.tortoise.tacotron_stft import TacotronSTFT

waveform, sr = torchaudio.load("/home/ubuntu/jsdir/neets-tts/en-us-eric-18.wav")

hop_length = 256

device = "cuda"

stft = TacotronSTFT(
  filter_length=1024,
  hop_length=hop_length,
  win_length=1024,
  n_mel_channels=80,
  sampling_rate=22050,
  mel_fmin=0,
  mel_fmax=11050
).to(device)

waveform_stft = stft.mel_spectrogram(waveform.to(device))

from neets_tts.models.tortoise.hifigan_decoder import HifiganGenerator
# from neets_tts.models.tortoise.tortoise import Tortoise, TortoiseConfig


#tortoise_config = TortoiseConfig()
#gpt = Tortoise(tortoise_config).to(device)
#from safetensors.torch import load_model
#load_model(gpt, "/var/neets/checkpoints/tortoise-gpt-small-jsdir-4-19.pt/model.safetensors")

decoder = HifiganGenerator(
  in_channels=1024,
  out_channels = 1,
  resblock_type = "1",
  resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
  resblock_kernel_sizes = [3, 7, 11],
  upsample_kernel_sizes = [16, 16, 4, 4],
  upsample_initial_channel = 512,
  upsample_factors = [8, 8, 2, 2],
  cond_channels=1024
).to(device).eval()


import torch

optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-4)

bs = 1
conditioning = torch.rand(bs, 1024, 1).to(device)
inp = torch.rand(1,1024, 986).to(device)

while True:
  optimizer.zero_grad()

  res = decoder(inp, conditioning)

  res_mel = stft.mel_spectrogram(res.squeeze(-2))

  img = res_mel[0]
  import matplotlib.pyplot as plt
  plt.imshow(img.cpu().detach().numpy())
  plt.savefig("res.png")

  target = waveform_stft[0]
  img = target
  plt.imshow(img.cpu().detach().numpy())
  plt.savefig("target.png")


  # get l2 loss between res and waveform_stft
  loss = torch.nn.functional.mse_loss(res_mel, waveform_stft)

  print("Loss", loss.item())
  loss.backward()
  optimizer.step()
  print("Done with step")
