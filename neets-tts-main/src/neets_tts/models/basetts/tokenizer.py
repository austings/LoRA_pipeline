class Tokenizer(nn.Module):
  """
  Similar to:

  Enhancing the Stability of LLM-based Speech Generation Systems through Self-Supervised Representations
  https://arxiv.org/abs/2402.03407

  with some modifications described in the main paper.

  @algomancer recommends removing VQ in favor of Disentanglement via Latent Quantization
  (https://arxiv.org/abs/2305.18378) with FSQ (https://github.com/lucidrains/vector-quantize-pytorch/blob/master/vector_quantize_pytorch/residual_fsq.py)
  or LFQ (https://github.com/lucidrains/vector-quantize-pytorch/blob/master/vector_quantize_pytorch/residual_lfq.py)
  as a natural extension.
  """

  def __init__(
    self,
    # > The SSVC model has been trained with λ0 = 1/301 and λ1 = λ2 = λ3 = 100/301,
    # as those values provided the best results in terms of speaker disentanglement,
    # compensating the reconstruction loss of BigVGAN.
    reconstruction_loss_weight: float = 1 / 301,
    contrastive_loss_weight: float = 100 / 301,
    cosine_loss_weight: float = 100 / 301,
    commitment_loss_weight: float = 100 / 301,
    speaker_encoding_features: int = 768, # TODO find optimal size for this
    codebook_size: int = 256,
  ):
    super().__init__()

    self.reconstruction_loss_weight = reconstruction_loss_weight
    self.contrastive_loss_weight = contrastive_loss_weight
    self.cosine_loss_weight = cosine_loss_weight
    self.commitment_loss_weight = commitment_loss_weight
    self.speaker_encoding_features = speaker_encoding_features

    self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").cuda()
    # self.resnet = ResNetModel(ResNetConfig(num_channels=1)).cuda()
    self.bigvgan = BigVGAN(bigvgan_config)

    self.residual_encoder = ResidualEncoder(
      channels=768,
      hidden_dim=768,
      codebook_dim=768)

    self.conv_decoder = ConvDecoder(input_feature_dim=769, output_mel_dim=80)

    self.quantizer = VectorQuantize(
      dim = 768,
      codebook_size = codebook_size,
      decay = 0.8, # the exponential moving average decay, lower means the dictionary will change faster
      commitment_weight = 1.0 # don't change the commitment loss here, we'll change it later with our own weights
    ).cuda()

    self.speaker_extractor = self.make_speaker_extractor()

    #self.content_speaker_extractor = self.make_speaker_extractor()
    #self.speaker_extractor = self.make_speaker_extractor()

    wavlm_hidden_size = 768
    num_hidden_layers = 13

    self.speaker_weights = [
      nn.Sequential(
        nn.Linear(wavlm_hidden_size, speaker_encoding_features),
        nn.Softmax(dim=-1)
      ).cuda()
      for _ in range(num_hidden_layers)
    ]

    self.content_weights = [
      nn.Sequential(
        nn.Linear(wavlm_hidden_size, speaker_encoding_features),
        nn.Softmax(dim=-1),
        SubtractFromOne()
      ).cuda()
      for _ in range(num_hidden_layers)
    ]

  def make_speaker_extractor(self):
    encoder_layer = nn.TransformerEncoderLayer(d_model=self.speaker_encoding_features, nhead=8)
    return nn.TransformerEncoder(encoder_layer, num_layers=6).cuda()

  def get_contrastive_loss(
    self,
    speaker_features: torch.Tensor, # TODO what shape?
  ):
    """
    Implementation from:

    Robust speech recognition via largescale weak supervision.
    https://arxiv.org/abs/2212.04356

    CLAP: Learning Audio Concepts From Natural Language Supervision
    https://arxiv.org/abs/2206.04769
    """
    # TODO create 2 chunks per element in batch
    output1, output2 = self.contrastive_speaker_extractor(speaker_features, speaker_features)

    # TODO contrastive_speaker_extractor
    euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
    # loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
    #                               (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

    # return loss_contrastive
    return 0

  # Tokenizer forward process as described in 2.2.2 WavLM-based speechcodes
  def forward(
    self,
    waveform: torch.Tensor, # [batch, sample]
  ):
    # > We first pass the waveform through the WavLM model and extract the hidden states.
    with torch.no_grad():
      wavlm_output = self.wavlm(waveform, output_hidden_states=True)

    # wavlm_output.hidden_states
    # [hidden_state, batch, sequence, feature]

    # > These hidden states are then passed through separate content and speaker linear regressors.
    speaker_regression = torch.stack([
      self.speaker_weights[i](hs)
      for i, hs in enumerate(wavlm_output.hidden_states)
    ]).mean(dim=0)
    # [batch, sequence, feature]

    fig = plt.figure()
    plt.imshow(speaker_regression[0].detach().T.cpu().numpy())
    fig.savefig("speaker_regression.png")

    content_regression = torch.stack([
      self.content_weights[i](hs)
      for i, hs in enumerate(wavlm_output.hidden_states)
    ]).mean(dim=0)

    # TODO single weight for now
    content_regression = wavlm_output.last_hidden_state
    speaker_regression = wavlm_output.last_hidden_state
    # [batch, sequence, feature]

    plt.imshow(content_regression[0].detach().T.cpu().numpy())
    fig.savefig("content_regression.png")

    # > The output of these regressors is then fed into a convolutional residual encoder [44].
    content_encoding = self.residual_encoder(content_regression.permute(0,2,1))
    plt.imshow(content_encoding[0].squeeze().detach().T.cpu().numpy())
    plt.savefig("content_encoding.png")

    speaker_encoding = self.residual_encoder(speaker_regression.permute(0,2,1))
    plt.imshow(speaker_encoding[0].squeeze().detach().T.cpu().numpy())
    plt.savefig("speaker_encoding.png")

    # > The content encodings are passed through a vector quantization module
    # > that outputs one speechcode per one WavLM frame (i.e. 20ms of speech).
    _, codes, commitment_loss = self.quantizer(content_encoding.permute(0,2,1))

    # > The speaker encodings are passed through a Transformer-based speaker extractor [15]
    # to obtain the speaker embeddings. The model only extracts, and we only use
    # non-specific features that cannot be used for identification.
    speaker_embeddings = self.speaker_extractor(speaker_encoding.permute(0,2,1))
    plt.imshow(speaker_embeddings[0].squeeze().detach().T.cpu().numpy())
    fig.savefig("speaker_embeddings.png")

    # TODO normalize the codes to fit with the speaker embeddings
    # > The speaker embeddings are concatenated with the speechcodes,
    codebook_size = float(256)
    cinput = torch.cat([speaker_embeddings, (codes/codebook_size).unsqueeze(-1)], dim=-1)
    plt.imshow(cinput[0].squeeze().detach().T.cpu().numpy())
    fig.savefig("convinput.png")

    # and decoded into a spectrogram using a convolutional decoder.
    reconstruction = self.conv_decoder(cinput).permute(0, 2, 1)

    # TODO skip this
    # pad torch.Size([32, 80, 99]) to torch.Size([32, 80, 100])
    reconstruction = F.pad(reconstruction, (0, 1))

    target_fig = plot_spectrogram(reconstruction[0].detach().cpu().numpy())
    target_fig.savefig("reconstruction.png")

    # Build target spectrogram
    target = mel_spectrogram(
      y=waveform,
      n_fft=1024,
      num_mels=80,
      sampling_rate=16000,
      hop_size=int(16000/100*2), # 256,
      win_size=1024,
      fmin=0,
      fmax=12000,
      center=False
    )

    target_fig = plot_spectrogram(target[0].cpu().numpy())
    target_fig.savefig("target.png")

    # > We then compute L1 distance between decoded and target spectrograms
    # and use it as the reconstruction loss. While L1 is not the optimal
    # reconstruction objective, we prioritize representations that are
    # conducive for autoregressive modeling [45], and demonstrate accordingly
    # that the final audio quality can be kept high when this learned
    # representation is decoded with our speechcode decoder, in Section 2.4.
    reconstruction_loss = nn.L1Loss()(target, reconstruction)

    loss = reconstruction_loss \
      + (self.commitment_loss_weight * commitment_loss) \
      # + (self.beta * contrastive_loss) \
      # + (self.gamma * cosine_loss)

    return {
      "codes": codes,
      "reconstruction_loss": reconstruction_loss,
      "commitment_loss": commitment_loss,
      "loss": loss
    }

    #conv_input = torch.cat([speaker_encoding, content_encoding], dim=-1) # [batch, frame, feature1*2]
    # add single channel for resnet
    #conv_input = conv_input.unsqueeze(-3) # [batch, channel, frame, feature1*2]
    # conv_input.shape torch.Size([10, 2999, 128])
    # res = self.resnet(conv_input).pooler_output.squeeze() # [batch, feature2*2]

    # ada.last_hidden_state.shape torch.Size([10, 2048, 94, 4])
    # ada.pooler_output.shape torch.Size([10, 2048, 1, 1])

    # res.shape torch.Size([10, 2048])
    # TODO no way this is right.
    # speaker_encoding = res[:, 1024:]
    # content_encoding = res[:, :1024]

    with torch.no_grad():
      # TODO reverse the gradient here
      content_features = self.content_speaker_extractor(content_encoding)
    speaker_features = self.speaker_extractor(speaker_encoding)

    cosine_loss = F.cosine_similarity(speaker_features, content_features)
    # TODO contrastive_loss = self.get_contrastive_loss(speaker_features)
    contrastive_loss = torch.zeros_like(cosine_loss)

    # why isn't this batched?
    _, quantized_features, commitment_loss = self.quantizer(content_features)
    reconstruction_loss = self.get_reconstruction_loss(speaker_features, quantized_features)

    loss = reconstruction_loss \
      + (self.alpha * commitment_loss) \
      + (self.beta * contrastive_loss) \
      + (self.gamma * cosine_loss)

    return loss, quantized_features

