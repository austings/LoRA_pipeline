class TorchMelSpectrogramInjector(Injector):
    def __init__(self, opt, env):
        super().__init__(opt, env)
        # These are the default tacotron values for the MEL spectrogram.
        self.filter_length = opt_get(opt, ['filter_length'], 1024)
        self.hop_length = opt_get(opt, ['hop_length'], 256)
        self.win_length = opt_get(opt, ['win_length'], 1024)
        self.n_mel_channels = opt_get(opt, ['n_mel_channels'], 80)
        self.mel_fmin = opt_get(opt, ['mel_fmin'], 0)
        self.mel_fmax = opt_get(opt, ['mel_fmax'], 8000)
        self.sampling_rate = opt_get(opt, ['sampling_rate'], 22050)
        norm = opt_get(opt, ['normalize'], False)
        self.true_norm = opt_get(opt, ['true_normalization'], False)
        self.mel_stft = torchaudio.transforms.MelSpectrogram(n_fft=self.filter_length, hop_length=self.hop_length,
                                                             win_length=self.win_length, power=2, normalized=norm,
                                                             sample_rate=self.sampling_rate, f_min=self.mel_fmin,
                                                             f_max=self.mel_fmax, n_mels=self.n_mel_channels,
                                                             norm="slaney")
        self.mel_norm_file = opt_get(opt, ['mel_norm_file'], None)
        if self.mel_norm_file is not None:
            self.mel_norms = torch.load(self.mel_norm_file)
        else:
            self.mel_norms = None

    def forward(self, state):
        with torch.no_grad():
            inp = state[self.input]
            # Automatically squeeze out the channels dimension if it is present (assuming mono-audio)
            if len(inp.shape) == 3:
                inp = inp.squeeze(1)
            assert len(inp.shape) == 2
            self.mel_stft = self.mel_stft.to(inp.device)
            mel = self.mel_stft(inp)
            # Perform dynamic range compression
            mel = torch.log(torch.clamp(mel, min=1e-5))
            if self.mel_norms is not None:
                self.mel_norms = self.mel_norms.to(mel.device)
                mel = mel / self.mel_norms.unsqueeze(0).unsqueeze(-1)
            if self.true_norm:
                mel = normalize_torch_mel(mel)
            return {self.output: mel}