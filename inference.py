import dlauto as dlco

import os
import random
from tortoise.utils.audio import load_voices
import torch
import torch.nn.functional as F
import torchaudio
from tortoise.models.diffusion_decoder import DiffusionTts
from tortoise.models.arch_util import TorchMelSpectrogram
from tortoise.models.clvp import CLVP

from tortoise.models.random_latent_generator import RandomLatentConverter
from tortoise.models.vocoder import UnivNetGenerator
from tortoise.utils.audio import wav_to_univnet_mel, denormalize_tacotron_mel, TacotronSTFT
from tortoise.utils.diffusion import SpacedDiffusion, space_timesteps, get_named_beta_schedule
from tortoise.utils.tokenizer import VoiceBpeTokenizer
from tortoise.utils.wav2vec_alignment import Wav2VecAlignment
from huggingface_hub import hf_hub_download
import concurrent.futures
import statistics
from peft import LoraConfig, TaskType
#from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model_state_dict
from peft import PeftModel, LoraModel

pbar = None

from time import time

average_values = []
average_valuesD = []
DEFAULT_MODELS_DIR = os.path.join(os.path.expanduser('~'), '.cache', 'tortoise', 'models')
MODELS_DIR = os.environ.get('TORTOISE_MODELS_DIR', DEFAULT_MODELS_DIR)
MODELS = {
    'autoregressive.pth': 'https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/autoregressive.pth',
    'classifier.pth': 'https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/classifier.pth',
    'clvp2.pth': 'https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/clvp2.pth',
    'cvvp.pth': 'https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/cvvp.pth',
    'diffusion_decoder.pth': 'https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/diffusion_decoder.pth',
    'vocoder.pth': 'https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/vocoder.pth',
    'rlg_auto.pth': 'https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/rlg_auto.pth',
    'rlg_diffuser.pth': 'https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/rlg_diffuser.pth',
    'bigvgan_base_24khz_100band.pth': 'https://huggingface.co/ecker/tortoise-tts-models/resolve/main/models/bigvgan_base_24khz_100band.pth',
    'bigvgan_24khz_100band.pth': 'https://huggingface.co/ecker/tortoise-tts-models/resolve/main/models/bigvgan_24khz_100band.pth',
    'bigvgan_base_24khz_100band.json': 'https://huggingface.co/ecker/tortoise-tts-models/resolve/main/models/bigvgan_base_24khz_100band.json',
    'bigvgan_24khz_100band.json': 'https://huggingface.co/ecker/tortoise-tts-models/resolve/main/models/bigvgan_24khz_100band.json',
}


def get_model_path(model_name, models_dir=MODELS_DIR):
    model_path = hf_hub_download(repo_id="Manmay/tortoise-tts", filename=model_name, cache_dir=models_dir)
    return model_path


def pad_or_truncate(t, length):
    if t.shape[-1] == length:
        return t
    elif t.shape[-1] < length:
        return F.pad(t, (0, length - t.shape[-1]))
    else:
        return t[..., :length]


def load_discrete_vocoder_diffuser(trained_diffusion_steps=4000, desired_diffusion_steps=200, cond_free=True,
                                   cond_free_k=1):
    return SpacedDiffusion(use_timesteps=space_timesteps(trained_diffusion_steps, [desired_diffusion_steps]),
                           model_mean_type='epsilon',
                           model_var_type='learned_range', loss_type='mse',
                           betas=get_named_beta_schedule('linear', trained_diffusion_steps),
                           conditioning_free=cond_free, conditioning_free_k=cond_free_k)


def format_conditioning(clip, cond_length=132300, device="cuda" if not torch.backends.mps.is_available() else 'mps'):
    gap = clip.shape[-1] - cond_length
    if gap < 0:
        clip = F.pad(clip, pad=(0, abs(gap)))
    elif gap > 0:
        rand_start = 0  # deterministic for testing
        # rand_start = random.randint(0, gap)
        clip = clip[:, rand_start:rand_start + cond_length]
    mel_clip = TorchMelSpectrogram()(clip.unsqueeze(0)).squeeze(0)
    return mel_clip.unsqueeze(0).to(device)


def fix_autoregressive_output(codes, stop_token, complain=True):
    stop_token_indices = (codes == stop_token).nonzero()
    if len(stop_token_indices) == 0:
        return codes
    else:
        codes[stop_token_indices] = 83
    stm = stop_token_indices.min().item()
    codes[stm:] = 83
    if stm - 3 < codes.shape[0]:
        codes[-3] = 45
        codes[-2] = 45
        codes[-1] = 248
    return codes


def do_spectrogram_diffusion(diffusion_model, diffuser, latents, conditioning_latents, temperature=1, verbose=True):
    with torch.no_grad():
        output_seq_len = latents.shape[
                             1] * 4 * 24000 // 22050  # This diffusion model converts from 22kHz spectrogram codes to a 24kHz spectrogram signal.

        output_shape = (latents.shape[0], 100, output_seq_len)
        precomputed_embeddings = diffusion_model.timestep_independent(latents, conditioning_latents, output_seq_len,
                                                                      False)
        noise = torch.randn(output_shape, device=latents.device) * temperature
        mel = diffuser.p_sample_loop(diffusion_model, output_shape, noise=noise,
                                     model_kwargs={'precomputed_aligned_embeddings': precomputed_embeddings},
                                     progress=verbose)
        return denormalize_tacotron_mel(mel)[:, :, :output_seq_len]


class TextToSpeech:
    def __init__(self, models_dir=MODELS_DIR, enable_redaction=True,half=False, tokenizer_vocab_file=None, tokenizer_basic=False):
        self.auto_latent, self.diffusion_conditioning = self.decouple_Tuple()
        self.auto_latent = self.auto_latent.to('cuda')
        self.diffusion_conditioning = self.diffusion_conditioning.to('cuda')
        self.models_dir = models_dir
        self.autoregressive_batch_size = 1
        self.enable_redaction = enable_redaction
        self.device = torch.device('cuda')
        self.aligner = Wav2VecAlignment()
        self.tokenizer = VoiceBpeTokenizer(vocab_file=tokenizer_vocab_file, use_basic_cleaners=tokenizer_basic, )
        self.half = half
        self.conditioning_cache = None

        def load_diffusion_model():
            diffusion = DiffusionTts(model_channels=1024, num_layers=10, in_channels=100, out_channels=200,
                                     in_latent_channels=1024, in_tokens=8193, dropout=0, use_fp16=False,
                                     num_heads=16,
                                     layer_drop=0, unconditioned_percentage=0).to('cuda').eval()
            diffusion.load_state_dict(torch.load(get_model_path('diffusion_decoder.pth', models_dir)))
            return diffusion

        def load_clvp_model():
            clvp = CLVP(dim_text=768, dim_speech=768, dim_latent=768, num_text_tokens=256, text_enc_depth=20,
                        text_seq_len=350, text_heads=12,
                        num_speech_tokens=8192, speech_enc_depth=20, speech_heads=12, speech_seq_len=430,
                        use_xformers=True).to(
                'cuda').eval()
            clvp.load_state_dict(torch.load(get_model_path('clvp2.pth', models_dir)))
            return clvp

        def load_vocoder():
            vocoder = UnivNetGenerator().to('cuda')
            vocoder.load_state_dict(
                torch.load(get_model_path('vocoder.pth', models_dir), map_location=torch.device('cuda'))['model_g'])
            vocoder.eval(inference=True)
            return vocoder

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            diffusion_future = executor.submit(load_diffusion_model)
            clvp_future = executor.submit(load_clvp_model)
            vocoder_future = executor.submit(load_vocoder)

        self.autoregressive = dlco.execute_autoregressive_model()
        self.diffusion = diffusion_future.result()
        self.clvp = clvp_future.result()
        self.vocoder = vocoder_future.result()

        self.stft = None  # TacotronSTFT is only loaded if used
        self.rlg_auto = None
        self.rlg_diffusion = None

    def decouple_Tuple(self):
        c: tuple[torch.Tensor, torch.Tensor] = torch.load("./LoRA_pipeline/lora_data/dt/dt_latent.pth", map_location="cpu")#torch.load("./LoRA_pipeline/lora_data/dt/dt_latent.pth", map_location="cpu")
        return c

    def get_conditioning_latents(self, voice_samples, return_mels=False):
        if self.conditioning_cache is not None:
            return self.conditioning_cache
        with torch.no_grad():
            voice_samples = [v.to(self.device) for v in voice_samples]
            self.stft = TacotronSTFT(1024, 256, 1024, 100, 24000, 0, 12000).to('cuda')
            diffusion_conds = []
            for sample in voice_samples:
                sample = torchaudio.functional.resample(sample, 22050, 24000)
                sample = pad_or_truncate(sample, 102400)
                cond_mel = wav_to_univnet_mel(sample.to('cuda'), do_normalization=False, device='cuda', stft=self.stft)
                diffusion_conds.append(cond_mel)

            diffusion_conds = torch.stack(diffusion_conds, dim=1).to(torch.float32)
            self.diffusion = self.diffusion.to('cuda')
            diffusion_latent = self.diffusion.get_conditioning(diffusion_conds)
            self.conditioning_cache = ( diffusion_latent,diffusion_conds)
        return self.conditioning_cache

    def get_random_conditioning_latents(self):
        # Lazy-load the RLG models.
        if self.rlg_auto is None:
            self.rlg_auto = RandomLatentConverter(1024).eval()
            self.rlg_auto.load_state_dict(
                torch.load(get_model_path('rlg_auto.pth', self.models_dir), map_location=torch.device('cpu')))
            self.rlg_diffusion = RandomLatentConverter(2048).eval()
            self.rlg_diffusion.load_state_dict(
                torch.load(get_model_path('rlg_diffuser.pth', self.models_dir), map_location=torch.device('cpu')))
        with torch.no_grad():
            return self.rlg_auto(torch.tensor([0.0])), self.rlg_diffusion(torch.tensor([0.0]))

    def tts_with_preset(self, text, preset='fast', **kwargs):
        settings = {'temperature': .8, 'length_penalty': 1.0, 'repetition_penalty': 2.0,
                    'top_p': .8, 'cond_free_k': 2.0, 'diffusion_temperature': 1.0}
        presets = {
            'ultra_fast': {'num_autoregressive_samples': 1, 'diffusion_iterations': 30, 'cond_free': False},
            'high_quality': {'num_autoregressive_samples': 1, 'diffusion_iterations': 30},
        }
        settings.update(presets[preset])
        settings.update(kwargs)  # allow overriding of preset settings with kwargs
        return self.tts(text, **settings)

    def tts(self, text, voice_samples=None, verbose=True, diffusion_iterations=30, cond_free=False, cond_free_k=2, diffusion_temperature=1.0,**hf_generate_kwargs):
        diff_start_time = time()

        diffuser = load_discrete_vocoder_diffuser(desired_diffusion_steps=diffusion_iterations, cond_free=cond_free,
                                                  cond_free_k=cond_free_k)
        
        self.autoregressive.prepare_inference_tts(text, self.auto_latent)
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.half):
                #l_config = LoraConfig(
                #    r = 16,
                #    lora_alpha = 32,
                #    lora_dropout = .05,
                #    task_type = TaskType.CAUSAL_LM,
                #    target_modules = [
                #        "c_attn","c_proj","c_fc",
                #    ]
                #)
                #best_latents = self.autoregressive.pre_config_module.generate(self.auto_latent, self.autoregressive.text_inputs, self.autoregressive.mel_codes)
                best_latents = self.autoregressive.pre_config_module(self.auto_latent, self.autoregressive.text_inputs, self.autoregressive.mel_codes)
            wav_candidates = []
            for b in range(self.autoregressive.mel_codes.shape[0]):
                codes = self.autoregressive.mel_codes[b].unsqueeze(0)
                latents = best_latents[b].unsqueeze(0)
                ctokens = 0
                for k in range(codes.shape[-1]):
                    if codes[0, k] == 83:
                        ctokens += 1
                    else:
                        ctokens = 0
                    if ctokens > 8:  # 8 tokens gives the diffusion model some "breathing room" to terminate speech.
                        latents = latents[:, :k]
                        break

                mel = do_spectrogram_diffusion(self.diffusion, diffuser, latents, self.diffusion_conditioning,
                                               temperature=diffusion_temperature, verbose=verbose)
                average = time() - diff_start_time

                clip_results = []
                wav = self.vocoder.inference(mel)
                wav_candidates.append(wav.cpu())

            def potentially_redact(clip, text):
                if self.enable_redaction:
                    return self.aligner.redact(clip.squeeze(1), text).unsqueeze(1)
                return clip

            #wav_candidates = [potentially_redact(wav_candidate, text) for wav_candidate in wav_candidates]
            #print('diffuser average time: ', average)
            #WHEN DONE DO THIS:             self.gpt = self.gpt.base_model.merge_and_unload()
            return wav_candidates[0]