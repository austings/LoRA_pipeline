import torch
from dataclasses import dataclass
from torch import nn
import torch.nn.functional as F
from transformers import GPT2Config
from neets_tts.models.tortoise.transformer_builders import build_hf_gpt_transformer
from neets_tts.models.tortoise.arch_utils import AttentionBlock
from neets_tts.models.tortoise.tortoise_inference import GPT2InferenceModel
@dataclass
class TortoiseConfig:
    layers: int = 30
    model_dim: int = 1024
    heads: int = 16
    max_text_tokens: int = 600  # 402
    max_mel_tokens: int  = 1015 # 604
    max_conditioning_inputs: int = 2
    mel_length_compression: int = 1024
    number_text_tokens: int = 256
    start_text_token: int = 255
    stop_text_token: int = 0
    number_mel_codes: int = 8194
    start_mel_token: int = 8192
    stop_mel_token: int = 8193
    checkpointing: bool = True
    num_mels: int = 80

class ResBlock(nn.Module):
    def __init__(self, chan):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(chan, chan, kernel_size=3, padding=1),
            nn.GroupNorm(chan // 8, chan),
            nn.ReLU(),
            nn.Conv1d(chan, chan, kernel_size=3, padding=1),
            nn.GroupNorm(chan // 8, chan)
        )

    def forward(self, x):
        return F.relu(self.net(x) + x)

class ConditioningEncoder(nn.Module):
    def __init__(self, spec_dim, embedding_dim, attn_blocks=6, num_attn_heads=4, do_checkpointing=False, mean=False):
        super().__init__()
        self.init = nn.Conv1d(spec_dim, embedding_dim, kernel_size=1)
        self.attn = nn.Sequential(*[AttentionBlock(embedding_dim, num_attn_heads, do_checkpoint=do_checkpointing)
                                    for _ in range(attn_blocks)])
        self.mean = mean

    def forward(self, x):
        h = self.init(x)
        h = self.attn(h)
        return h.mean(dim=2) if self.mean else h[:, :, 0]

class Tortoise(nn.Module):
    def __init__(self, config: TortoiseConfig):
        super().__init__()
        self.config = config

        self.conditioning_encoder = ConditioningEncoder(config.num_mels, config.model_dim, num_attn_heads=config.heads)
        self.text_embedding = nn.Embedding(config.number_text_tokens, config.model_dim)
        self.mel_embedding = nn.Embedding(config.number_mel_codes, config.model_dim)

        self.gpt, self.mel_pos_embedding, self.text_pos_embedding, _, _ = build_hf_gpt_transformer(
            config.layers, config.model_dim, config.heads, config.max_mel_tokens + 2 + config.max_conditioning_inputs,
            config.max_text_tokens + 2, config.checkpointing)

        self.final_norm = nn.LayerNorm(config.model_dim)
        self.text_head = nn.Linear(config.model_dim, config.number_text_tokens)
        self.mel_head = nn.Linear(config.model_dim, config.number_mel_codes)

        self._init_embeddings()

    def _init_embeddings(self):
        for module in [self.text_embedding, self.mel_embedding]:
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, speech_conditioning_input, text_inputs, text_lengths, mel_codes, wav_lengths,
                return_latent=False):
        wav_lengths *= self.config.mel_length_compression
        text_inputs = self._pad_inputs(text_inputs, text_lengths, self.config.stop_text_token)
        mel_codes = self._pad_inputs(mel_codes, wav_lengths // self.config.mel_length_compression,
                                     self.config.stop_mel_token)
        mel_codes = self._set_padding(mel_codes, wav_lengths)

        conds = self._encode_conditioning(speech_conditioning_input)

        text_inputs, text_targets = self._align_inputs_and_targets(
            text_inputs, self.config.start_text_token, self.config.stop_text_token)
        text_emb = self.text_embedding(text_inputs) + self.text_pos_embedding(text_inputs)

        mel_codes, mel_targets = self._align_inputs_and_targets(
            mel_codes, self.config.start_mel_token, self.config.stop_mel_token)
        mel_emb = self.mel_embedding(mel_codes) + self.mel_pos_embedding(mel_codes)

        text_logits, mel_logits = self._get_logits(conds, text_emb, mel_emb, return_latent)

        if return_latent:
            return mel_logits[:, :-1]

        loss_text = F.cross_entropy(text_logits, text_targets.long()).mean()
        loss_mel = F.cross_entropy(mel_logits, mel_targets.long()).mean()
        # print(mel_logits.shape, mel_targets.shape)
        # print(text_logits.shape, text_targets.shape)
        return loss_text.mean(), loss_mel.mean(), mel_logits

    def _pad_inputs(self, inputs, lengths, pad_value):
        max_len = lengths.max()
        return F.pad(inputs[:, :max_len], (0, 1), value=pad_value)

    def _set_padding(self, mel_input_tokens, wav_lengths):
        mel_lengths = wav_lengths // self.config.mel_length_compression
        for b in range(len(mel_lengths)):
            actual_end = mel_lengths[b] + 1
            if actual_end < mel_input_tokens.shape[-1]:
                mel_input_tokens[b, actual_end:] = self.config.stop_mel_token
        return mel_input_tokens

    def _encode_conditioning(self, speech_conditioning_input):
        if len(speech_conditioning_input.shape) == 3:
            speech_conditioning_input = speech_conditioning_input.unsqueeze(1)
        conds = [self.conditioning_encoder(speech_conditioning_input[:, j])
                 for j in range(speech_conditioning_input.shape[1])]
        return torch.stack(conds, dim=1).mean(dim=1).unsqueeze(1)

    def _align_inputs_and_targets(self, input, start_token, stop_token):
        inp = F.pad(input, (1, 0), value=start_token)
        tar = F.pad(input, (0, 1), value=stop_token)
        return inp, tar

    def _get_logits(self, speech_conditioning_inputs, text_tokens, speech_tokens, return_latent=False):
        emb = torch.cat([speech_conditioning_inputs, text_tokens, speech_tokens], dim=1)
        gpt_out = self.gpt(inputs_embeds=emb, return_dict=True)
        enc = self.final_norm(gpt_out.last_hidden_state[:, speech_conditioning_inputs.shape[1]:])

        if return_latent:
            return enc[:, :text_tokens.shape[1]], enc[:, -speech_tokens.shape[1]:]

        text_logits = self.text_head(enc[:, :text_tokens.shape[1]]).permute(0, 2, 1)
        speech_logits = self.mel_head(enc[:, -speech_tokens.shape[1]:]).permute(0, 2, 1)
        return text_logits, speech_logits

    def inference_speech(self, speech_conditioning_input, text_inputs, return_attentions=False, **hf_generate_kwargs):
        seq_length = self.config.max_mel_tokens + self.config.max_text_tokens + 2 if self.config.max_mel_tokens != -1 else 2002

        if not hasattr(self, 'inference_model'):
            gpt_config = GPT2Config(vocab_size=self.config.max_mel_tokens,
                                    n_positions=seq_length,
                                    n_ctx=seq_length,
                                    n_embd=self.config.model_dim,
                                    n_layer=self.config.layers,
                                    n_head=self.config.heads,
                                    gradient_checkpointing=False,
                                    use_cache=True)
            self.inference_model = GPT2InferenceModel(
                gpt_config, self.gpt, self.mel_pos_embedding, self.mel_embedding, self.final_norm, self.mel_head)
            self.gpt.wte = self.mel_embedding

        text_inputs = F.pad(text_inputs, (0, 1), value=self.config.stop_text_token)
        text_inputs, _ = self._align_inputs_and_targets(text_inputs, self.config.start_text_token,
                                                        self.config.stop_text_token)
        text_emb = self.text_embedding(text_inputs) + self.text_pos_embedding(text_inputs)

        if len(speech_conditioning_input.shape) == 3:
            speech_conditioning_input = speech_conditioning_input.unsqueeze(1)
        conds = torch.stack([self.conditioning_encoder(speech_conditioning_input[:, j])
                             for j in range(speech_conditioning_input.shape[1])], dim=1).mean(dim=1).unsqueeze(1)

        emb = torch.cat([conds, text_emb], dim=1)
        self.inference_model.store_mel_emb(emb)

        fake_inputs = torch.full((emb.shape[0], conds.shape[1] + emb.shape[1],),
                                 fill_value=1, dtype=torch.long, device=text_inputs.device)
        fake_inputs[:, -1] = self.config.start_mel_token

        gen = self.inference_model.generate(fake_inputs, bos_token_id=self.config.start_mel_token,
                                            pad_token_id=self.config.stop_mel_token,
                                            eos_token_id=self.config.stop_mel_token,
                                            max_length=seq_length, output_attentions=return_attentions,
                                            return_dict_in_generate=True, **hf_generate_kwargs)
        return gen.sequences[:, fake_inputs.shape[1]:], gen.attentions if return_attentions else gen.sequences[:, fake_inputs.shape[1]:]