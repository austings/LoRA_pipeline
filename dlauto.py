import argparse
import deepspeed
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import functools
import numpy as np
import re


from deepspeed.runtime.zero.config import DeepSpeedZeroConfig
from huggingface_hub import hf_hub_download
from tortoise.models.arch_util import AttentionBlock, TorchMelSpectrogram
from tortoise.models.diffusion_decoder import DiffusionTts
from tortoise.models.random_latent_generator import RandomLatentConverter
from tortoise.models.clvp import CLVP
from transformers.utils.model_parallel_utils import get_device_map, assert_device_map
from tortoise.utils.tokenizer import VoiceBpeTokenizer         
from tortoise.utils.typical_sampling import TypicalLogitsWarper
from tortoise.utils.audio import load_voices, wav_to_univnet_mel, denormalize_tacotron_mel, TacotronSTFT
import concurrent.futures
from transformers import LogitsProcessorList, GPT2PreTrainedModel, GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
#from peft import AutoPeftModelForCausalLM
from safetensors import safe_open

from peft import PeftModel

#                           #
#       Global Defaults     #
#                           #

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

autoregressive_model_path = hf_hub_download(
  "jbetker/tortoise-tts-v2",
  ".models/autoregressive.pth"
)

def null_position_embeddings(range, dim):
    return torch.zeros((range.shape[0], range.shape[1], dim), device=range.device)

#                                  #
#       Configuration Class        #
#                                  #


class GenerationConfig:
    def __init__(self, use_path_model=False, use_deterministic_seed=False, temperature=.8, length_penalty=1, repetition_penalty=2.0, top_p=.8, max_mel_tokens=604,**kwargs):
        super().__init__()
        # note make this more loosely coupled this is jeetin ugly
        weights_key = []
        weights_shapes = []
        past_keys_values = {}
        self.layers = 30
        if use_path_model:
            model = torch.load('./LoRA_pipeline/lora_data/500_gpt.pth')
            #model = torch.load(get_model_path('autoregressive.pth', MODELS_DIR))
            #model = PeftModel.from_pretrained(get_model_path('autoregressive.pth', MODELS_DIR), peft_model_id)
            #base_model = AutoModelForCausalLM.from_pretrained(autoregressive_model_path)

            #model = PeftModel.from_pretrained(self.autoregressive.pre_config_module.gpt, peft_model_id)
            #self.autoregressive.pre_config_module.gpt = model.merge_and_unload()
            #model.load_adapter(peft_model_id)

            #UNCOMMENT THIS IF YOU WANT TO PASS ATTENTION/POSITIONAL IDS
            #past_shape = [1, 16, 606, 1024] #model.config.batch_size, # model.config.n_head (heads) #model.config.n_ctx (max_mel_tokens+2) #model_dim

            # Iterate over the model parameters and save their shapes
            # Regular expression pattern to save feedforward weights only
            #pattern = re.compile(r'^gpt\.h\.\d+\.mlp\.c_fc\.weight$')
            #pattern2 = re.compile(r'^gpt\.h\.\d+\.mlp\.c_proj\.weight$')

            #for i in range(self.layers):
            #    weights_key.append(f"past_key{i}")
            #    weights_key.append(f"past_value{i}")

            #for k in model.keys(): #this is the jeetin ugly thing but efficent. relies on the weights being in order
            #    if pattern.match(k) or pattern2.match(k):
            #         value = model[k]
            #         weights_shapes.append(value.cpu().numpy())
            
            #for i in range(self.layers*2):
            #    past_keys_values[weights_key[i]] = weights_shapes[i]
    
            #for i in range(self.layers):
            #    weights_key.append(f"{i:02d}")
            #i_sub = 0
            #i = 0
            #for k in model.keys():
            #    # Create keys for past keys and values by creating a 'tuple' from the state dict
            #    if pattern.match(k) or pattern2.match(k):
            #        value = model[k]
            #        weights_shapes.append(value)
            #        if(i_sub==0):    
            #            i_sub=1
            #        else:
            #            past_keys_dict[weights_key[i]] = weights_shapes
            #            weights_shapes = []
            #            i_sub=0
            #            i = i+1

        self.tokenizer = VoiceBpeTokenizer(vocab_file=None, use_basic_cleaners=False, )                        
        self.pre_config_module = self.ConfigModule(max_mel_tokens=max_mel_tokens, max_text_tokens=402,         
                                                   max_conditioning_inputs=2, weights_shapes=past_keys_values,   
                                                   layers=self.layers, model_dim=1024, heads=16, number_text_tokens=255,
                                                   start_text_token=255, checkpointing=False,                  
                                                   train_solo_embeddings=False).to('cuda').eval()              
        

        self.pre_config_module.load_state_dict(model, strict=False)

        # model parameters
        self.deterministic_seed = use_deterministic_seed
        self.stop_token = 8193
        self.stop_mel_token = 8193
        self.max_mel_tokens_pad = 500
        self.dim_text = 768
        self.dim_speech = 768
        self.dim_latent = 768
        self.calm_token = 83
        self.num_text_tokens = 256
        self.text_enc_depth = 20
        self.text_seq_len = 350
        self.text_heads = 12
        self.num_speech_tokens = 8192
        self.speech_enc_depth = 20
        self.speech_heads = 12
        self.speech_seq_len = 430
        self.types = 1
        self.k = 1
        self.num_return_sequences = 1
        self.autoregressive_batch_size = 1
        self.length_penalty = length_penalty
        self.repetition_penalty = repetition_penalty
        self.typical_mass = .9
        self.temperature = temperature
        self.top_p = top_p
        self.std = .02
        self.mean = 0.0
        self.device = torch.device('cuda')
        self.checkpointing = False
        self.train_solo_embeddings = False
        self.return_loss = False
        self.use_mel_codes_as_input = True
        self.do_sample = True
        self.use_xformers = True
        self.typical_sampling = False
        self.input_tokens = None
        self.max_generate_length = self.max_mel_tokens_pad
        self.conditioning_cache = None
        self.stft = None
        self.clvp = self.load_clvp_model()

    class ConfigModule(nn.Module):
        def __init__(self, layers=8, model_dim=512, heads=8, max_text_tokens=120, max_mel_tokens=250,
                     max_conditioning_inputs=2, weights_shapes=None,
                     mel_length_compression=1024, number_text_tokens=256,
                     start_text_token=None, number_mel_codes=8194, start_mel_token=8192,
                     stop_mel_token=8193, train_solo_embeddings=False, use_mel_codes_as_input=True,
                     checkpointing=False, types=1):
            super().__init__()
            self.number_text_tokens = number_text_tokens
            self.weights_shapes = weights_shapes
            self.start_text_token = number_text_tokens * types if start_text_token is None else start_text_token
            self.stop_text_token = 0
            self.number_mel_codes = number_mel_codes
            self.start_mel_token = start_mel_token
            self.stop_mel_token = stop_mel_token
            self.layers = layers
            self.heads = heads
            self.max_mel_tokens = max_mel_tokens
            self.max_text_tokens = max_text_tokens
            self.model_dim = model_dim
            self.max_conditioning_inputs = max_conditioning_inputs
            self.mel_length_compression = mel_length_compression

            self.conditioning_encoder = self.ConditioningEncoder(spec_dim=80, embedding_dim=model_dim, num_attn_heads=heads)
            self.text_embedding = nn.Embedding(self.number_text_tokens * types + 1, model_dim)
            if use_mel_codes_as_input:
                self.mel_embedding = nn.Embedding(self.number_mel_codes, model_dim)
            else:
                self.mel_embedding = self.MelEncoder(model_dim, resblocks_per_reduction=1)
            self.gpt, self.mel_pos_embedding, self.text_pos_embedding, self.mel_layer_pos_embedding, self.text_layer_pos_embedding = \
                self.build_hf_gpt_transformer(layers, model_dim, heads,self.max_mel_tokens + 2 + self.max_conditioning_inputs, self.max_text_tokens + 2, checkpointing )
            if train_solo_embeddings:
                self.mel_solo_embedding = nn.Parameter(torch.randn(1, 1, model_dim) * .02, requires_grad=True)
                self.text_solo_embedding = nn.Parameter(torch.randn(1, 1, model_dim) * .02, requires_grad=True)
            else:
                self.mel_solo_embedding = 0
                self.text_solo_embedding = 0
            self.position_ids = self.text_embedding
            self.final_norm = nn.LayerNorm(model_dim)
            self.text_head = nn.Linear(model_dim, self.number_text_tokens * types + 1)
            self.mel_head = nn.Linear(model_dim, self.number_mel_codes)

            # Initialize the embeddings per the GPT-2 scheme
            embeddings = [self.text_embedding]
            if use_mel_codes_as_input:
                embeddings.append(self.mel_embedding)
            for module in embeddings:
                module.weight.data.normal_(mean=0.0, std=.02)



        def set_mel_padding(self, mel_input_tokens, wav_lengths):
            # Set padding areas within MEL (currently it is coded with the MEL code for <zero>).
            mel_lengths = np.floor_divide(wav_lengths, 1024)
            actual_ends = mel_lengths + 1  # Due to the convolutional nature of how these tokens are generated, it would be best if the model predicts a token past the actual last token.
            max_length = mel_input_tokens.shape[-1]
            mask = np.arange(max_length) >= actual_ends[:, np.newaxis]
            mel_input_tokens = mel_input_tokens.masked_fill(torch.tensor(mask).to("cuda"), 8193)  # self.stop_mel_token, needs optimize for cuda cause it gets sent to cpu
            return mel_input_tokens



        def get_logits(self, input_ids, speech_conditioning_inputs, first_inputs, first_head, second_inputs=None, second_head=None,
                       get_attns=False, return_latent=False):

            emb = torch.cat([speech_conditioning_inputs, first_inputs, second_inputs], dim=1)
            gpt_out = self.gpt(inputs_embeds=emb, return_dict=True, output_attentions=False)

            enc = gpt_out.last_hidden_state[:, 1:]  # The first logit is tied to the speech_conditioning_input
            enc = self.final_norm(enc)

            return enc[:,speech_conditioning_inputs.shape[1]:speech_conditioning_inputs.shape[1] + first_inputs.shape[
                           1]], enc[:, -second_inputs.shape[1]:]


        def forward(self, speech_conditioning_latent, text_inputs, mel_codes, types=None,
                    text_first=True, raw_mels=None, return_attentions=False,
                    return_latent=True, clip_inputs=False):

            text_inputs = text_inputs.repeat(1, 1)
            mel_codes = self.set_mel_padding(mel_codes, np.array([mel_codes.shape[-1] * 1024]))
            text_inputs = F.pad(text_inputs, (0, 1), value=self.stop_text_token)
            mel_codes = F.pad(mel_codes, (0, 1), value=self.stop_mel_token)

            conds = speech_conditioning_latent.unsqueeze(1)
            text_inputs, text_targets = self.build_aligned_inputs_and_targets(text_inputs, self.start_text_token,self.stop_text_token)
            text_emb = self.text_embedding(text_inputs) + self.text_pos_embedding(text_inputs)
            mel_codes, mel_targets = self.build_aligned_inputs_and_targets( mel_codes, self.start_mel_token, self.stop_mel_token)

            mel_inp = mel_codes

            mel_emb = self.mel_embedding(mel_inp)
            mel_emb = mel_emb + self.mel_pos_embedding(mel_codes)

            text_logits, mel_logits = self.get_logits(text_inputs, conds, text_emb, self.text_head, mel_emb, self.mel_head,
                                                      get_attns=return_attentions, return_latent=return_latent)

            return mel_logits[:, :-2]  # Despite the name, these are not logits. Strip off the two tokens added by this forward pass.



        #                                       #
        #      Encoder + Embedding Classes      #
        #                                       #


        def build_hf_gpt_transformer(self, layers, model_dim, heads, max_mel_seq_len, max_text_seq_len, checkpointing):
            """
            GPT-2 implemented by the HuggingFace library.
            """
            from transformers import GPT2Config, GPT2Model

            gpt_config = GPT2Config(vocab_size=256,  # Unused.
                                    n_positions=max_mel_seq_len + max_text_seq_len,
                                    n_ctx=max_mel_seq_len + max_text_seq_len,
                                    n_embd=model_dim, n_layer=layers, n_head=heads,
                                    gradient_checkpointing=False, use_cache=True)
            gpt = GPT2Model(gpt_config)
            # Override the built in positional embeddings
            del gpt.wpe
            gpt.wpe = functools.partial(null_position_embeddings, dim=model_dim)
            del gpt.wte
           
            return gpt, self.LearnedPositionEmbeddings(max_mel_seq_len, model_dim), self.LearnedPositionEmbeddings(max_text_seq_len, model_dim),\
                None, None



        def build_aligned_inputs_and_targets(self, input, start_token, stop_token):
            inp = F.pad(input, (1, 0), value=start_token)
            tar = F.pad(input, (0, 1), value=stop_token)
            return inp, tar


        def inference_speech(self, auto_latent, text_in, typical_mass=.9, max_generate_length=None, max_mel_tokens_pad=500, typical_sampling=False, num_return_sequences=1, **hf_generate_kwargs):
            text_inputs_fix = F.pad(text_in, (0, 1), value=self.stop_text_token)
            text_inputs_fix, _ = self.build_aligned_inputs_and_targets(text_inputs_fix,self.start_text_token,self.stop_text_token)
            # move everything to cuda
            text_inputs_fix = text_inputs_fix.to("cuda")
            self.text_embedding = self.text_embedding.to("cuda")
            self.text_pos_embedding = self.text_pos_embedding.to("cuda")

            text_emb = self.text_embedding(text_inputs_fix) + self.text_pos_embedding(text_inputs_fix)
            conds = auto_latent.unsqueeze(1)
            emb = torch.cat([conds, text_emb], dim=1)
            self.inference_model.store_mel_emb(emb)
            self.inference_model.to('cuda')
            fake_inputs = torch.full((emb.shape[0], conds.shape[1] + emb.shape[1],), fill_value=1, dtype=torch.long,
                                     device='cuda')
            fake_inputs[:, -1] = self.start_mel_token
            trunc_index = fake_inputs.shape[1]
            inputs = fake_inputs
            logits_processor = LogitsProcessorList(
                [TypicalLogitsWarper(mass=typical_mass)]) if typical_sampling else LogitsProcessorList()
            max_length = trunc_index + max_mel_tokens_pad - 1 if max_generate_length is None else trunc_index + max_mel_tokens_pad
            gen = self.inference_model.generate(inputs, bos_token_id=self.start_mel_token,
                                                pad_token_id=self.stop_mel_token, eos_token_id=self.stop_mel_token,
                                                max_length=max_length, logits_processor=logits_processor,
                                                num_return_sequences=num_return_sequences,
                                                **hf_generate_kwargs).to('cuda')
            return gen[:, trunc_index:]
            
        # Function to load safetensors adapter
        def load_adapter(self, model, adapter_path):
            with safe_open(adapter_path, framework="pt", device="cpu") as f:
                adapter_weights = {k: torch.tensor(f.get_tensor(k)) for k in f.keys()}
                
            # Assuming the adapter weights should be applied to the base model
            # This may involve setting specific layers or modules within the base model
            i = 0
            alpha = 128
            for i, layer in enumerate(model.h):
                if isinstance(layer, GPT2Block):
                    key  = f'base_model.model.h.{i}.attn.c_attn.lora_A.weight.' 
                    key2 = f'base_model.model.h.{i}.attn.c_attn.lora_B.weight.' 
                    key3 = f'base_model.model.h.{i}.attn.c_proj.lora_A.weight.' 
                    key4 = f'base_model.model.h.{i}.attn.c_proj.lora_B.weight.' 
                    key5 = f'base_model.model.h.{i}.mlp.c_fc.lora_A.weight.' 
                    key6 = f'base_model.model.h.{i}.mlp.c_fc.lora_B.weight.' 
                    key7 = f'base_model.model.h.{i}.mlp.c_proj.lora_A.weight.' 
                    key8 = f'base_model.model.h.{i}.mlp.c_proj.lora_B.weight.' 
                    #layer.attn.c_attn.weight =  + 
                    lora_update = (adapter_weights[key] @ adapter_weights[key2]) * (alpha / adapter_weights[key].size(0))
                    layer.attn.c_attn.weight = adapter_weights[key]
                    layer.attn.c_attn.weight = adapter_weights[key]
                    layer.attn.c_attn.weight = adapter_weights[key]
                    layer.attn.c_attn.weight = adapter_weights[key]
                    layer.attn.c_attn.weight = adapter_weights[key]
                    layer.attn.c_attn.weight = adapter_weights[key]
                    if layer in adapter_weights:
                        model.data = adapter_weights[layer].data
            return model
          

        def post_init(self):
            seq_length = self.max_mel_tokens + self.max_text_tokens + 2
            gpt_config = GPT2Config(
                vocab_size=self.max_mel_tokens,
                n_positions=seq_length,
                n_ctx=seq_length,
                n_embd=self.model_dim,
                n_layer=self.layers,
                n_head=self.heads,
                gradient_checkpointing=False,
                use_cache=True,
            )

            self.tensor_zero_config = DeepSpeedZeroConfig(
                stage=1,
                offload_optimizer={
                    "device": "cpu",  # Offload optimizer state to CPU
                    "pin_memory": "True",
                    "fast_init": "True"
                },
                reduce_scatter=True,
                reduce_bucket_size=32,
                contiguous_gradients=True,
                cpu_offload_use_pin_memory=True,
                use_multi_rank_bucket_allreduce=True,
                allgather_partitions=True,
                allgather_bucket_size=32,
                overlap_comm=True,
                load_from_fp32_weights=False
            )

            #with open('output1.txt', 'a') as file:
            #    for i, layer in enumerate(self.gpt.h):
            #        print(f"\nLayer {i + 1} attention details:", file=file)
            #        print(layer.attn, file=file)
            #        print(layer.attn.c_attn.weight.data, file=file)
            #peft_model_id = "./LoRA_pipeline/tortoise_mod/alpha128"
            #"./LoRA_pipeline/tortoise_mod"
            #self.gpt = PeftModel.from_pretrained(self.gpt, peft_model_id, device_map="auto",use_safetensors=True).base_model
            #self.gpt.add_weighted_adapter(
            #    adapters=["default"], #define the name in lora config
            #    weights=[1],
            #    adapter_name="combined",
            #    combination_type="svd",
            #)
            # Load the adapter into the base model
            #self.gpt = self.load_adapter(self.gpt, "./LoRA_pipeline/tortoise_mod/220/adapter_model.safetensors")
            #self.gpt = self.gpt.merge_and_unload() #WHEN DONE DO THIS
            #print("NEW:\n")
            #with open('output2.txt', 'a') as file:
            #    for i, layer in enumerate(self.gpt.h):
            #        print(f"\nLayer {i + 1} attention details:", file=file)
            #        print(layer.attn, file=file)
            #        print(layer.attn.c_attn.weight.data, file=file)
            #        print("\nlorab:\n",file=file)
            #        print(layer.attn.c_attn.lora_B.default.weight,file=file)
            #        print("\nloraa:\n",file=file)
            #        print(layer.attn.c_attn.lora_A.default.weight,file=file)
            #
            #    #print(layer.attn.attn_dropout)
                #print(layer.attn.resid_dropout)
            print("\n\n\n")
            #print(self.gpt)

            self.inference_model = self.GPT2InferenceModel(
                gpt_config,
                self.gpt,
                self.mel_pos_embedding,
                self.mel_embedding,
                self.final_norm,
                self.mel_head,
                kv_cache=True,
            )

            #self.inference_model = PeftModel.from_pretrained(self.inference_model, peft_model_id, device_map="auto",use_safetensors=True).base_model
            #self.inference_model.merge_and_unload()
            self.ds_engine = deepspeed.init_inference(model=self.inference_model, replace_with_kernel_inject=True,
                                                      dtype=torch.float32, zero=self.tensor_zero_config)
            self.inference_model = self.ds_engine.module.eval()
            #self.inference_model = self.inference_model.eval()
            self.gpt.wte = self.mel_embedding



        class MelEncoder(nn.Module):
            def __init__(self, channels, mel_channels=80, resblocks_per_reduction=2):
                super().__init__()
                self.channels = channels

                # Define functions for asynchronous initialization

                def create_conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
                    return nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)


                def create_group_norm(num_groups, num_channels):
                    return nn.GroupNorm(num_groups, num_channels)


                def create_relu():
                    return nn.ReLU()


                def create_resblocks(channels):
                    return nn.Sequential(*[self.ResBlock(channels) for _ in range(resblocks_per_reduction)])

                    # Use ThreadPoolExecutor to asynchronously initialize the modules

                with torch.cuda.stream(torch.cuda.current_stream()):
                    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                        # Submit tasks to the executor
                        conv1_future = executor.submit(create_conv_block, mel_channels, channels // 4, 3, 1, 1)
                        resblocks1_future = executor.submit(create_resblocks, channels // 4)
                        conv2_future = executor.submit(create_conv_block, channels // 4, channels // 2, 3, 2, 1)
                        group_norm_future1 = executor.submit(create_group_norm, channels // 16, channels // 2)
                        relu1_future = executor.submit(create_relu)
                        resblocks2_future = executor.submit(create_resblocks, channels // 2)
                        conv3_future = executor.submit(create_conv_block, channels // 2, channels, 3, 2, 1)
                        group_norm_future2 = executor.submit(create_group_norm, channels // 8, channels)
                        relu2_future = executor.submit(create_relu)
                        resblocks3_future = executor.submit(create_resblocks, channels)

                    # Retrieve results
                conv1 = conv1_future.result().cuda()
                resblocks1 = resblocks1_future.result().cuda()
                conv2 = conv2_future.result().cuda()
                group_norm1 = group_norm_future1.result().cuda()
                relu1 = relu1_future.result().cuda()
                resblocks2 = resblocks2_future.result().cuda()
                conv3 = conv3_future.result().cuda()
                group_norm2 = group_norm_future2.result().cuda()
                relu2 = relu2_future.result().cuda()
                resblocks3 = resblocks3_future.result().cuda()

                # Define the network architecture
                self.encoder = nn.Sequential(
                    conv1,
                    resblocks1,
                    conv2,
                    group_norm1,
                    relu1,
                    resblocks2,
                    conv3,
                    group_norm2,
                    relu2,
                    resblocks3
                )
                self.reduction = 4


            def forward(self, x):
                for e in self.encoder:
                    x = e(x)
                return x.permute(0, 2, 1)


        class ConditioningEncoder(nn.Module):
            def __init__(self, spec_dim, embedding_dim, attn_blocks=6, num_attn_heads=4, do_checkpointing=False,
                         mean=False):
                super().__init__()
                attn = []
                self.init = nn.Conv1d(spec_dim, embedding_dim, kernel_size=1)
                for a in range(attn_blocks):
                    attn.append(AttentionBlock(embedding_dim, num_attn_heads))
                self.attn = nn.Sequential(*attn)
                self.dim = embedding_dim
                self.do_checkpointing = do_checkpointing
                self.mean = mean


            def forward(self, x):
                h = self.init(x)
                h = self.attn(h)
                if self.mean:
                    return h.mean(dim=2)
                else:
                    return h[:, :, 0]


        class LearnedPositionEmbeddings(nn.Module):

            def __init__(self, seq_len, model_dim, init=.02):
                super().__init__()
                self.emb = nn.Embedding(seq_len, model_dim)
                # Initializing this way is standard for GPT-2
                self.emb.weight.data.normal_(mean=0.0, std=init)


            def forward(self, x):
                sl = x.shape[1]
                return self.emb(torch.arange(0, sl, device=x.device)) #arange not supported?


            def get_fixed_embedding(self, ind, dev):
                return self.emb(torch.tensor([ind], device=dev)).unsqueeze(0)


        class GPT2InferenceModel(GPT2PreTrainedModel):

            def __init__(self, config, gpt, text_pos_emb, embeddings, norm, linear, kv_cache=False):
                super().__init__(config)
                self.transformer = gpt
                self.text_pos_embedding = text_pos_emb
                self.embeddings = embeddings
                #self.embeddingsA = embeddingsA
                #self.embeddingsB = embeddingsB
                self.final_norm = norm
                self.lm_head = nn.Sequential(norm, linear)
                self.kv_cache = kv_cache
                self.model_parallel = False
                self.device_map = None
                self.cached_mel_emb = None

            def parallelize(self, device_map=None):
                self.device_map = (
                    get_device_map(len(self.transformer.h), range(max(1, torch.cuda.device_count())))
                    if device_map is None
                    else device_map)
                assert_device_map(self.device_map, len(self.transformer.h))
                self.transformer.parallelize(self.device_map)
                self.lm_head = self.lm_head.to(self.transformer.first_device)
                self.model_parallel = True

            def deparallelize(self):
                self.transformer.deparallelize()
                self.transformer = self.transformer.to("cpu")
                self.lm_head = self.lm_head.to("cpu")
                self.model_parallel = False
                torch.cuda.empty_cache()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()

            def get_output_embeddings(self):
                return self.lm_head

            def set_output_embeddings(self, new_embeddings):
                self.lm_head = new_embeddings

            def store_mel_emb(self, mel_emb):
                self.cached_mel_emb = mel_emb

            def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
                token_type_ids = kwargs.get("token_type_ids", None)  # usually None
                if not self.kv_cache:
                    past_key_values = None
                # only last token for inputs_ids if past is defined in kwargs
                if past_key_values:
                    input_ids = input_ids[:, -1].unsqueeze(-1)
                    if token_type_ids is not None:
                        token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

                attention_mask = kwargs.get("attention_mask", None)
                position_ids = kwargs.get("position_ids", None)

                if attention_mask is not None and position_ids is None:
                    # create position_ids on the fly for batch generation
                    position_ids = attention_mask.long().cumsum(-1) - 1
                    position_ids.masked_fill_(attention_mask == 0, 1)
                    if past_key_values:
                        position_ids = position_ids[:, -1].unsqueeze(-1)
                else:
                    position_ids = None
                return {
                    "input_ids": input_ids,
                    "past_key_values": past_key_values,
                    "use_cache": kwargs.get("use_cache"),
                    "position_ids": position_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids,
                }


            def forward(self, input_ids=None, past_key_values=None, attention_mask=None, token_type_ids=None,
                        position_ids=None,
                        head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None,
                        labels=None, use_cache=None, output_attentions=None, output_hidden_states=None,
                        return_dict=None):
                mel_len = self.cached_mel_emb.shape[1]
                if input_ids.shape[1] != 1:
                    text_inputs = input_ids[:, mel_len:]
                    text_emb = self.embeddings(text_inputs)
                    text_emb = text_emb + self.text_pos_embedding(text_emb)
                    if self.cached_mel_emb.shape[0] != text_emb.shape[0]:
                        mel_emb = self.cached_mel_emb.repeat_interleave(
                            text_emb.shape[0] // self.cached_mel_emb.shape[0], 0)
                    else:  # this outcome only occurs once per loop in most cases
                        mel_emb = self.cached_mel_emb
                    emb = torch.cat([mel_emb, text_emb], dim=1)
                else:
                    emb = self.embeddings(input_ids)
                    emb = emb + self.text_pos_embedding.get_fixed_embedding(attention_mask.shape[1] - mel_len,
                                                                            attention_mask.device)

                transformer_outputs = self.transformer(inputs_embeds=emb)
                hidden_states = transformer_outputs[0]
                lm_logits = self.lm_head(hidden_states)
                return CausalLMOutputWithCrossAttentions(loss=None, logits=lm_logits,
                                                         past_key_values=transformer_outputs.past_key_values,
                                                         attentions=transformer_outputs.attentions,
                                                         cross_attentions=transformer_outputs.cross_attentions)

            @staticmethod
            def _reorder_cache(past, beam_idx):
                """
                This function is used to re-order the :obj:`past_key_values` cache if
                :meth:`~transformers.PreTrainedModel.beam_search` or :meth:`~transformers.PreTrainedModel.beam_sample` is
                called. This is required to match :obj:`past_key_values` with the correct beam_idx at every generation step.
                """
                return tuple(
                    tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past) for
                    layer_past in past)

        class ResBlock(nn.Module):
            def __init__(self, chan):
                super().__init__()

                def create_conv_block(chan):
                    return nn.Conv1d(chan, chan, kernel_size=3, padding=1)


                def create_group_norm(chan):
                    return nn.GroupNorm(chan // 8, chan)


                def create_relu():
                    return nn.ReLU()

                with torch.cuda.stream(torch.cuda.current_stream()):
                    # Submit tasks to the executor
                    conv_block_future1 = create_conv_block(chan).cuda()
                    group_norm_future1 = create_group_norm(chan).cuda()
                    relu_future1 = create_relu().cuda()

                    conv_block_future2 = create_conv_block(chan).cuda()
                    group_norm_future2 = create_group_norm(chan).cuda()

                # Define the network architecture
                self.net = nn.Sequential(
                    conv_block_future1, group_norm_future1, relu_future1,
                    conv_block_future2, group_norm_future2
                )


            def forward(self, x):
                return F.relu(self.net(x) + x)



    def prepare_inference_tts(self, text_inputs,auto_latents):
        self.text_inputs = self.get_random_text_inputs(text_inputs)
        self.mel_codes = self.get_random_mel_codes(auto_latents)
        #self.auto_latent = self.auto_latent.repeat(self.k, 1)

    #                                         #
    #       Generate Latents and Text         #
    #                                         #

    def get_random_text_inputs(self,text_inputs):
        return torch.IntTensor(self.tokenizer.encode(text_inputs)).unsqueeze(
            0).to('cuda')

    def format_conditioning(self,clip, cond_length=132300,device="cuda" if not torch.backends.mps.is_available() else 'mps'):
        gap = clip.shape[-1] - cond_length
        if gap < 0:
            clip = F.pad(clip, pad=(0, abs(gap)))
        elif gap > 0:
            #if self.deterministic_seed:
                #rand_start = 0  # deterministic for testing
            #else:
            rand_start = random.randint(0, gap)
            clip = clip[:, rand_start:rand_start + cond_length]
        mel_clip = TorchMelSpectrogram()(clip.unsqueeze(0)).squeeze(0)
        return mel_clip.unsqueeze(0).to(device)


    def pad_or_truncate(self,t, length):
        if t.shape[-1] == length:
            return t
        elif t.shape[-1] < length:
            return F.pad(t, (0, length - t.shape[-1]))
        else:
            return t[..., :length]


    #                                  #
    #       Generate Mel Codes         #
    #                                  #

    def fix_autoregressive_output(self, codes):
        stop_token_indices = (codes == self.stop_mel_token).nonzero()
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

    def load_clvp_model(self):
        clvp = CLVP(dim_text=self.dim_text, dim_speech=self.dim_speech, dim_latent=self.dim_latent,
                    num_text_tokens=self.num_text_tokens, text_enc_depth=self.text_enc_depth,
                    text_seq_len=self.text_seq_len, text_heads=self.text_heads,
                    num_speech_tokens=self.num_speech_tokens, speech_enc_depth=self.speech_enc_depth,
                    speech_heads=self.speech_heads, speech_seq_len=self.speech_seq_len,
                    use_xformers=self.use_xformers).to(
            'cuda').eval()
        clvp.load_state_dict(torch.load(get_model_path('clvp2.pth', MODELS_DIR)))
        return clvp


    def get_random_mel_codes(self, auto_latents):
        with torch.no_grad():
            samples = []
            #stop_mel_token = self.stop_token #THIS MIGHT NEED TO BE -1 ? Index array
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                codes = self.pre_config_module.inference_speech(auto_latents, self.text_inputs,do_sample=True, top_p=self.top_p, temperature=self.temperature,
                                                                num_return_sequences=self.autoregressive_batch_size, length_penalty=self.length_penalty,
                                                                repetition_penalty=self.repetition_penalty, max_generate_length=500,)
                padding_needed = self.max_mel_tokens_pad - codes.shape[1]
                codes = F.pad(codes, (0, padding_needed), value=self.pre_config_module.stop_mel_token)
                samples.append(codes)
            clip_results = []

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                for batch in samples:
                    for i in range(batch.shape[0]):
                        batch[i] = self.fix_autoregressive_output(batch[i])
                    clvp_out = self.clvp(self.text_inputs.repeat(batch.shape[0], 1), batch, return_loss=self.return_loss)
                    clip_results.append(clvp_out)
                clip_results = torch.cat(clip_results, dim=0)
                samples = torch.cat(samples, dim=0)
                best_results = samples[torch.topk(clip_results, k=self.k).indices]
            del samples
            return best_results


#                                   #
#       Load Stuff Functions        #
#                                   #


def get_model_path(model_name, models_dir=MODELS_DIR):
    model_path = hf_hub_download(repo_id="Manmay/tortoise-tts", filename=model_name, cache_dir=models_dir)
    return model_path




#                         #
#       Execute           #
#                         #

def load_autoregressive_model(auto_latents, text_inputs="Ishmael discusses cetology (the zoological classification and natural history of the whale), and describes the crew members.",auto_conds=None):
    
   
    config = GenerationConfig(use_path_model=True,use_deterministic_seed=True)
    config.pre_config_module.post_init()
    config.prepare_inference_tts(text_inputs, auto_latents)
    return config #.half().eval()

def execute_autoregressive_model():
    config = GenerationConfig(use_path_model=True,use_deterministic_seed=True)
    config.pre_config_module.post_init()
    return config #.half().eval()

def init_autoregressive_model(config, auto_latents, text_inputs="Ishmael discusses cetology (the zoological classification and natural history of the whale), and describes the crew members.",auto_conds=None):
    config.prepare_inference_tts(text_inputs, auto_latents)
    return config 



