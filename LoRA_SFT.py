'''
Austin Sierra
DL Software
the program begins and ends here
Last edit May 8 2024
'''
import sys
import torch
from tortoise.api import UnifiedVoice
from transformers import DataCollatorForLanguageModeling, TrainingArguments, BitsAndBytesConfig
# GPT2Config, GPT2PreTrainedModel, LogitsProcessorList, AutoModelForCausalLM, TFAutoModelForCausalLM,
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.pytorch_utils import Conv1D
from peft import PeftConfig, LoraConfig, LoraModel, get_peft_model, TaskType #, prepare_model_for_kbit_training, PeftModel
#import peft
import argparse

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from huggingface_hub import hf_hub_download
#from bitsandbytes import 
from tortoise.utils.tokenizer import VoiceBpeTokenizer
from trl import SFTTrainer
from accelerate import Accelerator
from datasets import load_dataset
from dlas.utils import options as option
from dlas.train import Trainer
import functools

#torch.cuda.empty_cache()
MODELS_DIR = os.environ.get('TORTOISE_MODELS_DIR', os.path.realpath(os.path.join(os.getcwd(), './models/tortoise/')))
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

model_path = hf_hub_download(
"jbetker/tortoise-tts-v2",
".models/dvae.pth",
revision="hf"
)

mel_norms_path = hf_hub_download(
"jbetker/tortoise-tts-v2",
"tortoise/data/mel_norms.pth",
)


class DL_LoRA:
    def __init__(self):
        print("begin init")
        self.load_tokenizer_json(None)
        self.accelerator = Accelerator()

    def load_tokenizer_json(self, tokenizer_json):
        if hasattr(self,"tokenizer_json") and os.path.samefile(self.tokenizer_json, tokenizer_json):
            return

        self.loading = True
        self.tokenizer_json = tokenizer_json if tokenizer_json else os.path.join(os.path.dirname(os.path.realpath(__file__)), '../LoRA_pipeline/lora_data/tokenizer.json')
        print("Loading tokenizer JSON:", self.tokenizer_json)

        if hasattr(self, 'tokenizer'):
            del self.tokenizer

        self.tokenizer = VoiceBpeTokenizer(vocab_file=self.tokenizer_json)

        self.loading = False
        print(f"Loaded tokenizer")

    def freeze_weights(self, model):
        print("begin freeze")

        parameters = model.parameters()

        for param in parameters:
            param.requires_grad = False  # freeze the model - train adapters later
            if param.ndim == 1:
                # cast the small parameters (e.g. layernorm) to fp32 for stability
                param.data = param.data.to(torch.float32)

        model.gradient_checkpointing_enable()  # reduce number of stored activations
        model.enable_input_require_grads()
        return model
 
    def null_position_embeddings(self, range, dim):
        return torch.zeros((range.shape[0], range.shape[1], dim), device=range.device)

    def train_data(self):
        print("begin train")

        rel_path = "gpt_finetune.yml"
        config_path = rel_path 
        opt = option.parse(config_path, is_train=True)

        trainer = Trainer()
        trainer.rank = -1

        launcher = "none"
        mode = ""

        opt["path"]["pretrain_model_gpt"] = autoregressive_model_path
        opt["path"]["experiments_root"] = os.path.join(os.path.abspath(''), "experiments")
        opt["datasets"]["train"]["path"] = [
        "/tmp/donald-trump-prepared-clips/train.txt"
        ]
        opt["datasets"]["val"]["path"] = [
        "/tmp/donald-trump-prepared-clips/validation.txt"
        ]

        #notebook train
        opt["steps"]["gpt_train"]["injectors"]["paired_to_mel"]["mel_norm_file"] = mel_norms_path
        opt["steps"]["gpt_train"]["injectors"]["paired_cond_to_mel"]["mel_norm_file"] = mel_norms_path
        opt["steps"]["gpt_train"]["injectors"]["to_codes"]["dvae_config"] = "train_diffusion_vocoder_22k_level.yml"

        trainer.init(config_path, opt, launcher, mode)
        l_config = LoraConfig(
            r = 16,
            lora_alpha = 32,
            lora_dropout = .05,
            task_type = TaskType.CAUSAL_LM,
            target_modules = [
                "c_attn","c_proj","c_fc",
            ],
            modules_to_save=["lm_head"],
        )

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )   

        # Load the Lora model

        unified_voice = trainer.model.networks["gpt"].module
        gpt_model = unified_voice.gpt
        gpt_model.wte = unified_voice.text_embedding
        gpt_model.wpe = functools.partial(self.null_position_embeddings, dim=1024)
        gpt_model =  self.freeze_weights(gpt_model)
        
        #gpt_model..module.gpt.add_adapter(l_config)#?
        #text_lora_parameters_one = list(filter(lambda p: p.requires_grad, gpt_model.parameters()))

        gpt_model = get_peft_model(gpt_model, l_config)
        gpt_model = self.accelerator.prepare(gpt_model)  # Wrap with Accelerate for distributed training
        gpt_model.print_trainable_parameters()
        gpt_model.config.use_cache = False
        del gpt_model.base_model.model.wte
        #replace the trainer and save the new model
        trainer = SFTTrainer(
         model=trainer.model.networks["gpt"].module.gpt,
         train_dataset="/tmp/donald-trump-prepared-clips/train.txt",
         eval_dataset="/tmp/donald-trump-prepared-clips/validation.txt",
         dataset_text_field="prompt",
         peft_config=l_config,
         args=TrainingArguments(
             per_device_train_batch_size=2,
             gradient_accumulation_steps=4,
             warmup_steps=0.03,
             max_steps=100000,
             learning_rate=2e-4,
             logging_steps=500,
             output_dir="outputs",
             optim="adamw_hf",
             save_strategy="epoch",
         ),
         data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
        )

        trainer.train()
        trainer.model.save(10)# niter in gpt_finetune.yml default is 10000


#generate peft
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, default="The expressiveness of autoregressive transformers is literally nuts! I absolutely adore them.")
    adapter = DL_LoRA()
    adapter.load_peft_lora()
    adapter.train_data()

if __name__ == "__main__":
    main()

