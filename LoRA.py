'''
Austin Sierra
DL Software
the program begins and ends here
Last edit May 8 2024
'''
import sys
import torch
from tortoise.api import UnifiedVoice
from transformers import GPT2Config, GPT2PreTrainedModel, LogitsProcessorList, AutoModelForCausalLM, TFAutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
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
#from trl import SFTTrainer
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

    #used for retrieving applicable modules from a model
    #input: model- the model to retrieve modules from
    #output: layer_names - list with the layer names
    def get_specific_layer_names(self, model):
        layer_names = []
        
        # Recursively visit all modules and submodules
        for name, module in model.named_modules():
            # Check if the module is an instance of the specified layers
            if isinstance(module, (torch.nn.Linear, torch.nn.Embedding, torch.nn.Conv2d, Conv1D)):
                layer_names.append('.'.join(name.split('.')[4:]).split('.')[0])
        
        return layer_names



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

    def merge_columns(self, entry):
        entry["prediction"] = entry["quote"] + " ->: " + str(entry["tags"])
        return entry

    def load_peft_lora(self):
        print("begin lora")

 
    def null_position_embeddings(self, range, dim):
        return torch.zeros((range.shape[0], range.shape[1], dim), device=range.device)

    def load_data(self):
        print("begin load")

    def train_data(self):
        print("begin train")

        # TrainingArguments and Trainer
        training_arguments = TrainingArguments(
            per_device_train_batch_size=1,
            max_steps=30,
            gradient_accumulation_steps=4,
            per_device_eval_batch_size=1,
            output_dir="./",
            report_to=["none"],
            fp16=False,
        )

        rel_path = "gpt_finetune.yml"
        config_path = rel_path 
        opt = option.parse(config_path, is_train=True)

        trainer = Trainer()
        trainer.rank = -1

        opt["path"]["pretrain_model_gpt"] = autoregressive_model_path
        opt["path"]["experiments_root"] = os.path.join(os.path.abspath(''), "experiments")
        opt["datasets"]["train"]["path"] = [
        "/tmp/donald-trump-prepared-clips/train.txt"
        ]
        opt["datasets"]["val"]["path"] = [
        "/tmp/donald-trump-prepared-clips/validation.txt"
        ]

        file_path_train = "/tmp/donald-trump-prepared-clips/train.txt"
        file_path_validate = "/tmp/donald-trump-prepared-clips/validation.txt"

        #notebook train
        opt["steps"]["gpt_train"]["injectors"]["paired_to_mel"]["mel_norm_file"] = mel_norms_path
        opt["steps"]["gpt_train"]["injectors"]["paired_cond_to_mel"]["mel_norm_file"] = mel_norms_path
        opt["steps"]["gpt_train"]["injectors"]["to_codes"]["dvae_config"] = "train_diffusion_vocoder_22k_level.yml"

        launcher = "none"
        mode = ""
        trainer.init(config_path, opt, launcher, mode)
        # Create a dictionary from the base model's state dictionary
        #base_state_dict = dict(unified_voice.gpt.state_dict().items()) #dict(

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
        #train and save the new model
        trainer.do_training()
        #gpt_model = trainer.model.networks["gpt"].module
        gpt_model.save_pretrained("./tortoise_mod")
        trainer.model.save(10)# n_iter in gpt_finetune.yml default is 10000
        #trainer.model.networks["gpt"].module.gpt.save_model("./adapters")

        from safetensors.torch import load_model, save_model

        #save_model(trainer.model, "model.safetensors")
        save_model(trainer.model.networks["gpt"].module.gpt, "model_weight.safetensors")

        lora_weights = {}
        for key, value in trainer.model.networks["gpt"].module.gpt.state_dict().items():
            if "c_attn" in key or "c_proj" in key or "c_fc" in key:
                lora_weights[key] = value

        # Save the LoRA weights to a safetensor file
        torch.save(lora_weights, "model_weight_ft.safetensors")
        


#generate peft
def main():
    adapter = DL_LoRA()
    adapter.load_peft_lora()
    adapter.train_data()

#swap loras
def pipeline_main():
    adapter = DL_LoRA()
    adapter.extract_LoRA_from_peft("lora_data/dt/10_gpt.pth")


def extract_LoRA_from_peft( peft_model_path):

    import safetensors.torch
    # Load the PyTorch model
    model = AutoModelForCausalLM.from_pretrained(peft_model_path, torch_dtype=torch.float16)

    # Extract the model weights
    model_weights = model.state_dict()
    safetensors.torch.save_file(model_weights, "./tortoise_complete.safetensors")
    #from transformers import pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, default="The expressiveness of autoregressive transformers is literally nuts! I absolutely adore them.")

    #extract_LoRA_from_peft("./model_weight.safetensors")#lora_data/dt/10_gpt.pth")
    #pipeline_main()
    main()



# save weights seperately from complete model
        
        #torch.save(trainer.model.networks["gpt"].module.gpt.state_dict(), "./adapters/weights.bin")

    #generator = pipeline("text-to-speech", model="peft_model_path")
    #peft_model = PeftConfig.from_pretrained(peft_model_path)
    #peft_model.save_pretrained("adapters/")
    #pipeline.load_lora_weights(
    #    peft_model_path,
    #    adapter_name="lcm",
    #    #cache_dir=cache_dir,
    #    local_files_only=True,
    #    tokenizer = self.tokenizer
    #)
    '''
        #create a lora adapter from said model
        #lora_adapter = LoraModel(
        #    trainer.model.networks["gpt"].module.gpt,
        #    l_config,
        #    "trump"
        #)
        #fine_tuned_weights = trainer.model.networks["gpt"].module.gpt.state_dict()
        # Merge the fine-tuned weights into the LoRA adapter
        #lora_adapter.merge_weights(fine_tuned_weights)
        #lora_adapter.save_pretrained("./adapters")
        weights = dict(trainer.model.networks["gpt"].module.gpt.state_dict().items())
        print(weights)
        # Initialize a dictionary to store the modified tensors
        modified_tensors = {}
        # Loop through the base model's state dictionary
        for key, base_tensor in base_state_dict.items():
            # Check if the tensor has been modified during fine-tuning
            if key in weights:
                if torch.any(base_tensor != weights[key]):
                    modified_tensors[key] = weights[key]
        safetensors.torch.save_file(modified_tensors, "./tortoise_complete.safetensors")
        '''

        # trainer = SFTTrainer(
        #     model=self.model,
        #     train_dataset=train_data,
        #     eval_dataset=test_data,
        #     dataset_text_field="prompt",
        #     peft_config=lora_config,
        #     args=transformers.TrainingArguments(
        #         per_device_train_batch_size=2,
        #         gradient_accumulation_steps=4,
        #         warmup_steps=0.03,
        # #        max_steps=100000,
        #         learning_rate=2e-4,
        #         logging_steps=500,
        #         output_dir="outputs",
        #         optim="paged_adamw_8bit",
        #         save_strategy="epoch",
        #     ),
        #     data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
        # )

        #gpt_model.wte = unified_voice.mel_embedding #unified_voice.text_embedding.emb #= torch.zeros() #del g
        #gpt_model.resize_token_embeddings(len(self.tokenizer))
        #gpt_model.wpe = unified_voice.text_pos_embedding.emb# functools.partial(torch.zeros((range.shape[0], range.shape[1], 1024), device=range.device))
        #gpt_model.save_pretrained("./tortoise_mod")
        #gpt_model = prepare_model_for_kbit_training(gpt_model)
       
        # Prepare model for training with PEFT
        #AutoModelForCausalLM.register("new-model", unified_voice.gpt)
        #AutoModelForCausalLM.register(unified_voice.gpt, unified_voice.gpt)
        # gpt_model.save_pretrained("./tortoise_mod")
        # gpt_model = prepare_model_for_kbit_training(AutoModelForCausalLM.from_pretrained("./tortoise_mod", ignore_mismatched_sizes=True))#, quantization_config=bnb_config))
        #print(list(set(self.get_specific_layer_names(unified_voice.gpt))))
        #print("\n\n")
        #print(unified_voice.gpt)
        #unified_voice.gpt = gpt_model

    #from ntortoise.clone.finetune import build_trainer, ValidatingTrainer
    # trainer = build_trainer(
    #     local_clips_path="/tmp/tmpha5w_m77/training/prepared_clips",
    #     output_voice_s3_path="s3://neets/tmp/output",
    #     validation_diffusion_iterations=30,
    #     validation_out_dir="/tmp/validation",
    #     output_model_path="/tmp/output",
    #     voice_id_prefix="spongebob")
    #quantization_config = transformers.BitsAndBytesConfig(load_in_8bit=True)
    #self.lora_model = LoraModel(self.lora_model,l_config,"default")
    #model = PeftModel.from_pretrained(model, peft_model_id)
    #unified_voice = trainer.model.networks["gpt"].module   
    #sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 16}
    #model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
    #ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
    #tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    #config = PeftConfig.from_pretrained(pretrained_model_name_or_path=self.autoregressive.gpt)
    #model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, return_dict=True, load_in_8bit=True, device_map='auto')
    #Non-Lora Adapter
    #config = PPOConfig(
    #    model_name="lvwerra/gpt2-imdb",
    #    learning_rate=1.41e-5,
    #    log_with="wandb")
    #tokenizer.pad_token = tokenizer.eos_token
    #ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator)

    #non used lora config
                    #"conditioning_encoder.init", Conv1D shape not supported in current lora branch. add later
            #"conditioning_encoder.attn.0.qkv",
            # "conditioning_encoder.attn.0.proj_out",
            # "conditioning_encoder.attn.1.qkv",
            # "conditioning_encoder.attn.1.proj_out",
            # "conditioning_encoder.attn.2.qkv",
            # "conditioning_encoder.attn.2.proj_out",
            # "conditioning_encoder.attn.3.qkv",
            # "conditioning_encoder.attn.3.proj_out",
            # "conditioning_encoder.attn.4.qkv",
            # "conditioning_encoder.attn.4.proj_out",
            # "conditioning_encoder.attn.5.qkv",
            # "conditioning_encoder.attn.5.proj_out",
            #  "text_embedding",
            #  "mel_embedding",
            #  "gpt.text_pos_embedding",
            #  "gpt.wte",
            #  "mel_pos_embedding.emb",
            #  "text_pos_embedding.emb",
            #  "text_head",
            #  "mel_head",
            #  "inference_model.text_pos_embedding",
            #  "inference_model.embeddings",
            #  "inference_model.lm_head.1",
            #  "ds_engine.module.text_pos_embedding",
            #  "ds_engine.module.embeddings",
            #  "ds_engine.module.lm_head.1"
