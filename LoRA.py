import torch
from tortoise.api import UnifiedVoice
from peft import PeftConfig, LoraModel, LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import argparse
import os
from transformers import GPT2Config, GPT2PreTrainedModel, LogitsProcessorList, AutoModelForCausalLM, TFAutoModelForCausalLM,   TrainingArguments, BitsAndBytesConfig
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.pytorch_utils import Conv1D
from huggingface_hub import hf_hub_download
#from bitsandbytes import 
from tortoise.utils.tokenizer import VoiceBpeTokenizer
from trl import SFTTrainer
from accelerate import Accelerator
from datasets import load_dataset

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

class DL_LoRA:
    def __init__(self):
        print("begin init")
        #initialize autoregressive
        autoregressive = UnifiedVoice(
            max_mel_tokens=604,
            max_text_tokens=402,
            max_conditioning_inputs=2,
            layers=30,
            model_dim=1024,
            heads=16,
            number_text_tokens=255,
            start_text_token=255,
            checkpointing=False,
            train_solo_embeddings=False
        )
        autoregressive.load_state_dict(torch.load(autoregressive_model_path), strict=False)
        autoregressive.post_init_gpt2_config(use_deepspeed=True, kv_cache=True)
        #initialize tokenizer 
        self.load_tokenizer_json(None)
        self.model =  self.freeze_weights(autoregressive)
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
        # Iterate through the parameters
        #for param in parameters:
        #    print(param)
        #print(model)

        for param in parameters:
            param.requires_grad = False  # freeze the model - train adapters later
            if param.ndim == 1:
                # cast the small parameters (e.g. layernorm) to fp32 for stability
                param.data = param.data.to(torch.float32)

        model.gpt.gradient_checkpointing_enable()  # reduce number of stored activations
        model.gpt.enable_input_require_grads()
        return model

    def merge_columns(self, entry):
        entry["prediction"] = entry["quote"] + " ->: " + str(entry["tags"])
        return entry

    def load_peft_lora(self):
        print("begin lora")

        l_config = LoraConfig(
            r = 16,
            lora_alpha = 32,
            lora_dropout = .05,
            task_type = TaskType.CAUSAL_LM,
            target_modules = [
                "c_attn","c_proj","c_fc",
            ]    
        )

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )   

        # Load the Lora model
        self.model.inference_model.save_pretrained("./tortoise_mod")
        #base_model = 
        # Prepare model for training with PEFT
        self.model = prepare_model_for_kbit_training(AutoModelForCausalLM.from_pretrained("./tortoise_mod", ignore_mismatched_sizes=True))#, quantization_config=bnb_config))
        print(list(set(self.get_specific_layer_names(self.model))))
        self.model = get_peft_model(self.model, l_config)

        self.model = self.accelerator.prepare(self.model)  # Wrap with Accelerate for distributed training
        self.model.print_trainable_parameters()
        print("\n\nComplete\n\n")
   

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

        #peft_config = PeftConfig.from_pretrained("./tortoise_mod")


        data = load_dataset("Abirate/english_quotes")
        data['train'] = data['train'].map(self.merge_columns)
        print(data['train']['prediction'][:5])

        data = data.map(lambda samples: self.tokenizer(samples['prediction']), batched=True)


        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=data,
            dataset_text_field="messages",
            peft_config=self.model.peft_config,
            packing=True, # pack samples together for efficient training
            max_seq_length=1024,
            args=training_arguments,
        )
        # Start training
        self.model.config.use_cache = False
        trainer.train()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, default="The expressiveness of autoregressive transformers is literally nuts! I absolutely adore them.")

    adapter = DL_LoRA()
    adapter.load_peft_lora()
    #adapter.load_data()
    adapter.train_data()

if __name__ == "__main__":
    main()





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
