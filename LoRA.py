import torch
from tortoise.api import UnifiedVoice
from peft import PeftConfig, LoraConfig, get_peft_model, TaskType
import argparse
import os
from transformers import GPT2Config, GPT2PreTrainedModel, LogitsProcessorList
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from huggingface_hub import hf_hub_download

autoregressive_model_path = hf_hub_download(
  "jbetker/tortoise-tts-v2",
  ".models/autoregressive.pth"
)


class DL_LoRA:
    def __init__(self):
        print("begin init")
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
        peft_model_id = "samwit/bloom-7b1-lora-tagger"
        config = PeftConfig.from_pretrained(peft_model_id)
        model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, return_dict=True, load_in_8bit=True, device_map='auto')
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

        # Load the Lora model
        #model = PeftModel.from_pretrained(model, peft_model_id)
        unified_voice = trainer.model.networks["gpt"].module

    def freeze_weights(self):
        print("begin freeze")

    def config_lora(self):
        print("begin lora")
        config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=.05,
            bias = None,
            task_type = "Causal_LM"
        )

    def load_data(self):
        print("begin load")

    def train_data(self):
        print("begin train")

        from ntortoise.clone.finetune import build_trainer, ValidatingTrainer

        trainer = build_trainer(
            local_clips_path="/tmp/tmpha5w_m77/training/prepared_clips",
            output_voice_s3_path="s3://neets/tmp/output",
            validation_diffusion_iterations=30,
            validation_out_dir="/tmp/validation",
            output_model_path="/tmp/output",
            voice_id_prefix="spongebob"
        )




def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, default="The expressiveness of autoregressive transformers is literally nuts! I absolutely adore them.")

    adapter = DL_LoRA()
    adapter.freeze_weights()
    adapter.config_lora()
    adapter.load_data()
    adapter.train_data()

if __name__ == "__main__":
    main()
