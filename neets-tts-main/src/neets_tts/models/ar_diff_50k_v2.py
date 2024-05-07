# from neets_tts.datasets.gigaspeech import GigaSpeech, GigaSpeechItem
from neets_tts.datasets.libriheavy import LibriHeavy, LibriHeavySubset

# from neets_tts.models import register_model
# from neets_tts.models.base import BaseModel
from neets_tts.trainers.tortoise.gpt import GPTTrainer, GPTDataset
from neets_tts.trainers.tortoise.diffuser import DiffuserDataset, DiffuserTrainer

def create_speech_dataset():
  # TODO eventually use a sampler to mix datasets based on gender and language.
  # TODO sampler can also apply a uniform duration distribution filter to the dataset.

  libriheavy = LibriHeavy(
    subset=LibriHeavySubset.large,
    dataset_path="/var/neets/datasets/libriheavy"
  )

  return libriheavy.get_dataset()

def create_gpt_trainer():
  gpt_dataset = GPTDataset(
    dataset_path="/var/neets/datasets/ar_diff_50k_v2/gpt",
    speech_dataset=create_speech_dataset()
  )
  return GPTTrainer(dataset=gpt_dataset.get_dataset())

def create_diffuser_trainer():
  diffuser_dataset = DiffuserDataset(
    dataset_path="/var/neets/datasets/ar_diff_50k_v2/diffuser",
    gpt_dataset_path="/var/neets/datasets/ar_diff_50k_v2/gpt",
    speech_dataset=create_speech_dataset()
  )
  diffuser_dataset.build_cache()
  return DiffuserTrainer(dataset=diffuser_dataset)

if __name__ == '__main__':
  gpt_trainer = create_gpt_trainer()
  gpt_trainer.train()
