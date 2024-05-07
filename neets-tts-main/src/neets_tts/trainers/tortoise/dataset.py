import torch
import multiprocessing

from torch.nn.utils.rnn import pad_sequence

from datasets import concatenate_datasets

from neets_tts.datasets.libriheavy import LibriHeavy
from neets_tts.trainers.tortoise.gpt2 import SPEECH_EOS_TOKEN
from neets_tts.trainers.tortoise.injectors import (
  create_vectorizer, load_waveforms, build_tokenizer,
)

def get_libriheavy_dataset(sample_rate: int):
  from multiprocess import set_start_method # type: ignore
  set_start_method("spawn")

  dataset_path = "/var/neets/datasets/libriheavy"
  dataset = LibriHeavy(dataset_path).get_dataset("small")

  tokenize = build_tokenizer()

  def cpu_stage(batch: dict):
    return tokenize(load_waveforms(batch, dataset_path, sample_rate))

  # CPU stage
  dataset = dataset.map(
    # load_from_cache_file=False,
    desc="Loading waveforms",
    function=cpu_stage,
    batch_size=64,
    batched=True,
    num_proc=multiprocessing.cpu_count(),
    writer_batch_size=100000,
  )

  # GPU stage
  # Numpy format outperforms torch 10 to 1. This might be a bug with huggingface.
  dataset.set_format("numpy", columns=["waveform"], output_all_columns=True)

  dataset_no_waveform = dataset.map(
    desc="generate vq codes",
    function=create_vectorizer(),
    batch_size=128,
    num_proc=torch.cuda.device_count(),
    batched=True,
    with_rank=True,
    # cache_file_name=f"/tmp/arrow_cache.pq",
    remove_columns=["waveform"]
  )

  dataset = concatenate_datasets([
    dataset.select_columns(['waveform']),
    dataset_no_waveform
  ], axis=1).with_format("torch")

  return dataset


SPEECH_EOS_TOKEN = 8193
def collate_fn(batch: list[dict]):
  return {
    "speaker_id": [item["speaker_id"] for item in batch],
    "waveform": pad_sequence([item["waveform"] for item in batch], batch_first=True),
    "vq_codes": pad_sequence([item["vq_codes"] for item in batch], batch_first=True, padding_value=SPEECH_EOS_TOKEN),
    "text_tokens": pad_sequence([item["text_tokens"] for item in batch], batch_first=True),
    "vq_codes_size": torch.IntTensor([item["vq_codes_size"] for item in batch]),
    "waveform_size": torch.LongTensor([item["waveform_size"] for item in batch]),
    "text_tokens_size": torch.IntTensor([item["text_tokens_size"] for item in batch]),
  }

