import os

from datasets import load_dataset, Dataset

def extract_audio_batch(batch: dict):
  return {
    "text": [
      supervisions[0]["custom"]["texts"][0]
      for supervisions in batch["supervisions"]
    ],
    "speaker_id": [
      supervisions[0]["speaker"]
      for supervisions in batch["supervisions"]
    ],
    "audio_path": [
      recording["sources"][0]["source"]
      for recording in batch["recording"]
    ],
    "start": batch["start"],
    "duration": batch["duration"]
  }

class LibriHeavy:
  def __init__(self, dataset_path: str):
    self.dataset_path = dataset_path

  def setup(self):
    os.system(f"""
git clone https://github.com/k2-fsa/libriheavy.git
cd libriheavy
# download audio files
bash run.sh --stage -1 --stop-stage -1
# download transcripts
bash run.sh --stage 1 --stop-stage 1
# unzip clip files
gunzip -k *.jsonl.gz
mkdir -p {self.dataset_path}
mv download {self.dataset_path}/
""")

  # TODO standardize to SpeechDatasetItemBatch
  def get_dataset(self, split: str, old = False) -> Dataset:
    subsets = [
      "small",  # 509   hours
      "medium", # 5042  hours
      "large",  # 50794 hours
      "dev",
      "test_clean",
      "test_clean_large",
      "test_other",
      "test_other_large",
    ]

    dataset: Dataset = load_dataset(
      "json",
      cache_dir=f"{self.dataset_path}/cache",
      data_files={
        s: f"{self.dataset_path}/libriheavy_cuts_{s}.jsonl"
        for s in subsets
      },
      split=split,
    ) # type: ignore

    if old:
      return dataset

    print(set(dataset.column_names) - set(["text", "speaker_id", "audio_path", "start", "duration"]))
    dataset = dataset.map(
      function=extract_audio_batch,
      batched=True,
      desc="Extracting audio batches",
      remove_columns=[
        'type', 'id', 'channel', 'recording', 'custom', 'supervisions'
      ]
    )

    return dataset
