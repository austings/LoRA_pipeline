from enum import Enum
from typing import TypedDict
from datasets import Dataset

import numpy as np
from datasets import load_dataset

from neets_tts.datasets.speech import SpeechDataset

class GigaSpeechSubset(Enum):
  xs = "xs" #    10 hours
  s  = "s"  #   250 hours
  m  = "m"  #  1000 hours
  l  = "l"  #  2500 hours
  xl = "xl" # 10000 hours

class GigaSpeechAudio(TypedDict):
  array: np.ndarray
  sampling_rate: int

# features: ['segment_id', 'speaker', 'text', 'audio', 'begin_time', 'end_time', 'audio_id', 'title', 'url', 'source', 'category', 'original_full_path'],
class GigaSpeechItem(TypedDict):
  begin_time: float
  end_time: float
  segment_id: str
  audio_id: str
  text: str
  speaker_id: str
  audio: GigaSpeechAudio

class GigaSpeech(SpeechDataset):
  def __init__(self, subset: GigaSpeechSubset = GigaSpeechSubset.xs):
    super().__init__()
    self.dataset = load_dataset("speechcolab/gigaspeech", subset.value, token="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")

  def map_to_report(self, item: GigaSpeechItem):
    return {
      "segment_id": item["segment_id"],
      "audio_id": item["audio_id"],
      "speaker_id": item["speaker_id"],
      "text": item["text"],
      "sample_rate": item["audio"]["sampling_rate"],
      "waveform": item["audio"]["array"]
    }

  def create_dataset(self) -> Dataset:
    return self.dataset

# TODO add report
# - number of speakers in the dataset, % with speaker
# - total number of hours of audio
# - duration distribution of audio
