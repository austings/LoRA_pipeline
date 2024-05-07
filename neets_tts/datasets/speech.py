import numpy as np
from typing import TypedDict

from datasets import Dataset

class SpeechDatasetItemBatch(TypedDict):
  text: list[str]
  clip_id: list[str]
  segment_id: list[str]
  duration: list[float]
  speaker_id: list[str]
  language: list[str]
  sample_rate: list[int]
  waveform: list[np.ndarray]

SpeechDataset = Dataset # items of type SpeechDatasetItem
