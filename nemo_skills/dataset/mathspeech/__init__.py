# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MathSpeech: Spoken mathematical expression benchmark (AAAI 2025).

Two sub-benchmarks share the same audio (1,101 samples from MIT OpenCourseWare
lectures). Models are prompted differently and scored with different metrics:

- ``mathspeech.asr``    – Speech \u2192 Spoken English transcription (WER).
- ``mathspeech.latex``  – Speech \u2192 LaTeX formula (CER, ROUGE-1, ROUGE-L, BLEU).

Dataset: https://huggingface.co/datasets/AAAI2025/MathSpeech
Paper:   https://arxiv.org/abs/2412.15655
"""

REQUIRES_DATA_DIR = True
IS_BENCHMARK_GROUP = True
SCORE_MODULE = "nemo_skills.dataset.mathspeech.mathspeech_score"

BENCHMARKS = {
    "mathspeech.asr": {},
    "mathspeech.latex": {},
}
