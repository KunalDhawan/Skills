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

"""MathSpeech LaTeX evaluator: CER, ROUGE-1, ROUGE-L, BLEU on space-free strings.

Implements the evaluation protocol from the MathSpeech paper (AAAI 2025):
  > "LaTeX can represent the same formula in multiple ways. For example,
  > $A B$ and $AB$ result in the same LaTeX compilation output. So the
  > evaluation was conducted after removing all spaces."

Normalization:
  1. Strip the outer ``$...$`` delimiters if present.
  2. Remove all whitespace characters.

Metrics are then computed character-by-character on the normalized strings.
Per-sample counts are accumulated so the metrics class can produce corpus-level
CER and mean-per-sample BLEU/ROUGE.
"""

import re

import editdistance

from nemo_skills.evaluation.evaluator.base import BaseEvaluator, BaseEvaluatorConfig
from nemo_skills.utils import nested_dataclass

_DELIM_RE = re.compile(r"^\s*\$+(.*?)\$+\s*$", re.DOTALL)


def normalize_latex(text):
    """Strip ``$`` delimiters and remove all whitespace (paper's scoring protocol)."""
    if text is None:
        return ""
    text = text.strip()
    match = _DELIM_RE.match(text)
    if match:
        text = match.group(1)
    return re.sub(r"\s+", "", text)


def _char_tokens(text):
    """Return the string as a list of single-character tokens."""
    return list(text)


def _lcs_length(a, b):
    """Length of the longest common subsequence of sequences ``a`` and ``b``."""
    if not a or not b:
        return 0
    n, m = len(a), len(b)
    prev = [0] * (m + 1)
    for i in range(1, n + 1):
        curr = [0] * (m + 1)
        ai = a[i - 1]
        for j in range(1, m + 1):
            if ai == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = prev[j] if prev[j] >= curr[j - 1] else curr[j - 1]
        prev = curr
    return prev[m]


def _f1(precision, recall):
    """Standard F1 score from precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def rouge1_f1(ref_tokens, hyp_tokens):
    """ROUGE-1 F1: unigram overlap F-measure (ROUGE treats multiset overlap)."""
    if not ref_tokens and not hyp_tokens:
        return 1.0
    if not ref_tokens or not hyp_tokens:
        return 0.0
    from collections import Counter

    ref_counts = Counter(ref_tokens)
    hyp_counts = Counter(hyp_tokens)
    overlap = sum((ref_counts & hyp_counts).values())
    precision = overlap / len(hyp_tokens)
    recall = overlap / len(ref_tokens)
    return _f1(precision, recall)


def rougeL_f1(ref_tokens, hyp_tokens):
    """ROUGE-L F1 based on longest common subsequence length."""
    if not ref_tokens and not hyp_tokens:
        return 1.0
    if not ref_tokens or not hyp_tokens:
        return 0.0
    lcs = _lcs_length(ref_tokens, hyp_tokens)
    precision = lcs / len(hyp_tokens)
    recall = lcs / len(ref_tokens)
    return _f1(precision, recall)


def sentence_bleu_chars(reference, hypothesis):
    """Character-level sentence BLEU (sacrebleu) on strings that already had spaces removed.

    We insert spaces between characters so sacrebleu's whitespace tokenizer treats each
    character as a token \u2014 matching the paper's "remove all spaces, then score" protocol.
    """
    from sacrebleu.metrics import BLEU

    if not reference or not hypothesis:
        return 0.0 if reference != hypothesis else 1.0
    bleu = BLEU(effective_order=True, tokenize="none", lowercase=False)
    ref_spaced = " ".join(reference)
    hyp_spaced = " ".join(hypothesis)
    return bleu.sentence_score(hyp_spaced, [ref_spaced]).score / 100.0


def evaluate_mathspeech_latex_sample(data_point):
    """Score a single MathSpeech LaTeX prediction.

    Returns per-sample metrics plus raw char-level counts so the metrics class can
    compute corpus-level CER alongside mean-per-sample BLEU/ROUGE.
    """
    reference = data_point["expected_answer"]
    generation = data_point["generation"].strip()

    norm_ref = normalize_latex(reference)
    norm_hyp = normalize_latex(generation)

    ref_chars = _char_tokens(norm_ref)
    hyp_chars = _char_tokens(norm_hyp)

    cer_errors = editdistance.eval(hyp_chars, ref_chars)
    cer_ref_chars = len(ref_chars)
    cer = cer_errors / cer_ref_chars if cer_ref_chars > 0 else (0.0 if not hyp_chars else 1.0)

    rouge1 = rouge1_f1(ref_chars, hyp_chars)
    rougel = rougeL_f1(ref_chars, hyp_chars)
    bleu = sentence_bleu_chars(norm_ref, norm_hyp)

    return {
        "cer": cer,
        "cer_errors": int(cer_errors),
        "cer_ref_chars": cer_ref_chars,
        "rouge1_f1": rouge1,
        "rougeL_f1": rougel,
        "bleu": bleu,
        "is_correct": norm_hyp == norm_ref,
        "normalized_reference": norm_ref,
        "normalized_hypothesis": norm_hyp,
    }


@nested_dataclass(kw_only=True)
class MathSpeechLatexEvaluatorConfig(BaseEvaluatorConfig):
    """Configuration for the MathSpeech LaTeX evaluator (no options yet)."""

    pass


class MathSpeechLatexEvaluator(BaseEvaluator):
    """Evaluator for the ``mathspeech.latex`` sub-benchmark."""

    def __init__(self, config: dict, num_parallel_requests=10):
        """Initialize with evaluator config and parallelism settings."""
        super().__init__(config, num_parallel_requests)

    async def eval_single(self, data_point: dict) -> dict:
        """Score a single data point and return per-sample metric fields."""
        return evaluate_mathspeech_latex_sample(data_point)
