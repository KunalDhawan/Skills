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

"""Aggregation metrics for the MathSpeech LaTeX sub-benchmark.

- CER is computed corpus-level (total char errors / total reference chars), so a
  short sample with many errors does not dominate a long correct sample.
- BLEU / ROUGE-1 / ROUGE-L are averaged per-sample (matches the MathSpeech paper,
  which reports per-sample mean ROUGE/BLEU).
"""

from nemo_skills.evaluation.metrics.base import BaseMetrics, as_int, as_percentage


class MathSpeechLatexMetrics(BaseMetrics):
    """Corpus-level CER + mean-per-sample BLEU/ROUGE for MathSpeech LaTeX."""

    def __init__(self, compute_no_answer: bool = True, max_k: int = 1):
        """Initialize accumulators for CER counts and per-sample BLEU/ROUGE sums."""
        super().__init__(compute_no_answer=compute_no_answer)
        self.max_k = max_k

        self.cer_total_errors = 0
        self.cer_total_ref_chars = 0

        self.bleu_sum = 0.0
        self.rouge1_sum = 0.0
        self.rougeL_sum = 0.0
        self.metric_samples = 0

    def _get_score_dict(self, prediction):
        """Binary correctness: exact match after LaTeX normalization."""
        return {"is_correct": prediction.get("is_correct", False)}

    def get_incorrect_sample(self, prediction):
        """Return a copy of the prediction marked as incorrect (for no-answer handling)."""
        prediction = prediction.copy()
        prediction["is_correct"] = False
        return prediction

    def update_common_metrics(self, agg_dict):
        """Populate num_entries, avg_tokens, and gen_seconds into the aggregation dict."""
        agg_dict["num_entries"] = self.total
        agg_dict["avg_tokens"] = int(self.avg_tokens / self.total) if self.total > 0 else 0
        if self.max_end_time > float("-inf") and self.min_start_time < float("inf"):
            agg_dict["gen_seconds"] = int(self.max_end_time - self.min_start_time)

    def update(self, predictions):
        """Accumulate per-sample CER error counts and BLEU/ROUGE sums."""
        super().update(predictions)

        predicted_answers = [pred.get("generation", "").strip() or None for pred in predictions]

        for pred in predictions:
            self.cer_total_errors += pred.get("cer_errors", 0)
            self.cer_total_ref_chars += pred.get("cer_ref_chars", 0)
            self.bleu_sum += pred.get("bleu", 0.0)
            self.rouge1_sum += pred.get("rouge1_f1", 0.0)
            self.rougeL_sum += pred.get("rougeL_f1", 0.0)
            self.metric_samples += 1

        self._compute_pass_at_k(predictions=predictions, predicted_answers=predicted_answers)
        self._compute_majority_at_k(predictions=predictions, predicted_answers=predicted_answers)

    def get_metrics(self):
        """Compute corpus-level CER and mean-per-sample BLEU/ROUGE from accumulators."""
        metrics_dict = super().get_metrics()

        for _agg_mode, agg_metrics in metrics_dict.items():
            if "correct" in agg_metrics:
                agg_metrics["success_rate"] = agg_metrics["correct"]

            if self.cer_total_ref_chars > 0:
                agg_metrics["cer"] = round(100.0 * self.cer_total_errors / self.cer_total_ref_chars, 2)

            if self.metric_samples > 0:
                agg_metrics["bleu"] = round(100.0 * self.bleu_sum / self.metric_samples, 2)
                agg_metrics["rouge1"] = round(100.0 * self.rouge1_sum / self.metric_samples, 2)
                agg_metrics["rougeL"] = round(100.0 * self.rougeL_sum / self.metric_samples, 2)

        return metrics_dict

    def evaluations_to_print(self):
        """Return the list of evaluation mode names to display."""
        evals = [f"pass@{self.max_k}"]
        if self.max_k > 1:
            evals.extend([f"majority@{self.max_k}", f"pass@1[avg-of-{self.max_k}]"])
        return evals

    def metrics_to_print(self):
        """Return ordered dict of metric names to formatters for display."""
        base_metrics = {
            "avg_tokens": as_int,
            "gen_seconds": as_int,
            "success_rate": as_percentage,
        }
        if self.compute_no_answer:
            base_metrics["no_answer"] = as_percentage

        if self.cer_total_ref_chars > 0:
            base_metrics["cer"] = as_percentage
        if self.metric_samples > 0:
            base_metrics["bleu"] = as_percentage
            base_metrics["rouge1"] = as_percentage
            base_metrics["rougeL"] = as_percentage

        base_metrics["num_entries"] = as_int
        return base_metrics
