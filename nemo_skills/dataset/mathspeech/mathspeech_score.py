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

"""Group-level aggregation for the MathSpeech benchmark.

The two sub-benchmarks (``asr`` and ``latex``) report disjoint metric sets (WER
vs. CER/BLEU/ROUGE), so aggregation is a straight union of per-sub-benchmark
metrics weighted by sample count, not a single pooled score.
"""

SUB_BENCHMARKS = ("asr", "latex")
_AGG_METRICS = ("wer", "cer", "bleu", "rouge1", "rougeL", "success_rate", "avg_tokens")


def compute_score(combined_metrics: dict) -> dict:
    """Weighted-average each metric across the two MathSpeech sub-benchmarks."""
    benchmarks = {k: v for k, v in combined_metrics.items() if k.split(".")[-1] in SUB_BENCHMARKS}
    if not benchmarks:
        return {}

    first_benchmark = next(iter(benchmarks.values()))
    eval_modes = list(first_benchmark.keys())

    aggregated = {}
    for eval_mode in eval_modes:
        total_entries = 0
        total_gen_seconds = 0
        weighted = {metric: 0.0 for metric in _AGG_METRICS}
        metric_weights = {metric: 0 for metric in _AGG_METRICS}

        for benchmark_data in benchmarks.values():
            metrics = benchmark_data.get(eval_mode)
            if not metrics:
                continue

            num_entries = metrics.get("num_entries", 0)
            if num_entries == 0:
                continue

            total_entries += num_entries
            total_gen_seconds += metrics.get("gen_seconds", 0)

            for metric in _AGG_METRICS:
                if metric in metrics:
                    weighted[metric] += metrics[metric] * num_entries
                    metric_weights[metric] += num_entries

        if total_entries == 0:
            continue

        agg = {
            "gen_seconds": total_gen_seconds,
            "num_entries": total_entries,
        }
        for metric, total_weight in metric_weights.items():
            if total_weight > 0:
                value = weighted[metric] / total_weight
                agg[metric] = int(value) if metric == "avg_tokens" else round(value, 2)

        aggregated[eval_mode] = agg

    return aggregated
