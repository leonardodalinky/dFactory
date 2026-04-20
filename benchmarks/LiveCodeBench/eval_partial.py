#!/usr/bin/env python3
"""Evaluate partial results from the generation cache.

Usage (from the LiveCodeBench directory):
    PYTHONPATH=../.. python eval_partial.py

Or from the dFactory root:
    PYTHONPATH=$(pwd):$(pwd)/VeOmni python benchmarks/LiveCodeBench/eval_partial.py
"""

import json
import os

from lcb_runner.benchmarks import load_code_generation_dataset
from lcb_runner.prompts import format_prompt_generation
from lcb_runner.utils.extraction_utils import extract_code
from lcb_runner.lm_styles import LanguageModelStore
from lcb_runner.evaluation import codegen_metrics, extract_instance_results

# ── Configuration (match run_lcb.slurm settings) ──
CACHE_FILE = "cache/iCoder-LLaDA2-MoE/Scenario.codegeneration_1_0.0.json"
MODEL_NAME = "icoder-llada2-moe"
NUM_PROCESS_EVALUATE = 12
TIMEOUT = 6
OUTPUT_DIR = "output/iCoder-LLaDA2-MoE"

# ── Load cache ──
with open(CACHE_FILE) as f:
    cache = json.load(f)
print(f"Loaded {len(cache)} cached entries")

# ── Load benchmark ──
model = LanguageModelStore[MODEL_NAME]
benchmark = load_code_generation_dataset("release_latest")
benchmark = sorted(benchmark, key=lambda x: x.question_id)
print(f"Total benchmark: {len(benchmark)} problems")

# ── Match prompts to cache ──
matched_problems = []
combined_results = []

for problem in benchmark:
    prompt = format_prompt_generation(problem, model.model_style)
    prompt_cache_key = json.dumps(prompt)
    if prompt_cache_key in cache:
        outputs_list = cache[prompt_cache_key]
        extracted_list = [
            extract_code(output, model.model_style) for output in outputs_list
        ]
        matched_problems.append(problem)
        combined_results.append((outputs_list, extracted_list))

print(f"Matched {len(matched_problems)} / {len(benchmark)} problems from cache")

# ── Save output file ──
os.makedirs(OUTPUT_DIR, exist_ok=True)
save_results = [
    problem.insert_output(outputs_list, extracted_list)
    for problem, (outputs_list, extracted_list) in zip(
        matched_problems, combined_results
    )
]
output_path = os.path.join(OUTPUT_DIR, "Scenario.codegeneration_1_0.0_partial.json")
with open(output_path, "w") as f:
    json.dump(save_results, f, indent=4)
print(f"Saved output to {output_path}")

# ── Evaluate ──
print(f"\nRunning evaluation on {len(matched_problems)} problems...")
eval_samples = [p.get_evaluation_sample() for p in matched_problems]
generations = [extracted for _, extracted in combined_results]

metrics = codegen_metrics(
    eval_samples,
    generations,
    num_process_evaluate=NUM_PROCESS_EVALUATE,
    timeout=TIMEOUT,
)
graded = extract_instance_results(metrics[1])

# ── Build eval results ──
metadatas = metrics[2]
save_eval_results = [
    problem.insert_output_evaluation(
        outputs_list, extracted_list, graded_list, metadata=meta
    )
    for problem, (outputs_list, extracted_list), graded_list, meta in zip(
        matched_problems, combined_results, graded, metadatas
    )
]

# ── Print results ──
pass_1_list = [r["pass@1"] for r in save_eval_results]
print(f"\n{'='*50}")
print(f"Results on {len(matched_problems)} / {len(benchmark)} problems")
print(f"Pass@1: {sum(pass_1_list) / len(pass_1_list):.4f}")

easy = [r for r in save_eval_results if r.get("difficulty") == "easy"]
medium = [r for r in save_eval_results if r.get("difficulty") == "medium"]
hard = [r for r in save_eval_results if r.get("difficulty") == "hard"]
if easy:
    print(f"  Easy   Pass@1: {sum(r['pass@1'] for r in easy) / len(easy):.4f}  ({len(easy)} problems)")
if medium:
    print(f"  Medium Pass@1: {sum(r['pass@1'] for r in medium) / len(medium):.4f}  ({len(medium)} problems)")
if hard:
    print(f"  Hard   Pass@1: {sum(r['pass@1'] for r in hard) / len(hard):.4f}  ({len(hard)} problems)")
print(f"{'='*50}")

# ── Save eval files ──
eval_path = output_path.replace(".json", "_eval.json")
eval_all_path = output_path.replace(".json", "_eval_all.json")
with open(eval_path, "w") as f:
    json.dump(metrics, f, indent=4)
with open(eval_all_path, "w") as f:
    json.dump(save_eval_results, f, indent=4)
print(f"Saved eval to {eval_path}")
print(f"Saved eval_all to {eval_all_path}")
