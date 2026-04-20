"""
Scripts for transformering Ling-Coder-SFT dataset into interleaved format.
See https://huggingface.co/datasets/inclusionAI/Ling-Coder-SFT.
"""

import argparse
import asyncio
import glob
import json
import os
import re

import jinja2
import jsonlines
import litellm
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from jsonschema import validate
from loguru import logger
from tqdm import tqdm

load_dotenv()

DEBUG = False
USE_LOCAL_API = os.getenv("USE_LOCAL_API", False) in ["1", "true", "True", "TRUE"]
LITELLM_MODEL_NAME = os.getenv("LITELLM_MODEL_NAME", "gemini/gemini-2.5-flash-lite")
REASONING_EFFORT = os.getenv("REASONING_EFFORT", None)

if USE_LOCAL_API:
    litellm.api_base = os.getenv("LOCAL_API_BASE")
    litellm.api_key = os.getenv("LOCAL_API_KEY")
    model_name = os.getenv("LOCAL_MODEL_NAME", LITELLM_MODEL_NAME)
    if not model_name.startswith("openai/"):
        model_name = f"openai/{model_name}"
else:
    model_name = LITELLM_MODEL_NAME
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

SCHEMA = {
    "type": "object",
    "properties": {
        "think": {"type": "string"},
    },
    "required": ["think"],
    "additionalProperties": False,
}


class TransformError(Exception):
    """Custom exception for transformation errors."""

    pass


SYSTEM_PROMPT: jinja2.Template = jinja2.Template(
    """\
You are a helpful assistant that generate reasoning part for a specific segment of code in the context of a full program and a user instruction.
You will be given:
1. A user instruction describing the programming task.
2. The full program written to fulfill that instruction.
3. A specific code segment (a contiguous part of the full program) to focus on.

Your task is to produce a chain-of-thought (CoT) style explanation of what the given code segment does and why should it be generated, grounded in the context of the full program and the user instruction.
Please follow these instructions:
1. Read the full program to understand the overall structure and intent before analyzing the segment.
2. Explain the specific code segment: briefly describe why it should be generated and what it should accomplish, as if you are going to reason about how to generate this segment based on its previous code chunks.
3. Do not repeat the code in your explanation. Only produce reasoning text.

**NOTE**:
- Your answer should be concise and to the point.
- Act as like you are thinking about what code to be generated next. Avoid explaining the code, instead, explain why this code should be generated like you are thinking and reasoning!

You MUST respond with ONLY a single fenced code block labeled as `json`. Do not output any text before or after the code block. The JSON object must have only a "think" key containing your full CoT reasoning as a string, matching this schema:
{{ json_schema }}

## Example

```json
{"think": "The user instruction asks for a function that computes the GCD of two integers. This segment defines the core recursive helper used by the public API. The base case returns `b` when `a` is zero, correctly handling the termination condition of the Euclidean algorithm. The recursive call passes `b` and `a % b`, reducing the problem size monotonically — termination is guaranteed since `a % b < b` always holds for positive integers. The result propagates back unchanged through the call stack, so the top-level wrapper in the rest of the program receives the actual GCD."}
```
"""
)

USER_PROMPT: jinja2.Template = jinja2.Template(
    """\
Generate reasoning part for the following code segment based on the full program and user instruction, as described in the system prompt.
[User Instruction]
{{ instruction }}
[/User Instruction]
[Full Program]
{{ full_program }}
[/Full Program]
{% if chunk_index and chunk_total %}
[Chunk Info]
The segment below is chunk {{ chunk_index }} of {{ chunk_total }} of the full program after splitting by empty lines.
Keep continuity with other chunks and avoid contradicting the broader context.
[/Chunk Info]
{% endif %}
[Code Segment to Explain]
{{ segment }}
[/Code Segment to Explain]

Your response must be exactly one ```json ... ``` code block containing a single JSON object with a "think" key. No other text.
Like:
```json
{"think": "Your reasoning here"}
```
"""
)


def split_by_empty_lines(text: str) -> list[str]:
    """Split text into chunks by empty lines and drop empty fragments."""
    if not isinstance(text, str):
        return []
    # split text by empty lines
    chunks = [chunk for chunk in re.split(r"\n\s*\n", text) if chunk.strip() != ""]
    # chunks = [chunk.strip() for chunk in re.split(r"\n\s*\n", text) if chunk.strip() != ""]
    return chunks if chunks else [text]


def normalize_row(row: pd.Series, idx: int, source_name: str) -> dict:
    """Normalize Ling-Coder and EpiCoder rows into a common structure."""
    if "instruction" in row and "output" in row:
        raw_data_id = row.get("id", row.get("idx", f"epicoder_{idx}"))
        return {
            "raw_data_id": str(raw_data_id),
            "user": row["instruction"],
            "assistant": row["output"],
            "tags": ["dataset/epicoder-func-380k"],
            "data_source": source_name,
            "row_format": "epicoder",
            "row_record": row.to_dict(),
        }

    raise TransformError("Unsupported row schema. Expected Ling-Coder or EpiCoder fields.")


def list_dataset_parquet_files(repo_id: str, revision: str | None = None) -> list[str]:
    """Download metadata/files and return sorted dataset file paths for a dataset repo."""
    local_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        allow_patterns=[
            "**/*.parquet",
            "**/*.jsonl",
            "**/*.jsonl.gz",
            "*.parquet",
            "*.jsonl",
            "*.jsonl.gz",
        ],
    )
    parquet_files = sorted(glob.glob(os.path.join(local_dir, "**", "*.parquet"), recursive=True))
    jsonl_files = sorted(glob.glob(os.path.join(local_dir, "**", "*.jsonl"), recursive=True))
    dataset_files = parquet_files + jsonl_files
    if not dataset_files:
        raise TransformError(f"No parquet/jsonl files found in dataset repo: {repo_id}")
    return dataset_files


def read_dataset_file_to_df(dataset_path: str) -> pd.DataFrame:
    """Read parquet/jsonl/jsonl.gz files into a DataFrame."""
    if dataset_path.endswith(".parquet"):
        return pd.read_parquet(dataset_path)

    if dataset_path.endswith(".jsonl"):
        with jsonlines.open(dataset_path) as reader:
            rows = list(reader)
        return pd.DataFrame(rows)

    raise TransformError(f"Unsupported dataset file format: {dataset_path}")


async def transform_code_response(
    segment: str,
    instruction: str = "",
    full_program: str = "",
    chunk_index: int | None = None,
    chunk_total: int | None = None,
    max_attempts: int = 3,
) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.render(json_schema=SCHEMA)},
        {
            "role": "user",
            "content": USER_PROMPT.render(
                segment=segment,
                instruction=instruction,
                full_program=full_program,
                chunk_index=chunk_index,
                chunk_total=chunk_total,
            ),
        },
    ]
    for attempt in range(max_attempts):
        try:
            response = await litellm.acompletion(
                model=model_name,
                messages=messages,
                temperature=0.5,
                top_p=0.9,
                reasoning_effort=REASONING_EFFORT,
                stream=False,
                allowed_openai_params=["reasoning_effort"],
            )
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_attempts - 1:
                raise TransformError(f"Failed to transform after {max_attempts} attempts")
            continue
        except asyncio.exceptions.CancelledError:
            logger.warning(f"Attempt {attempt + 1}: Asyncio task was cancelled unexpectedly.")
            continue

        ret = response.choices[0].message.content

        if not ret:
            logger.warning(f"Empty response received. Retrying... {attempt + 1}/{max_attempts}")
            continue

        # extract json
        json_match = re.search(r"```json(.+?)```", ret, re.DOTALL)
        if json_match:
            ret_json = json_match.group(1).strip()
        else:
            ret_json = ret

        # parse json to ensure it's valid
        try:
            parsed = json.loads(ret_json)
        except Exception as e:
            logger.warning(f"JSON parsing error: {e}. Retrying... {attempt + 1}/{max_attempts}")
            continue

        try:
            validate(instance=parsed, schema=SCHEMA)
        except Exception as e:
            logger.warning(
                f"JSON schema validation error: {e}. Retrying... {attempt + 1}/{max_attempts}"
            )
            continue

        return parsed
    else:
        raise TransformError(f"Failed to transform after {max_attempts} attempts.")


async def transform(
    data_df: pd.DataFrame,
    existing_lines: list,
    output_path: str,
    data_source: str,
    use_llm: bool = True,
    max_concurrent: int = 4,
    skip_first_n: int = 0,
):
    """Transforms a DataFrame into interleaved JSONL format with concurrent async processing.

    Args:
        data_df: DataFrame with raw data to transform
        existing_lines: List of already processed lines to skip
        output_path: Path to the output JSONL file
        max_concurrent: Maximum number of concurrent transformation tasks (default: 4)
    """
    skip_first_n = max(skip_first_n, 0)
    existing_raw_data_ids = [
        str(line.get("raw_data_id", line.get("id", line.get("idx", "")))) for line in existing_lines
    ]
    original_len = len(data_df)
    logger.info(
        f"Original data length: {original_len}, existing processed lines: {len(existing_raw_data_ids)}"
    )
    # remove existing lines from data_df
    if "mid" in data_df.columns:
        data_df = data_df[~data_df["mid"].isin(existing_raw_data_ids)]
    elif "id" in data_df.columns:
        data_df = data_df[~data_df["id"].astype(str).isin(existing_raw_data_ids)]
    elif "idx" in data_df.columns:
        data_df = data_df[~data_df["idx"].astype(str).isin(existing_raw_data_ids)]
    logger.info(f"Data length after removing existing lines: {len(data_df)}")

    total_rows = len(data_df)

    if skip_first_n > 0:
        data_df = data_df.iloc[skip_first_n:]
        total_rows = len(data_df)
        logger.info(f"Data length after skipping first {skip_first_n} rows: {total_rows}")

    # Create task queue for row indices
    task_queue: asyncio.Queue = asyncio.Queue()
    for idx in range(total_rows):
        task_queue.put_nowait(idx)

    # Create lock to protect writer operations
    writer_lock = asyncio.Lock()
    # Progress bar
    pbar = tqdm(total=total_rows + skip_first_n, desc="Transforming", initial=skip_first_n)

    async def process_worker(writer):
        """Worker coroutine that processes tasks from the queue."""
        while True:
            try:
                idx = task_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            try:
                # Get the row by index
                row = data_df.iloc[idx]
                normalized = normalize_row(row, idx, data_source)
                raw_data_id: str = normalized["raw_data_id"]
                raw_human_content: str = normalized["user"]
                raw_assistant_content: str = normalized["assistant"]
                tags: list[str] = normalized["tags"]

                # Split each datum on blank lines and transform each chunk with CoT.
                assistant_chunks = split_by_empty_lines(raw_assistant_content)
                transformed_code_res: list[dict] = []

                for chunk_idx, assistant_chunk in enumerate(assistant_chunks, start=1):
                    if use_llm:
                        try:
                            _res = await transform_code_response(
                                assistant_chunk,
                                instruction=raw_human_content,
                                full_program=raw_assistant_content,
                                chunk_index=chunk_idx,
                                chunk_total=len(assistant_chunks),
                            )
                            chunk_res = {
                                "think": _res.get("think", ""),
                                "output": assistant_chunk,
                            }

                        except TransformError as e:
                            logger.error(
                                f"TransformError for raw_data_id {raw_data_id}, chunk {chunk_idx}/{len(assistant_chunks)}: {e}"
                            )
                            transformed_code_res = []
                            break
                    else:
                        # Fast deterministic mode for testing without any LLM call.
                        chunk_res = {"think": "<CoT>", "output": assistant_chunk}
                    transformed_code_res.append(chunk_res)

                if not transformed_code_res:
                    pbar.update(1)
                    continue

                out = {
                    "raw_data_id": raw_data_id,
                    "messages": [
                        {"role": "user", "content": raw_human_content},
                        {"role": "assistant", "icontent": transformed_code_res},
                    ],
                    "tags": tags,
                    "data_source": normalized["data_source"],
                }

                # Acquire lock before writing to ensure thread-safe file operations
                async with writer_lock:
                    writer.write(out)
            finally:
                pbar.update(1)

    # Open writer and process all rows concurrently
    with jsonlines.open(output_path, "a") as writer:
        workers = [asyncio.create_task(process_worker(writer)) for _ in range(max_concurrent)]
        await asyncio.gather(*workers)

    pbar.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo_id",
        type=str,
        default="microsoft/EpiCoder-func-380k",
        help="HF dataset repo id. Example: microsoft/EpiCoder-func-380k",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="",
        help="Dataset revision/commit. Use empty string for latest.",
    )
    parser.add_argument(
        "--data_source",
        type=str,
        default="",
        help="Optional data source name in output. Defaults to repo_id.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=f"{os.path.dirname(os.path.abspath(__file__))}/transformed/epicoder",
        help="Path to save the transformed dataset.",
    )
    parser.add_argument(
        "--data_index_start",
        type=int,
        default=0,
        help="Starting index of data files to process, inclusive.",
    )
    parser.add_argument(
        "--data_index_end",
        type=int,
        help="Ending index of data files to process, exclusive.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to resume from existing output file.",
    )
    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=4,
        help="Maximum number of concurrent transformation tasks.",
    )
    parser.add_argument(
        "--skip_first_n",
        type=int,
        default=0,
        help="Number of initial rows to skip in each data file.",
    )
    parser.add_argument(
        "--disable_llm",
        action="store_true",
        help="Disable LLM calls and use '<CoT>' as the think text for each chunk.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: only process the first 10 rows from the first data file.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    data_source = args.data_source.strip() if args.data_source else args.repo_id
    revision = args.revision if args.revision else None

    dataset_files = list_dataset_parquet_files(args.repo_id, revision=revision)
    if args.test:
        selected_files = dataset_files[:1]
    else:
        data_index_end = len(dataset_files) if args.data_index_end is None else args.data_index_end
        selected_files = dataset_files[args.data_index_start : data_index_end]

    for dataset_path in tqdm(selected_files, desc="Data Files"):
        fstem = os.path.basename(dataset_path)
        if fstem.endswith(".jsonl.gz"):
            fstem = fstem[: -len(".jsonl.gz")]
        else:
            fstem = os.path.splitext(fstem)[0]

        logger.info(f"Processing file: {dataset_path}")
        data_df = read_dataset_file_to_df(dataset_path)
        if args.test:
            data_df = data_df.iloc[:10]
            logger.info("Test mode: using first 10 rows only.")
        output_filename = f"{fstem}.jsonl"
        output_path = os.path.join(output_dir, output_filename)

        if args.resume and os.path.exists(output_path):
            logger.info(f"Resuming from existing output file: {output_path}")
            with jsonlines.open(output_path) as reader:
                existing_lines = list(reader)
        else:
            existing_lines = []

        try:
            asyncio.run(
                transform(
                    data_df,
                    existing_lines,
                    output_path,
                    data_source,
                    not args.disable_llm,
                    args.max_concurrent,
                    args.skip_first_n,
                )
            )
        except KeyboardInterrupt:
            logger.info("\n\nReceived Ctrl+C, gracefully shutting down...")
            logger.info(f"Progress saved to: {output_path}")
            logger.info("You can resume with --resume flag.")
            break
