"""Parse raw interleaved inference output into the messages / icontent format
used by the iCoder training pipeline.

Typical usage
-------------
>>> from models.llada2_moe.parse_response import parse_interleaved_response
>>> icontent = parse_interleaved_response(open("results/3sum").read())
>>> # [{"think": "...", "output": "..."}, ...]

>>> messages = build_messages(prompt_text, raw_response)
>>> # [{"role": "user", "content": "..."}, {"role": "assistant", "icontent": [...]}]
"""

import re
from typing import List, Dict, Optional

# Regex for all <|...|> special tokens emitted by the tokenizer
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|]+\|>")


def parse_interleaved_response(text: str) -> List[Dict[str, str]]:
    """Parse a raw interleaved assistant response into a list of icontent dicts.

    Handles common malformations from model generation:
    - Missing ``<ithink>`` or ``<ioutput>`` opening tags
    - Missing ``<iconv>`` wrapper tags
    - Output-only blocks (no think content)

    Parameters
    ----------
    text : str
        Raw decoded text of the assistant response (may include special tokens
        such as ``<|role_end|>``, ``<|endoftext|>``, ``<|interleaved|>``).

    Returns
    -------
    list[dict]
        Each element is ``{"think": str, "output": str}``.
    """
    # Strip special tokens and role markers
    text = _SPECIAL_TOKEN_RE.sub("", text)
    text = re.sub(r"<role>.*?</role>", "", text)
    text = text.strip()

    # Split into blocks by <iconv> / </iconv> boundaries
    blocks = re.split(r"</?iconv>", text)
    blocks = [b for b in blocks if b.strip()]

    results: List[Dict[str, str]] = []
    for block in blocks:
        think = ""
        output = ""

        # Extract think content (try proper tag first, fall back to missing opening tag)
        think_match = re.search(r"<ithink>(.*?)</ithink>", block, re.DOTALL)
        if not think_match:
            think_match = re.search(r"(.*?)</ithink>", block, re.DOTALL)
        if think_match:
            think = think_match.group(1).strip()
            remaining = block[think_match.end():]
        else:
            remaining = block

        # Extract output content (try proper tag first, fall back to missing opening tag)
        output_match = re.search(r"<ioutput>(.*?)</ioutput>", remaining, re.DOTALL)
        if not output_match:
            output_match = re.search(r"(.*?)</ioutput>", remaining, re.DOTALL)
        if output_match:
            output = output_match.group(1).strip()
        elif remaining.strip():
            # No output tags at all — treat the remainder as output
            output = remaining.strip()

        # Clean any stray opening tags left over from malformed generation
        think = re.sub(r"^<ithink>", "", think).strip()
        output = re.sub(r"^<ioutput>", "", output).strip()

        if think or output:
            results.append({"think": think, "output": output})

    return results


def build_messages(
    prompt: str,
    raw_response: str,
    system_prompt: Optional[str] = None,
) -> List[Dict]:
    """Combine a user prompt and raw inference output into the training-data
    ``messages`` format.

    Parameters
    ----------
    prompt : str
        The user question / instruction text.
    raw_response : str
        Raw decoded assistant response (as saved by inference script).
    system_prompt : str, optional
        Optional system message prepended to the conversation.

    Returns
    -------
    list[dict]
        ``[{role, content}, {role, icontent}, ...]`` matching the schema used
        by the training pipeline.
    """
    messages: List[Dict] = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    messages.append({
        "role": "assistant",
        "icontent": parse_interleaved_response(raw_response),
    })
    return messages


def parse_result_dir(
    question_dir: str,
    result_dir: str,
    system_prompt: Optional[str] = None,
) -> List[Dict]:
    """Batch-parse all question–result file pairs into messages format.

    Parameters
    ----------
    question_dir : str
        Directory containing the input question files.
    result_dir : str
        Directory containing the corresponding inference result files
        (filenames must match those in *question_dir*).
    system_prompt : str, optional
        Optional system message prepended to every conversation.

    Returns
    -------
    list[dict]
        One element per file, each containing ``{"name": str, "messages": list}``.
    """
    import os

    results = []
    for fname in sorted(os.listdir(result_dir)):
        result_path = os.path.join(result_dir, fname)
        question_path = os.path.join(question_dir, fname)
        if not os.path.isfile(result_path):
            continue

        with open(result_path, "r") as f:
            raw_response = f.read()

        prompt = ""
        if os.path.isfile(question_path):
            with open(question_path, "r") as f:
                prompt = f.read()

        messages = build_messages(prompt, raw_response, system_prompt)
        results.append({"name": fname, "messages": messages})

    return results


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Parse interleaved inference results")
    parser.add_argument("--question-dir", type=str, required=True)
    parser.add_argument("--result-dir", type=str, required=True)
    parser.add_argument("--output", type=str, default=None, help="Output JSONL path (default: stdout)")
    args = parser.parse_args()

    parsed = parse_result_dir(args.question_dir, args.result_dir)

    if args.output:
        with open(args.output, "w") as f:
            for item in parsed:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Wrote {len(parsed)} entries to {args.output}")
    else:
        for item in parsed:
            print(json.dumps(item, indent=2, ensure_ascii=False))