"""Tool for converting JSONL files (with .gz) to Parquet format."""

import argparse
import gzip
from pathlib import Path

import logging

import jsonlines
import pyarrow as pa
import pyarrow.parquet as pq

_ICONTENT_TYPE = pa.list_(
    pa.struct(
        [
            pa.field("think", pa.string()),
            pa.field("output", pa.string()),
        ]
    )
)

_MESSAGE_TYPE = pa.list_(
    pa.struct(
        [
            pa.field("role", pa.string()),
            pa.field("content", pa.string()),
            pa.field("icontent", _ICONTENT_TYPE),
        ]
    )
)

SCHEMA = pa.schema(
    [
        pa.field("raw_data_id", pa.string()),
        pa.field("messages", _MESSAGE_TYPE),
        pa.field("tags", pa.list_(pa.string())),
        pa.field("data_source", pa.string()),
    ]
)

logger = logging.getLogger(__name__)


def convert_jsonl_to_parquet(
    input_path: str, output_path: str = None, chunk_size: int = None
) -> list[str]:
    """Convert JSONL file (with optional .gz compression) to Parquet format.

    Args:
        input_path: Path to input JSONL file (.jsonl or .jsonl.gz)
        output_path: Path to output Parquet file. If None, uses same directory
                     and filename as input with .parquet extension
        chunk_size: If provided, split output into multiple files each containing
                    at most this many records (e.g. 100000). Files are named
                    <stem>_001.parquet, <stem>_002.parquet, ...

    Returns:
        List of paths to the created Parquet file(s)
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Derive base stem and output dir
    stem = input_path.name
    if stem.endswith(".gz"):
        stem = stem[:-3]
    if stem.endswith(".jsonl"):
        stem = stem[:-6]

    if output_path is None:
        out_base = input_path.parent / stem
    else:
        out_base = Path(output_path)
        # Strip .parquet suffix so we can append chunk indices when needed
        if out_base.suffix == ".parquet":
            out_base = out_base.with_suffix("")

    logger.info(f"Reading JSONL from: {input_path}")

    # Read JSONL file
    data = []
    if str(input_path).endswith(".gz"):
        with gzip.open(input_path, "rt", encoding="utf-8") as f:
            with jsonlines.Reader(f) as reader:
                data = list(reader)
    else:
        with jsonlines.open(input_path) as reader:
            data = list(reader)

    logger.info(f"Read {len(data)} records from JSONL")

    def _write(records: list, path: Path) -> str:
        table = pa.Table.from_pylist(records, schema=SCHEMA)
        pq.write_table(table, path, compression="snappy")
        logger.info(f"Written {len(records)} records → {path}")
        return str(path)

    if chunk_size is None or chunk_size <= 0:
        out_path = Path(str(out_base) + ".parquet")
        return [_write(data, out_path)]

    # Split into chunks
    total = len(data)
    n_chunks = (total + chunk_size - 1) // chunk_size
    width = len(str(n_chunks))
    output_paths = []
    for i in range(n_chunks):
        chunk = data[i * chunk_size : (i + 1) * chunk_size]
        idx = str(i + 1).zfill(width)
        out_path = Path(f"{out_base}_{idx}.parquet")
        output_paths.append(_write(chunk, out_path))

    logger.info(f"Split into {n_chunks} chunks of up to {chunk_size} records each")
    return output_paths


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Convert JSONL files (with optional .gz compression) to Parquet format"
    )
    parser.add_argument(
        "input",
        type=str,
        nargs="+",
        help="Path(s) to input JSONL file(s) (.jsonl or .jsonl.gz). Multiple files supported.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Path to output Parquet file (only used when converting a single file; default: same directory and filename with .parquet extension)",
    )
    parser.add_argument(
        "-c",
        "--chunk-size",
        type=int,
        default=None,
        metavar="N",
        help="Split output into chunks of at most N records each (e.g. 100000). "
        "Files are named <stem>_001.parquet, <stem>_002.parquet, ... "
        "If omitted, a single file is produced.",
    )

    args = parser.parse_args()

    # Process multiple files
    success_count = 0
    fail_count = 0

    for input_file in args.input:
        try:
            # Only use custom output path if processing a single file
            output_path_arg = args.output if len(args.input) == 1 else None
            output_paths = convert_jsonl_to_parquet(input_file, output_path_arg, args.chunk_size)
            for p in output_paths:
                logger.info(f"Conversion completed: {p}")
            success_count += 1
        except Exception as e:
            logger.error(f"Conversion failed for {input_file}: {e}")
            fail_count += 1

    # Summary
    logger.info(f"\nConversion summary: {success_count} succeeded, {fail_count} failed")

    if fail_count > 0:
        raise RuntimeError(f"{fail_count} file(s) failed to convert")


if __name__ == "__main__":
    main()
