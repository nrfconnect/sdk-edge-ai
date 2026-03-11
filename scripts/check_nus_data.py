#!/usr/bin/env python3
#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#

import argparse
from pathlib import Path


def check_consecutive_ids(log_path: Path) -> tuple[list[str], int]:
    """Return (issues, parsed_count) after validating consecutive IDs."""
    issues: list[str] = []
    parsed_count = 0
    prev_id: int | None = None

    with log_path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()

            if not line:
                continue

            first_token = line.split(maxsplit=1)[0]
            try:
                curr_id = int(first_token)
            except ValueError:
                issues.append(
                    f"Line {line_no}: could not parse entry ID from first token '{first_token}'"
                )
                continue

            parsed_count += 1

            if prev_id is not None:
                expected = prev_id + 1
                if curr_id != expected:
                    issues.append(f"Line {line_no}: expected ID {expected}, got {curr_id}")

            prev_id = curr_id

    return issues, parsed_count


def create_raw_log(log_path: Path, *, overwrite: bool = False) -> tuple[Path, int]:
    """Create a copy with leading ID removed from each parsable line."""
    raw_path = log_path.with_name(f"{log_path.stem}_data{log_path.suffix}")
    converted_count = 0

    if not overwrite and raw_path.exists():
        raise FileExistsError(f"Output file already exists: {raw_path}")

    with log_path.open("r", encoding="utf-8") as src, raw_path.open("w", encoding="utf-8") as dst:
        for raw_line in src:
            line = raw_line.rstrip("\n")
            parts = line.split(maxsplit=1)

            if len(parts) == 2:
                try:
                    int(parts[0])
                except ValueError:
                    dst.write(raw_line)
                else:
                    dst.write(parts[1] + "\n")
                    converted_count += 1
            else:
                dst.write(raw_line)

    return raw_path, converted_count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="""
            Check NUS serial output for consecutive entry IDs.
            Expected format is "<id> <data>".
            If no problems were detected, then creates a '_data' file with IDs removed.
        """,
        allow_abbrev=False,
    )
    parser.add_argument(
        "log_file",
        help="Path to serial log file",
    )
    parser.add_argument(
        "-f", "--force",
        action="store_true",
        help="Overwrite existing output file",
    )
    args = parser.parse_args()

    log_path = Path(args.log_file)
    if not log_path.is_file():
        print(f"Error: file not found: {log_path}")
        return 2

    issues, parsed_count = check_consecutive_ids(log_path)

    if parsed_count == 0:
        print("No valid log entries found.")
        return 1

    if issues:
        print(f"Found {len(issues)} issue(s) in {parsed_count} parsed entries:")
        for issue in issues:
            print(f"- {issue}")
        return 1

    print(f"OK: {parsed_count} entries have consecutive IDs.")

    try:
        raw_path, converted_count = create_raw_log(log_path, overwrite=args.force)
    except FileExistsError as error:
        print(f"Error: {error}. Use -f/--force to overwrite it.")
        return 1

    print(f"Created raw log: {raw_path} ({converted_count} entries converted)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
