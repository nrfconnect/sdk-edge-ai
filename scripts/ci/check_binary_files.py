#!/usr/bin/env python3
#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause

"""Check that a git patch does not add disallowed binary files."""

import argparse
import re
import shlex
import subprocess
import sys
from pathlib import Path

import jsonschema
import yaml
from jsonschema.exceptions import ValidationError

TITLE = 'BinaryFiles'
DOC = 'No binary files allowed.'

# Safe charset for --commits.
_COMMIT_RANGE_RE = re.compile(r'^[0-9A-Za-z._/@^~:-]+$')

_CONFIG_SCHEMA = {
    'type': 'object',
    'required': ['allowed_paths', 'allowed_extensions', 'preapproved_binaries'],
    'additionalProperties': False,
    'properties': {
        'allowed_paths': {
            'type': 'array',
            'items': {'type': 'string'},
        },
        'allowed_extensions': {
            'type': 'array',
            'items': {'type': 'string'},
        },
        'preapproved_binaries': {
            'type': 'array',
            'items': {'type': 'string'},
        },
    },
}


def parse_args():
    default_config = Path(__file__).parent / 'check_binary_files.yaml'
    parser = argparse.ArgumentParser(
        description='Check that the diff contains no disallowed binary files.',
        allow_abbrev=False,
    )
    parser.add_argument(
        '-c',
        '--commits',
        default='HEAD~1..',
        help='Commit range in the form: a..[b], default is HEAD~1..HEAD',
    )
    parser.add_argument(
        '-f',
        '--config',
        type=Path,
        default=default_config,
        help=f'Configuration file, default is {default_config}',
    )
    parser.add_argument(
        '--github',
        action='store_true',
        help='Print GitHub Actions workflow commands to stdout.',
    )
    return parser.parse_args()


def validate_commit_range(commits: str) -> str:
    """Reject values that could be interpreted as git CLI options."""
    if not commits or commits.startswith('-') or '--' in commits:
        raise ValueError(f'Invalid commit range: {commits!r}')

    if not _COMMIT_RANGE_RE.fullmatch(commits):
        raise ValueError(f'Invalid commit range: {commits!r}')

    return commits


def run_git(*args: str) -> str:
    run_cmd = ('git',) + args
    run_str = ' '.join(shlex.quote(arg) for arg in run_cmd)
    process = subprocess.run(run_cmd, capture_output=True, text=True, check=False)
    stdout = process.stdout
    stderr = process.stderr
    if process.returncode:
        raise RuntimeError(
            f'Command "{run_str}" exited with {process.returncode}\n'
            f'==stdout==\n{stdout}\n==stderr==\n{stderr}'
        )
    return stdout.rstrip()


def checked_repo_root() -> Path:
    """Return the git top directory for the repository being checked (cwd)."""
    return Path(run_git('rev-parse', '--show-toplevel'))


def resolve_config_path(config_file: Path) -> Path:
    """Only allow config files inside the git repository being checked."""
    resolved = config_file.expanduser().resolve(strict=False)
    repo_root = checked_repo_root()

    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(
            f'Config file must be inside the checked repository ({repo_root}), not {config_file}',
        ) from exc

    if not resolved.is_file():
        raise ValueError(f'Config file not found: {resolved}')

    return resolved


def load_config(config_file: Path) -> dict:
    config_file = resolve_config_path(config_file)

    with open(config_file, encoding='utf-8') as fd:
        data = yaml.safe_load(fd)

    try:
        jsonschema.validate(data, _CONFIG_SCHEMA)
    except ValidationError as exc:
        raise ValueError(f'Invalid config file {config_file}: {exc.message}') from exc

    return data


def is_allowed(fname: str, config: dict) -> bool:
    allowed_paths = tuple(config['allowed_paths'])
    allowed_extensions = tuple(config['allowed_extensions'])
    preapproved_binaries = set(config['preapproved_binaries'])

    if fname in preapproved_binaries:
        return True

    return fname.startswith(allowed_paths) and fname.endswith(allowed_extensions)


def escape_github_message(message: str) -> str:
    return message.replace('%', '%25').replace('\n', '%0A').replace('\r', '%0D')


def report_error(fname: str, message: str, github: bool) -> None:
    if github:
        msg = escape_github_message(message)
        doc = escape_github_message(DOC)
        print(f'::error file={fname},title={TITLE}::{msg}%0A{doc}')
    else:
        print(f'ERROR: {fname}: {message}', file=sys.stderr)


def check_binary_files(commits: str, config: dict, github: bool) -> bool:
    output = run_git('diff', '--numstat', '--diff-filter=A', commits)
    errors = []

    for stat in output.splitlines():
        if not stat.strip():
            continue

        parts = stat.split('\t')
        if len(parts) != 3:
            continue

        added, deleted, fname = parts
        if added != '-' or deleted != '-':
            continue

        if is_allowed(fname, config):
            continue

        message = f'Binary file not allowed: {fname}'
        report_error(fname, message, github)
        errors.append(fname)

    if errors:
        if not github:
            print(f'Binary file check failed with {len(errors)} error(s).', file=sys.stderr)
        return False

    print('Binary file check successful.')
    return True


def main() -> None:
    args = parse_args()

    try:
        commits = validate_commit_range(args.commits)
        config = load_config(args.config)
        success = check_binary_files(commits, config, args.github)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)

    if not success:
        sys.exit(1)


if __name__ == '__main__':
    main()
