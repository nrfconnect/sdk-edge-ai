#!/usr/bin/env bash
#
# Format C/C++ source files using Zephyr's checkpatch.pl --fix-inplace.
#
# This is the canonical "formatter" for this repo: it produces output that
# matches what the pre-push hook (and CI's Checkpatch test) require, which
# clang-format alone cannot reproduce (e.g. designated-initializer tables,
# LINE_SPACING after declarations, BLOCK_COMMENT_STYLE).
#
# Usage:
#   scripts/checkpatch-format.sh FILE [FILE...]
#
# Exit status:
#   0  - files were processed (whether or not anything was changed)
#   1  - bad arguments / missing tooling
#
# Requirements:
#   - perl in /usr/bin/perl (matches the pre-push hook).
#
# ZEPHYR_BASE resolution order:
#   1) use existing $ZEPHYR_BASE from environment.
#   2) fallback to sibling zephyr repo next to sdk-edge-ai checkout
#      (../zephyr from this repo root).

set -u

if [[ $# -lt 1 ]]; then
	echo "usage: $0 FILE [FILE...]" >&2
	exit 1
fi

if [[ -z "${ZEPHYR_BASE:-}" ]]; then
	SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
	REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
	CANDIDATE_ZEPHYR_BASE="$(cd -- "$REPO_ROOT/../zephyr" 2>/dev/null && pwd || true)"

	if [[ -n "$CANDIDATE_ZEPHYR_BASE" && -f "$CANDIDATE_ZEPHYR_BASE/scripts/checkpatch.pl" ]];
    then
		ZEPHYR_BASE="$CANDIDATE_ZEPHYR_BASE"
		export ZEPHYR_BASE
	else
		echo "checkpatch-format: ZEPHYR_BASE is not set and auto-detection failed." >&2
		echo "  Expected checkpatch at: $REPO_ROOT/../zephyr/scripts/checkpatch.pl" >&2
		echo "  Open the nRF Connect SDK terminal, or set ZEPHYR_BASE manually." >&2
		exit 1
	fi
fi

CHECKPATCH="${ZEPHYR_BASE}/scripts/checkpatch.pl"
if [[ ! -x "$CHECKPATCH" && ! -r "$CHECKPATCH" ]]; then
	echo "checkpatch-format: cannot find $CHECKPATCH" >&2
	exit 1
fi

# Filter to files that actually exist and look like C/C++ sources we want to
# touch. checkpatch on a non-source file is harmless but noisy.
files=()
for f in "$@"; do
	if [[ ! -f "$f" ]]; then
		echo "checkpatch-format: skipping (not a file): $f" >&2
		continue
	fi
	case "$f" in
	*.c | *.h | *.cpp | *.cc | *.hpp) files+=("$f") ;;
	*)
		echo "checkpatch-format: skipping (unsupported extension): $f" >&2
		;;
	esac
done

if [[ ${#files[@]} -eq 0 ]]; then
	exit 0
fi

# Same flags as the pre-push hook's --fix-inplace pass.
/usr/bin/perl "$CHECKPATCH" \
	--fix-inplace \
	--no-tree \
	--quiet \
	--max-line-length=100 \
	--min-conf-desc-length=1 \
	--ignore SPDX_LICENSE_TAG \
	-f "${files[@]}" \
	>/dev/null 2>&1 || true

# checkpatch leaves *.EXPERIMENTAL-checkpatch-fixes alongside the originals
# in some cases; clean them up so they don't pollute the tree.
for f in "${files[@]}"; do
	rm -f -- "${f}.EXPERIMENTAL-checkpatch-fixes"
done

exit 0
