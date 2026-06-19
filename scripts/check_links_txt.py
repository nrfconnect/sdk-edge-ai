#!/usr/bin/env python3
#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#

"""Validate doc/links.txt for the Edge AI add-on documentation.

Static checks (default):
  - URL syntax (no trailing punctuation)
  - unresolved |substitution| markers after expansion
  - duplicate link target names
  - links defined in links.txt but not referenced in RST sources (warning;
    use --fail-on-unused to treat as an error)

Optional HTTP checks (--online):
  - Detect permanent redirects (301/308) unless benign
  - HTTP 403 is warned by default; use --fail-on-403 to treat it as an error (CI)
  - Other HTTP failures (404, 5xx, ...) still fail
  - With --fail-on-permanent-redirect: fail on meaningful permanent redirects
  - With --changed-only: probe only link lines added since --changed-since (CI / PR diffs)

Compliance CI runs --online --changed-only --fail-on-permanent-redirect --fail-on-403.
Doc build CI runs static checks only.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

LINK_RE = re.compile(r"^\.\. _`(?P<name>[^`]+)`: (?P<url>https?://\S+)\s*$")
MARKER_RE = re.compile(r"\|[A-Za-z_][A-Za-z0-9_]*\|")
TRAILING_JUNK_RE = re.compile(r"[>?.,;:!)\]}>]+$")
INCLUDE_RE = re.compile(r"^\.\.\s+include::\s+(\S+)", re.MULTILINE)
ANON_HYPERLINK_REF_RE = re.compile(r"`([^<`][^`]*)`_")
EXPLICIT_HYPERLINK_REF_RE = re.compile(r"`[^`]*<([^>]+)_>`")

DOC_SOURCE_GLOBS = (
    "doc/**/*.rst",
    "applications/**/*.rst",
    "samples/**/*.rst",
    "tests/**/*.rst",
)
DOC_INCLUDE_DIR = "doc/includes"

ONLINE_SKIP_PATTERNS = (
    re.compile(r"^https://app\.lab\.nordicsemi\.com/"),
)

PERMANENT_REDIRECT_CODES = frozenset({301, 308})
REDIRECT_CODES = PERMANENT_REDIRECT_CODES | frozenset({302, 303, 307})
MAX_REDIRECT_HOPS = 10
USER_AGENT = "edge-ai-link-check/1.0"


@dataclass(frozen=True)
class RedirectHop:
    code: int
    from_url: str
    to_url: str


@dataclass
class HttpProbeResult:
    status: int | str
    hops: list[RedirectHop] = field(default_factory=list)


@dataclass(frozen=True)
class LinkEntry:
    line_no: int
    name: str
    url: str


@dataclass(frozen=True)
class WorkspaceRoots:
    edge_ai: Path
    ncs_version_number: str


def _find_workspace_roots(start: Path | None) -> WorkspaceRoots:
    edge_ai = start.resolve() if start else Path.cwd().resolve()
    for candidate in (edge_ai, *edge_ai.parents):
        if (candidate / "doc" / "links.txt").is_file() and (candidate / "west.yml").is_file():
            edge_ai = candidate
            break
    else:
        raise SystemExit("Could not locate sdk-edge-ai workspace root (need doc/links.txt and west.yml)")

    version_file = edge_ai.parent / "nrf" / "VERSION"
    if not version_file.is_file():
        raise SystemExit(
            f"Missing {version_file}. Run west update so |ncs_version_number| can be expanded."
        )

    return WorkspaceRoots(
        edge_ai,
        version_file.read_text(encoding="utf-8").strip(),
    )


def _parse_links(links_path: Path) -> list[LinkEntry]:
    entries: list[LinkEntry] = []
    for line_no, raw_line in enumerate(links_path.read_text(encoding="utf-8").splitlines(), start=1):
        match = LINK_RE.match(raw_line)
        if match:
            entries.append(LinkEntry(line_no, match.group("name"), match.group("url")))
    return entries


def _expand_url(url: str, ncs_version_number: str) -> str:
    return url.replace("|ncs_version_number|", ncs_version_number)


def _check_syntax(entries: list[LinkEntry], ncs_version_number: str) -> list[str]:
    issues: list[str] = []
    seen: dict[str, int] = {}

    for entry in entries:
        if TRAILING_JUNK_RE.search(entry.url):
            issues.append(
                f"Line {entry.line_no}: URL for '{entry.name}' has trailing punctuation: "
                f"{entry.url!r}"
            )

        expanded = _expand_url(entry.url, ncs_version_number)
        leftovers = MARKER_RE.findall(expanded)
        if leftovers:
            issues.append(
                f"Line {entry.line_no}: unresolved substitution(s) {sorted(set(leftovers))} "
                f"in URL for '{entry.name}'"
            )

        if entry.name in seen:
            issues.append(
                f"Line {entry.line_no}: duplicate link name '{entry.name}' "
                f"(first defined on line {seen[entry.name]})"
            )
        else:
            seen[entry.name] = entry.line_no

    return issues


def _resolve_include_target(include_path: str, source_file: Path, doc_dir: Path) -> Path | None:
    raw = include_path.strip()
    candidates: list[Path]
    if raw.startswith("/"):
        candidates = [doc_dir / raw.lstrip("/")]
    else:
        candidates = [source_file.parent / raw, doc_dir / raw]

    include_name = Path(raw).name
    if include_name:
        candidates.append(doc_dir / "includes" / include_name)

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    return None


def _collect_doc_source_files(edge_ai: Path) -> list[Path]:
    doc_dir = edge_ai / "doc"
    seen: set[Path] = set()
    pending: list[Path] = []

    for pattern in DOC_SOURCE_GLOBS:
        pending.extend(sorted(edge_ai.glob(pattern)))

    include_dir = edge_ai / DOC_INCLUDE_DIR
    if include_dir.is_dir():
        pending.extend(sorted(include_dir.glob("**/*")))

    sources: list[Path] = []
    while pending:
        path = pending.pop()
        resolved = path.resolve()
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)
        sources.append(resolved)

        try:
            text = resolved.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        for match in INCLUDE_RE.finditer(text):
            included = _resolve_include_target(match.group(1), resolved, doc_dir)
            if included is not None:
                pending.append(included)

    return sources


def _extract_link_references(text: str) -> set[str]:
    refs: set[str] = set()
    refs.update(name.strip() for name in ANON_HYPERLINK_REF_RE.findall(text))
    refs.update(name.strip() for name in EXPLICIT_HYPERLINK_REF_RE.findall(text))
    return refs


def _collect_referenced_link_names(edge_ai: Path) -> set[str]:
    referenced: set[str] = set()
    for path in _collect_doc_source_files(edge_ai):
        for name in _extract_link_references(path.read_text(encoding="utf-8")):
            referenced.add(name.casefold())

    shortcuts_path = edge_ai / "doc" / "shortcuts.txt"
    if shortcuts_path.is_file():
        for name in _extract_link_references(shortcuts_path.read_text(encoding="utf-8")):
            referenced.add(name.casefold())

    return referenced


def _check_unused_links(
    entries: list[LinkEntry],
    edge_ai: Path,
    *,
    fail_on_unused: bool,
) -> tuple[list[str], list[str]]:
    defined = {entry.name: entry.line_no for entry in entries}
    referenced = _collect_referenced_link_names(edge_ai)
    errors: list[str] = []
    warnings: list[str] = []

    for name, line_no in sorted(defined.items(), key=lambda item: item[1]):
        if name.casefold() in referenced:
            continue
        message = (
            f"Line {line_no}: link '{name}' is not referenced in documentation "
            f"(remove from links.txt or add a reference)"
        )
        if fail_on_unused:
            errors.append(message)
        else:
            warnings.append(message)

    return errors, warnings


class _NoFollowRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _normalize_url(url: str) -> str:
    parsed = urlparse(url.strip())
    path = parsed.path.rstrip("/") or "/"
    query = parsed.query.rstrip("?&")
    return urlunparse((
        parsed.scheme.lower(),
        parsed.netloc.lower(),
        path,
        parsed.params,
        query,
        "",
    ))


def _same_host(left: str, right: str) -> bool:
    def strip_www(host: str) -> str:
        host = host.lower()
        return host[4:] if host.startswith("www.") else host

    return strip_www(left) == strip_www(right)


def _is_nordic_files_url(url: str) -> bool:
    return _same_host(urlparse(url).netloc, "files.nordicsemi.com")


def _is_benign_redirect(from_url: str, to_url: str) -> bool:
    if _normalize_url(from_url) == _normalize_url(to_url):
        return True

    source = urlparse(from_url)
    target = urlparse(to_url)

    if _same_host(source.netloc, "youtu.be"):
        youtube_host = target.netloc.lower()
        if youtube_host.startswith("www."):
            youtube_host = youtube_host[4:]
        if youtube_host == "youtube.com" or youtube_host.endswith(".youtube.com"):
            return True

    if (
        source.scheme == "http"
        and target.scheme == "https"
        and _same_host(source.netloc, target.netloc)
        and source.path == target.path
        and source.query == target.query
    ):
        return True

    return (
        source.scheme == target.scheme
        and _same_host(source.netloc, target.netloc)
        and source.path == target.path
        and source.query == target.query
    )


def _request_once(url: str, *, method: str) -> tuple[int, dict[str, str]]:
    req = Request(url, method=method, headers={"User-Agent": USER_AGENT})
    opener = build_opener(_NoFollowRedirectHandler())

    try:
        with opener.open(req, timeout=30) as response:
            return response.status, {k.lower(): v for k, v in response.headers.items()}
    except HTTPError as error:
        return error.code, {k.lower(): v for k, v in error.headers.items()}
    except URLError as error:
        return str(error.reason), {}


def _probe_url(url: str) -> HttpProbeResult:
    method = "GET" if _is_nordic_files_url(url) else "HEAD"
    current = url
    hops: list[RedirectHop] = []

    for _ in range(MAX_REDIRECT_HOPS):
        status, headers = _request_once(current, method=method)

        if status in REDIRECT_CODES:
            location = headers.get("location")
            if not location:
                return HttpProbeResult(status, hops)
            next_url = urljoin(current, location)
            hops.append(RedirectHop(status, current, next_url))
            current = next_url
            method = "HEAD"
            continue

        if status in (403, 405, 501) and method == "HEAD" and not _is_nordic_files_url(current):
            status, headers = _request_once(current, method="GET")
            if status in REDIRECT_CODES:
                location = headers.get("location")
                if not location:
                    return HttpProbeResult(status, hops)
                next_url = urljoin(current, location)
                hops.append(RedirectHop(status, current, next_url))
                current = next_url
                continue

        return HttpProbeResult(status, hops)

    return HttpProbeResult("too many redirects", hops)


def _check_online(
    entries: list[LinkEntry],
    ncs_version_number: str,
    *,
    fail_on_permanent_redirect: bool,
    fail_on_403: bool,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    for entry in entries:
        url = _expand_url(entry.url, ncs_version_number)
        if any(pattern.search(url) for pattern in ONLINE_SKIP_PATTERNS):
            continue

        probe = _probe_url(url)
        if probe.status != 200:
            if probe.status == 403:
                message = (
                    f"Line {entry.line_no}: HTTP 403 for '{entry.name}' "
                    f"(automated request blocked; link not verified): {url}"
                )
                if fail_on_403:
                    errors.append(message)
                else:
                    warnings.append(message)
            else:
                errors.append(
                    f"Line {entry.line_no}: HTTP check failed for '{entry.name}' "
                    f"(status {probe.status}): {url}"
                )

        for hop in probe.hops:
            if hop.code not in PERMANENT_REDIRECT_CODES or _is_benign_redirect(hop.from_url, hop.to_url):
                continue

            message = (
                f"Line {entry.line_no}: permanent redirect for '{entry.name}' "
                f"({hop.code}): {hop.from_url} -> {hop.to_url} "
                f"(update links.txt to the final URL)"
            )
            if fail_on_permanent_redirect:
                errors.append(message)
            else:
                warnings.append(message)

    return errors, warnings


def _default_changed_since() -> str:
    base_ref = os.environ.get("GITHUB_BASE_REF")
    if base_ref:
        return f"origin/{base_ref}"
    return "origin/main"


def _filter_changed_entries(
    entries: list[LinkEntry],
    *,
    since: str,
    links_path: Path,
    edge_ai: Path,
) -> list[LinkEntry]:
    try:
        rel_path = links_path.resolve().relative_to(edge_ai.resolve())
    except ValueError:
        rel_path = links_path

    proc = subprocess.run(
        ["git", "diff", "--unified=0", f"{since}..HEAD", "--", str(rel_path)],
        cwd=edge_ai,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(
            f"git diff {since}..HEAD -- {rel_path} failed: {proc.stderr.strip() or proc.stdout.strip()}"
        )

    added_names: set[str] = set()
    for line in proc.stdout.splitlines():
        if not line.startswith("+") or line.startswith("+++"):
            continue
        match = LINK_RE.match(line[1:])
        if match:
            added_names.add(match.group("name"))

    return [entry for entry in entries if entry.name in added_names]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--links-file",
        type=Path,
        default=None,
        help="Path to links.txt (default: <edge-ai>/doc/links.txt)",
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help="Probe each URL for HTTP failures and permanent redirects",
    )
    parser.add_argument(
        "--fail-on-permanent-redirect",
        action="store_true",
        help="With --online, fail when a URL permanently redirects to a different URL",
    )
    parser.add_argument(
        "--fail-on-403",
        action="store_true",
        help="With --online, fail on HTTP 403 (default: warn only)",
    )
    parser.add_argument(
        "--fail-on-unused",
        action="store_true",
        help="Fail when a links.txt entry is not referenced in RST sources (default: warn only)",
    )
    parser.add_argument(
        "--skip-unused",
        action="store_true",
        help="Skip the unused-link reference scan",
    )
    parser.add_argument(
        "--changed-only",
        action="store_true",
        help="With --online, probe only link definitions added in links.txt since --changed-since",
    )
    parser.add_argument(
        "--changed-since",
        default=None,
        metavar="REF",
        help="Base git ref for --changed-only (default: origin/$GITHUB_BASE_REF or origin/main)",
    )
    args = parser.parse_args()

    if args.changed_only and not args.online:
        parser.error("--changed-only requires --online")

    roots = _find_workspace_roots(args.links_file.parent.parent if args.links_file else None)
    links_path = args.links_file or (roots.edge_ai / "doc" / "links.txt")
    if not links_path.is_file():
        raise SystemExit(f"Missing links file: {links_path}")

    entries = _parse_links(links_path)
    issues: list[str] = []
    warnings: list[str] = []
    online_entries = entries

    issues.extend(_check_syntax(entries, roots.ncs_version_number))

    if not args.skip_unused:
        unused_errors, unused_warnings = _check_unused_links(
            entries,
            roots.edge_ai,
            fail_on_unused=args.fail_on_unused,
        )
        issues.extend(unused_errors)
        warnings.extend(unused_warnings)

    if args.online:
        online_entries = entries
        if args.changed_only:
            since = args.changed_since or _default_changed_since()
            online_entries = _filter_changed_entries(
                entries,
                since=since,
                links_path=links_path,
                edge_ai=roots.edge_ai,
            )
            if not online_entries:
                print(f"No added links in {links_path} since {since}; skipping online checks")

        online_errors, online_warnings = _check_online(
            online_entries,
            roots.ncs_version_number,
            fail_on_permanent_redirect=args.fail_on_permanent_redirect,
            fail_on_403=args.fail_on_403,
        )
        issues.extend(online_errors)
        warnings.extend(online_warnings)

    if warnings:
        print(f"Warnings ({len(warnings)}):")
        for warning in warnings:
            print(f"- {warning}")

    if issues:
        print(f"Found {len(issues)} issue(s) in {links_path}:")
        for issue in issues:
            print(f"- {issue}")
        return 1

    modes = ["static"]
    if not args.skip_unused:
        modes.append("no-unused" if args.fail_on_unused else "unused-warn")
    if args.online:
        modes.append("online")
        if args.changed_only:
            modes.append("changed-only")
        if args.fail_on_permanent_redirect:
            modes.append("no-permanent-redirects")
        if args.fail_on_403:
            modes.append("no-403")
    checked = len(online_entries) if args.online else len(entries)
    print(
        f"OK: {checked} link(s) passed {' + '.join(modes)} checks "
        f"(ncs_version_number={roots.ncs_version_number})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
