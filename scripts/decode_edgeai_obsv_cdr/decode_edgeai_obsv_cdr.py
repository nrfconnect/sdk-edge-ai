#!/usr/bin/env python3
#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#
"""Decode a Memfault chunk carrying an nrf_edgeai_obsv CDR payload.

Three input modes:

1. Local chunk (the default) - decode a raw Memfault chunk dumped from UART
   or sniffed on the BLE link. Layout per chunk:

       <chunk-header>  1 byte    Memfault chunk framing byte.
       <event-type>    1 byte    Event type marker (0x04 == CDR).
                                 Present only in the FIRST chunk of a message;
                                 continuation chunks omit this byte.
       <cbor-event>    N bytes   CBOR map fragment (full event in single-chunk;
                                 partial in multi-chunk messages).
       <crc>           2 bytes   CRC-16 trailer per chunk.

   Memfault chunk_transport sets header bit 7 for CONTINUATION fragments;
   single-chunk local mode only accepts an INIT fragment (bit 7 clear). Pass
   all fragments via ``--chunks`` when the message is split.

   For CDR events the reassembled CBOR event map has the following keys (per
   memfault-firmware-sdk serializer):

       2  event type tag (5 == CDR)
       3  event format version
       6  hardware version
       9  firmware version
       10 SDK type
       11 device id / serial
       4  CDR sub-map:
              1  schema version
              2  mimetype list
              3  collection reason
              4  raw payload (byte string)

2. Multi-chunk reassembly (--chunks) - when a large CDR payload is split
   across two or more BLE/UART chunks. Provide each chunk's hex separately;
   the script strips framing from every chunk, concatenates the payloads, and
   decodes the reassembled event as a single message.

3. Cloud fetch (--from-cloud) - pull a CDR payload directly from the
   Memfault REST API. The cloud strips the chunk/event envelope during
   ingest, so the downloaded bytes are the inner payload only.

4. Plain CBOR file (--binary --file) - decode a raw ``.bin`` saved from the
   Memfault web UI "Download" button. Those files are the inner nrf_edgeai_obsv
   CBOR array/map with no Memfault chunk or event wrapper. Pass ``--binary``.

   CAVEAT: as of 2026-04 the Memfault public API (api-docs.memfault.com)
   documents only *upload* endpoints for CDRs (POST /upload/... /Commit
   Custom Data Recording). The list and download endpoints used below
   are the UNDOCUMENTED ones the web UI consumes; they work today but
   Memfault may change them without notice. If that happens, run with
   --verbose, confirm which paths 404, and update the constants below.
   The official alternative is manual download via the Memfault web UI's
   per-CDR "Download" button, or asking Memfault support for a supported
   programmatic route.

The inner payload is the CBOR blob produced by nrf_edgeai_obsv_memfault.
It is a CBOR array with one entry per observed model:

    [
      {
        "format_version": ...,
        "num_inferences": ...,   # PROBS-stream updates since last reset
        "num_features": ...,     # FEATURES-stream updates since last reset (counter)
        "model": { "id": ..., "num_classes": ..., "num_features": ..., "version": ... },
        "metrics": [
          { "id": metric_id, "v": metric_version, "d": [[...], [...], ...] },
          ...
        ]
      },
      ...  # one entry per model
    ]

Single-model deployments produce a one-element array.

Usage (paths relative to the sdk-edge-ai tree):

    # Local chunk (UART hexdump, file, or stdin)
    ./scripts/decode_edgeai_obsv_cdr/decode_edgeai_obsv_cdr.py <hex-string>
    ./scripts/decode_edgeai_obsv_cdr/decode_edgeai_obsv_cdr.py --file chunk.hex
    echo <hex> | ./scripts/decode_edgeai_obsv_cdr/decode_edgeai_obsv_cdr.py -

    # Plain CBOR downloaded from Memfault's "CDR Payloads" menu (.bin)
    ./scripts/decode_edgeai_obsv_cdr/decode_edgeai_obsv_cdr.py --binary --file recording.bin
    ./scripts/decode_edgeai_obsv_cdr/decode_edgeai_obsv_cdr.py --binary - < recording.bin

    # Multi-chunk reassembly (each arg is one complete chunk)
    ./scripts/decode_edgeai_obsv_cdr/decode_edgeai_obsv_cdr.py --chunks <hex-chunk1> <hex-chunk2> ...

    # From Memfault cloud. Auth options (pick one):
    #   a) Admin: MEMFAULT_ORG_TOKEN
    #   b) Any user: MEMFAULT_USER_EMAIL + MEMFAULT_USER_API_KEY
    # Plus MEMFAULT_ORG and MEMFAULT_PROJECT for both.
    ./scripts/decode_edgeai_obsv_cdr/decode_edgeai_obsv_cdr.py --from-cloud --device model-obsv-dev-nrf54lm20
    ./scripts/decode_edgeai_obsv_cdr/decode_edgeai_obsv_cdr.py --from-cloud --device <SN> --limit 5
    ./scripts/decode_edgeai_obsv_cdr/decode_edgeai_obsv_cdr.py --from-cloud --cdr-id 12345
    # Optional --device with --cdr-id probes device-scoped download URLs if org-level paths 404.
    ./scripts/decode_edgeai_obsv_cdr/decode_edgeai_obsv_cdr.py --from-cloud --cdr-id 12345 --device <SN>

    # Fleet mode (all devices, mirrors https://app.memfault.com/.../custom-data-recordings)
    ./scripts/decode_edgeai_obsv_cdr/decode_edgeai_obsv_cdr.py --from-cloud --fleet
    ./scripts/decode_edgeai_obsv_cdr/decode_edgeai_obsv_cdr.py --from-cloud --fleet --limit 20
    ./scripts/decode_edgeai_obsv_cdr/decode_edgeai_obsv_cdr.py --from-cloud --fleet --reason "" --limit 5

Requires Python 3.9+, with dependencies ``cbor2`` and ``requests`` installed.
Install with: ``pip install -r scripts/decode_edgeai_obsv_cdr/requirements.txt``
(from the sdk-edge-ai root), or ``pip install cbor2 requests``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

import cbor2
import requests

log = logging.getLogger(__name__)


def _redact_url_for_log(url: str) -> str:
    """Drop query/fragment from URLs in debug logs (presigned links often embed secrets)."""
    p = urlparse(url)
    if not p.query and not p.fragment:
        return url
    return urlunparse((p.scheme, p.netloc, p.path, p.params, "<redacted>", ""))


# Mirrors nrf_edgeai_obsv_metric_id enum in nrf_edgeai_obsv_metrics.h.
# Update here whenever a new metric ID is added on the firmware side.
METRIC_NAMES = {
    2: "transition_matrix",
    3: "probs_distribution",
    4: "prediction_switching_rate",
    5: "probs_entropy_dist",
    6: "probs_top2_margin_dist",
    7: "mel_energy_desc",
    8: "mel_spectral_desc",
    9: "class_streak_dist",
}

EVENT_TYPES = {
    2: "heartbeat",
    4: "cdr",
    5: "log",
    6: "trace",
}

DEFAULT_API_BASE = "https://api.memfault.com"
DEFAULT_REASON = "edgeai_observability"

# Listing: Memfault may ignore ``per_page`` or return pages oldest-first; see
# ``_cloud_list_all_cdrs`` (all pages) and ``_fetch_cloud_list`` (newest-first sort).
CDR_LIST_PER_PAGE_DEFAULT = 100
CDR_LIST_PER_PAGE_CAP = 250


# ---------------------------------------------------------------------------
# Shared payload decoding (used by both local and cloud modes)
# ---------------------------------------------------------------------------


def _decode_one_obsv_payload(decoded: dict, validate: bool = False) -> dict[str, Any]:
    """Annotate a single already-decoded CBOR obsv-payload map."""
    result: dict[str, Any] = {
        "format_version": decoded.get("format_version"),
        "num_inferences": decoded.get("num_inferences"),
        "num_features": decoded.get("num_features"),
        "model": decoded.get("model"),
        "metrics": [],
    }

    n = decoded.get("num_inferences")
    nf = decoded.get("num_features")
    for metric in decoded.get("metrics", []):
        if not isinstance(metric, dict):
            continue
        mid = metric.get("id")
        data = metric.get("d") or []

        entry: dict[str, Any] = {
            "id": mid,
            "name": METRIC_NAMES.get(mid, f"unknown_{mid}"),
            "version": metric.get("v"),
            "shape": [len(data), max((len(r) for r in data if isinstance(r, list)), default=0)],
            "data": data,
        }

        if validate:
            row_sums = [sum(row) for row in data if isinstance(row, list)]
            entry["row_sums"] = row_sums
            if mid == 3 and isinstance(n, int):
                # probs_distribution: every class histogram must total num_inferences.
                entry["row_sums_match_n"] = all(s == n for s in row_sums)
            if mid == 2 and isinstance(n, int):
                total = sum(row_sums)
                entry["total_transitions"] = total
                # First inference has no predecessor => (n - 1) transitions total.
                entry["matches_n_minus_one"] = total == max(n - 1, 0)
            if mid in (7, 8) and isinstance(nf, int):
                # FEATURES-stream descriptors histogram one entry per feature
                # update, so every row must total num_features.
                entry["row_sums_match_num_features"] = all(s == nf for s in row_sums)
            if mid == 9 and isinstance(n, int):
                # class_streak_dist records one count per COMPLETED streak, not
                # per inference, so the matrix totals the number of finished
                # streaks. Unlike pd/ped/pmd, rows do NOT sum to num_inferences;
                # the streak count is bounded by (and in practice far below) n.
                total = sum(row_sums)
                entry["total_streaks"] = total
                entry["streaks_le_n"] = total <= n

        result["metrics"].append(entry)

    return result


def _decode_obsv_cdr(payload: bytes, validate: bool = False) -> list[dict[str, Any]]:
    """Decode the CDR inner payload into a list of per-model obsv-payload dicts.

    The wire format is a CBOR array ``[obsv-payload, ...]`` (one element per
    observed model). Single-model deployments produce a one-element list.

    For backward compatibility, a bare CBOR map is accepted and wrapped in a
    one-element list so callers always get a list regardless of CDR version.
    """
    decoded = cbor2.loads(payload)

    if isinstance(decoded, list):
        entries = decoded
    elif isinstance(decoded, dict):
        # Old single-map format (pre-multi-model).
        entries = [decoded]
    else:
        raise ValueError(
            f"CDR payload is neither a CBOR array nor a map "
            f"(got {type(decoded).__name__})"
        )

    result: list[dict[str, Any]] = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(
                f"CDR array element {i} is not a CBOR map "
                f"(got {type(entry).__name__})"
            )
        result.append(_decode_one_obsv_payload(entry, validate=validate))

    return result


# ---------------------------------------------------------------------------
# Local chunk path
# ---------------------------------------------------------------------------

# Memfault chunk_transport.h: bit 7 = continuation (INIT chunks have this clear).
_CHUNK_HDR_CONTINUATION_BIT = 0x80


def _chunk_header_is_continuation(header: int) -> bool:
    return (header & _CHUNK_HDR_CONTINUATION_BIT) != 0


def _strip_hex(text: str) -> bytes:
    """Accept hex with whitespace, newlines, or a leading 0x; ignore the rest."""
    cleaned = re.sub(r"(?i)0x|\s+|:|,", "", text)
    if not cleaned:
        raise ValueError("empty hex input")
    if len(cleaned) % 2 != 0:
        raise ValueError(f"hex input has odd length ({len(cleaned)} chars)")
    try:
        return bytes.fromhex(cleaned)
    except ValueError as exc:
        raise ValueError(f"invalid hex characters: {exc}") from None


def decode_plain_cbor(raw: bytes, validate: bool = False) -> dict[str, Any]:
    """Decode a raw inner CDR CBOR blob (Memfault web download or API body)."""
    return {
        "source": "plain_cbor",
        "payload_size_bytes": len(raw),
        "payloads": _decode_obsv_cdr(raw, validate=validate),
    }


def _unwrap_chunk(raw: bytes) -> tuple[int, int, bytes, bytes]:
    """Split a single-chunk Memfault frame into (hdr, event_type, body, crc).

    The CRC is not verified here: the Memfault chunk protocol uses a CRC-16
    variant whose parameters differ across firmware-sdk versions. We treat the
    last 2 bytes as the trailer and leave integrity checking to the caller.
    """
    if len(raw) < 5:
        raise ValueError(f"chunk too short: {len(raw)} bytes")
    return raw[0], raw[1], raw[2:-2], raw[-2:]


def _decode_outer(event: Any, validate: bool = False) -> dict[str, Any]:
    """Pull the interesting fields out of the Memfault event map."""
    if not isinstance(event, dict):
        raise ValueError(f"outer CBOR is not a map (got {type(event).__name__})")

    cdr = event.get(4)
    if not isinstance(cdr, dict):
        raise ValueError("event has no CDR sub-map under key 4")

    payload = cdr.get(4)
    if not isinstance(payload, (bytes, bytearray)):
        raise ValueError("CDR sub-map has no payload byte string under key 4")

    _dev = event.get(11)
    if isinstance(_dev, (bytes, bytearray, memoryview)):
        device_id: str | None = bytes(_dev).hex()
    elif _dev is not None:
        device_id = str(_dev)
    else:
        device_id = None

    outer = {
        "event_type": event.get(2),
        "event_format_version": event.get(3),
        "sdk_type": event.get(10),
        "firmware_version": event.get(9),
        "hardware_version": event.get(6),
        "device_id": device_id,
        "cdr": {
            "schema_version": cdr.get(1),
            "mimetypes": cdr.get(2),
            "collection_reason": cdr.get(3),
            "payload_size": len(payload),
        },
    }

    outer["cdr"]["payloads"] = _decode_obsv_cdr(bytes(payload), validate=validate)
    return outer


def decode_chunk(raw: bytes, validate: bool = False) -> dict[str, Any]:
    if len(raw) >= 1 and _chunk_header_is_continuation(raw[0]):
        raise ValueError(
            "chunk header has bit 7 set (Memfault CONTINUATION). "
            "Local single-chunk decode expects the INIT (first) fragment, or pass "
            "all fragments in order via --chunks. "
            "If your capture uses varint offsets after the header on continuations, "
            "strip or reassemble to match this script's layout (see module docstring)."
        )
    header, event_type, body, crc = _unwrap_chunk(raw)
    try:
        event = cbor2.loads(body)
    except cbor2.CBORDecodeError as exc:
        raise ValueError(f"outer CBOR decode failed: {exc}") from None

    return {
        "source": "local_chunk",
        "chunk_total_bytes": len(raw),
        "chunk_header_byte": f"0x{header:02x}",
        "event_type": {
            "value": event_type,
            "name": EVENT_TYPES.get(event_type, f"unknown_{event_type}"),
        },
        "chunk_crc_trailer": crc.hex(),
        "event": _decode_outer(event, validate=validate),
    }


def decode_chunks(hex_list: list[str], validate: bool = False) -> dict[str, Any]:
    """Reassemble a multi-chunk Memfault message and decode it.

    Each string in *hex_list* is the full hex of one Memfault chunk
    (header + payload + CRC-16 trailer).  The function strips the 1-byte
    chunk header and the 2-byte CRC from every chunk, concatenates the
    resulting payloads, and then decodes the combined byte stream exactly
    like ``decode_chunk`` does for a single-chunk message.

    Chunk layout assumed here:
      - First chunk:        [hdr(1)][event_type(1)][cbor_fragment][crc(2)]
      - Continuation chunks:[hdr(1)][cbor_fragment][crc(2)]

    After stripping, concatenation yields [event_type(1)][full_cbor_event],
    which is identical to what ``decode_chunk`` expects at raw[1:].
    """
    if not hex_list:
        raise ValueError("--chunks requires at least one hex chunk")

    first_raw = _strip_hex(hex_list[0])
    if first_raw and _chunk_header_is_continuation(first_raw[0]):
        raise ValueError(
            "first --chunks fragment must be the INIT chunk (header bit 7 clear), "
            "not a CONTINUATION fragment"
        )

    combined = b""
    chunk_headers: list[str] = []
    chunk_sizes: list[int] = []

    for i, hex_str in enumerate(hex_list):
        raw = _strip_hex(hex_str)
        if len(raw) < 4:
            raise ValueError(
                f"chunk {i}: too short ({len(raw)} bytes); "
                "expected at least header(1) + 1 byte payload + crc(2)"
            )
        chunk_headers.append(f"0x{raw[0]:02x}")
        chunk_sizes.append(len(raw))
        combined += raw[1:-2]  # strip header and CRC

    if not combined:
        raise ValueError("no payload after stripping chunk framing")

    event_type = combined[0]
    cbor_body = combined[1:]

    try:
        event = cbor2.loads(cbor_body)
    except cbor2.CBORDecodeError as exc:
        raise ValueError(
            f"outer CBOR decode failed after reassembly: {exc}"
        ) from None

    return {
        "source": "local_chunks_reassembled",
        "num_chunks": len(hex_list),
        "chunk_sizes_bytes": chunk_sizes,
        "chunk_headers": chunk_headers,
        "reassembled_cbor_bytes": len(cbor_body),
        "event_type": {
            "value": event_type,
            "name": EVENT_TYPES.get(event_type, f"unknown_{event_type}"),
        },
        "event": _decode_outer(event, validate=validate),
    }


def _read_file_bytes(path: str) -> bytes:
    return Path(path).read_bytes()


def _read_hex_input(args: argparse.Namespace) -> str:
    if args.hex == "-":
        return sys.stdin.read()
    if args.hex:
        return args.hex
    if not sys.stdin.isatty():
        text = sys.stdin.read()
        if text.strip():
            return text
    raise SystemExit(
        "error: provide a hex chunk as an argument, via --file, on stdin, "
        "or use --from-cloud"
    )


def _decode_local_input(args: argparse.Namespace) -> dict[str, Any]:
    """Decode local input: Memfault chunk hex or plain CBOR binary (--binary)."""
    if args.binary:
        if args.file:
            raw = _read_file_bytes(args.file)
        elif args.hex == "-" or not sys.stdin.isatty():
            raw = sys.stdin.buffer.read()
        else:
            raise SystemExit(
                "error: --binary requires --file PATH or binary data on stdin"
            )
        if not raw:
            raise ValueError("empty binary input")
        return decode_plain_cbor(raw, validate=args.validate)

    if args.file:
        try:
            text = Path(args.file).read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise SystemExit(
                "error: file is not UTF-8 hex text; for a Memfault web "
                "download (.bin), pass --binary"
            ) from exc
        raw = _strip_hex(text)
        return decode_chunk(raw, validate=args.validate)

    raw = _strip_hex(_read_hex_input(args))
    return decode_chunk(raw, validate=args.validate)


# ---------------------------------------------------------------------------
# Cloud fetch path
# ---------------------------------------------------------------------------


@dataclass
class CloudConfig:
    org: str
    project: str
    api_base: str = DEFAULT_API_BASE
    # Exactly one of these auth paths is populated:
    org_token: str | None = None
    user_email: str | None = None
    user_api_key: str | None = None


def _cloud_config(args: argparse.Namespace) -> CloudConfig:
    org_token = args.org_token or os.environ.get("MEMFAULT_ORG_TOKEN")
    user_email = args.user_email or os.environ.get("MEMFAULT_USER_EMAIL")
    user_api_key = args.user_api_key or os.environ.get("MEMFAULT_USER_API_KEY")
    org = args.org or os.environ.get("MEMFAULT_ORG")
    project = args.project or os.environ.get("MEMFAULT_PROJECT")
    api_base = (
        args.api_base or os.environ.get("MEMFAULT_API_BASE") or DEFAULT_API_BASE
    ).rstrip("/")

    has_org_token = bool(org_token)
    has_user_auth = bool(user_email and user_api_key)
    if not (has_org_token or has_user_auth):
        raise SystemExit(
            "error: missing Memfault auth. Provide either:\n"
            "  - MEMFAULT_ORG_TOKEN (or --org-token), OR\n"
            "  - MEMFAULT_USER_EMAIL + MEMFAULT_USER_API_KEY "
            "(or --user-email + --user-api-key)"
        )
    if has_org_token and has_user_auth:
        raise SystemExit(
            "error: pass either an org token or user-email+user-api-key, not both"
        )

    missing = [
        name
        for name, value in (
            ("--org / MEMFAULT_ORG", org),
            ("--project / MEMFAULT_PROJECT", project),
        )
        if not value
    ]
    if missing:
        raise SystemExit("error: missing Memfault config: " + ", ".join(missing))

    return CloudConfig(
        org=org,
        project=project,
        api_base=api_base,
        org_token=org_token,
        user_email=user_email if has_user_auth else None,
        user_api_key=user_api_key if has_user_auth else None,
    )


def _cloud_session(cfg: CloudConfig):
    session = requests.Session()
    session.headers.update({"Accept": "application/json"})
    if cfg.org_token:
        session.headers["Memfault-Organization-Token"] = cfg.org_token
    else:
        # User-scoped auth: HTTP Basic with email:user_api_key.
        session.auth = (cfg.user_email, cfg.user_api_key)
    return session


def _cloud_get(session, url: str, params: dict | None = None, raw: bool = False):
    """GET a Memfault API URL. Retries once with underscore-style path if the
    dash-style endpoint returns 404, since Memfault has used both conventions.
    Returns parsed JSON (raw=False) or the raw response (raw=True).
    """
    log.debug("GET %s params=%s", _redact_url_for_log(url), params)
    response = session.get(url, params=params, timeout=60)
    if response.status_code == 404 and "custom-data-recordings" in url:
        alt = url.replace("custom-data-recordings", "custom_data_recordings")
        log.debug("  404; retrying with underscore form: %s", _redact_url_for_log(alt))
        response = session.get(alt, params=params, timeout=60)
    if response.status_code != 200:
        body = response.text[:500]
        raise SystemExit(
            f"error: Memfault API {response.status_code} for {response.url}\n{body}"
        )
    if raw:
        log.debug("  -> %d bytes binary", len(response.content))
        return response
    try:
        payload = response.json()
    except ValueError as exc:
        raise SystemExit(f"error: Memfault API returned non-JSON: {exc}") from None
    log.debug("  -> JSON: %r", payload)
    return payload


def _cdr_list_url(cfg: CloudConfig, device: str) -> str:
    return (
        f"{cfg.api_base}/api/v0/organizations/{cfg.org}"
        f"/projects/{cfg.project}/devices/{device}/custom-data-recordings"
    )


def _cdr_fleet_list_url(cfg: CloudConfig) -> str:
    """Project-level CDR list; mirrors the Fleet CDR Viewer UI (2026-04).

    UI:  https://app.memfault.com/organizations/{org}/projects/{project}/custom-data-recordings
    API: https://api.memfault.com/api/v0/organizations/{org}/projects/{project}/custom-data-recordings
    """
    return (
        f"{cfg.api_base}/api/v0/organizations/{cfg.org}"
        f"/projects/{cfg.project}/custom-data-recordings"
    )


def _unwrap_paginated(payload: Any) -> list[Any]:
    """Memfault APIs return either {"data": [...]} or a bare list."""
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return payload["data"]
    raise SystemExit(
        f"error: unexpected list payload shape: {json.dumps(payload)[:400]}"
    )


def _cdr_list_paging_meta(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        p = payload.get("paging")
        if isinstance(p, dict):
            return p
    return {}


def _dedupe_cdr_rows_by_id(rows: list[Any]) -> list[Any]:
    """Keep one dict per ``id`` (last wins). Skips rows without ``id``."""
    by_id: dict[Any, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        rid = row.get("id")
        if rid is None:
            continue
        by_id[rid] = row
    return list(by_id.values())


def _cloud_list_cdrs_page(
    session: Any,
    list_url: str,
    *,
    page: int,
    per_page: int,
    reason: str | None,
) -> tuple[list[Any], dict[str, Any]]:
    params: dict[str, Any] = {"page": page, "per_page": per_page}
    if reason:
        params["collection_reason"] = reason
    payload = _cloud_get(session, list_url, params=params)
    return _unwrap_paginated(payload), _cdr_list_paging_meta(payload)


def _cloud_list_all_cdrs(
    session: Any,
    list_url: str,
    *,
    per_page: int,
    reason: str | None,
) -> tuple[list[Any], int]:
    """Fetch every page of a paginated CDR list; return deduplicated rows and page count."""
    items_first, paging = _cloud_list_cdrs_page(
        session, list_url, page=1, per_page=per_page, reason=reason,
    )
    total_pages = max(int(paging.get("total_pages") or 1), 1)
    all_items: list[Any] = list(items_first)

    for page in range(2, total_pages + 1):
        page_items, _ = _cloud_list_cdrs_page(
            session, list_url, page=page, per_page=per_page, reason=reason,
        )
        all_items.extend(page_items)
        log.debug("CDR list page %s/%s: %s row(s)", page, total_pages, len(page_items))

    return _dedupe_cdr_rows_by_id(all_items), total_pages


def _parse_timestamp_for_sort(value: Any) -> float | None:
    """Return POSIX seconds for ordering list rows, or None if not parseable."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        sec = float(value)
        if sec > 1e12:
            sec /= 1000.0
        return sec
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


_CDR_LIST_TIME_FIELDS = (
    "created_date",
    "created_at",
    "updated_at",
    "start_time",
    "end_time",
)


def _best_timestamp_unix(item: dict[str, Any]) -> float | None:
    for key in _CDR_LIST_TIME_FIELDS:
        ts = _parse_timestamp_for_sort(item.get(key))
        if ts is not None:
            return ts
    return None


def _cdr_newest_first_sort_key(item: Any) -> tuple[float, int]:
    """Sort key with reverse=True → newest first; tie-break by numeric id."""
    if not isinstance(item, dict):
        return (float("-inf"), -1)
    ts = _best_timestamp_unix(item)
    ts_part = ts if ts is not None else float("-inf")
    raw_id = item.get("id")
    try:
        id_part = int(raw_id)
    except (TypeError, ValueError):
        id_part = -1
    return (ts_part, id_part)


def _unwrap_single(payload: Any) -> dict[str, Any]:
    if (
        isinstance(payload, dict)
        and "data" in payload
        and isinstance(payload["data"], dict)
    ):
        return payload["data"]
    if isinstance(payload, dict):
        return payload
    raise SystemExit(
        f"error: unexpected single-item payload: {json.dumps(payload)[:400]}"
    )


DOWNLOAD_URL_KEYS = (
    "data_url",
    "download_url",
    "url",
    "raw_url",
    "payload_url",
    "presigned_url",
)

# Empirically-verified (but officially UNDOCUMENTED) Memfault CDR endpoints:
#
#   list:     GET /api/v0/.../devices/{sn}/custom-data-recordings      (plural)
#   download: GET /api/v0/.../custom-data-recording/{id}/download      (singular)
#
# See the module docstring for the caveat. The small fallback sweep below is
# kept so that if Memfault renames a path, --verbose shows the new shape.
_PRIMARY_RESOURCE = "custom-data-recording"
_PRIMARY_SUFFIX = "/download"
_FALLBACK_RESOURCES = ("custom-data-recordings", "custom_data_recording")
_FALLBACK_SUFFIXES = ("/download", "/data", "/raw", "")


def _looks_like_binary(content_type: str) -> bool:
    ct = content_type.lower()
    return not any(k in ct for k in ("json", "text/", "html"))


def _try_download_via_url(session, url: str) -> bytes | None:
    """GET url accepting binary. Returns bytes on 200-binary, None on 404.
    Follows an embedded JSON {url/download_url/...} once if that's what comes back."""
    log.debug("try download: %s", _redact_url_for_log(url))
    response = session.get(
        url,
        timeout=120,
        headers={"Accept": "application/octet-stream, */*"},
    )
    if response.status_code == 404:
        return None
    if response.status_code in (401, 403):
        raise SystemExit(
            f"error: Memfault API {response.status_code} "
            f"(authentication/permission denied) for {response.url}\n"
            f"{response.text[:400]}"
        )
    if response.status_code != 200:
        log.debug("  -> HTTP %d: %s", response.status_code, response.text[:200])
        return None
    content_type = response.headers.get("Content-Type", "")
    if _looks_like_binary(content_type):
        log.debug("  -> %d binary bytes (%s)", len(response.content), content_type)
        return response.content
    # Might be JSON wrapping a signed download URL.
    try:
        payload = response.json()
    except ValueError:
        log.debug("  -> non-JSON text, treating as payload (%s)", content_type)
        return response.content
    log.debug("  -> JSON wrapper: %r", payload)
    wrapper = payload.get("data") if isinstance(payload, dict) else payload
    if isinstance(wrapper, dict):
        for key in DOWNLOAD_URL_KEYS:
            signed = wrapper.get(key)
            if signed:
                log.debug(
                    "  following wrapper['%s'] -> %s",
                    key,
                    _redact_url_for_log(str(signed)),
                )
                follow = session.get(signed, timeout=120)
                if follow.status_code == 200:
                    return follow.content
                log.debug("    follow HTTP %d", follow.status_code)
    return None


def _download_cdr_bytes(
    session, cfg: CloudConfig, cdr_id: int | str, device: str | None
) -> bytes:
    """Fetch a CDR's raw bytes. Uses the verified primary URL first and falls
    back to a small set of alternates if that ever 404s.
    """
    base = f"{cfg.api_base}/api/v0/organizations/{cfg.org}/projects/{cfg.project}"
    candidates: list[str] = [
        f"{base}/{_PRIMARY_RESOURCE}/{cdr_id}{_PRIMARY_SUFFIX}",
    ]
    for resource in _FALLBACK_RESOURCES:
        for suffix in _FALLBACK_SUFFIXES:
            candidates.append(f"{base}/{resource}/{cdr_id}{suffix}")
            if device:
                candidates.append(
                    f"{base}/devices/{device}/{resource}/{cdr_id}{suffix}"
                )

    attempted: list[str] = []
    for url in candidates:
        if url in attempted:
            continue
        attempted.append(url)
        data = _try_download_via_url(session, url)
        if data is not None:
            return data

    raise SystemExit(
        "error: could not locate a download URL for CDR id "
        f"{cdr_id}. All probe paths returned 404.\n"
        "       Tried:\n         " + "\n         ".join(attempted)
    )


def _count_cloud(args: argparse.Namespace) -> dict[str, Any]:
    """List-only: hit the list endpoint with per_page=1 and read total_count
    from the paging block. One GET, no downloads.
    """
    cfg = _cloud_config(args)
    session = _cloud_session(cfg)
    if args.fleet:
        list_url = _cdr_fleet_list_url(cfg)
        scope = "fleet"
    elif args.device:
        list_url = _cdr_list_url(cfg, args.device)
        scope = args.device
    else:
        raise SystemExit("error: --count requires --device or --fleet")
    _, paging = _cloud_list_cdrs_page(
        session,
        list_url,
        page=1,
        per_page=1,
        reason=args.reason,
    )
    if "total_count" not in paging:
        raise SystemExit(
            "error: list response has no paging.total_count: "
            f"{json.dumps(paging, default=str)[:400]}"
        )
    return {
        "scope": scope,
        "reason": args.reason or None,
        "total_count": paging["total_count"],
    }


def _fetch_cloud_list(
    session: Any,
    cfg: CloudConfig,
    list_url: str,
    *,
    limit: int,
    reason: str | None,
    device_hint: str | None,
    validate: bool,
) -> list[dict[str, Any]]:
    """Shared list-and-download logic for per-device and fleet modes."""
    per_page = min(max(limit, CDR_LIST_PER_PAGE_DEFAULT), CDR_LIST_PER_PAGE_CAP)

    items, total_pages = _cloud_list_all_cdrs(
        session, list_url, per_page=per_page, reason=reason,
    )
    log.debug(
        "CDR list: per_page=%s pages=%s unique row(s)=%s",
        per_page, total_pages, len(items),
    )

    dict_rows = [row for row in items if isinstance(row, dict)]
    if dict_rows and all(_best_timestamp_unix(row) is None for row in dict_rows):
        log.warning(
            "CDR list rows had no parseable time fields %s; "
            "falling back to numeric id descending (largest id first)",
            _CDR_LIST_TIME_FIELDS,
        )
    items = sorted(items, key=_cdr_newest_first_sort_key, reverse=True)
    items = items[:limit]

    log.debug(
        "CDR list after sort: downloading id(s) %s",
        [row.get("id") if isinstance(row, dict) else None for row in items],
    )

    results: list[dict[str, Any]] = []
    n_dl = len(items)
    for i, summary in enumerate(items, start=1):
        cdr_id = summary.get("id")
        if cdr_id is None:
            raise SystemExit(
                f"error: list item missing 'id': {json.dumps(summary)[:400]}"
            )
        print(
            f"decode_edgeai_obsv_cdr: fetching CDR id={cdr_id} ({i}/{n_dl})...",
            file=sys.stderr,
            flush=True,
        )
        raw = _download_cdr_bytes(session, cfg, cdr_id, device=device_hint)
        results.append(
            _decode_cloud_item(summary, raw, device_hint=device_hint, validate=validate)
        )
    return results


def _fetch_cloud(args: argparse.Namespace) -> list[dict[str, Any]]:
    cfg = _cloud_config(args)
    session = _cloud_session(cfg)

    if args.cdr_id:
        # No list metadata; fabricate a stub so _decode_cloud_item still works.
        raw = _download_cdr_bytes(session, cfg, args.cdr_id, device=args.device)
        stub = {"id": args.cdr_id}
        return [
            _decode_cloud_item(
                stub, raw, device_hint=args.device, validate=args.validate
            )
        ]

    if args.fleet:
        return _fetch_cloud_list(
            session,
            cfg,
            _cdr_fleet_list_url(cfg),
            limit=args.limit,
            reason=args.reason,
            device_hint=None,
            validate=args.validate,
        )

    if not args.device:
        raise SystemExit(
            "error: --from-cloud requires --device, --fleet, or --cdr-id"
        )

    return _fetch_cloud_list(
        session,
        cfg,
        _cdr_list_url(cfg, args.device),
        limit=args.limit,
        reason=args.reason,
        device_hint=args.device,
        validate=args.validate,
    )


def _decode_cloud_item(
    item: dict[str, Any],
    raw: bytes,
    device_hint: str | None = None,
    validate: bool = False,
) -> dict[str, Any]:
    device_value = item.get("device")
    if isinstance(device_value, dict):
        device_value = device_value.get("device_serial") or device_value.get("serial")
    return {
        "source": "memfault_cloud",
        "cdr": {
            "id": item.get("id"),
            # Memfault's CDR list uses 'reason', 'start_time', 'end_time';
            # older/other endpoints may use 'collection_reason', 'created_date'.
            "reason": item.get("reason") or item.get("collection_reason"),
            "start_time": item.get("start_time") or item.get("created_date"),
            "end_time": item.get("end_time"),
            "mimetypes": item.get("mimetypes"),
            "size_bytes": item.get("size_bytes") or len(raw),
            "device": device_value or device_hint,
        },
        "payloads": _decode_obsv_cdr(raw, validate=validate),
    }


def _cli_positive_int(text: str) -> int:
    """argparse type for integers >= 1 (used by --limit)."""
    try:
        n = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None
    if n < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return n


def _limit_flag_in_argv(argv: list[str]) -> bool:
    return any(a == "--limit" or a.startswith("--limit=") for a in argv)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Decode an nrf_edgeai_obsv CDR payload, either from a "
            "locally captured Memfault chunk or directly from the Memfault cloud."
        ),
        allow_abbrev=False,
    )
    # Local mode inputs.
    parser.add_argument(
        "hex",
        nargs="?",
        help="Hex-encoded chunk (local mode). Use '-' to read from stdin.",
    )
    parser.add_argument(
        "--file",
        "-f",
        help="Read a hex-encoded Memfault chunk from a file (use with --binary for .bin).",
    )
    parser.add_argument(
        "--binary",
        action="store_true",
        help=(
            "Read raw CBOR bytes from --file or stdin (inner nrf_edgeai_obsv "
            "payload from a Memfault web download)."
        ),
    )
    parser.add_argument(
        "--chunks",
        nargs="+",
        metavar="HEX",
        help=(
            "Two or more hex-encoded Memfault chunks to reassemble before "
            "decoding. Each argument is one complete chunk (header + payload "
            "+ CRC). Use when a large CDR payload is split across multiple "
            "BLE notifications or UART frames."
        ),
    )

    # Cloud mode.
    parser.add_argument(
        "--from-cloud",
        action="store_true",
        help="Fetch CDRs from the Memfault REST API instead of decoding local hex.",
    )
    parser.add_argument(
        "--device",
        help="Device serial (used with --from-cloud for per-device listing or --cdr-id fallback).",
    )
    parser.add_argument(
        "--fleet",
        action="store_true",
        help=(
            "List CDRs across all devices in the project (fleet-wide). "
            "Mirrors https://app.memfault.com/.../custom-data-recordings. "
            "Used with --from-cloud; mutually exclusive with --device."
        ),
    )
    parser.add_argument(
        "--cdr-id",
        help=(
            "Specific CDR id (used with --from-cloud). Optional --device helps download "
            "URL fallbacks when org-level paths 404."
        ),
    )
    parser.add_argument(
        "--limit",
        type=_cli_positive_int,
        default=1,
        help=(
            "Number of most-recent CDRs to fetch (--from-cloud with --device or "
            "--fleet; default: 1). Ignored for local hex, --chunks, and --cdr-id."
        ),
    )
    parser.add_argument(
        "--reason",
        default=DEFAULT_REASON,
        help=(
            f"Filter by collection_reason. Default: {DEFAULT_REASON!r}. "
            "Pass an empty string to disable filtering."
        ),
    )
    parser.add_argument(
        "--count",
        action="store_true",
        help=(
            "List-only mode: print how many CDRs match --device or --fleet (+--reason) "
            "and exit, without downloading payloads. Implies --from-cloud."
        ),
    )

    # Memfault config. Each falls back to MEMFAULT_* env vars.
    parser.add_argument("--org", help="Overrides MEMFAULT_ORG (org slug).")
    parser.add_argument("--project", help="Overrides MEMFAULT_PROJECT (project slug).")
    parser.add_argument(
        "--api-base",
        help=f"Overrides MEMFAULT_API_BASE (default: {DEFAULT_API_BASE}).",
    )

    # Auth: use either an Organization Auth Token (admin-only) or a personal
    # User API Key + email (any user; scoped to your permissions).
    parser.add_argument(
        "--org-token",
        help="Organization Auth Token (admin). Overrides MEMFAULT_ORG_TOKEN.",
    )
    parser.add_argument(
        "--user-email",
        help="Your Memfault login email (user-auth). Overrides MEMFAULT_USER_EMAIL.",
    )
    parser.add_argument(
        "--user-api-key",
        help="Personal User API Key (user-auth). Overrides MEMFAULT_USER_API_KEY.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print each HTTP request/response (to stderr) for debugging.",
    )

    # Output.
    parser.add_argument(
        "--output",
        "-o",
        help="Write JSON output to FILE instead of stdout.",
    )
    parser.add_argument(
        "--jsonl",
        action="store_true",
        help=(
            "Emit JSON Lines (one record per line, no outer wrapper). "
            "Handy for pandas.read_json(lines=True), jq -c, and concat."
        ),
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help=(
            "Add decoder-side integrity checks to each metric entry "
            "(row_sums, row_sums_match_n, total_transitions, "
            "matches_n_minus_one). Off by default."
        ),
    )
    return parser


def _write_output(records: list[dict[str, Any]], args: argparse.Namespace) -> None:
    """Serialise `records` as either pretty JSON or JSON Lines, to stdout or a file."""
    if args.jsonl:
        lines = [json.dumps(r, default=str) + "\n" for r in records]
        text = "".join(lines)
    elif len(records) == 1:
        text = json.dumps(records[0], indent=2, default=str) + "\n"
    else:
        text = (
            json.dumps(
                {"count": len(records), "results": records},
                indent=2,
                default=str,
            )
            + "\n"
        )

    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        log.debug(
            "wrote %d record(s) (%s) to %s",
            len(records),
            "jsonl" if args.jsonl else "json",
            args.output,
        )
    else:
        sys.stdout.write(text)


def main() -> int:
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(message)s",
        stream=sys.stderr,
    )

    if args.device and args.fleet:
        print("error: --device and --fleet are mutually exclusive", file=sys.stderr)
        return 2

    if (
        _limit_flag_in_argv(sys.argv)
        and not args.from_cloud
        and not args.count
    ):
        log.warning(
            "--limit applies only to --from-cloud when listing by --device or --fleet; "
            "local decoding ignores it."
        )

    try:
        if args.count:
            records = [_count_cloud(args)]
        elif args.from_cloud:
            records = _fetch_cloud(args)
        elif args.chunks:
            records = [decode_chunks(args.chunks, validate=args.validate)]
        else:
            records = [_decode_local_input(args)]
        _write_output(records, args)
    except (ValueError, cbor2.CBORDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
