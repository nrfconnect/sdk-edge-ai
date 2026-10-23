#!/usr/bin/env python3
#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#
"""Check Memfault OTA releases for a Bluetooth LE device and flash them over SMP.

Extends the nRF Connect Device Manager Memfault FOTA flow with Memfault multi-component
releases, in which non-primary components (for example model images) are separate software
types with their own versions and artifacts:

1. Over SMP, read the device identity and project key from the Memfault MCUmgr group
   (CONFIG_MEMFAULT_MCUMGR_GRP) and the installed MCUboot images.
2. Ask Memfault for the latest release of the primary software type and of every component
   given with --component SOFTWARE_TYPE=IMAGE_INDEX. A component's current version is the
   version of its installed image as reported by MCUmgr, for example "0.1.2.1".
3. Download the artifacts. The primary artifact may be a dfu_application.zip with several
   images; a component artifact is a single signed MCUboot binary.
4. Upload the images that are not installed yet, confirm them by hash, reset the device and
   check that the new images are installed.

Examples::

    ./memfault_ota.py scan
    ./memfault_ota.py info -d "WW KWS"
    ./memfault_ota.py check -d "WW KWS" --component ww-model=1 --component kws-model=2
    ./memfault_ota.py update -d "WW KWS" --component ww-model=1 --component kws-model=2
    ./memfault_ota.py update -d "WW KWS" --file build/dfu_application.zip
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import requests
from bleak import BleakScanner
from bleak.exc import BleakError
from smp import error as smperror
from smp import message as smpmsg
from smpclient import SMPClient
from smpclient.exceptions import SMPClientException
from smpclient.generics import error
from smpclient.mcuboot import IMAGE_TLV, ImageInfo, TLVNotFound
from smpclient.requests.image_management import ImageStatesRead, ImageStatesWrite
from smpclient.requests.os_management import ResetWrite
from smpclient.transport import SMP_SERVICE_UUID, SMPTransportDisconnected
from smpclient.transport.ble import SMPBLETransport

MEMFAULT_API_URL = "https://device.memfault.com/api/v0"
PROJECT_KEY_HEADER = "Memfault-Project-Key"

MEMFAULT_MGMT_GROUP_ID = 128
MEMFAULT_MGMT_ID_DEVICE_INFO = 0
MEMFAULT_MGMT_ID_PROJECT_KEY = 1

SMP_TIMEOUT_S = 10.0
FIRST_UPLOAD_TIMEOUT_S = 60.0
REBOOT_TIMEOUT_S = 90.0

log = logging.getLogger("memfault_ota")


class OtaError(Exception):
    pass


class MEMFAULT_MGMT_ERR(IntEnum):
    OK = 0
    UNKNOWN = 1
    NO_PROJECT_KEY = 2


class MemfaultErrorV1(smperror.ErrorV1):
    _GROUP_ID = MEMFAULT_MGMT_GROUP_ID


class MemfaultErrorV2(smperror.ErrorV2[MEMFAULT_MGMT_ERR]):
    _GROUP_ID = MEMFAULT_MGMT_GROUP_ID


class DeviceInfoReadResponse(smpmsg.ReadResponse):
    _GROUP_ID = MEMFAULT_MGMT_GROUP_ID
    _COMMAND_ID = MEMFAULT_MGMT_ID_DEVICE_INFO

    device_serial: str
    hardware_version: str
    software_type: str
    current_version: str


class DeviceInfoRead(smpmsg.ReadRequest):
    _GROUP_ID = MEMFAULT_MGMT_GROUP_ID
    _COMMAND_ID = MEMFAULT_MGMT_ID_DEVICE_INFO
    _Response = DeviceInfoReadResponse
    _ErrorV1 = MemfaultErrorV1
    _ErrorV2 = MemfaultErrorV2


class ProjectKeyReadResponse(smpmsg.ReadResponse):
    _GROUP_ID = MEMFAULT_MGMT_GROUP_ID
    _COMMAND_ID = MEMFAULT_MGMT_ID_PROJECT_KEY

    project_key: str


class ProjectKeyRead(smpmsg.ReadRequest):
    _GROUP_ID = MEMFAULT_MGMT_GROUP_ID
    _COMMAND_ID = MEMFAULT_MGMT_ID_PROJECT_KEY
    _Response = ProjectKeyReadResponse
    _ErrorV1 = MemfaultErrorV1
    _ErrorV2 = MemfaultErrorV2


@dataclass
class Device:
    serial: str
    hardware_version: str
    software_type: str
    current_version: str
    project_key: str | None
    images: list

    def image_version(self, index: int) -> str | None:
        for state in self.images:
            if (state.image or 0) == index and state.slot == 0:
                return state.version
        return None


@dataclass
class Component:
    software_type: str
    image_index: int | None
    """MCUboot image of a non-primary component; None for the primary software type."""
    current_version: str


@dataclass
class Artifact:
    component: Component
    release: str
    version: str
    url: str
    sha256: str | None


@dataclass
class Image:
    file: str
    index: int
    data: bytes
    hash: bytes
    version: str


async def read_image_states(client: SMPClient) -> list:
    response = await client.request(ImageStatesRead())
    if error(response):
        raise OtaError(f"Reading image states failed: {response}")
    return response.images


def installed_images(states: list) -> set[tuple[int, bytes]]:
    return {
        (state.image or 0, bytes(state.hash))
        for state in states
        if state.slot == 0 and state.confirmed and state.hash is not None
    }


async def read_device(target: str) -> Device:
    async with SMPClient(SMPBLETransport(), target, timeout_s=SMP_TIMEOUT_S) as client:
        info = await client.request(DeviceInfoRead())
        if error(info):
            raise OtaError(
                f"Reading Memfault device info failed ({info}), "
                "check that the firmware enables CONFIG_MEMFAULT_MCUMGR_GRP"
            )
        key = await client.request(ProjectKeyRead())
        images = await read_image_states(client)

    return Device(
        serial=info.device_serial,
        hardware_version=info.hardware_version,
        software_type=info.software_type,
        current_version=info.current_version,
        project_key=None if error(key) else key.project_key,
        images=images,
    )


def device_components(device: Device, mapping: list[tuple[str, int]]) -> list[Component]:
    components = [Component(device.software_type, None, device.current_version)]
    for software_type, index in mapping:
        version = device.image_version(index)
        if version is None:
            raise OtaError(f"Device has no MCUboot image {index} for component {software_type}")
        components.append(Component(software_type, index, version))
    return components


def resolve_project_key(device: Device, override: str | None) -> str:
    key = override or device.project_key
    if not key:
        raise OtaError(
            "No Memfault project key: the device has none configured. "
            "Pass --project-key or set MEMFAULT_PROJECT_KEY."
        )
    return key


def query_latest_release(device: Device, component: Component, project_key: str) -> Artifact | None:
    response = requests.get(
        f"{MEMFAULT_API_URL}/releases/latest",
        params={
            "device_serial": device.serial,
            "hardware_version": device.hardware_version,
            "software_type": component.software_type,
            "current_version": component.current_version,
        },
        headers={PROJECT_KEY_HEADER: project_key},
        timeout=30,
    )
    if response.status_code == 204:
        return None
    if not response.ok:
        raise OtaError(f"Memfault API returned {response.status_code}: {response.text.strip()}")

    release = response.json()
    log.debug("Release for %s: %s", component.software_type, json.dumps(release, indent=2))
    artifacts = release.get("artifacts") or []
    if not artifacts:
        log.info("Release %s has no payload for %s", release["version"], component.software_type)
        return None
    payload = artifacts[0]
    return Artifact(
        component,
        release=release["version"],
        version=payload.get("version") or release["version"],
        url=payload["url"],
        sha256=payload.get("sha256"),
    )


def download(artifact: Artifact, dest: Path) -> Path:
    path = dest / f"{artifact.component.software_type}-{artifact.version}"
    digest = hashlib.sha256()
    with requests.get(artifact.url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with path.open("wb") as f:
            for chunk in response.iter_content(64 * 1024):
                f.write(chunk)
                digest.update(chunk)
    if artifact.sha256 and digest.hexdigest() != artifact.sha256.lower():
        raise OtaError(f"SHA-256 mismatch for the {artifact.component.software_type} artifact")
    return path


def load_image(path: Path, index: int) -> Image:
    info = ImageInfo.load_file(str(path))
    for hash_tlv in (IMAGE_TLV.SHA512, IMAGE_TLV.SHA384, IMAGE_TLV.SHA256):
        try:
            digest = info.get_tlv(hash_tlv).value
            break
        except TLVNotFound:
            continue
    else:
        raise OtaError(f"{path.name}: MCUboot image has no hash TLV")
    return Image(path.name, index, path.read_bytes(), digest, str(info.header.ver))


def load_images(path: Path, work_dir: Path, index: int) -> list[Image]:
    """Load a signed MCUboot binary for image `index`, or every image of a DFU zip."""
    if not zipfile.is_zipfile(path):
        return [load_image(path, index)]

    images = []
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        for entry in manifest["files"]:
            image_path = work_dir / Path(entry["file"]).name
            image_path.write_bytes(archive.read(entry["file"]))
            images.append(load_image(image_path, int(entry.get("image_index", 0))))
    return images


def print_progress(image: Image, offset: int, start: float) -> None:
    total = len(image.data)
    elapsed = max(time.monotonic() - start, 1e-3)
    filled = 30 * offset // total
    sys.stderr.write(
        f"\r  {image.file:<32} [{'#' * filled}{'.' * (30 - filled)}] "
        f"{100 * offset // total:3d}%  {offset / elapsed / 1024:6.1f} KiB/s"
    )
    sys.stderr.flush()


async def flash_images(target: str, images: list[Image]) -> list[Image]:
    async with SMPClient(SMPBLETransport(), target, timeout_s=SMP_TIMEOUT_S) as client:
        installed = installed_images(await read_image_states(client))
        pending = [image for image in images if (image.index, image.hash) not in installed]
        for image in images:
            if (image.index, image.hash) in installed:
                print(f"  {image.file}: image {image.index} already installed, skipping")

        for image in pending:
            start = time.monotonic()
            async for offset in client.upload(
                image.data,
                slot=image.index,
                first_timeout_s=FIRST_UPLOAD_TIMEOUT_S,
                subsequent_timeout_s=SMP_TIMEOUT_S,
            ):
                print_progress(image, offset, start)
            sys.stderr.write("\n")

        for image in pending:
            response = await client.request(ImageStatesWrite(hash=image.hash, confirm=True))
            if error(response):
                raise OtaError(f"Confirming {image.file} failed: {response}")

        if pending:
            print("Resetting device")
            try:
                await client.request(ResetWrite())
            except (TimeoutError, SMPTransportDisconnected):
                pass
    return pending


async def wait_for_reboot(target: str) -> list:
    deadline = time.monotonic() + REBOOT_TIMEOUT_S
    await asyncio.sleep(5)
    while True:
        try:
            async with SMPClient(SMPBLETransport(), target, timeout_s=SMP_TIMEOUT_S) as client:
                return await read_image_states(client)
        except (BleakError, SMPClientException, TimeoutError) as e:
            if time.monotonic() > deadline:
                raise OtaError(f"Device did not come back after reset: {e}") from e
            await asyncio.sleep(2)


def print_device(device: Device) -> None:
    print(f"Serial            {device.serial}")
    print(f"Hardware version  {device.hardware_version}")
    print(f"Software type     {device.software_type}")
    print(f"Firmware version  {device.current_version}")
    print(f"Project key       {'set' if device.project_key else 'not configured'}")


async def cmd_scan(args: argparse.Namespace) -> None:
    found = await BleakScanner.discover(timeout=args.timeout, return_adv=True)
    smp_uuid = str(SMP_SERVICE_UUID)
    devices = [(d, adv) for d, adv in found.values() if smp_uuid in adv.service_uuids]
    if not devices:
        print("No SMP devices found")
        return
    for device, adv in sorted(devices, key=lambda d: d[1].rssi, reverse=True):
        print(f"{device.address}  {adv.rssi:4d} dBm  {device.name or '?'}")


async def cmd_info(args: argparse.Namespace) -> None:
    device = await read_device(args.device)
    print_device(device)
    print("Images")
    for state in device.images:
        flags = [f for f in ("active", "confirmed", "pending", "permanent") if getattr(state, f)]
        print(
            f"  image {state.image or 0} slot {state.slot}  {state.version:<12} "
            f"{bytes(state.hash).hex()[:16] if state.hash else '-':<16}  {' '.join(flags)}"
        )


def find_updates(
    device: Device, components: list[Component], key: str
) -> list[tuple[Component, Artifact | None, str]]:
    """Query each component, dropping artifacts from a release other than the app's.

    Non-primary components are queried by their own version only, so Memfault may offer
    an artifact built for a different application than the one the device ends up running.
    """
    app_release = device.current_version
    updates = []
    for component in components:
        artifact = query_latest_release(device, component, key)
        if artifact is None:
            updates.append((component, None, "up to date"))
            continue
        status = f"-> {artifact.version} (release {artifact.release})"
        if component.image_index is None:
            app_release = artifact.release
        elif artifact.release != app_release:
            status += f" skipped, not for application release {app_release}"
            artifact = None
        updates.append((component, artifact, status))
    return updates


async def cmd_check(args: argparse.Namespace) -> None:
    device = await read_device(args.device)
    print_device(device)
    key = resolve_project_key(device, args.project_key)
    print("Components")
    for component, _, status in find_updates(device, device_components(device, args.component), key):
        print(f"  {component.software_type:<16} {component.current_version:<12} {status}")


async def cmd_update(args: argparse.Namespace) -> None:
    device = await read_device(args.device)
    print_device(device)

    with tempfile.TemporaryDirectory(prefix="memfault_ota_") as tmp:
        work_dir = Path(tmp)
        if args.file:
            images = load_images(args.file, work_dir, 0)
        else:
            key = resolve_project_key(device, args.project_key)
            components = device_components(device, args.component)
            images_by_index = {}
            print("Components")
            for component, artifact, status in find_updates(device, components, key):
                print(f"  {component.software_type:<16} {component.current_version:<12} {status}")
                if artifact is None:
                    continue
                path = download(artifact, work_dir)
                for image in load_images(path, work_dir, component.image_index or 0):
                    images_by_index[image.index] = image
            images = sorted(images_by_index.values(), key=lambda image: image.index)

        if not images:
            print("No updates to flash")
            return

        print("Images")
        for image in images:
            print(f"  {image.file}: image {image.index}, {image.version}, {len(image.data)} B")

        if not args.yes and input("Flash this update? [y/N] ").strip().lower() != "y":
            print("Aborted")
            return

        flashed = await flash_images(args.device, images)
        if not flashed:
            print("Nothing to flash, all images are already installed")
            return

    print("Waiting for the device to boot the new firmware")
    installed = installed_images(await wait_for_reboot(args.device))
    missing = [image.file for image in flashed if (image.index, image.hash) not in installed]
    if missing:
        raise OtaError(f"Device did not install: {', '.join(missing)}")
    print("Update complete")


def parse_component(value: str) -> tuple[str, int]:
    software_type, sep, index = value.partition("=")
    if not sep or not software_type or not index.isdigit():
        raise argparse.ArgumentTypeError(f"expected SOFTWARE_TYPE=IMAGE_INDEX, got '{value}'")
    return software_type, int(index)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="List nearby devices advertising the SMP service")
    scan.add_argument("-t", "--timeout", type=float, default=10.0, help="Scan time in seconds")

    def add_device_parser(name: str, help_text: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("-d", "--device", required=True, help="Device name or Bluetooth address")
        return p

    add_device_parser("info", help_text="Show device identity and MCUboot image slots")

    for name, help_text in (
        ("check", "Check Memfault for newer releases of each component"),
        ("update", "Download the latest Memfault releases and flash them"),
    ):
        p = add_device_parser(name, help_text)
        p.add_argument(
            "--component",
            type=parse_component,
            action="append",
            default=[],
            metavar="SOFTWARE_TYPE=IMAGE_INDEX",
            help="Non-primary Memfault component and the MCUboot image it updates (repeatable)",
        )
        p.add_argument(
            "--project-key",
            default=os.environ.get("MEMFAULT_PROJECT_KEY"),
            help="Memfault project key, overrides the key read from the device "
            "(default: $MEMFAULT_PROJECT_KEY)",
        )

    update = sub.choices["update"]
    update.add_argument(
        "--file", type=Path, help="Flash a local dfu_application.zip or signed .bin instead"
    )
    update.add_argument("-y", "--yes", action="store_true", help="Do not ask for confirmation")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if not args.verbose:
        logging.getLogger("smpclient.transport.ble").setLevel(logging.ERROR)
    command = {"scan": cmd_scan, "info": cmd_info, "check": cmd_check, "update": cmd_update}
    try:
        asyncio.run(command[args.command](args))
    except (OtaError, BleakError, SMPClientException, requests.RequestException, TimeoutError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
