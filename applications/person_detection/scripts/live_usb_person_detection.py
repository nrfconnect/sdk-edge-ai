#!/usr/bin/env python3
"""
Copyright (c) 2026 Nordic Semiconductor
SPDX-License-Identifier: LicenseRef-Nordic-5-Clause

Live USB CDC ACM viewer for person_detection firmware.

Receives RGB565 frames and detection boxes streamed over USB HS CDC ACM,
converts to RGB888, overlays bounding boxes, and displays in an OpenCV window.

Binary protocol (little-endian):

  Frame message (type 0x01):
    [4B magic 0xA55A3412] [1B version] [1B type=0x01] [2B width] [2B height]
    [4B payload_len] [4B frame_id] [4B crc32(payload)] [payload: RGB565 BE]

  Detection message (type 0x02):
    [4B magic] [1B version] [1B type=0x02] [4B frame_id] [2B model_w] [2B model_h]
    [2B pad_left] [2B pad_top] [1B box_count]
    [per box: 4x float32 x1,y1,x2,y2 + float32 score + uint8 head]
    [4B crc32(version..last box byte)]

Dependencies: pyserial, opencv-python, numpy
"""
from __future__ import annotations

import argparse
import struct
import sys
import threading
import time
import zlib
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

try:
    import serial
except ImportError as exc:
    raise SystemExit("pyserial required: pip install pyserial") from exc

MAGIC = b"\xa5\x5a\x34\x12"
MAGIC_INT = 0x12345AA5
VERSION = 1
TYPE_FRAME = 0x01
TYPE_DETECT = 0x02

FRAME_HDR_SIZE = 22  # magic(4)+ver(1)+type(1)+w(2)+h(2)+plen(4)+fid(4)+crc(4)
DETECT_HDR_SIZE = 19  # magic(4)+ver(1)+type(1)+fid(4)+mw(2)+mh(2)+pl(2)+pt(2)+cnt(1)
BOX_SIZE = 21  # 4*f32 + f32 + u8

HEAD_NAMES = {0: "s32", 1: "s16", 2: "s8"}


@dataclass
class DetBox:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    head: int


@dataclass
class FrameData:
    frame_id: int
    width: int
    height: int
    rgb: np.ndarray


@dataclass
class DetectionData:
    frame_id: int
    model_w: int
    model_h: int
    pad_left: int
    pad_top: int
    boxes: list[DetBox] = field(default_factory=list)


def rgb565_be_to_rgb(buf: bytes, width: int, height: int) -> np.ndarray:
    """Convert RGB565 big-endian buffer to numpy RGB888 array."""
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 2)
    hi = arr[:, :, 0].astype(np.uint16)
    lo = arr[:, :, 1].astype(np.uint16)
    pix = (hi << 8) | lo

    r5 = ((pix >> 11) & 0x1F).astype(np.uint8)
    g6 = ((pix >> 5) & 0x3F).astype(np.uint8)
    b5 = (pix & 0x1F).astype(np.uint8)

    rgb = np.empty((height, width, 3), dtype=np.uint8)
    rgb[:, :, 0] = (r5 << 3) | (r5 >> 2)
    rgb[:, :, 1] = (g6 << 2) | (g6 >> 4)
    rgb[:, :, 2] = (b5 << 3) | (b5 >> 2)
    return rgb


class StreamParser:
    """Thread-safe parser for the binary stream protocol."""

    def __init__(self):
        self._buf = bytearray()
        self._lock = threading.Lock()
        self.frames: deque[FrameData] = deque(maxlen=4)
        self.detections: deque[DetectionData] = deque(maxlen=8)
        self.stats = {"frames_ok": 0, "frames_crc_err": 0, "dets_ok": 0, "dets_crc_err": 0,
                      "resync": 0}

    def feed(self, data: bytes) -> None:
        with self._lock:
            self._buf.extend(data)
            self._parse()

    def _parse(self) -> None:
        while True:
            idx = self._buf.find(MAGIC)
            if idx < 0:
                if len(self._buf) > len(MAGIC):
                    self._buf = self._buf[-(len(MAGIC) - 1):]
                return
            if idx > 0:
                self.stats["resync"] += 1
                del self._buf[:idx]

            if len(self._buf) < 6:
                return

            ver = self._buf[4]
            msg_type = self._buf[5]

            if ver != VERSION:
                del self._buf[:4]
                continue

            if msg_type == TYPE_FRAME:
                if not self._try_parse_frame():
                    return
            elif msg_type == TYPE_DETECT:
                if not self._try_parse_detection():
                    return
            else:
                del self._buf[:4]

    def _try_parse_frame(self) -> bool:
        if len(self._buf) < FRAME_HDR_SIZE:
            return False

        _magic, ver, mtype, w, h, plen, fid, crc_expect = struct.unpack_from(
            "<IBBHHIII", self._buf, 0
        )

        total = FRAME_HDR_SIZE + plen
        if len(self._buf) < total:
            return False

        payload = bytes(self._buf[FRAME_HDR_SIZE:total])
        crc_got = zlib.crc32(payload) & 0xFFFFFFFF

        if crc_got != crc_expect:
            self.stats["frames_crc_err"] += 1
            del self._buf[:4]
            return True

        try:
            rgb = rgb565_be_to_rgb(payload, w, h)
        except (ValueError, Exception):
            del self._buf[:total]
            return True

        self.frames.append(FrameData(frame_id=fid, width=w, height=h, rgb=rgb))
        self.stats["frames_ok"] += 1
        del self._buf[:total]
        return True

    def _try_parse_detection(self) -> bool:
        if len(self._buf) < DETECT_HDR_SIZE:
            return False

        _magic, ver, mtype, fid, mw, mh, pl, pt, cnt = struct.unpack_from(
            "<IBBIHHHHB", self._buf, 0
        )

        total = DETECT_HDR_SIZE + cnt * BOX_SIZE + 4
        if len(self._buf) < total:
            return False

        # CRC covers bytes 4..total-4 (version through last box byte)
        crc_payload = bytes(self._buf[4:total - 4])
        crc_expect = struct.unpack_from("<I", self._buf, total - 4)[0]
        crc_got = zlib.crc32(crc_payload) & 0xFFFFFFFF

        if crc_got != crc_expect:
            self.stats["dets_crc_err"] += 1
            del self._buf[:4]
            return True

        boxes = []
        off = DETECT_HDR_SIZE
        for _ in range(cnt):
            x1, y1, x2, y2, score = struct.unpack_from("<fffff", self._buf, off)
            head = self._buf[off + 20]
            boxes.append(DetBox(x1=x1, y1=y1, x2=x2, y2=y2, score=score, head=head))
            off += BOX_SIZE

        self.detections.append(DetectionData(
            frame_id=fid, model_w=mw, model_h=mh, pad_left=pl, pad_top=pt, boxes=boxes
        ))
        self.stats["dets_ok"] += 1
        del self._buf[:total]
        return True


def reader_thread(ser: serial.Serial, parser: StreamParser, stop_event: threading.Event) -> None:
    """Continuously read from serial port and feed into parser."""
    while not stop_event.is_set():
        try:
            data = ser.read(4096)
            if data:
                parser.feed(data)
        except serial.SerialException:
            break
        except Exception as exc:
            print(f"Reader error: {exc}", file=sys.stderr)
            break


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def main() -> int:
    ap = argparse.ArgumentParser(description="Live USB viewer for person_detection")
    ap.add_argument("--port", default="/dev/ttyACM2", help="CDC ACM serial port")
    ap.add_argument("--baud", type=int, default=115200, help="Baud rate (CDC ACM ignores this)")
    ap.add_argument("--scale", type=int, default=4, help="Display upscale factor")
    args = ap.parse_args()

    try:
        ser = serial.Serial(args.port, args.baud, timeout=0.1)
    except serial.SerialException as exc:
        print(f"Cannot open {args.port}: {exc}", file=sys.stderr)
        return 1

    parser = StreamParser()
    stop_event = threading.Event()
    reader = threading.Thread(target=reader_thread, args=(ser, parser, stop_event), daemon=True)
    reader.start()

    print(f"Listening on {args.port} (scale={args.scale}x). Press 'q' to quit.")

    last_frame: FrameData | None = None
    last_det: DetectionData | None = None
    fps_times: deque[float] = deque(maxlen=30)
    window_name = "Person Detection Live"

    try:
        while True:
            # Get latest frame
            while parser.frames:
                last_frame = parser.frames.popleft()
                fps_times.append(time.monotonic())

            # Get latest detection
            while parser.detections:
                last_det = parser.detections.popleft()

            if last_frame is None:
                # No frame yet — show waiting message
                blank = np.zeros((128 * args.scale, 128 * args.scale, 3), dtype=np.uint8)
                cv2.putText(blank, "Waiting for frames...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                cv2.putText(blank, f"port: {args.port}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
                s = parser.stats
                cv2.putText(blank, f"rx: {s['frames_ok']}f {s['dets_ok']}d "
                            f"err: {s['frames_crc_err']}f {s['dets_crc_err']}d "
                            f"resync: {s['resync']}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
                cv2.imshow(window_name, blank)
                key = cv2.waitKey(50) & 0xFF
                if key == ord("q"):
                    break
                continue

            # Prepare display image (BGR for OpenCV)
            display = cv2.cvtColor(last_frame.rgb, cv2.COLOR_RGB2BGR)

            # Overlay detection boxes
            if last_det is not None and last_det.frame_id == last_frame.frame_id:
                for box in last_det.boxes:
                    x1 = int(clamp(box.x1 - last_det.pad_left, 0, last_frame.width))
                    y1 = int(clamp(box.y1 - last_det.pad_top, 0, last_frame.height))
                    x2 = int(clamp(box.x2 - last_det.pad_left, 0, last_frame.width))
                    y2 = int(clamp(box.y2 - last_det.pad_top, 0, last_frame.height))
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    label = f"{HEAD_NAMES.get(box.head, '?')} {box.score:.2f}"
                    cv2.putText(display, label, (x1, max(y1 - 3, 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

            # Upscale
            if args.scale > 1:
                display = cv2.resize(display, None, fx=args.scale, fy=args.scale,
                                     interpolation=cv2.INTER_NEAREST)

            # FPS overlay
            if len(fps_times) >= 2:
                dt = fps_times[-1] - fps_times[0]
                fps = (len(fps_times) - 1) / dt if dt > 0 else 0
                cv2.putText(display, f"{fps:.1f} FPS", (5, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

            # Stats overlay
            s = parser.stats
            cv2.putText(display, f"fid:{last_frame.frame_id} f:{s['frames_ok']} d:{s['dets_ok']}",
                        (5, display.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        ser.close()
        cv2.destroyAllWindows()
        reader.join(timeout=2)

    s = parser.stats
    print(f"\nSession stats: frames={s['frames_ok']} dets={s['dets_ok']} "
          f"crc_err_f={s['frames_crc_err']} crc_err_d={s['dets_crc_err']} "
          f"resync={s['resync']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
