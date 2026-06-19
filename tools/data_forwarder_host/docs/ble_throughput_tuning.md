<!--
Copyright (c) 2026 Nordic Semiconductor ASA
SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
-->

# BLE throughput: why it varies and how to make it faster

You opened this guide because the live BLE stream is dropping frames or running
slower than you expect. This document explains **what actually limits BLE
throughput** for the `data_forwarder_host` tool, and gives you an **ordered,
do-this-next procedure** to get more out of the link.

Read §1 to understand the tested baseline and why your numbers may differ, skim
§2–§3 for the data path and the levers, then follow §4 step by step. §5 lists
the firmware-side changes for when the host is exhausted.

> ## OS applicability — read this first
>
> The concepts (§1–§3, §5) are **OS-independent**. The hands-on host commands in
> §4 are **not**, because each operating system exposes the Bluetooth stack
> differently. Every host step is tagged with the OS it applies to:
>
> | Tag | Platform | Status in this tool |
> |-----|----------|---------------------|
> | **[Linux]**   | Linux / BlueZ        | ✅ Supported and validated (tested on Ubuntu). Most tunable from the host. |
> | **[macOS]**   | macOS / CoreBluetooth | ⚠️ **Should run but not validated** (untested). The OS auto-negotiates aggressively, so almost nothing is user-tunable. |
> | **[Windows]** | Windows / WinRT       | ⚠️ **Should run but not validated** (untested). Like macOS, no host-side throughput knobs apply. Notes are included so you know what *would* apply, but treat them as untested. |
>
> If a step is not tagged for your OS, it does not apply to you — skip to the
> next one. When the host platform offers no knob (macOS, and most of Windows),
> the only remaining wins are **firmware-side (§5)**.

## 1. The tested baseline (and why yours may be slower)

The default device acquisition this tool is validated against is **500 frames
per second, 10 channels per frame** — i.e. **500 BLE notifications/s**, each
carrying one CBOR `sensor-data` sample. That is the number the pipeline, the
decoder and the Host BT stage were sized and tested for.

**This is a reference point, not a guarantee.** The *actual* throughput you see
depends on a stack of factors that are mostly **outside the host application's
control**, because they live in the Bluetooth controllers and the radio
environment. In rough order of real-world impact:

1. **Connection interval.** The single biggest lever. A short interval
   (7.5–15 ms) gives many connection events per second; a long one (30–50 ms)
   starves a 500 Hz stream and forces many notifications into each event, which
   the controller then coalesces or drops. The **peripheral's preferred
   interval usually wins**, so this is often a firmware question (§5.1).

2. **Other Bluetooth activity on the same adapter.** This is the effect you may
   have observed. A single Bluetooth controller **time-division-multiplexes all
   of its links** — there is one radio. If the host already has another BLE
   connection, a Classic-BT transfer, A2DP audio, or even an HID mouse/keyboard
   active on the same adapter, those links **steal connection events and airtime
   from your stream**, so its effective rate drops. Disconnect or move competing
   peripherals to a different adapter when you need full bandwidth.

3. **Wi-Fi / 2.4 GHz coexistence.** Most laptops use a **combo Wi-Fi+BT chip
   that shares one antenna and radio**. When Wi-Fi is busy, the chip's
   coexistence arbitration **blanks BLE** during Wi-Fi slots — delaying or
   dropping connection events even with no other BT device in sight. Heavy Wi-Fi
   traffic, a nearby microwave oven, Zigbee, or other BLE/2.4 GHz devices all
   eat into your effective throughput. Using a **dedicated USB BLE dongle** (a
   separate radio) sidesteps most of this.

4. **Signal margin / distance.** Weak signal → more link-layer retransmissions →
   lower *effective* throughput even though the nominal rate is unchanged. Keep
   the device close and in line of sight while tuning.

5. **Negotiated radio parameters — PHY, Data Length, ATT MTU.** If the link
   comes up at 1M PHY, 27-byte packets, or the 23-byte minimum MTU, each ~90 B
   frame is fragmented across several packets/notifications and the per-packet
   overhead multiplies. These are controller-negotiated (§3).

6. **Host-side delivery cost.** The OS Bluetooth stack must hand all 500
   notifications/s up to the application. On Linux the delivery path matters
   (D-Bus vs. a dedicated socket — the app already picks the fast one, §3.1); a
   slow/old USB controller, USB power-saving (autosuspend), or a CPU-starved
   host can also coalesce or drop notifications.

The takeaway: **a clean radio environment and a dedicated adapter matter as much
as any setting.** Before blaming the firmware, eliminate competing BT links and
Wi-Fi contention, then work through §4.

## 2. How the data path works

The host is a BLE **central**. It scans, connects to the device-side
`samples/data_forwarder` peripheral, and subscribes to **notifications** on the
Nordic UART Service (NUS) **TX** characteristic. Each notification payload is a
COBS/CBOR frame, decoded by the same pipeline the UART source uses.

The path has more layers than it looks:

```
[Device radio] → [Host BT stack] → [bleak] → [decode] → [pipeline]
   firmware       controller +      Python     COBS/      IoWorker
                  kernel HCI +       async      CBOR
                  BlueZ/WinRT/CB
```

The easily-overlooked layer is the **Host BT stack** between the device
radiating a frame and `bleak` handing over bytes. When that stack is busy
(multiple connections, Wi-Fi coexistence, a competing transfer — see §1), it
coalesces or drops notifications. The bandwidth-details pipeline therefore shows
a dedicated **Host BT** stage:

* its **input rate** is the device's produced frame rate (sampling Hz) and its
  **output rate** is the rate the app actually receives — a host-stack shortfall
  shows up as a produced-vs-received rate shrink;
* its **drop counter** is the confirmed transport loss: the `sensor_data`
  **sequence-gap** losses detected by `core/sequence_tracker.py` (frames the
  device sent that never arrived). These are over-the-air *plus* host-stack
  losses, distinct from the device's own reported drops.

The bottleneck detector flags this stage when it is the limiter, so a losing
host link points you at the host Bluetooth side rather than the sensor board.

## 3. The workload, the levers, and who owns each one

Each sample is a CBOR `sensor-data` map, COBS-framed, with an optional CRC-16
trailer (see `cddl/data_forwarder.cddl`):

```
sensor-data = { "seq": uint32, "ts": uint32, "lbl": uint8, "val": [10 × float32] }
```

Rough on-wire size per frame:

| Part                                   | Bytes (approx.) |
|----------------------------------------|-----------------|
| 10 channels × float32 values           | 40              |
| CBOR keys + `seq`/`ts`/`lbl` + headers | ~25–35          |
| COBS framing + delimiter               | ~2              |
| CRC-16 trailer (if enabled)            | 2               |
| **Total per frame**                    | **~80–95**      |

At 500 Hz that is **~40–47 KB/s ≈ 320–380 kbps of application payload**. The
link is *not* limited by raw radio bandwidth so much as by **per-packet and
per-event overhead**. The biggest wins therefore come from sending **fewer,
larger packets per second** — not a "faster radio" alone.

The classic levers, and **who controls each** (which tells you where the change
must be made):

| Lever | Effect | Owner | Settable from the host app? |
|-------|--------|-------|-----------------------------|
| **Data Length Extension (27 → 251 B)** | One LL packet carries up to 251 B, so a ~90 B notification rides in **one** PDU instead of 4–5. | Controller, both ends; auto-negotiated. | **No** — not exposed by `bleak`/BlueZ. |
| **2M PHY** | Doubles symbol rate (1 → 2 Msym/s); higher ceiling, shorter air-time. | Controller, both ends; central requests it. | **No** — automatic if both controllers + the OS stack support it. |
| **ATT MTU (23 → 247 B)** | Largest single notification payload (MTU − 3). At 247 a ~90 B frame fits in one notification; at 23 it fragments. | Negotiated at connect. | **No** — the app can only **read** `client.mtu_size`; it is auto-maxed on macOS/Windows. |
| **Connection interval** | Shorter interval → far more connection events/s → far more notifications/s. **The single biggest non-firmware lever.** | Central proposes; **peripheral's preferred params usually win**. | **No** — no portable `bleak` API; OS/firmware config only. |
| **TX power (dBm)** | More signal margin → fewer retransmits at range. Helps *reliability*, not the ceiling. | Controller/adapter, both ends. | **No** — not exposed by `bleak`. |

**Key point:** none of these are settable from portable Python — they are
negotiated in the controllers. The app's job is to (a) make sure the OS stack
actually *uses* the big ones and (b) consume 500 notifications/s efficiently.
Everything the app *can* do is already done (§3.1).

### 3.1 What the host application already does for you

* **[Linux] Prefers BlueZ `AcquireNotify`.** `bleak` can deliver notifications
  either via D-Bus `PropertiesChanged` signals (the default) or via a dedicated
  SEQPACKET socket (`AcquireNotify`). At 500 notifications/s the D-Bus path is a
  real bottleneck and a source of coalesced/dropped frames. The source passes
  `bluez={"use_start_notify": False}` to select `AcquireNotify`, falling back
  automatically when unsupported. The argument is inert on macOS/Windows, so it
  is cross-platform safe.
* **[All] Logs the negotiated ATT MTU** (`client.mtu_size`) after connect, so
  you can confirm the link came up with a large MTU rather than silently at 23.
* **[All] Surfaces the Host BT stage** in the pipeline (§2) so transport loss is
  attributed to the host side, not the device.

Because the app's levers are already pulled, the procedure below is about
**radio hygiene, OS configuration, and finally firmware.**

## 4. Speed-up procedure (do this in order)

Work top to bottom. Re-measure after each step (via the **Host BT** pipeline
stage, and on Linux with `btmon`).

### 4.1 [All] Remove competing radio load first — free and high-impact

Before touching any setting, clean up the environment (this is the most common
real-world fix; see §1, points 2–3):

* **Disconnect other Bluetooth devices** from the same adapter — other BLE
  peripherals, BT audio (A2DP), BT mice/keyboards. They share one radio with
  your stream.
* **Reduce 2.4 GHz contention** — pause large Wi-Fi transfers, move away from
  busy access points / microwaves, or prefer a **5 GHz** Wi-Fi network so the
  combo chip's coexistence logic stops blanking BLE.
* **Use a dedicated USB BLE dongle** instead of the built-in combo Wi-Fi+BT
  chip when you can — a separate radio avoids Wi-Fi coexistence entirely.
* **Move the device closer / into line of sight** to cut retransmissions.

Reconnect and re-measure. If the Host BT stage now shows no drops, you are done.

### 4.2 [Linux/macOS] Measure what the link actually negotiated

Knowing which lever is limiting you avoids guesswork.

**[Linux]** capture the negotiation with `btmon`:

```bash
sudo btmon > btmon.log
# now connect with the tool, wait a few seconds, then Ctrl-C
grep -iE "Connection interval|Max TX octets|Max RX octets|PHY:" btmon.log
```

Interpret it:

| What you see | Meaning |
|--------------|---------|
| `TX PHY: LE 2M` / `RX PHY: LE 2M` | ✅ 2M PHY active. |
| `Max TX octets: 251` / `Max RX octets: 251` | ✅ DLE active. |
| `Connection interval: 7.50–15.00 msec` | ✅ Short interval — good for 500 Hz. |
| `Connection interval: 30–50 msec` | ❌ **Almost always the bottleneck** at 500 Hz. |

**[macOS]** there is no `btmon`. Use the **PacketLogger** tool from Apple's
*Additional Tools for Xcode* to capture an HCI trace, or rely on the tool's own
**Host BT** pipeline stage and the logged ATT MTU. macOS negotiates 2M PHY, DLE
and a large MTU automatically, so the only realistic limiter you can see is the
connection interval — which on macOS you cannot change from the host (go to
§5.1).

**[Windows]** *(not validated)* the equivalent capture is a **Bluetooth
ETW/Btvs** trace via the Windows Driver Kit, but this path is untested for
this tool.

If PHY shows `LE 1M` or DLE octets are 27, your **adapter or OS stack is the
limiter** — see §4.5. If the interval is 30–50 ms, continue with §4.3.

### 4.3 [Linux] Shorten the connection interval from the host

A 45 ms interval gives only ~22 connection events/s — fragile at 500 Hz. Lower
BlueZ's proposed defaults **before connecting** (root; values reset on reboot):

```bash
# units of 1.25 ms → 6 = 7.5 ms, 12 = 15 ms. Replace hci0 with your adapter.
echo 6  | sudo tee /sys/kernel/debug/bluetooth/hci0/conn_min_interval
echo 12 | sudo tee /sys/kernel/debug/bluetooth/hci0/conn_max_interval
cat /sys/kernel/debug/bluetooth/hci0/conn_min_interval   # expect 6
cat /sys/kernel/debug/bluetooth/hci0/conn_max_interval   # expect 12
```

Reconnect and re-check `btmon`. **Two outcomes:**

* Interval drops to ~7.5–15 ms → done, throughput should rise sharply.
* Interval springs back to ~30–50 ms → the **device's preferred parameters are
  overriding you** (the central only *proposes*). Go to §5.1 (firmware).

> **If `tee` fails with `Operation not permitted` even under `sudo`:** the kernel
> is in **lockdown** mode (triggered by Secure Boot), which forbids debugfs
> writes:
> ```bash
> cat /sys/kernel/security/lockdown   # "[integrity]"/"[confidentiality]" = locked
> mokutil --sb-state                  # "SecureBoot enabled" = the cause
> ```
> Disable Secure Boot in UEFI to lift lockdown — but if the device is requesting
> 30–50 ms anyway (very common), the host setting is overridden regardless, so
> prefer the firmware fix in §5.1.

**[macOS] / [Windows]** there is no supported host-side knob for the connection
interval; the firmware must request a short one (§5.1).

### 4.4 [Linux/macOS] Confirm the device is not requesting a long interval

**[Linux]** look for a **Connection Parameter Update Request** from the
peripheral:

```bash
grep -iA4 "Connection Parameter" btmon.log
```

If you see `Min: 24 / Max: 40` (= 30–50 ms), the firmware is asking for a long
interval and the OS honours it. No host setting beats this — the fix is §5.1.
(On macOS the same is true; you just confirm it from a PacketLogger trace.)

### 4.5 [Linux/macOS] Verify the adapter and stack can do 2M PHY + DLE

If §4.2 showed `LE 1M` or 27-byte octets, the adapter or stack is the cap:

```bash
bluetoothctl --version     # [Linux] prefer a recent BlueZ (5.50+)
hciconfig                  # [Linux] confirm the adapter is present and UP
```

Use a modern USB controller that supports **2M PHY + DLE + a small connection
interval**; an old dongle caps throughput regardless of every other setting. On
macOS the built-in controller already supports these — there is nothing to
configure.

### 4.6 [All] Re-measure with the pipeline

Run a session and watch the **Host BT** stage:

* **No drops, produced ≈ received** → the host side is optimal; for more, go to
  §5.
* **Drops / rate shrink persist** → the bottleneck has moved to the device or
  the air; the highest-impact next step is sample batching (§5.2).

## 5. Going further: firmware (+ matching host) changes

When the host is exhausted — or your OS offers no knob (macOS, Windows) — these
raise throughput further but require a **firmware change** (sometimes plus a
matching host decoder change). Ordered by typical payoff.

### 5.1 Shorten the connection interval in firmware (highest-impact, do first)

Make the peripheral request a 7.5–15 ms interval so the link no longer falls
back to 30–50 ms. In the device `prj.conf`:

```ini
CONFIG_BT_PERIPHERAL_PREF_MIN_INT=6     # 6 × 1.25 ms = 7.5 ms
CONFIG_BT_PERIPHERAL_PREF_MAX_INT=12    # 12 × 1.25 ms = 15 ms
CONFIG_BT_PERIPHERAL_PREF_LATENCY=0
CONFIG_BT_PERIPHERAL_PREF_TIMEOUT=400   # 4 s supervision timeout
```

or call it explicitly after connect:

```c
struct bt_le_conn_param param = BT_LE_CONN_PARAM_INIT(6, 12, 0, 400);
bt_conn_le_param_update(conn, &param);
```

Reflash, reconnect, and confirm the interval dropped to ~7.5–15 ms.

### 5.2 Batch multiple samples per notification (biggest end-to-end win)

Instead of one CBOR frame per sample (500 notifications/s), pack N samples into
one notification (e.g. 5 → 100 notifications/s). This amortises every
per-packet/per-notification overhead on both ends — usually the single largest
end-to-end improvement. Requires a device-side framing change and a host-side
decoder that iterates samples within a frame.

### 5.3 Switch the high-rate stream to an L2CAP CoC

A connection-oriented channel has lower per-packet overhead and built-in flow
control versus GATT notifications. Requires a new device-side transport and a
host-side L2CAP client; note `bleak` does not currently expose L2CAP CoC, so the
host would need a BlueZ-specific socket path.

### 5.4 Explicitly drive 2M PHY + DLE from the device

Call `bt_conn_le_phy_update()` and `bt_conn_le_data_len_update()` right after
connect so the link does not rely on automatic negotiation timing. (The
`samples/data_forwarder` sample should enable the buffers —
`CONFIG_BT_L2CAP_TX_MTU=247`, `CONFIG_BT_BUF_ACL_RX_SIZE=251` — so this just
forces the update early.)

### 5.5 Compress the payload

Use int16 fixed-point instead of float32, or delta-encode samples, to roughly
halve bytes/sample. A device encoder + host decoder change.

### 5.6 Tune device TX power for range

Match TX power to the deployment range to cut retransmissions and improve
*effective* throughput at the edge of range.

## 6. Quick reference

| Symptom (from the pipeline / a trace) | Fix | Where |
|---------------------------------------|-----|-------|
| Throughput drops when another BT/Wi-Fi transfer is active | Free the radio: disconnect peers, cut 2.4 GHz load, use a dedicated dongle | §4.1 |
| Notifications coalesced/dropped on Linux | `AcquireNotify` | Host app (already done) |
| `Connection interval: 30–50 msec`, Linux host | Lower `conn_min/max_interval` | §4.3 |
| Interval bounces back; device requests 30–50 ms (any OS) | Lower peripheral preferred interval | §5.1 |
| `LE 1M` PHY or 27-byte octets | Modern adapter + recent stack | §4.5 |
| Still notification-bound after interval fix | Batch N samples/notification | §5.2 |
| Want lower per-packet overhead + flow control | L2CAP CoC | §5.3 |
| Bytes/sample too high | Compress payload (int16/delta) | §5.5 |
| Drops only at range | Raise device TX power | §5.6 |

**Bottom line:** the tool is validated at **500 frames/s × 10 channels**, but
real throughput is dominated by things the host app cannot set — the
**connection interval**, **competing BT/Wi-Fi activity on the same radio**, and
**signal quality**. First clean up the radio environment (§4.1); on **Linux**
try the connection-interval change (§4.3); on **macOS/Windows**, or when the
device overrides the host, shorten the interval in **firmware (§5.1)**. After
that, **sample batching (§5.2)** is the biggest remaining win.