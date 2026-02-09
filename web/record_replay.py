#!/usr/bin/env python3
"""Record ~150 ticks from the live Daphne WebSocket and save to replay-data.json."""

import asyncio
import json
import os
import sys

import websockets

WS_URL = "ws://localhost:8000/ws/simulation/"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(SCRIPT_DIR, "frontend", "public", "replay-data.json")
TARGET_TICKS = 150


async def record():
    ticks = []
    print(f"Connecting to {WS_URL} ...")
    async with websockets.connect(WS_URL) as ws:
        print(f"Connected. Recording {TARGET_TICKS} ticks ...")
        while len(ticks) < TARGET_TICKS:
            raw = await ws.recv()
            msg = json.loads(raw)
            if msg.get("type") == "tick":
                ticks.append(msg)
                n = len(ticks)
                if n % 10 == 0 or n == 1:
                    print(f"  tick {n}/{TARGET_TICKS}")
    with open(OUTPUT, "w") as f:
        json.dump(ticks, f, separators=(",", ":"))
    size_mb = len(json.dumps(ticks, separators=(",", ":"))) / 1_000_000
    print(f"Done. Saved {len(ticks)} ticks to {OUTPUT} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    try:
        asyncio.run(record())
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(1)
