#!/usr/bin/env python3

import argparse
import time

import serial


DEFAULT_PORT = "/dev/ttyS1"
DEFAULT_BAUDRATE = 115200
DEFAULT_TIMEOUT = 1.0
VALID_MESSAGES = {"OK", "LEFT", "RIGHT"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="RDK X5 UART receiver for gesture.py loopback verification"
    )
    parser.add_argument(
        "--port",
        default=DEFAULT_PORT,
        help=f"UART device path, default {DEFAULT_PORT}",
    )
    parser.add_argument(
        "--baudrate",
        type=int,
        default=DEFAULT_BAUDRATE,
        help=f"UART baudrate, default {DEFAULT_BAUDRATE}",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"serial read timeout seconds, default {DEFAULT_TIMEOUT}",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="exit after receiving the first valid gesture payload",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    ser = serial.Serial(
        port=args.port,
        baudrate=args.baudrate,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=args.timeout,
        write_timeout=args.timeout,
    )

    print(f"[UART-RECV] listening on {args.port} @ {args.baudrate}, 8N1")
    print("[UART-RECV] waiting for gesture.py to send OK / LEFT / RIGHT")

    try:
        ser.reset_input_buffer()
        ser.reset_output_buffer()

        while True:
            raw = ser.readline()
            if not raw:
                continue

            payload = raw.decode("ascii", errors="replace").strip()
            if not payload:
                continue

            now = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{now}] recv={payload}")

            if payload in VALID_MESSAGES:
                print(f"[UART-RECV] valid gesture payload received: {payload}")
                if args.once:
                    return 0
            else:
                print(f"[UART-RECV] ignored unexpected payload: {payload}")
    except KeyboardInterrupt:
        return 0
    finally:
        ser.close()
        print("[UART-RECV] uart closed")


if __name__ == "__main__":
    raise SystemExit(main())
