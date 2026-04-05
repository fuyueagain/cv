#!/usr/bin/env python3

import serial

PORT = "/dev/ttyS1"
BAUDRATE = 115200

COMMAND_MAP = {
    "1": "LEFT\n",
    "2": "RIGHT\n",
    "3": "OK\n",
    "LEFT": "LEFT\n",
    "RIGHT": "RIGHT\n",
    "OK": "OK\n",
}


def main():
    ser = serial.Serial(
        port=PORT,
        baudrate=BAUDRATE,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=1,
        write_timeout=1,
    )

    print(f"[RDK] UART opened: {PORT} @ {BAUDRATE}, 8N1")
    print("[RDK] 输入 1/2/3 或 LEFT/RIGHT/OK，输入 q 退出")

    try:
        while True:
            cmd = input("Send> ").strip().upper()
            if cmd == "Q":
                break

            payload = COMMAND_MAP.get(cmd)
            if payload is None:
                print("无效输入，请输入 1/2/3/LEFT/RIGHT/OK/q")
                continue

            ser.write(payload.encode("ascii"))
            ser.flush()
            print(f"[RDK] Sent: {payload.strip()}")
    finally:
        ser.close()
        print("[RDK] UART closed")


if __name__ == "__main__":
    main()
v