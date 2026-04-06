from __future__ import annotations

import threading
import time
from typing import Callable, Optional

try:
    import serial
except ImportError:  # pragma: no cover - depends on runtime environment
    serial = None

import face_detector
import gesture


UART_DEVICE = "/dev/ttyS1"
UART_BAUDRATE = 115200
UART_TIMEOUT_SECONDS = 0.2

FACE_WAIT_TIMEOUT_SECONDS = 30.0
FACE_POLL_TIMEOUT_SECONDS = 0.3
FACE_MISS_THRESHOLD = 2

GESTURE_SCAN_TIMEOUT_SECONDS = 1.2
GESTURE_RELEASE_TIMEOUT_SECONDS = 0.35
OK_HOLD_SECONDS = 0.8


class RdkVisionController:
    def __init__(self) -> None:
        self._serial: Optional[serial.Serial] = None
        self._serial_lock = threading.Lock()
        self._worker_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._current_mode = "IDLE"

    def open(self) -> None:
        if serial is None:
            raise RuntimeError("未安装 pyserial，无法启动 UART 控制器。")

        self._serial = serial.Serial(
            port=UART_DEVICE,
            baudrate=UART_BAUDRATE,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=UART_TIMEOUT_SECONDS,
            write_timeout=UART_TIMEOUT_SECONDS,
        )
        self._serial.reset_input_buffer()
        self._serial.reset_output_buffer()

    def close(self) -> None:
        self._stop_worker()
        self._release_all_resources()

        if self._serial is not None:
            self._serial.close()
            self._serial = None

    def serve_forever(self) -> None:
        if self._serial is None:
            self.open()

        self._send_line("READY")

        while True:
            line = self._read_line()
            if line is None:
                continue

            command = line.strip().upper()
            if not command:
                continue

            try:
                self._handle_command(command)
            except Exception as exc:
                self._send_line(f"ERR:{self._format_error(exc)}")

    def _handle_command(self, command: str) -> None:
        if command == "PING":
            self._send_line("PONG")
            return

        if command in {"STOP", "RESET"}:
            self._stop_worker()
            self._send_line("STOPPED")
            return

        if command == "FACE_WAIT":
            self._start_worker("FACE_WAIT", self._run_face_wait)
            return

        if command == "INTERACT":
            self._start_worker("INTERACT", self._run_interact)
            return

        if command == "PRESENCE":
            self._start_worker("PRESENCE", self._run_presence_monitor)
            return

        self._send_line(f"ERR:UNKNOWN_CMD:{command}")

    def _start_worker(self, mode: str, target: Callable[[], None]) -> None:
        self._stop_worker()

        worker = threading.Thread(
            target=self._worker_entry,
            args=(mode, target),
            daemon=True,
            name=f"rdk-{mode.lower()}",
        )

        with self._worker_lock:
            self._stop_event = threading.Event()
            self._worker = worker
            self._current_mode = mode

        worker.start()
        self._send_line(f"ACK:{mode}")

    def _worker_entry(self, mode: str, target: Callable[[], None]) -> None:
        try:
            target()
        except InterruptedError:
            pass
        except Exception as exc:
            self._send_line(f"ERR:{self._format_error(exc)}")
        finally:
            self._release_all_resources()

            with self._worker_lock:
                current = self._worker
                if current is threading.current_thread():
                    self._worker = None
                    self._current_mode = "IDLE"

    def _stop_worker(self) -> None:
        with self._worker_lock:
            worker = self._worker
            stop_event = self._stop_event

        if worker is None:
            self._release_all_resources()
            return

        stop_event.set()
        worker.join(timeout=2.0)
        self._release_all_resources()

        if worker.is_alive():
            raise RuntimeError("视觉任务停止超时，拒绝并发切换模式。")

        with self._worker_lock:
            if self._worker is worker:
                self._worker = None
                self._current_mode = "IDLE"

    def _run_face_wait(self) -> None:
        while not self._should_stop():
            try:
                face_detector.detect_face(
                    timeout=FACE_WAIT_TIMEOUT_SECONDS,
                    emit_uart=False,
                    should_stop=self._should_stop,
                )
                self._send_line("START")
                return
            except TimeoutError:
                face_detector.release_camera()

    def _run_interact(self) -> None:
        miss_count = 0

        while not self._should_stop():
            if self._face_present():
                miss_count = 0
            else:
                miss_count += 1
                if miss_count >= FACE_MISS_THRESHOLD:
                    self._send_line("LEAVE")
                    return

            detected = self._try_detect_gesture()
            if detected is None:
                continue

            self._send_line(detected)
            if detected == "OK":
                return

            self._wait_for_gesture_release()

    def _run_presence_monitor(self) -> None:
        miss_count = 0

        while not self._should_stop():
            if self._face_present():
                miss_count = 0
                time.sleep(0.05)
                continue

            miss_count += 1
            if miss_count >= FACE_MISS_THRESHOLD:
                self._send_line("LEAVE")
                return

    def _face_present(self) -> bool:
        try:
            face_detector.detect_face(
                timeout=FACE_POLL_TIMEOUT_SECONDS,
                emit_uart=False,
                should_stop=self._should_stop,
            )
            return True
        except TimeoutError:
            return False
        finally:
            face_detector.release_camera()

    def _try_detect_gesture(self) -> Optional[gesture.TargetGesture]:
        try:
            return gesture.detect_gesture(
                timeout=GESTURE_SCAN_TIMEOUT_SECONDS,
                ok_hold_seconds=OK_HOLD_SECONDS,
                emit_uart=False,
                should_stop=self._should_stop,
            )
        except TimeoutError:
            return None
        finally:
            gesture.release_camera()

    def _wait_for_gesture_release(self) -> None:
        while not self._should_stop():
            try:
                gesture.detect_gesture(
                    timeout=GESTURE_RELEASE_TIMEOUT_SECONDS,
                    ok_hold_seconds=OK_HOLD_SECONDS,
                    emit_uart=False,
                    should_stop=self._should_stop,
                )
            except TimeoutError:
                return
            finally:
                gesture.release_camera()

    def _should_stop(self) -> bool:
        return self._stop_event.is_set()

    def _release_all_resources(self) -> None:
        face_detector.release_camera()
        gesture.release_camera()

    def _read_line(self) -> Optional[str]:
        if self._serial is None:
            raise RuntimeError("UART 尚未初始化。")

        raw = self._serial.readline()
        if not raw:
            return None

        return raw.decode("ascii", errors="ignore")

    def _send_line(self, payload: str) -> None:
        if self._serial is None:
            raise RuntimeError("UART 尚未初始化。")

        data = f"{payload}\n".encode("ascii", errors="ignore")
        with self._serial_lock:
            self._serial.write(data)
            self._serial.flush()

    @staticmethod
    def _format_error(exc: Exception) -> str:
        name = exc.__class__.__name__.upper()
        message = str(exc).strip().replace(" ", "_")
        return f"{name}:{message}" if message else name


def main() -> int:
    controller = RdkVisionController()

    try:
        controller.open()
        controller.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        controller.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
