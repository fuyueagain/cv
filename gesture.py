import math
import time
from typing import Any, Callable, Literal, Optional

import cv2
import mediapipe as mp

try:
    import serial
except ImportError:  # pragma: no cover - depends on runtime environment
    serial = None


TargetGesture = Literal["OK", "LEFT", "RIGHT"]


OK: TargetGesture = "OK"
LEFT: TargetGesture = "LEFT"
RIGHT: TargetGesture = "RIGHT"


CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MIRROR_INPUT = True
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.6
POLL_INTERVAL_SECONDS = 0.01
DETECTION_TIMEOUT_SECONDS = 10.0
OK_HOLD_SECONDS = 0.8
UART_DEVICE = "/dev/ttyS1"
UART_BAUDRATE = 115200
UART_TIMEOUT_SECONDS = 1
UART_APPEND_NEWLINE = True


_cap: Optional[Any] = None
_hands: Optional[Any] = None
_uart: Optional[Any] = None
_last_detection_elapsed_seconds: Optional[float] = None


def _finger_extended(lm, tip: int, pip_idx: int) -> bool:
    return lm[tip].y < lm[pip_idx].y


def _dist(lm, a: int, b: int) -> float:
    return math.hypot(lm[a].x - lm[b].x, lm[a].y - lm[b].y)


def _classify(lm) -> Optional[TargetGesture]:
    index_ext = _finger_extended(lm, 8, 6)
    mid_ext = _finger_extended(lm, 12, 10)
    ring_ext = _finger_extended(lm, 16, 14)
    pinky_ext = _finger_extended(lm, 20, 18)
    thumb_index_d = _dist(lm, 4, 8)

    if thumb_index_d < 0.07 and mid_ext and ring_ext and pinky_ext:
        return OK
    if index_ext and not mid_ext and not ring_ext and not pinky_ext:
        if lm[8].x < lm[0].x:
            return LEFT
        return RIGHT
    return None


def _create_hands():
    return mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )


def _initialize_resources() -> None:
    global _cap, _hands

    if _cap is None:
        _cap = cv2.VideoCapture(CAMERA_INDEX)
        _cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        _cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        if not _cap.isOpened():
            release_camera()
            raise RuntimeError(
                f"无法打开摄像头索引 {CAMERA_INDEX}，请检查 /dev/video*"
            )

    if _hands is None:
        _hands = _create_hands()


def _initialize_uart() -> None:
    global _uart

    if _uart is not None:
        return

    if serial is None:
        raise RuntimeError("未安装 pyserial，无法通过 UART 发送手势结果。")

    _uart = serial.Serial(
        port=UART_DEVICE,
        baudrate=UART_BAUDRATE,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=UART_TIMEOUT_SECONDS,
        write_timeout=UART_TIMEOUT_SECONDS,
    )


def _send_gesture_uart(gesture: TargetGesture) -> None:
    _initialize_uart()

    if _uart is None:
        raise RuntimeError("串口尚未初始化。")

    payload = gesture if not UART_APPEND_NEWLINE else f"{gesture}\n"
    _uart.write(payload.encode("ascii"))
    _uart.flush()


def _detect_target_gesture(frame) -> Optional[TargetGesture]:
    if frame is None:
        return None

    if _hands is None:
        raise RuntimeError("手势检测器尚未初始化。")

    working_frame = cv2.flip(frame, 1) if MIRROR_INPUT else frame
    rgb = cv2.cvtColor(working_frame, cv2.COLOR_BGR2RGB)
    result = _hands.process(rgb)

    if not result.multi_hand_landmarks:
        return None

    landmarks = result.multi_hand_landmarks[0].landmark
    return _classify(landmarks)


def detect_gesture(
    timeout: float = DETECTION_TIMEOUT_SECONDS,
    ok_hold_seconds: float = OK_HOLD_SECONDS,
    emit_uart: bool = True,
    should_stop: Optional[Callable[[], bool]] = None,
) -> TargetGesture:
    """
    阻塞式手势检测函数。

    行为：
    - 首次调用时自动初始化摄像头和 MediaPipe Hands。
    - 持续循环检测，同一目标手势连续两帧命中后返回。
    - emit_uart=True 时，识别成功后会通过 UART 发送手势信号。
    - 检测过程中对未检测到手部、其他手势、手势不明确均静默忽略。

    参数：
    - timeout: 最大等待秒数，默认 10 秒。超时后抛出 TimeoutError。
    - ok_hold_seconds: OK 手势最短稳定保持时长，默认 0.8 秒。
    - emit_uart: 是否在识别成功后通过 UART 发送结果，默认发送。
    - should_stop: 可选的停止回调，返回 True 时立刻中断检测。

    返回值：
    - "OK" | "LEFT" | "RIGHT"
    """

    global _last_detection_elapsed_seconds

    if timeout <= 0:
        raise ValueError("timeout must be greater than 0")
    if ok_hold_seconds < 0:
        raise ValueError("ok_hold_seconds must not be negative")

    _initialize_resources()
    start_time = time.monotonic()
    previous_gesture: Optional[TargetGesture] = None
    current_gesture_started_at: Optional[float] = None
    _last_detection_elapsed_seconds = None

    while True:
        if should_stop is not None and should_stop():
            raise InterruptedError("手势检测已被上层停止。")

        if time.monotonic() - start_time >= timeout:
            raise TimeoutError(
                f"在 {timeout:.2f} 秒内未检测到 OK / LEFT / RIGHT"
            )

        if _cap is None:
            raise RuntimeError("摄像头尚未初始化。")

        ret, frame = _cap.read()
        if not ret:
            time.sleep(POLL_INTERVAL_SECONDS)
            continue

        gesture = _detect_target_gesture(frame)
        if gesture is None:
            previous_gesture = None
            current_gesture_started_at = None
            time.sleep(POLL_INTERVAL_SECONDS)
            continue

        if gesture == previous_gesture:
            if gesture == OK and current_gesture_started_at is not None:
                held_seconds = time.monotonic() - current_gesture_started_at
                if held_seconds < ok_hold_seconds:
                    time.sleep(POLL_INTERVAL_SECONDS)
                    continue

            _last_detection_elapsed_seconds = time.monotonic() - start_time
            if emit_uart:
                _send_gesture_uart(gesture)
            return gesture

        previous_gesture = gesture
        current_gesture_started_at = time.monotonic()
        time.sleep(POLL_INTERVAL_SECONDS)


def release_camera() -> None:
    """
    释放摄像头、MediaPipe Hands 检测器和 UART 资源。
    """

    global _cap, _hands, _uart, _last_detection_elapsed_seconds

    if _cap is not None:
        _cap.release()
        _cap = None

    if _hands is not None:
        _hands.close()
        _hands = None

    if _uart is not None:
        _uart.close()
        _uart = None

    _last_detection_elapsed_seconds = None


def get_last_detection_elapsed() -> Optional[float]:
    """
    返回最近一次成功识别目标手势的耗时秒数。
    """

    return _last_detection_elapsed_seconds


if __name__ == "__main__":
    try:
        print(detect_gesture())
    except TimeoutError:
        pass
    finally:
        release_camera()
