import time
from typing import Any, Callable, Literal, Optional

import cv2

try:
    import serial
except ImportError:  # pragma: no cover - depends on runtime environment
    serial = None


FaceSignal = Literal["START"]


START: FaceSignal = "START"


CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FACE_WIDTH = 150
POLL_INTERVAL_SECONDS = 0.01
DETECTION_TIMEOUT_SECONDS = 10.0
DETECTION_SCALE_FACTOR = 1.1
DETECTION_MIN_NEIGHBORS = 5
DETECTION_MIN_SIZE = (50, 50)
UART_DEVICE = "/dev/ttyS1"
UART_BAUDRATE = 115200
UART_TIMEOUT_SECONDS = 1
UART_APPEND_NEWLINE = True


_cap: Optional[Any] = None
_face_cascade: Optional[Any] = None
_uart: Optional[Any] = None
_last_detection_elapsed_seconds: Optional[float] = None


def _create_face_cascade():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise RuntimeError(f"无法加载 Haar 模型: {cascade_path}")
    return face_cascade


def _initialize_resources() -> None:
    global _cap, _face_cascade

    if _cap is None:
        _cap = cv2.VideoCapture(CAMERA_INDEX)
        _cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        _cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        if not _cap.isOpened():
            release_camera()
            raise RuntimeError(
                f"无法打开摄像头索引 {CAMERA_INDEX}，请检查 /dev/video*"
            )

    if _face_cascade is None:
        _face_cascade = _create_face_cascade()


def _initialize_uart() -> None:
    global _uart

    if _uart is not None:
        return

    if serial is None:
        raise RuntimeError("未安装 pyserial，无法通过 UART 发送人脸信号。")

    _uart = serial.Serial(
        port=UART_DEVICE,
        baudrate=UART_BAUDRATE,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=UART_TIMEOUT_SECONDS,
        write_timeout=UART_TIMEOUT_SECONDS,
    )


def _send_face_uart(signal: FaceSignal) -> None:
    _initialize_uart()

    if _uart is None:
        raise RuntimeError("串口尚未初始化。")

    payload = signal if not UART_APPEND_NEWLINE else f"{signal}\n"
    _uart.write(payload.encode("ascii"))
    _uart.flush()


def _detect_face_trigger(frame, target_width: int = TARGET_FACE_WIDTH) -> bool:
    if frame is None:
        return False

    if _face_cascade is None:
        raise RuntimeError("人脸检测器尚未初始化。")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(
        gray,
        scaleFactor=DETECTION_SCALE_FACTOR,
        minNeighbors=DETECTION_MIN_NEIGHBORS,
        minSize=DETECTION_MIN_SIZE,
    )

    if len(faces) == 0:
        return False

    largest_face = max(faces, key=lambda rect: rect[2])
    _, _, face_width, _ = largest_face
    return face_width >= target_width


def detect_face(
    timeout: float = DETECTION_TIMEOUT_SECONDS,
    target_width: int = TARGET_FACE_WIDTH,
    emit_uart: bool = True,
    should_stop: Optional[Callable[[], bool]] = None,
) -> FaceSignal:
    """
    阻塞式人脸检测函数。

    行为：
    - 首次调用时自动初始化摄像头和 Haar 人脸检测器。
    - 持续循环检测，同一触发条件连续两帧命中后返回 START。
    - emit_uart=True 时，识别成功后会通过 UART 发送 START。
    - 未检测到人脸、距离未达阈值时均静默忽略。

    参数：
    - timeout: 最大等待秒数，默认 10 秒。超时后抛出 TimeoutError。
    - target_width: 人脸宽度触发阈值，默认 150 像素。
    - emit_uart: 是否在识别成功后通过 UART 发送 START，默认发送。
    - should_stop: 可选的停止回调，返回 True 时立刻中断检测。

    返回值：
    - "START"
    """

    global _last_detection_elapsed_seconds

    if timeout <= 0:
        raise ValueError("timeout must be greater than 0")
    if target_width <= 0:
        raise ValueError("target_width must be greater than 0")

    _initialize_resources()
    start_time = time.monotonic()
    previous_triggered = False
    _last_detection_elapsed_seconds = None

    while True:
        if should_stop is not None and should_stop():
            raise InterruptedError("人脸检测已被上层停止。")

        if time.monotonic() - start_time >= timeout:
            raise TimeoutError(
                f"在 {timeout:.2f} 秒内未检测到达到阈值的人脸"
            )

        if _cap is None:
            raise RuntimeError("摄像头尚未初始化。")

        ret, frame = _cap.read()
        if not ret:
            time.sleep(POLL_INTERVAL_SECONDS)
            continue

        triggered = _detect_face_trigger(frame, target_width=target_width)
        if not triggered:
            previous_triggered = False
            time.sleep(POLL_INTERVAL_SECONDS)
            continue

        if previous_triggered:
            _last_detection_elapsed_seconds = time.monotonic() - start_time
            if emit_uart:
                _send_face_uart(START)
            return START

        previous_triggered = True
        time.sleep(POLL_INTERVAL_SECONDS)


def release_camera() -> None:
    """
    释放摄像头、人脸检测器和 UART 资源。
    """

    global _cap, _face_cascade, _uart, _last_detection_elapsed_seconds

    if _cap is not None:
        _cap.release()
        _cap = None

    _face_cascade = None

    if _uart is not None:
        _uart.close()
        _uart = None

    _last_detection_elapsed_seconds = None


def get_last_detection_elapsed() -> Optional[float]:
    """
    返回最近一次成功识别人脸触发的耗时秒数。
    """

    return _last_detection_elapsed_seconds


if __name__ == "__main__":
    try:
        print(detect_face())
    except TimeoutError:
        pass
    finally:
        release_camera()
