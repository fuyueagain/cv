import argparse
from pathlib import Path
from datetime import datetime

import cv2
import time


def parse_args():
    parser = argparse.ArgumentParser(description="摄像头标定与手动抓图工具")
    parser.add_argument("--camera-index", type=int, default=0, help="摄像头索引，默认 0")
    parser.add_argument("--save-dir", default="captures", help="抓拍图片保存目录")
    parser.add_argument("--gui", action="store_true", help="打开预览窗口，按 s 保存，按 q 退出")
    return parser.parse_args()


def save_snapshot(save_dir: Path, frame, annotated_frame):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    raw_path = save_dir / f"frame_{timestamp}.jpg"
    annotated_path = save_dir / f"frame_{timestamp}_annotated.jpg"

    ok_raw = cv2.imwrite(str(raw_path), frame)
    ok_annotated = cv2.imwrite(str(annotated_path), annotated_frame)
    if not ok_raw or not ok_annotated:
        raise RuntimeError("图片保存失败，请检查保存目录权限和磁盘空间")

    print(f"[SAVE] 原始图已保存: {raw_path}")
    print(f"[SAVE] 标注图已保存: {annotated_path}")


def main():
    args = parse_args()

    # 调用 OpenCV 自带 Haar 模型，脚本全程无 GUI，适合串口/SSH 标定。
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise RuntimeError(f"无法加载 Haar 模型: {cascade_path}")

    # 根据你跑通的示例，可能需要改成 /dev/video8 之类的，这里默认 0。
    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print("没抓到图，检查下摄像头节点")
        return 1

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("====== 终端测距标定开始 ======")
    print("请把脸凑到你设定的触发距离（也就是能看清手机屏幕的距离，大概30-40cm左右）...")
    print("保持姿势，看屏幕输出的像素宽度。")
    if args.gui:
        print("GUI 模式已开启：按 s 保存当前画面，按 q 退出。")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("没抓到图，检查下摄像头节点")
                time.sleep(1)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(50, 50),
            )

            annotated_frame = frame.copy()
            for (x, y, w, h) in faces:
                print(f"-> 捕捉到人脸！当前人脸像素宽度 w = {w} px (按Ctrl+C退出)")
                if args.gui:
                    cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(
                        annotated_frame,
                        f"w={w}px",
                        (x, max(y - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

            if args.gui:
                cv2.putText(
                    annotated_frame,
                    "Press S to save, Q to quit",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )
                cv2.imshow("Calibration Preview", annotated_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("s"):
                    save_snapshot(save_dir, frame, annotated_frame)
                elif key == ord("q"):
                    print("收到退出指令，结束标定。")
                    break

            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\n标定结束。")
    finally:
        cap.release()
        if args.gui:
            cv2.destroyAllWindows()
        print("摄像头已释放。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
