import cv2
import time
from face_detector import FaceProximityDetector

def main():
    # 1. 初始化外设 I/O
    print("[System] 初始化摄像头...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[Error] 无法打开摄像头，请检查 /dev/video 节点！")
        return

    # 2. 实例化算法模块 (这里设定默认参考w为150px)
    print("[System] 加载视觉检测模块...")
    detector = FaceProximityDetector(default_target_w=150)

    print("[System] 系统就绪，等待用户靠近...")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # 3. 调用模块处理当前帧。
        # 如果你想动态改变距离（比如根据不同场景），只需传入参数，例如：
        # is_triggered, face_info = detector.process_frame(frame, custom_target_w=180)
        is_triggered, face_info = detector.process_frame(frame)

        # 4. 业务逻辑分支
        if is_triggered:
            # ==== 拿到信号，开始执行你的后续流程 ====
            w = face_info[2]
            print(f"\n[SIGNAL] 触发！用户已到达最佳交互距离 (当前脸宽: {w}px)")
            
            # TODO: 在这里触发你的其他程序
            # 例如：通过 Socket 发送指令、启动语音识别、唤醒屏幕等
            # trigger_next_process()
            
            # 为了防止连续触发导致后续流程被狂刷，加上一个简单的状态锁或休眠防抖（Debounce）
            print("[System] 暂停视觉检测，处理后续流程中...")
            time.sleep(3) 
            print("[System] 流程处理完毕，重新开始检测环境...")
            
            # 可以在这里清除摄像头的旧缓存帧，保证恢复检测时是最新画面
            for _ in range(5): cap.read() 

if __name__ == "__main__":
    main()