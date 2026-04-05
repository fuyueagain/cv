import cv2
import time

# 调用X5自带OpenCV的基础Haar模型
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

# 根据你跑通的示例，可能需要改成/dev/video8之类的，这里默认0
cap = cv2.VideoCapture(0) 

print("====== 终端测距标定开始 ======")
print("请把脸凑到你设定的触发距离（也就是能看清手机屏幕的距离，大概30-40cm左右）...")
print("保持姿势，看屏幕输出的像素宽度。")

while True:
    ret, frame = cap.read()
    if not ret:
        print("没抓到图，检查下摄像头节点")
        time.sleep(1)
        continue
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 扫人脸
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    
    for (x, y, w, h) in faces:
        print(f"-> 捕捉到人脸！当前人脸像素宽度 w = {w} px (按Ctrl+C退出)")
        
    time.sleep(0.2) # 降一降打印频率，免得终端刷屏太快