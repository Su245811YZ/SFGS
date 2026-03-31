import cv2
import os

video_path = '/home/suyuze/workspace/ExAvatar_mine/data/XHumans/data/tvxq_mirotic/video+audio.mp4'
frames_dir = '/home/suyuze/workspace/ExAvatar_mine/data/XHumans/data/tvxq_mirotic/frames/'

os.makedirs(frames_dir, exist_ok=True)      # 自动建文件夹
cap = cv2.VideoCapture(video_path)

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:          # 读到结尾
        break
    save_path = os.path.join(frames_dir, f'{frame_idx:06d}.png')
    cv2.imwrite(save_path, frame)
    frame_idx += 1

cap.release()
print(f'Done! {frame_idx} frames saved to {frames_dir}')