

import cv2
import os


def video2imgs(videoPath, imgPath, merge=1):
    if not os.path.exists(imgPath):
        # 目标文件夹不存在，则创建
        os.makedirs(imgPath)
    # 获取视频
    cap = cv2.VideoCapture(videoPath)
    # 判断是否能打开成功
    judge = cap.isOpened()
    print("checking video: ", judge)
    # 帧率，视频每秒展示多少张图片
    fps = cap.get(cv2.CAP_PROP_FPS)
    print('fps is:', fps)

    frames = 1
    # 用于统计所有帧数
    count = 1
    # 用于统计保存的图片数量

    while(judge):
        # 读取每一张图片 flag表示是否读取成功，frame是图片
        flag, frame = cap.read()
        if not flag:
            print(flag)
            print("Process finished!")
            break
        else:
            # 每隔10帧抽一张
            if frames % merge == 0:
                imgname = str(count).zfill(4) + ".jpg"
                newPath = os.path.join(imgPath, imgname)
                print(imgname)
                cv2.imwrite(newPath, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
                # cv2.imencode('.jpg', frame)[1].tofile(newPath)
                count += 1
        frames += 1
    cap.release()
    print("共有 %d 张图片"%(count-1))


if __name__ == "__main__":

    # video_path = "../video/002.mp4"  # fps=30
    save_to_root = 'out-tmp'

    video_path = "output-002-cat.avi"  # fps=30

    video2imgs(video_path, save_to_root)

