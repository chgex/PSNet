

import cv2
import os

import numpy as np


def video2imgs(videoPath, imgPath, merge=2):
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



def concat2image_and2video(root1, root2, fps, avi_name="out.avi"):

    image_list = os.listdir(image_root)
    first_image = cv2.imread(os.path.join(image_root, image_list[0]))
    # 获取图片尺寸
    imgInfo = first_image.shape
    size = (imgInfo[1], imgInfo[0])
    print("image size: ", size)

    image_list.sort()
    # 设置视频编码格式
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(avi_name, fourcc, fps, size)
    # 视频保存在当前目录下
    total = len(image_list)
    for index, image_name in enumerate(image_list):
        image_path = os.path.join(image_root, image_name)
        image = cv2.imread(image_path)
        video.write(image)

        print("current: %d, total:%d, %s" % (index, total, image_path))
    video.release()


if __name__ == "__main__":

    output_root = "outputs-002"
    org_root = "images-002"
    image_list = os.listdir(output_root)

    total = len(image_list)
    for index, image_name in enumerate(image_list):
        img_path1 = os.path.join(output_root, image_name)
        image_out = cv2.imread(img_path1)
        image_h, image_w = image_out.shape[:2]
        # print(image_out.shape)

        img_path2 = os.path.join(org_root, image_name)
        image_org = cv2.imread(img_path2)

        image = np.zeros((image_h, image_w * 2 + 100, 3), dtype='int')
        # print(image.shape)
        image[:, :image_w, :] = image_org
        image[:, image_w + 100:image_w * 2 + 100, :] = image_out

        to_path = "./out2put/" + image_name
        cv2.imwrite(to_path, image)
        # break

        print("index: %d, total: %d" % (index, total))


