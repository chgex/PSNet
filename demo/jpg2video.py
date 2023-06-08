import cv2
import numpy as np
import os


def images_to_video(image_root, fps, avi_name="out.avi"):

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
    total = len(image_list) - 1
    for index, image_name in enumerate(image_list):
        image_path = os.path.join(image_root, image_name)
        image = cv2.imread(image_path)
        video.write(image)

        print("current: %d, total:%d, %s" % (index, total, image_path))
    video.release()


if __name__ == "__main__":

    # image_root = "output_images"
    # images_to_video(image_root, 12)

    # image_root = "outputs-002"
    # images_to_video(image_root, 30, "output-002.avi")

    # image_root = "out2put"
    # images_to_video(image_root, 30, "output-002-cat.avi")

    image_root = "out-tmp"
    images_to_video(image_root, 30, "t2.avi")

