

import os
from PIL import Image


def make_gif(image_root, to_name):

    image_list = os.listdir(image_root)
    image_list.sort()

    frames = [Image.open(os.path.join(image_root, image)) for image in image_list]
    frame_one = frames[0]
    frame_one.save(to_name + ".gif",
                   format="GIF",
                   append_images=frames,
                   save_all=True,
                   duration=400,
                   loop=0
                   )
    print('-' * 20)


if __name__ == "__main__":

    make_gif("out-tmp", "t2")

