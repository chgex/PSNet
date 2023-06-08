

import torch
import torchvision.transforms as T

import numpy as np
import os

import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

# import matplotlib.pyplot as plt
from PIL import Image

TRT_LOGGER = trt.Logger()

# Filenames of TensorRT plan file and input/output images.
engine_file = "psnet.trt"
# output_file = "output.ppm"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device ...".format(device.type))


# def preprocess(image):
#     # Mean normalization
#     mean = np.array([0.485, 0.456, 0.406]).astype('float32')
#     stddev = np.array([0.229, 0.224, 0.225]).astype('float32')
#     data = (np.asarray(image).astype('float32') / float(255.0) - mean) / stddev
#     # Switch from HWC to to CHW order
#     return np.moveaxis(data, 2, 0)


def preprocess(image_pil):
    image_pil = T.CenterCrop(size=(320, 448))(image_pil)
    data = np.asarray(image_pil).astype('float32')
    # Switch from HWC to to CHW order
    return np.moveaxis(data, 2, 0)

    # image = np.array(image_pil, dtype="float")
    # # [h, w, c] to [c, h, w]
    # image = np.transpose(image, (2, 0, 1))
    # image_input = torch.from_numpy(image).type(torch.FloatTensor)
    #
    # # original image shape to input size
    # image_input = T.CenterCrop(size=(320, 448))(image_input)
    # image_input = image_input.unsqueeze(0)
    #
    # return image_input


def postprocess(data, org_size, device):
    assert len(data) == 6

    data = [torch.from_numpy(d).type(torch.FloatTensor) for d in data]
    data = [d.unsqueeze(0).to(device) for d in data]
    cls, radii1, radii2, off1, off2, logics = data
    output_loc = [cls, radii1, radii2, off1, off2]

    from PSNet import decode_model_output

    detected_circle, predicted_mask = decode_model_output([output_loc, logics], (320, 448), org_size, device)

    return detected_circle, predicted_mask


def draw_circle_ndarray(image, circle, color=(255, 0, 0), thickness=1):
    """
    draw circle on image or mask image
    :param image: ndarray, shape is H W C
    :param circle: ndarray, shanpe is (6,0) or (3,0)
    :return: image
    """
    import cv2

    assert len(circle) == 3 or len(circle) == 6

    image = np.array(image, dtype='int')
    image = image.astype("int")
    circle = circle.astype("int")

    if len(circle) == 3:
        inner_center, inner_radius = (circle[0], circle[1]), circle[2]
        image = cv2.circle(image, inner_center, inner_radius, color=color, thickness=thickness)
    else:
        if circle[2] < circle[5]:
            inner_center, inner_radius = (circle[0], circle[1]), circle[2]
            outer_center, outer_radius = (circle[3], circle[4]), circle[5]
        else:
            inner_center, inner_radius = (circle[3], circle[4]), circle[5]
            outer_center, outer_radius = (circle[0], circle[1]), circle[2]
        image = cv2.circle(image, inner_center, inner_radius, color=color, thickness=thickness)
        image = cv2.circle(image, outer_center, outer_radius, color=color, thickness=thickness)

    return image


def draw_shadow_ndarray(image, mask, shadow_color=(255, 255, 0)):
    import cv2
    image = image.astype('int')
    mask = mask.astype("int")

    assert len(image.shape) == 3 and len(mask.shape) == 2

    if np.max(mask) == 255:
        mask /= 255

    image = image.astype(np.int32)
    seglap = image.copy()
    segout = image.copy()

    mask_img = np.array(mask, dtype="int32")
    mask_t = mask_img > 0
    for i in range(3):
        seglap[mask_t, i] = shadow_color[i]
    alpha = 0.5
    cv2.addWeighted(seglap, alpha, segout, 1 - alpha, 0, segout)

    # segout = Image.fromarray(segout.astype("uint8"))

    return segout


# load tensorrt engine
def load_engine(engine_file_path):
    assert os.path.exists(engine_file_path)
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


# inference pipeline
def infer(engine, image_path):
    print("Reading input image from file {}".format(image_path))
    with Image.open(image_path) as img:
        input_image = preprocess(img)
        image_width = img.width
        image_height = img.height
    # input_image = preprocess(input_file)
    print("---")
    with engine.create_execution_context() as context:
        # Set input shape based on image dimensions for inference
        context.set_binding_shape(engine.get_binding_index("inputs"), (1, 3, 320, 448))
        # Allocate host and device buffers
        bindings = []
        out_bufs = []
        out_mems = []
        for binding in engine:
            binding_idx = engine.get_binding_index(binding)
            size = trt.volume(context.get_binding_shape(binding_idx))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            if engine.binding_is_input(binding):
                input_buffer = np.ascontiguousarray(input_image)
                input_memory = cuda.mem_alloc(input_image.nbytes)
                bindings.append(int(input_memory))
            else:
                output_buffer = cuda.pagelocked_empty(size, dtype)
                output_memory = cuda.mem_alloc(output_buffer.nbytes)
                bindings.append(int(output_memory))

                out_bufs.append(output_buffer)
                out_mems.append(output_memory)

        stream = cuda.Stream()
        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(input_memory, input_buffer, stream)
        # Run inference
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        # Transfer prediction output from the GPU.
        # cuda.memcpy_dtoh_async(output_buffer, output_memory, stream)
        [cuda.memcpy_dtoh_async(out_buf, out_mem, stream) for out_buf, out_mem in zip(out_bufs, out_mems)]
        # Synchronize the stream
        stream.synchronize()

    outputs = [np.reshape(out, (-1, 320, 448)) for out in out_bufs]

    print("----------")
    circle, mask = postprocess(outputs, (image_height, image_width), device)
    print(circle)

    image = draw_circle_ndarray(img, circle[:3], color=(0, 255, 0), thickness=2)
    image = draw_circle_ndarray(image, circle[3:], color=(255, 0, 0), thickness=2)
    image = draw_shadow_ndarray(image, mask, shadow_color=(100, 50, 255))
    image_show = Image.fromarray(image.astype("uint8"))
    image_show.show()

    print("------------")


if __name__ == "__main__":
    print("Running TensorRT inference for FCN-ResNet101")

    image_path = '0003.jpg'
    with load_engine(engine_file) as engine:

        infer(engine, image_path)

        print('*' * 20)




