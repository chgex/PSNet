

import time
import cv2

from detect_images import *


# from PSNet import get_engine, detect_image_iris_trt
# engine = get_engine("psnet.trt")


def process_frame(org_cv2):

    start_time = time.time()

    org_cv2 = cv2.cvtColor(org_cv2, cv2.COLOR_BGR2RGB)
    image_show = copy.deepcopy(org_cv2)

    lr_list = detect_image_face(yolo_face, image_show, device, vis_face=True, vis_mark=False, vis_box=True)

    if len(lr_list) > 0:
        for lr_box in lr_list:
            for xyxy in lr_box:
                # generally, len(lr_box) == 2, indicate left eye and right eye.
                x1, y1, x2, y2 = xyxy
                if min(x1, y1, x2, y2) <= 10:
                    continue
                else:
                    cropped_region = org_cv2[y1:y2, x1:x2, :]
                    print(" 7 cropped_region size: ", cropped_region.shape)
                    circle, mask, vision = detect_image_iris(cropped_region, psnet, device, vis=True)
                    # circle, mask, vision = detect_image_iris_trt(cropped_region, engine, device, vis=True)
                    image_show[y1:y2, x1:x2, :] = vision

    # pil_show = Image.fromarray(image_show.astype('uint8'))
    # pil_show.show()
    end_time = time.time()
    fps = 2 / (end_time - start_time)

    image_show = cv2.putText(image_show, 'FPS: '+ str(int(fps)), (50, 50), 0, 0.5, [0, 255, 0], thickness=1, lineType=cv2.LINE_AA)
    image_show = cv2.cvtColor(image_show, cv2.COLOR_RGB2BGR)

    return image_show


# get camera, 0 indicate 'default'
cap = cv2.VideoCapture(0)
# open camera
cap.open(0)


# loop,
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Error")
        break
    # process frame
    frame = process_frame(frame)

    # show
    cv2.imshow('window', frame)

    if cv2.waitKey(1) in [ord('q'), 27]:
        break
cap.release()
cv2.destroyAllWindows()









