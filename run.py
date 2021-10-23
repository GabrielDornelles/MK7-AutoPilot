import cv2
import time
from PIL import ImageGrab
import numpy as np
from pynput.keyboard import Key, Controller
from model import steering_model
import torch
import torch.nn.functional as F
from PIL import Image
import albumentations
import threading
import torchvision
import matplotlib.pyplot as plt


def pre_process(frame,early_return=False):
    image = Image.fromarray(np.uint8(frame)).convert('RGB')
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    aug = albumentations.Compose(
                [
                    albumentations.Normalize(
                        mean, std, max_pixel_value=255.0, always_apply=True)
                ]
            )
    image = image.resize(
        (200,66), resample=Image.BILINEAR
    )

    image = np.array(image)
    augmented = aug(image=image)
    image = augmented["image"]
    if early_return:
        return image

    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    image = image[None,:,:,:]
    return torch.from_numpy(image)


path = "./mk8-model-roi-checkpoint-epoche-37.pth"
device = torch.device("cuda")
model = steering_model()
model.load_state_dict(torch.load(path, map_location=device))

prev = 0
framerate=5
steering = 0

# keyboard = Controller()

# def update_steering(steering):
#     keyboard.release(Key.right)
#     keyboard.release(Key.left)
#     if steering<-0.5:
#         keyboard.press(Key.left)
#     if steering>-0.2:
#         keyboard.press(Key.right)
       

# monitor_thread = threading.Thread(target=update_steering(steering), args=())
# monitor_thread.daemon = True

# monitor_thread.start()
if __name__ == '__main__':
    kernels = model.feature_maps(0)
    while True:
        screen = cv2.cvtColor(np.array(ImageGrab.grab(bbox=(0,300,1920,620))), cv2.COLOR_BGR2RGB)
        frame = pre_process(screen)
        convolved_inputs = F.conv2d(frame, kernels, stride=2, padding=1)[0]
        convolved_inputs = [convolved_image.detach().numpy() for convolved_image in convolved_inputs]

        # worst way to visualize as grids, but its real time at least. TODO: make something better
        images = np.concatenate((convolved_inputs[0], convolved_inputs[1], convolved_inputs[2], convolved_inputs[3], convolved_inputs[4], convolved_inputs[5]), axis=1)
        images_2 = np.concatenate((convolved_inputs[6], convolved_inputs[7], convolved_inputs[8], convolved_inputs[9], convolved_inputs[10], convolved_inputs[11]), axis=1)
        images_3 = np.concatenate((convolved_inputs[12], convolved_inputs[13], convolved_inputs[14], convolved_inputs[15], convolved_inputs[16], convolved_inputs[17]), axis=1)
        images_4 = np.concatenate((convolved_inputs[18], convolved_inputs[19], convolved_inputs[20], convolved_inputs[21], convolved_inputs[22], convolved_inputs[23]), axis=1)
        images = np.concatenate((images,images_2), axis=0)
        images_2 = np.concatenate((images_3,images_4), axis=0)
        images =np.concatenate((images,images_2), axis=0)
        images = cv2.resize(images, (594*4, 64*4))   #594 64
        cv2.imshow("convolved-feature-maps-first-layer", images)
        cv2.imshow("roi", screen)

        steering = model(frame)
        print(steering[0])
        if cv2.waitKey(33) & 0xFF in (
            ord('q'), 
            27, 
        ):
            break
        
        # frame = pre_process(screen)
        # steering = model(frame)
        # print(steering[0])
        #update_steering(steering)
          
