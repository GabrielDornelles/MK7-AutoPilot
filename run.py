import cv2
import time
from PIL import ImageGrab
import numpy as np
from pynput.keyboard import Key, Controller
from model import steering_model
import torch
from PIL import Image
import albumentations
import threading


def pre_process(frame):
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
        (66,200), resample=Image.BILINEAR
    )

    image = np.array(image)
    augmented = aug(image=image)
    image = augmented["image"]
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    image = image[None,:,:,:]
    return torch.from_numpy(image)


path = "./mk7-model-roi.pth"
device = torch.device("cuda")
model = steering_model()
model.load_state_dict(torch.load(path, map_location=device))

prev = 0
framerate=5
keyboard = Controller()

def update_steering(steering):
    keyboard.release(Key.right)
    keyboard.release(Key.left)
    if steering<-0.5:
        keyboard.press(Key.left)
    if steering>0.5:
        keyboard.press(Key.right)
       

steering = 0
monitor_thread = threading.Thread(target=update_steering(steering), args=())
monitor_thread.daemon = True

if __name__ == '__main__':
    monitor_thread.start()
    while True:
        # time_elapsed = time.time() - prev
        # if time_elapsed > 1./framerate:

        #     prev= time.time()
        screen = cv2.cvtColor(np.array(ImageGrab.grab(bbox=(0,340,1060,670))), cv2.COLOR_BGR2RGB)
        frame = pre_process(screen)
        steering = model(frame)
        print(steering[0])
        update_steering(steering)
          
