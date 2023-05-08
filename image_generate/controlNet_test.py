
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import DiffusionPipeline
import torch
import os
import uuid
import cv2
from PIL import Image
import numpy as np

def auto_canny(image, sigma=0.7):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    print(v)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    # edged = cv2.Canny(image, 50, 101)
    print(lower, upper)
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged

class CannyText2Image:

    """
    controlNet text2Image is a multi priors diffusion generation model, one is the text embedding
     prior, another one is the skeleton information which maintain the main component of the image
    """

    def __init__(self, device):

        self.device = device
        self.torch_dtype = torch.float16 if ("cuda" or "mps") in device else torch.float32
        controlnet = ControlNetModel.from_pretrained("fusing/stable-diffusion-v1-5-controlnet-canny", torch_dtype=torch.float16)
        self.pipline = StableDiffusionControlNetPipeline.from_pretrained("/Users/ample/Downloads/models--runwayml--stable-diffusion-v1-5/snapshots/39593d5650112b4cc580433f6b0435385882d819",
                                                                         trust_remote_code=True, controlnet = controlnet, torch_dtype=torch.float16,safety_checker = None)
        self.pipline.to(device)
        self.prompt = 'style, typical style, professional environment, cinematic lighting, 8K' \
                      'colorful ' \
                      'a high-quality, detailed, and professional image'


    def inference(self, image_path, prompt):
        canny_image = Image.open(image_path)

        image_filename = os.path.join('../image', f"{str(uuid.uuid4())[:8]}.png")

        prompt = prompt + self.prompt
        image = self.pipline(prompt, num_inference_steps=20, image=canny_image).images[0]
        image.save(image_filename)

        print(f"\nProcessed ControlNet_Text2Image, Input Text: {prompt}, Output Image: {image_filename}"
                )
        return image_filename

class ControlText2Image:
    """
    controlNet text2Image is a multi priors diffusion generation model, one is the text embedding
     prior, another one is the skeleton information which maintain the main component of the image
    """

    def __init__(self, device):

        self.device = device
        self.torch_dtype = torch.float16 if ("cuda" or "mps") in device else torch.float32
        controlnet = ControlNetModel.from_pretrained("fusing/stable-diffusion-v1-5-controlnet-canny", torch_dtype=torch.float16)
        self.pipline = StableDiffusionControlNetPipeline.from_pretrained("/Users/ample/Downloads/models--runwayml--stable-diffusion-v1-5/snapshots/39593d5650112b4cc580433f6b0435385882d819",
                                                                         trust_remote_code=True, controlnet = controlnet, torch_dtype=torch.float16,)
        self.pipline.to(device)
        self.prompt = f'style, strong target environment style' \
                    # f"8k, wallpaper, best quality, masterpiece, illustration, an extremely exquisite and beautiful, " \
                    # f"extremely detailed, CG, unified, wallpaper(realistic, photo realistic: 1.37), " \
                    # f"stunning, fine details, masterpiece, best quality, " \
                    # f"official art, extremely detailed CG unified, 8k wallpaper,"

    def inference(self, image_path, instruct_text):
        # image_path, instruct_text = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        control_image = Image.open(image_path)
        image = np.array(control_image)
        kernel = int(np.median(image) // 20)
        if kernel%2 == 0: kernel = kernel + 1
        image = cv2.GaussianBlur(image, (kernel, kernel), 0)
        # # transform img to edg map
        image = auto_canny(image)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        canny_image = Image.fromarray(image)
        prompt = instruct_text + self.prompt
        image_canny_filename = os.path.join('../image',
                                            f"{str(uuid.uuid4())[:8]}_canny.png")
        print(image_canny_filename)
        canny_image.save(image_canny_filename)

        image_filename = os.path.join('../image', f"{str(uuid.uuid4())[:8]}.png")

        image = self.pipline(prompt, num_inference_steps=100, image=canny_image).images[0]
        image.save(image_filename)


        print(f"\nProcessed ControlNet_Text2Image, Input Text: {instruct_text}, Output Image: {image_filename},"
              f"canny Image: {image_canny_filename}")
        return image_filename

if __name__ == "__main__":
    # image = cv2.imread("../image/d385ca7f.png")
    # print(np.unique(image))
    # image = auto_canny(image)
    # image = image[:, :, None]
    # image = np.concatenate([image, image, image], axis=2)
    # canny_image = Image.fromarray(image)
    # canny_image.save("image_canny_filename.png")
    # print(image.shape, np.unique(image))
    # exit()
    model = CannyText2Image("mps")
    model.inference("../image/8904d4c3_canny.png", "disco")