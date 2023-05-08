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

class Text2Image:
    """
    Text2Image is a text prior generation model, which use clip embedding associate with text and image
    """

    def __init__(self, device):

        self.device = device
        self.torch_dtype = torch.float16 if ("cuda" or "mps") in device else torch.float32
        self.pipline = StableDiffusionPipeline.from_pretrained("../../models--runwayml--stable-diffusion-v1-5/snapshots/39593d5650112b4cc580433f6b0435385882d819", trust_remote_code=True)
        self.pipline.to(device)
        # self.prompt = 'a high-quality, detailed, and professional image'
        # self.prompt = f"Complex 3 D rendering " \
        #               f"semi robot, in outer space, cyberpunk, robot parts, 150 mm, beautiful studio soft light, " \
        #               f"fine face, vibrant details, luxury cyberpunk, surrealism, anatomy, facial muscles, cables, " \
        #               f"microchips, electronic boards, current, elegance, beautiful background, octane rendering, H. R. Gig style, " \
        #               f"8k, wallpaper, best quality, masterpiece, illustration, an extremely exquisite and beautiful, extremely detailed, CG, unified, wallpaper, " \
        #               f"(realistic, photo realistic: 1.37), stunning, fine details, masterpiece, best quality, " \
        #               f"official art, extremely detailed CG unified, 8k wallpaper, robot, silver dummy,"

        self.prompt = f"monograph, full body, cyberpunk, pure linework, Parts assembly diagram," \
                      f" diagramatic sketch, Leonardo da Vinci sketch,  notation and Auxiliary line," \
                      f" gitantic scale,design spread sheet"


    def inference(self, text):

        prompt = text + "," + self.prompt
        image_name = f"{str(uuid.uuid4())[:8]}.png"
        image_filename = os.path.join('/Users/ample/Downloads/chat_sd/image', image_name)

        image = self.pipline(prompt, num_inference_steps = 20).images[0]
        image.save(image_filename)
        print(
            f"\nProcessed Text2Image, Input Text: {text}, Output Image: {image_filename}")
        return image

if __name__ == "__main__":


    prompt = "美少女"
    model = Text2Image("mps")
    img = model.inference(prompt)