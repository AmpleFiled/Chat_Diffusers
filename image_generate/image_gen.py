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

def prompts(name, description):
    def decorator(func):
        func.name = name
        func.description = description
        return func

    return decorator

class CyberPunkText2Image:
    """
    Text2Image is a text prior generation model, which use clip embedding associate with text and image
    """

    def __init__(self, device):

        self.device = device
        self.torch_dtype = torch.float16 if ("cuda" or "mps") in device else torch.float32
        self.pipline = StableDiffusionPipeline.from_pretrained("../models--runwayml--stable-diffusion-v1-5/snapshots/39593d5650112b4cc580433f6b0435385882d819", trust_remote_code=True)
        self.pipline.to(device)
        self.prompt = f"Complex 3 D rendering " \
                      f"semi robot, in outer space, cyberpunk, robot parts, 150 mm, beautiful studio soft light, " \
                      f"fine face, vibrant details, luxury cyberpunk, surrealism, anatomy, facial muscles, cables, " \
                      f"microchips, electronic boards, current, elegance, beautiful background, octane rendering, H. R. Gig style, " \
                      f"8k, wallpaper, best quality, masterpiece, illustration, an extremely exquisite and beautiful, extremely detailed, CG, unified, wallpaper, " \
                      f"(realistic, photo realistic: 1.37), stunning, fine details, masterpiece, best quality, " \
                      f"official art, extremely detailed CG unified, 8k wallpaper, robot, silver dummy,"

    @prompts(name="CyperPunk image",
             description="useful when you want to generate an cyberpunk image from prefixed prompt setting. "
                         "like: generate an image of an object or something, or generate an image that includes some objects. "
                         "The input to this tool should be a string, representing the text used to generate image. ")

    def inference(self, text):

        prompt = text + "," + self.prompt
        image_name = f"{str(uuid.uuid4())[:8]}.png"
        image_filename = os.path.join('/Users/ample/Downloads/chat_sd/image', image_name)
        print("the CyberPunk prompt is:", prompt)
        image = self.pipline(prompt, num_inference_steps = 20).images[0]
        image.save(image_filename)
        print(
            f"\nProcessed Text2Image, Input Text: {text}, Output Image: {image_filename}")
        return image_filename


class TextGuided_image2image:

    """
    similar latent diffusion model, which change nosie to noise + img_embedding info
    """

    def __init__(self, device):

        self.device = device
        self.torch_dtype = torch.float16 if ("cuda" or "mps") in device else torch.float32
        self.pipeline = StableDiffusionImg2ImgPipeline.from_pretrained("nitrosocke/Ghibli-Diffusion",torch_dtype=torch.float16).to(device)
        self.pipeline.to(device)
        self.prompt = 'high resolution image, detailed riched'

    @prompts(name="Generate Image From User Input Text",
             description="useful when you want to generate an image from a user input text and save it to a file. "
                         "like: generate an image of an object or something, or generate an image that includes some objects. "
                         "The input to this tool should be a string, representing the text used to generate image. ")

    def inference(self, text, image):

        prompt = text + "," + self.prompt
        image_filename = os.path.join('/Users/ample/Downloads/chat_sd/image', f"{str(uuid.uuid4())[:8]}.png")
        image = self.pipeline(prompt=prompt, image=image, strength=0.75, guidance_scale=7.5).images[0]
        image.save(image_filename)
        print(
            f"\nProcessed Text2Image, Input Text: {text}, Output Image: {image_filename}")
        return image_filename

class Text2Image:
    """
    Text2Image is a text prior generation model, which use clip embedding associate with text and image
    """

    def __init__(self, device):

        self.device = device
        self.torch_dtype = torch.float16 if ("cuda" or "mps") in device else torch.float32
        self.pipline = StableDiffusionPipeline.from_pretrained("../models--runwayml--stable-diffusion-v1-5/snapshots/39593d5650112b4cc580433f6b0435385882d819", trust_remote_code=True)
        self.pipline.to(device)
        # self.prompt = 'a high-quality, detailed, and professional image'
        self.prompt = "a high-quality, background clean"


    @prompts(name="Generate Image From User Input Text",
             description="useful when you want to generate an image from a user input text and save it to a file. "
                         "like: generate an image of an object or something, or generate an image that includes some objects. "
                         "The input to this tool should be a string, representing the text used to generate image. ")

    def inference(self, text):

        prompt = text + "," + self.prompt
        image_name = f"{str(uuid.uuid4())[:8]}.png"
        image_filename = os.path.join('./image', image_name)

        image = self.pipline(prompt, num_inference_steps = 20).images[0]
        image.save(image_filename)
        print(
            f"\nProcessed Text2Image, Input Text: {text}, Output Image: {image_filename}")
        return image_filename

class Dreambooth_Text2Image:
    """
    Dreambooth_Text2Image is a strone control image generation method,
     which can maintain the user target object. It can generate the various user taget object images,
     guided by the text prompt embedding.
     It is need retrain the sd models and provide a special target identifier

    """

    def __init__(self, device):

        self.device = device
        self.torch_dtype = torch.float16 if ("cuda" or "mps") in device else torch.float32
        model_id = "path_to_saved_model"
        self.pipline = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
        # self.pipline = StableDiffusionPipeline.from_pretrained("../models--runwayml--stable-diffusion-v1-5/snapshots/39593d5650112b4cc580433f6b0435385882d819", trust_remote_code=True)
        self.pipline.to(device)
        self.prompt = 'style, typical style, professional environment, cinematic lighting, 8K' \
                      'style, typical style, professional environment, cinematic lighting, 8K' \
                      'a high-quality, detailed, and professional image'
    @prompts(name="Generate Image From User Input Text",
             description="useful when you want to generate an image from a user input text and save it to a file. "
                         "like: generate an image of an object or something, or generate an image that includes some objects. "
                         "The input to this tool should be a string, representing the text used to generate image. ")

    def inference(self, text):

        prompt = text + "," + self.prompt
        image_filename = os.path.join('../image', f"{str(uuid.uuid4())[:8]}.png")
        image = self.pipline(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
        image.save(image_filename)
        print(
            f"\nProcessed Dreambooth_Text2Image, Input Text: {text}, Output Image: {image_filename}")
        return image


def auto_canny(image, sigma=0.9):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median

    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged

class ControlText2Image:
    """
    controlNet text2Image is a multi priors diffusion generation model, one is the text embedding
     prior, another one is the skeleton information which maintain the main component of the image
    """

    def __init__(self, device):

        self.device = device
        self.torch_dtype = torch.float16 if ("cuda" or "mps") in device else torch.float32
        controlnet = ControlNetModel.from_pretrained("/Users/ample/.cache/huggingface/hub/models--lllyasviel--sd-controlnet-canny/snapshots/7f2f69197050967007f6bbd23ab5e52f0384162a", torch_dtype=torch.float16)
        self.pipline = StableDiffusionControlNetPipeline.from_pretrained("/Users/ample/Downloads/models--runwayml--stable-diffusion-v1-5/snapshots/39593d5650112b4cc580433f6b0435385882d819",
                                                                         trust_remote_code=True, controlnet = controlnet, torch_dtype=torch.float16,safety_checker = None)
        self.pipline.to(device)
        self.prompt = 'style, typical style, professional environment, cinematic lighting, 8K' \
                      'style, typical style, professional environment, cinematic lighting, 8K' \
                      'a high-quality, detailed, and professional image'

    @prompts(name="transfer Image style using input image skeleton",
             description="useful when you want to generate an image from a user input text and maintain its component. "
                         "like: generate an image of an object or something with the component . "
                         "The input to this tool should be a string, representing the text used to generate image. "
                         "The input to this tool should be a comma separated string of two, ")

    def inference(self, inputs):
        image_path, instruct_text = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        control_image = Image.open(image_path)
        image = np.array(control_image)

        kernel = int(np.median(image) // 20)
        if kernel % 2 == 0: kernel = kernel + 1
        image = cv2.GaussianBlur(image, (kernel, kernel), 0)

        # # transform img to edg map
        image = auto_canny(image)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        canny_image = Image.fromarray(image)
        prompt = instruct_text + " " + self.prompt

        image_filename = os.path.join('/Users/ample/Downloads/chat_sd/image', f"{str(uuid.uuid4())[:8]}.png")
        image_canny_filename = os.path.join('/Users/ample/Downloads/chat_sd/image', f"{str(uuid.uuid4())[:8]}_canny.png")
        canny_image.save(image_canny_filename)
        print("the control net input is:",prompt)
        image = self.pipline(prompt, num_inference_steps=20, image=canny_image).images[0]
        image.save(image_filename)

        print(f"\nProcessed ControlNet_Text2Image, Input Text: {instruct_text}, Output Image: {image_filename}"
              f"canny Image: {image_canny_filename}")
        return image_filename

if __name__ == "__main__":

    # model = TextGuided_image2image("mps")
    # text = "rich detail, beautiful cloth"
    # image = cv2.imread("../image/47d25199.png")
    # model.inference(text, image)

    model = ControlText2Image("mps")
    text = "disco dancer with colorful lights"
    # image = cv2.imread("../image/a20c4f74.png")
    # image = np.array(image)
    model.inference("../image/650d66be.png")

    # model = Dreambooth_Text2Image("mps")
    # text = "rich detail, beautiful cloth"
    # image = cv2.imread("../image/47d25199.png")
    # image = np.array(image)
    # model.inference(text, image)
