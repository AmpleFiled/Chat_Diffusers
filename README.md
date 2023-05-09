# Chat_Diffusers



chat_diffusers is a image generation tool based on chat conversation interaction. It uses the LLM model as the skeleton and uses chat interaction to generate target images.

**This function contains**:

1. suport generate image from chat interaction 
2. suport transfer image style from chat interaction
3. support ControlNet image generation from chat interaction
4. suport preset special style(CyberPunk) image generation from chat interaction
5. personal stabel diffusion finetune from chat interaction(to do...
6. personal DreamBooth Control image generation form chat interaction(to do...
7. local personal LLM model(like ChatGLM) deploymen(to do..


## Goal:
Chat Diffuers can make image generation ealier!


## Install


```sh
conda create -n chatSD python=3.8
conda install --yes --file requirements.txt
```

## Usage


```sh
# prepare your private OpenAI key
export OPENAI_API_KEY={Your_Private_Openai_Key}

# Start ChatDiffusers !
# You can specify the GPU/CPU/M1/2 assignment by "--load", the parameter indicates which 
# Visual Foundation Model to use and where it will be loaded to
# The model and device are separated by underline '_', the different models are separated by comma ','
# The available Visual Foundation Models can be found in the following table

# Advice for cpu Users
python visual_chatgpt.py --load Text2Image_cpu, ControlText2Image_cpu, CyberPunkText2Image_cpu

# Advice for mac m1/2 Users
python visual_chatgpt.py --load Text2Image_mps, ControlText2Image_mps, CyberPunkText2Image_mps

# Advice for CUDA Users                    
python visual_chatgpt.py --load "Text2Image_cuda:0,ControlText2Image_cuda:0"
                                


```

### Demo

 - generate normal image

![截屏2023-05-08 23 20 37](https://user-images.githubusercontent.com/132820015/236863865-9462e23e-1281-4f6c-85be-4e68d53dac4d.jpg)

 
 - transfer image style
![WechatIMG1735](https://github.com/AmpleFiled/Chat_Diffusers/assets/132820015/a93298a7-fb46-4f20-8d4e-b1a923ee4986)

 
 - generate preset special style(CyberPunk) image
 ![截屏2023-05-09 00 26 04](https://user-images.githubusercontent.com/132820015/236877979-3feb2998-3a71-44d4-9d16-28c79c20041d.jpg)




## Acknowledgement

We appreciate the open source of the following projects:

[Hugging Face](https://github.com/huggingface) &#8194;
[TaskMatrix](https://github.com/microsoft/TaskMatrix) &#8194;
[LangChain](https://github.com/hwchase17/langchain) &#8194;
[Stable Diffusion](https://github.com/CompVis/stable-diffusion) &#8194; 
[ControlNet](https://github.com/lllyasviel/ControlNet) &#8194; 

```

## Contact Information

For help or issues using the ChatDiffusers, please submit a GitHub issue.

For other communications, please contact Bo Tian (btian2@yahoot.com).


  [1]: %E6%88%AA%E5%B1%8F2023-05-08%2002.06.04
