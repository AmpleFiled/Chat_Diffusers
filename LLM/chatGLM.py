
from langchain import LLMMathChain, SerpAPIWrapper
from langchain.agents import Tool, initialize_agent
from langchain.tools import BaseTool
from langchain.llms.base import LLM
from langchain.agents.loading import AGENT_TO_CLASS
from langchain.agents.conversational.base import ConversationalAgent
from langchain.chains.conversation.memory import ConversationBufferMemory

from diffusers import StableDiffusionPipeline
from PIL import Image
import torch
import argparse
from typing import Any, List, Tuple, Dict, Mapping, Optional
from pydantic import BaseModel, root_validator
import re

from transformers import AutoModel, AutoTokenizer

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def register_agent(cls):
    if not getattr(cls, '_agent_type'):
        raise AttributeError(
            f"{cls.__name__} has no attribute `_agent_type`. "
            f"register {cls.__name__} as agent failed. "
        )
    AGENT_TO_CLASS[cls._agent_type.fget(None)] = cls
    return cls

class CustomChatGLM6B:

    def __init__(self):
        # self.tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/chatglm-6b', trust_remote_code=True, cache_dir = '/root/autodl-tmp')
        # self.model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True, cache_dir = '/root/autodl-tmp').half()

        self.tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/models--THUDM--chatglm-6b/snapshots/f83182484538e663a03d3f73647f10f89878f438', trust_remote_code=True)
        self.model = AutoModel.from_pretrained('/root/autodl-tmp/models--THUDM--chatglm-6b/snapshots/f83182484538e663a03d3f73647f10f89878f438', trust_remote_code=True).half().to(device)
        # model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        #
        # torch.save(model_to_save.state_dict(), 'pytorch_model_unzip.bin', _use_new_zipfile_serialization=False)
        self.model = self.model.eval()

    def inference(self, prompt):
        response, history = self.model.chat(self.tokenizer, prompt, history=[])
        return response


class ChatGLM6BLLM(LLM, BaseModel):
    # client: Any
    #
    # @root_validator()
    # def validate_environment(cls, values: Dict) -> Dict:
    #     values['client'] = CustomChatGLM6B()
    #     return values


    def _llm_type(self) -> str:
        return "custom"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        llm = CustomChatGLM6B()
        out= "Thought: " + llm.inference(prompt)
        # if stop is not None:
        #     raise ValueError("stop kwargs are not permitted.")

        print("the chatGLM out is:", out)
        return out

# @register_agent
# class CustomConversationalAgent(ConversationalAgent):
#
#     @property
#     def _agent_type(self) -> str:
#         """Return Identifier of agent type."""
#         return "custom-conversational-react-description"
#
#     def _extract_tool_and_input(self, llm_output: str) -> Optional[Tuple[str, str]]:
#         # if f"{self.ai_prefix}:" in llm_output:
#         #     return self.ai_prefix, llm_output.split(f"{self.ai_prefix}:")[-1].strip()
#         regex = r"Action: (.*?)[\n]*Action Input: (.*)"
#         match = re.search(regex, llm_output)
#         print("12344", match)
#         action = "Generate Image From User Input Text"
#         if not match:
#             print(f"Could not parse LLM output: `{llm_output}`. "
#                 f"So using the default action input")
#             # action_input = text2prompt(text, max_retry_num=5)
#         else:
#             action_input = match.group(2)
#             # action_input = text2prompt(action_input, max_retry_num=5)
#         return action.strip(), action_input.strip(" ").strip('"')
        # return action.strip(), action.strip()

if __name__ == "__main__":
    llm = ChatGLM6BLLM()
    out = llm.generate(["I want change the world!"])
    print(out)