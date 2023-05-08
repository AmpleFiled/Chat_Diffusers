
# from langchain.agents import Tool, initialize_agent, LLMSingleActionAgent, AgentOutputParser
# from langchain.prompts import BaseChatPromptTemplate
# from langchain.prompts import StringPromptTemplate
# from langchain import SerpAPIWrapper, LLMChain
# from langchain.utilities import GoogleSerperAPIWrapper
# from langchain.chat_models import ChatOpenAI
# from typing import List, Union
# from langchain.schema import AgentAction, AgentFinish, HumanMessage

from langchain.agents import Tool, initialize_agent
from langchain.chains.conversation.memory import ConversationBufferMemory
# from langchain.chat_models import ChatOpenAI
from langchain.llms.openai import OpenAI

import re
import argparse
import inspect

import gradio as gr

import os

CHAT_DIFFUSERS_PREFIX_CN = "B"
CHAT_DIFFUSERS_FORMAT_INSTRUCTIONS_CN ="""用户使用中文和你进行聊天，但是工具的参数应当使用英文。如果要调用工具，你必须遵循如下格式:
```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```
当你不再需要继续调用工具，而是对观察结果进行总结回复时，你必须使用如下格式：
```
Thought: Do I need to use a tool? No
{ai_prefix}: [your response here]
```
"""
CHAT_DIFFUSERS_SUFFIX_CN = """
推理想法和观察结果只对Visual ChatGPT可见，需要记得在最终回复时把重要的信息重复给用户，你只能给用户返回中文句子。我们一步一步思考。在你使用工具时，工具的参数只能是英文。
聊天历史:
{chat_history}
新输入: {input}
Thought: Do I need to use a tool? {agent_scratchpad}
"""


os.environ["OPENAI_API_KEY"] = "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"

def cut_dialogue_history(history_memory, keep_last_n_words=500):
    if history_memory is None or len(history_memory) == 0:
        return history_memory
    tokens = history_memory.split()
    n_tokens = len(tokens)
    print(f"history_memory:{history_memory}, n_tokens: {n_tokens}")
    if n_tokens < keep_last_n_words:
        return history_memory
    paragraphs = history_memory.split('\n')
    last_n_tokens = n_tokens
    while last_n_tokens >= keep_last_n_words:
        last_n_tokens -= len(paragraphs[0].split(' '))
        paragraphs = paragraphs[1:]
    return '\n' + '\n'.join(paragraphs)

def prompts(name, description):
    def decorator(func):
        func.name = name
        func.description = description
        return func

    return decorator

class ConversationBot:
    def __init__(self, load_dict):

        self.models = {}
        # Load Basic Foundation Models
        for class_name, device in load_dict.items():
            self.models[class_name] = globals()[class_name](device=device)

        # Load Template Foundation Models
        for class_name, module in globals().items():
            if getattr(module, 'template_model', False):
                template_required_names = {k for k in inspect.signature(module.__init__).parameters.keys() if
                                           k != 'self'}
                loaded_names = set([type(e).__name__ for e in self.models.values()])
                if template_required_names.issubset(loaded_names):
                    self.models[class_name] = globals()[class_name](
                        **{name: self.models[name] for name in template_required_names})

        print(f"All the Available Functions: {self.models}")

        self.tools = []
        for instance in self.models.values():
            for e in dir(instance):
                if e.startswith('inference'):
                    func = getattr(instance, e)
                    self.tools.append(Tool(name=func.name, description=func.description, func=func))
        self.llm = OpenAI(temperature=0)
        self.memory = ConversationBufferMemory(memory_key="chat_history", output_key='output')

    def init_agent(self, lang):
        self.memory.clear()  # clear previous history

        PREFIX, FORMAT_INSTRUCTIONS, SUFFIX = CHAT_DIFFUSERS_PREFIX_CN, CHAT_DIFFUSERS_FORMAT_INSTRUCTIONS_CN, CHAT_DIFFUSERS_SUFFIX_CN
        place = "Enter text and press enter, or upload an image"
        label_clear = "Clear"
        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent="conversational-react-description",
            verbose=True,
            memory=self.memory,
            return_intermediate_steps=True,
            agent_kwargs={'prefix': PREFIX, 'format_instructions': FORMAT_INSTRUCTIONS,
                          'suffix': SUFFIX}, )
        return gr.update(visible=True), gr.update(visible=False), gr.update(placeholder=place), gr.update(
            value=label_clear)

    def run_text(self, text, state):
        self.agent.memory.buffer = cut_dialogue_history(self.agent.memory.buffer, keep_last_n_words=500)
        res = self.agent({"input": text.strip()})
        res['output'] = res['output'].replace("\\", "/")
        response = re.sub('(image/[-\w]*.png)', lambda m: f'![](file={m.group(0)})*{m.group(0)}*', res['output'])
        state = state + [(text, response)]
        print(f"\nProcessed run_text, Input text: {text}\nCurrent state: {state}\n"
              f"Current Memory: {self.agent.memory.buffer}")
        return state, state


if __name__ == '__main__':
    if not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str, default="Text2Image_mps, ControlText2Image_mps, CyberPunkText2Image_mps")
    args = parser.parse_args()
    load_dict = {e.split('_')[0].strip(): e.split('_')[1].strip() for e in args.load.split(',')}
    bot = ConversationBot(load_dict=load_dict)
    with gr.Blocks() as demo:
        lang = gr.Radio(choices=['begin enjoy'], value=None, label='Welcome to Chat Diffusers')
        chatbot = gr.Chatbot(elem_id="chatbot", label="Chat_Diffusers")
        state = gr.State([])
        with gr.Row(visible=False) as input_raws:
            with gr.Column(scale=0.7):
                txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter, or upload an image").style(
                    container=False)
            with gr.Column(scale=0.15, min_width=0):
                clear = gr.Button("Clear")

        lang.change(bot.init_agent, [lang], [input_raws, lang, txt, clear])
        txt.submit(bot.run_text, [txt, state], [chatbot, state])
        txt.submit(lambda: "", None, txt)
        clear.click(bot.memory.clear)
        clear.click(lambda: [], None, chatbot)
        clear.click(lambda: [], None, state)
    demo.launch(server_name="0.0.0.0")
