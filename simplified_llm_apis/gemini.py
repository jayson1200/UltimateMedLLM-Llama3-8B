PROJ_PATH = "/home/meribejayson/Desktop/Projects/UltimateMedLLM-Llama3-8B/"
import sys
sys.path.append(PROJ_PATH)

from utils.get_env_vars import get_env_vars
import base64
import vertexai
from vertexai.generative_models import GenerativeModel, Part
import vertexai.preview.generative_models as generative_models

from simplified_llm_apis.model import LLM_Model

ENV_PATH = "/home/meribejayson/Desktop/Projects/UltimateMedLLM-Llama3-8B/env.json"
env_dict = get_env_vars(ENV_PATH)
PROJECT_ID = env_dict["project-id"]
safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
}

class Gemini(LLM_Model):
    def __init__(self, system_instructions: str) -> None:
        vertexai.init(project=PROJECT_ID, location="us-central1")
        self.curr_gen_config = {
            "max_output_tokens": 8192,
            "temperature": 1,
            "top_p": 1,
        }

        self.init_gen_config = self.curr_gen_config.copy()
        
        self.curr_model = GenerativeModel(
            "gemini-1.5-pro-preview-0514",
            system_instruction=[system_instructions]
        )

        self.curr_chat = self.curr_model.start_chat()

    def set_curr_temperature(self, temp: float):
        self.curr_gen_config["temperature"] = temp

    def set_curr_top_p(self, top_p: float):
        self.curr_gen_config["top_p"] = top_p
    
    def send_chat_message(self, message: str):
        return self.curr_chat.send_message(
            [message],
            generation_config=self.curr_gen_config,
            safety_settings=safety_settings
        ).candidates[0].content.parts[0].text
    
    def clear_chat(self):
        self.curr_chat = self.curr_model.start_chat()

    def reset_curr_gen_config(self):
        self.curr_gen_config = self.init_gen_config.copy()