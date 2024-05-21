PROJ_PATH = "/home/meribejayson/Desktop/Projects/UltimateMedLLM-Llama3-8B/"
import sys
sys.path.append(PROJ_PATH)

from simplified_llm_apis.model import LLM_Model
from simplified_llm_apis.gemini import Gemini

gemini = Gemini("You are a cat. Answer the following questions as best you can.")

print(gemini.send_chat_message("How are you?"))

print(gemini.send_chat_message("Do you want some milk?"))

class EnsembleRefinement:
    def __init__(self, system_instructions: str, model: LLM_Model) -> None:
        pass