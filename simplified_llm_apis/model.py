import abc
from abc import ABC, abstractmethod

class LLM_Model:
    
    @abstractmethod
    def __init__(self, system_instructions: str) -> None:
        pass

    @abstractmethod
    def set_curr_temperature(self, temp: float):
        pass

    @abstractmethod
    def set_curr_top_p(self, top_p: float):
        pass

    @abstractmethod
    def send_chat_message(self, message: str) -> str:
        pass
    
    @abstractmethod
    def clear_chat(self):
        pass
    
    @abstractmethod
    def reset_curr_gen_config(self):
        pass