PROJ_PATH = "/home/meribejayson/Desktop/Projects/UltimateMedLLM-Llama3-8B/"
STEP_ONE_SYS_PROMPT = """The following are multiple choice questions about medical knowledge. Solve them in a step-by-step fashion,
starting by summarizing the available information. Then provide an explanation of your reasoning. Then provide a single option from the four options as the final answer. 
We provide two example prompts, summaries, explanations, and answers, fill in the last example continuing from what was provided."""

STEP_TWO_SYS_PROMPT = """The following are multiple choice questions about medical knowledge. Solve them in a step-by-step fashion,
starting by summarizing the available information. Then provide an explanation of your reasoning. Then provide the answer. 
Output a single option from the four options as the final answer. We provide several student reasonings for the last question as well as two example prompts, summaries, explanations, and answers. 
Some of them may be correct and some incorrect. You can use the best correct arguments from these reasonings. 
Beware of wrong reasoning and do not repeat wrong reasoning. We provide two example prompts, summaries, explanations, and answers, fill in the last example continuing from what was provided."""

import sys
import numpy as np
sys.path.append(PROJ_PATH)

from simplified_llm_apis.model import LLM_Model

class EnsembleRefinement:
    def __init__(self, 
                 model: LLM_Model, 
                 prompt: str, 
                 num_resp_first_stage=11, 
                 num_resp_second_stage=33) -> None:
        
        self.step_one_model = model(STEP_ONE_SYS_PROMPT)
        self.step_two_model = model(STEP_TWO_SYS_PROMPT)
        self.num_resp_first_stage = num_resp_first_stage
        self.num_resp_second_stage = num_resp_second_stage
        self.prompt = prompt
        self.first_stage_concat_out = """"""

        self.second_stage_resp = []
        
    def run_first_stage(self): 
        for i in range(self.num_resp_first_stage):
            self.step_one_model.set_curr_temperature(np.random.uniform(0.0, 2.0))
            self.first_stage_concat_out += ("\n" + self.step_one_model.send_chat_message(self.prompt))
            self.step_one_model.reset_curr_gen_config()

    def run_second_stage(self):
        model_input = "Student Reasonings:\n" + self.first_stage_concat_out + "\n" + self.prompt

        for i in range(self.num_resp_second_stage):
            self.step_two_model.clear_chat()
            self.second_stage_resp.append(self.step_two_model.send_chat_message(model_input))

        return self.second_stage_resp
    
    def run_ensemble_refinement(self):
        self.run_first_stage()
        return self.run_second_stage()