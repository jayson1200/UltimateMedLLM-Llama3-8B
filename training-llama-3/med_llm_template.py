import torchtune.data.InstructTemplate

from typing import Any, Dict, Mapping, Optional

class MedLLMTemplate(InstructTemplate):
    template = {
        "mcq_prompt_input": (
            "Below is a multiple choice questions about medical knowledge. Solve it in a step-by-step fashion, starting by summarizing the available information."
            "Output a single option from the four options as the final answer. \n\n"
            "### Question:\n{question}\n\n### Options:\n{options}\n\n### Answer:\n"
        ),
        "open_end_prompt_input": (
            "You are a helpful medical knowledge assistant. Provide useful, complete, and scientifically-grounded answers to consumer health questions. \n\n"
            "### Question:\n{question}\n\n### Answer:\n"
        )
    }
    
    def format(
        cls, sample: Mapping[str, Any], column_map: Optional[Dict[str, str]] = None
    ) -> str:
        column_map = column_map or {}
        key_input = column_map.get("input", "input")
        key_options = column_map.get("options", "options")

        if key_input in sample and sample[key_input]:
            return cls.template["mcq_prompt_input"].format(
                question=sample[key_input],
                options=sample[key_options],
            )
        else:
            return cls.template["open_end_prompt_input"].format(
                question=sample[key_input],
            )