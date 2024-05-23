# UltimateMedLLM-Llama3-8B

A medical question answering LLM built

## Benchmarking example:

```python
PROJ_PATH = "/home/meribejayson/Desktop/Projects/UltimateMedLLM-Llama3-8B/"
import sys
sys.path.append(PROJ_PATH)

from prompting_response_parsers.mcq_response_parser import MCQPARSER
from evaluators.mcq_evaluator import MCQEvaluator
from prompting_strategies.ensemble_refinement import EnsembleRefinement
from simplified_llm_apis.gemini import Gemini


exps = """[Question]
A 21-year-old man presents to the physician with complaint of fever and non-bloody ..."""


exp_q = """
[Question]
A 57-year-old man presents to his primary ..."""

correct_answer = "C"
choices = ["A", "B", "C", "D", "E"]

er = EnsembleRefinement(Gemini,
                        exps + "\n\n" + exp_q,
                        num_resp_first_stage=3,
                        num_resp_second_stage=4)
parser = MCQPARSER(choices)
evaluator = MCQEvaluator()

resp = er.run_ensemble_refinement()
print(resp[0])
ans = parser.get_answers_for_each_question([resp])
print(ans)
evaluator.add_acc([correct_answer],ans)
evaluator.print_curr_acc()
```
