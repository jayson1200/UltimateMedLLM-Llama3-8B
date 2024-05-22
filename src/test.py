PROJ_PATH = "/home/meribejayson/Desktop/Projects/UltimateMedLLM-Llama3-8B/"
import sys
sys.path.append(PROJ_PATH)

from prompting_response_parsers.mcq_response_parser import MCQPARSER
from evaluators.mcq_evaluator import MCQEvaluator
from prompting_strategies.ensemble_refinement import EnsembleRefinement
from simplified_llm_apis.gemini import Gemini

exp_q = """Question: A 21-year-old man presents to the physician with complaint of fever and non-bloody diarrhea for the past 3 days, after a week of constipation. He and his family recently returned from a summer spent in New Delhi, India visiting relatives. Physical examination reveals abdominal tenderness and a pink macular rash extending from his trunk to his upper arms. His vital signs are as follows: temperature is 99.7°F (37.6°C), blood pressure is 120/72 mmHg, pulse is 85/min, and respirations are 16/min. Which of the following drugs would be most effective in treating this patient’s condition?

Choices: A: Ciprofloxacin, B: Metronidazole, C: Oral rehydration solution, D: Oral vancomycin, E: Penicillin

Answer: A

Question: A 43-year-old man with a history of untreated HIV presents with fever, shortness of breath, and a nonproductive cough for the past week. Past medical history is significant for HIV diagnosed 10 years ago and never treated. His most recent CD4+ T cell count was 105/µL. Physical examination reveals bilateral crepitus over all lobes. No lymphadenopathy is present. A chest radiograph reveals bilateral infiltrates. Which of the following is the best treatment for this patient?

Choices: A: Highly active antiretroviral therapy (HAART), B: Trimethoprim-sulfamethoxazole, C: Ganciclovir, D: Azithromycin, E: Amphotericin B

Answer: B

Question: A 57-year-old man presents to his primary care physician with a 2-month history of right upper and lower extremity weakness. He noticed the weakness when he started falling far more frequently while running errands. Since then, he has had increasing difficulty with walking and lifting objects. His past medical history is significant only for well-controlled hypertension, but he says that some members of his family have had musculoskeletal problems. His right upper extremity shows forearm atrophy and depressed reflexes while his right lower extremity is hypertonic with a positive Babinski sign. Which of the following is most likely associated with the cause of this patient's symptoms?

Choices: A: HLA-B8 haplotype, B: HLA-DR2 haplotype, C: Mutation in SOD1, D: Mutation in SMN1, E: Viral infection

Answer:"""

correct_answer = "C"
choices = ["A", "B", "C", "D", "E"]

er = EnsembleRefinement(Gemini,
                        exp_q,
                        num_resp_first_stage=2,
                        num_resp_second_stage=3)
parser = MCQPARSER(choices)
evaluator = MCQEvaluator()

resp = er.run_ensemble_refinement()
print(resp)
ans = parser.get_answers_for_each_question([resp])
print(ans)
evaluator.add_acc([correct_answer],ans)
evaluator.print_curr_acc()