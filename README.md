# UltimateMedLLM-Llama3-8B

A medical question answering LLM built

Benchmarking example usage:

```python
PROJ_PATH = "/home/meribejayson/Desktop/Projects/UltimateMedLLM-Llama3-8B/"
import sys
sys.path.append(PROJ_PATH)

from prompting_response_parsers.mcq_response_parser import MCQPARSER
from evaluators.mcq_evaluator import MCQEvaluator
from prompting_strategies.ensemble_refinement import EnsembleRefinement
from simplified_llm_apis.gemini import Gemini


exps = """[Question]
A 21-year-old man presents to the physician with complaint of fever and non-bloody diarrhea for the past 3 days, after a week of constipation. He and his family recently returned from a summer spent in New Delhi, India visiting relatives. Physical examination reveals abdominal tenderness and a pink macular rash extending from his trunk to his upper arms. His vital signs are as follows: temperature is 99.7°F (37.6°C), blood pressure is 120/72 mmHg, pulse is 85/min, and respirations are 16/min. Which of the following drugs would be most effective in treating this patient’s condition?

[Choices]
A: Ciprofloxacin, B: Metronidazole, C: Oral rehydration solution, D: Oral vancomycin, E: Penicillin

[Summary]
A 21-year-old man presents with fever and diarrhea after constipation. After returning from a summer trip, he shows abdominal tenderness and a pink macular rash. Vital signs include temperature, blood pressure, pulse, and respirations.

[Explanation]
In the context of recent travel to India and the described symptoms, this presentation is highly suggestive of typhoid fever, which is caused by Salmonella typhi. Typhoid fever commonly presents with a stepwise fever, abdominal pain, and a characteristic "rose spots" rash, which are pink macules that typically appear on the trunk and spread to the extremities.
The most effective treatment for typhoid fever in this scenario is Ciprofloxacin (Choice A), a fluoroquinolone antibiotic. Ciprofloxacin is commonly used to treat enteric fever, including typhoid fever, due to its efficacy against Salmonella typhi and its ability to reach high intracellular concentrations.
Other Choices:
Metronidazole (Choice B) is primarily used to treat infections caused by anaerobic bacteria and certain protozoa, such as Giardia and Entamoeba histolytica, but is not effective against Salmonella typhi.
Oral rehydration solution (Choice C) is important for managing dehydration associated with diarrhea but does not address the underlying bacterial infection.
Oral vancomycin (Choice D) is used for treating Clostridioides difficile infections and is not effective against Salmonella typhi.
Penicillin (Choice E) is not the first-line treatment for typhoid fever and is less effective against Salmonella typhi compared to ciprofloxacin.

Given the patient’s travel history, clinical presentation, and the need for effective antibiotic therapy, Ciprofloxacin (Choice A) is the most appropriate treatment for this patient's condition.

[Answer]
A

[Question]
A 43-year-old man with a history of untreated HIV presents with fever, shortness of breath, and a nonproductive cough for the past week. Past medical history is significant for HIV diagnosed 10 years ago and never treated. His most recent CD4+ T cell count was 105/µL. Physical examination reveals bilateral crepitus over all lobes. No lymphadenopathy is present. A chest radiograph reveals bilateral infiltrates. Which of the following is the best treatment for this patient?

[Choices]
A: Highly active antiretroviral therapy (HAART), B: Trimethoprim-sulfamethoxazole, C: Ganciclovir, D: Azithromycin, E: Amphotericin B
[Summary]
A 21-year-old man presents with fever, diarrhea, abdominal tenderness, and a pink macular rash after constipation, requiring monitoring of vital signs after a summer trip.

[Explanation]
This 43-year-old man with untreated HIV presents with fever, shortness of breath, and a nonproductive cough, symptoms that have been ongoing for the past week. His CD4+ T cell count is notably low at 105/µL, indicating severe immunosuppression. Physical examination reveals bilateral crepitus over all lung lobes, and a chest radiograph shows bilateral infiltrates.
In the context of HIV/AIDS and a CD4+ T cell count below 200/µL, these symptoms are highly suggestive of Pneumocystis jirovecii pneumonia (PJP), a common opportunistic infection in immunocompromised individuals, particularly those with untreated HIV/AIDS.
The standard treatment for PJP is Trimethoprim-sulfamethoxazole (Bactrim). This antibiotic is effective against Pneumocystis jirovecii and is the first-line treatment for this condition.
Other Choices:
Highly active antiretroviral therapy (HAART) (Choice A) is crucial for long-term management of HIV/AIDS but is not the immediate treatment for the acute infection described here.
Ganciclovir (Choice C) is an antiviral medication used to treat cytomegalovirus (CMV) infections, not typically for PJP.
Azithromycin (Choice D) is an antibiotic that might be used for bacterial infections or prophylaxis for Mycobacterium avium complex (MAC) but is not the treatment of choice for PJP.
Amphotericin B (Choice E) is an antifungal medication used for severe fungal infections such as cryptococcal meningitis, not typically for PJP.
Thus, given the patient's symptoms, history, and radiographic findings, the best treatment option is Trimethoprim-sulfamethoxazole (B).

[Answer]
B"""


exp_q = """
[Question]
A 57-year-old man presents to his primary care physician with a 2-month history of right upper and lower extremity weakness. He noticed the weakness when he started falling far more frequently while running errands. Since then, he has had increasing difficulty with walking and lifting objects. His past medical history is significant only for well-controlled hypertension, but he says that some members of his family have had musculoskeletal problems. His right upper extremity shows forearm atrophy and depressed reflexes while his right lower extremity is hypertonic with a positive Babinski sign. Which of the following is most likely associated with the cause of this patient's symptoms?

[Choices]
A: HLA-B8 haplotype, B: HLA-DR2 haplotype, C: Mutation in SOD1, D: Mutation in SMN1, E: Viral infection
"""

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
