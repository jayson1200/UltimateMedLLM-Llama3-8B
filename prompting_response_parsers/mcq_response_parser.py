class MCQPARSER:
    def __init__(self,
                 choices):
        self.choices = choices
    
    """
        @param generations_per_response takes a list of list of generations where each list of generations is a prompting response to one of a sequence of MCQs
    """
    def get_answers_for_each_question(self, generations_per_response: list[list[str]]):
        
        final_answers = []

        for generations in generations_per_response:
            curr_score = dict(zip(self.choices, [0] * len(self.choices)))

            for generation in generations:
                splits = generation.split("[Answer]")

                for choice in self.choices:
                    if choice in splits[-1]:
                        curr_score[choice] += 1
                        break
            
            final_answers.append(max(curr_score, key=curr_score.get))


        return final_answers