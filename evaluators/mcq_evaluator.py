class MCQEvaluator:
    def __init__(self):
        self.num_questions_per_batch = []
        self.num_correct_per_batch = []

    
    def print_curr_acc(self):
        print(f"The current accuracy is {sum(self.num_correct_per_batch) / sum(self.num_questions_per_batch)}")

    def add_acc(self, correct_choices, responses):
        self.num_questions_per_batch.append(len(correct_choices))
        self.num_correct_per_batch.append(sum(1 for x, y in zip(correct_choices, responses) if x.lower() == y.lower()))
        
    