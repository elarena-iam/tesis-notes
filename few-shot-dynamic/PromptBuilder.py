from DevDataset import DevDataset
from PromptExamplesDB import PromptExamplesDB

class PromptBuilder:

    def __init__(self):
        self.devDataset = DevDataset("../dev.json")
        self.promptExamples = PromptExamplesDB()

    def process(self, n):

        dataset = self.devDataset.data
        improvedPrompts = []

        for sample in dataset:
            prompt = sample["nl"]
            improvedPrompt = self.promptExamples.search(prompt, n)
            improvedPrompts.append(improvedPrompt)

        return improvedPrompts
