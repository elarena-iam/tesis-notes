from DevDataset import DevDataset
from PromptExamplesDB import PromptExamplesDB
from LlmHandler import LlmHandler
from ResultManager import ResultManager

class PromptBuilder:

    def __init__(self):
        self.devDataset = DevDataset("../dev.json")
        self.promptExamples = PromptExamplesDB()
        self.llmHandler = LlmHandler()

    def process(self, n):

        dataset = self.devDataset.data
        improvedPrompts = []
        resultManager = ResultManager()

        for sample in dataset:
            prompt = sample["nl"].split("concode")[0]
            improvedPrompt = self.promptExamples.search(prompt, n)
            improvedPrompts.append(improvedPrompt)
            result = self.llmHandler.call(improvedPrompt)
            resultManager.register(sample, improvedPrompt, result)

        return improvedPrompts
