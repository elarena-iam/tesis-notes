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
        improvedPromptsFewShorLearning = []
        resultManager = ResultManager()

        for sample in dataset:
            prompt = sample["nl"].split("concode")[0]
            improvedPrompt, fewShotLearning = self.promptExamples.search(prompt, n)
            improvedPrompts.append(improvedPrompt)
            result = self.llmHandler.call(improvedPrompt)
            fewShotLearningImprovedPrompt = self.llmHandler.call(fewShotLearning)
            fewShotLearningImprovedPromptResult = self.llmHandler.call(fewShotLearningImprovedPrompt)
            improvedPromptsFewShorLearning.append(fewShotLearningImprovedPrompt)
            resultManager.register(sample, improvedPrompt, result)
            resultManager.register(sample, f"prompt:\n{fewShotLearningImprovedPrompt}\n\nget improved prompt:\n{fewShotLearning}", fewShotLearningImprovedPromptResult)

        return improvedPrompts, improvedPromptsFewShorLearning
