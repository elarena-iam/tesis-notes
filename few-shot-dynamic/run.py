from PromptBuilder import PromptBuilder

promptBuilder = PromptBuilder()

prompts, fewShotPrompts = promptBuilder.process(5)

for prompt in prompts:
    print(prompt)

for prompt in fewShotPrompts:
    print(prompt)
