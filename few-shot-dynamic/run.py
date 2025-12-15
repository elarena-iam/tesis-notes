from PromptBuilder import PromptBuilder

promptBuilder = PromptBuilder()

prompts = promptBuilder.process(5)

for prompt in prompts:
    print(prompt)
