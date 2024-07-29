from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from together import Together
import os


# Initialize the model
client = Together(api_key=os.getenv("TOGETHER_API_KEY"))

# Define the prompt templates for each technique

# Few-shot Prompting
few_shot_examples = [
    "Explain classical computing.",
    "Classical computing relies on bits, which can be either 0 or 1. Operations are performed on these bits using logic gates to process information."
]

few_shot_prompt = PromptTemplate(
    input_variables=["product"],
    template="Explain quantum computing.\nExamples:\n{examples}\n\nPlease explain quantum computing in a similar format.",
    examples=few_shot_examples
)

# Chain-of-Thought Prompting
chain_of_thought_prompt = PromptTemplate(
    input_variables=["product"],
    template="Let's break it down step-by-step. First, define what quantum computing is. Then, explain how it differs from classical computing. Finally, discuss its potential applications."
)

# Instruction Prompting
instruction_prompt = PromptTemplate(
    input_variables=["product"],
    template="Provide a detailed explanation of quantum computing. Include definitions, differences from classical computing, and potential applications. Ensure clarity and comprehensiveness."
)

# Define a function to generate responses using the model
def generate_response(prompt_template, context):
    prompt = prompt_template.format(product=context)
    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct-Turbo",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response["choices"][0]["message"]["content"]

# Example context for each technique
context_few_shot = "Explain quantum computing."
context_chain_of_thought = "Let's break it down step-by-step. First, define what quantum computing is. Then, explain how it differs from classical computing. Finally, discuss its potential applications."
context_instruction = "Provide a detailed explanation of quantum computing. Include definitions, differences from classical computing, and potential applications. Ensure clarity and comprehensiveness."

# Generate responses
response_few_shot = generate_response(few_shot_prompt, context_few_shot)
response_chain_of_thought = generate_response(chain_of_thought_prompt, context_chain_of_thought)
response_instruction = generate_response(instruction_prompt, context_instruction)

# Print responses
print("Few-Shot Prompting Response:\n", response_few_shot)
print("\nChain-of-Thought Prompting Response:\n", response_chain_of_thought)
print("\nInstruction Prompting Response:\n", response_instruction)
