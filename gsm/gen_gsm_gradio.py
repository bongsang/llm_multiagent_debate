import pandas as pd
from openai import OpenAI
import json
import numpy as np
import random
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
client = OpenAI()

def construct_message(agents, question, idx):
    if len(agents) == 0:
        return {"role": "user", "content": "Can you double check that your answer is correct? Please reiterate your answer, with your final answer a single numerical number, in the form \\boxed{{answer}}."}

    prefix_string = "These are the solutions to the problem from other agents: "

    for agent in agents:
        agent_response = agent[idx]["content"]
        response = "\n\n One agent solution: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + """\n\n Using the solutions from other agents as additional information, can you provide your answer to the math problem? \n The original math problem is {}. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response.""".format(question)
    return {"role": "user", "content": prefix_string}


def construct_assistant_message(completion):
    content = completion.choices[0].message.content
    return {"role": "assistant", "content": content}


def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]


def gradio_display(aggregated_data):
    # Create a DataFrame from the aggregated data
    df = pd.DataFrame(aggregated_data)

    # Apply styling to differentiate agents and evaluations
    styler = df.style.applymap(
        lambda val: "color: green" if "assistant" in val else "color: blue",
        subset=["Role"],
    )
    return styler


def main(num_agents, num_rounds, num_evals, model_name, selected_question=None):
    agents = num_agents
    rounds = num_rounds
    random.seed(0)

    generated_description = {}

    # Load questions from jsonl file
    questions = read_jsonl(BASE_DIR / "test.jsonl")

    # If a question is selected from the dropdown, use it; otherwise, shuffle questions
    if selected_question:
        selected_questions = [q for q in questions if q['question'] == selected_question]
    else:
        random.shuffle(questions)
        selected_questions = questions[:num_evals]

    aggregated_data = []  # To store the GSM data for display

    for eval_idx, data in enumerate(selected_questions):
        question = data['question']
        answer = data['answer']

        # Initialize agent contexts with the math question
        agent_contexts = [[{"role": "user", "content": f"Can you solve the following math problem? {question}. Explain your reasoning. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response."}] for agent in range(agents)]

        for round in range(num_rounds):
            for i, agent_context in enumerate(agent_contexts):

                if round != 0:
                    agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
                    message = construct_message(agent_contexts_other, question, 2*round - 1)
                    agent_context.append(message)

                completion = client.chat.completions.create(
                          model=model_name,
                          messages=agent_context,
                          n=1)

                assistant_message = construct_assistant_message(completion)
                agent_context.append(assistant_message)

        # Collect answers and append to aggregated data
        for i, agent_context in enumerate(agent_contexts):
            for message in agent_context:
                aggregated_data.append(
                    {
                        "Evaluation": eval_idx + 1,  # Current evaluation round
                        "Agent": f"Agent {i + 1}",  # Current agent number
                        "Role": message["role"],
                        "Content": message["content"],
                    }
                )

        generated_description[question] = (agent_contexts, answer)

    # Save to JSON for reference
    json.dump(generated_description, open(BASE_DIR / f"gsm_{agents}_{rounds}.json", "w"))

    # Return the display of the aggregated data
    return gradio_display(aggregated_data)
