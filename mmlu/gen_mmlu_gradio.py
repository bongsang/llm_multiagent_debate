import pandas as pd
import json
import time
import random
from glob import glob
from openai import OpenAI
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
client = OpenAI()


def construct_message(agents, question, idx):
    if len(agents) == 0:
        return {
            "role": "user",
            "content": "Can you double check that your answer is correct? Put your final answer in the form (X) at the end of your response.",
        }

    prefix_string = "These are the solutions to the problem from other agents: "

    for agent in agents:
        agent_response = agent[idx]["content"]
        response = f"\n\n One agent solution: ```{agent_response}```"

        prefix_string += response

    prefix_string += (
        "\n\nUsing the reasoning from other agents as additional advice, can you give an updated answer? "
        "Examine your solution and that of other agents step by step. Put your answer in the form (X) at the end of your response."
    )
    return {"role": "user", "content": prefix_string}


def construct_assistant_message(completion):
    content = completion.choices[0].message.content
    return {"role": "assistant", "content": content}


def generate_answer(answer_context, model_name):
    try:
        completion = client.chat.completions.create(
            model=model_name, messages=answer_context, n=1
        )
    except:
        print("retrying due to an error......")
        time.sleep(20)
        return generate_answer(answer_context, model_name)
    return completion


def parse_question_answer(df, ix):
    question = df.iloc[ix, 0]
    a = df.iloc[ix, 1]
    b = df.iloc[ix, 2]
    c = df.iloc[ix, 3]
    d = df.iloc[ix, 4]

    question = f"Can you answer the following question as accurately as possible? {question}: A) {a}, B) {b}, C) {c}, D) {d}. Explain your answer, putting the answer in the form (X) at the end of your response."
    answer = df.iloc[ix, 5]

    return question, answer


def format_output(response_dict):
    output = ""
    for question, (agent_contexts, answer) in response_dict.items():
        output += f"### Question:\n**{question}**\n"
        output += f"- **Correct Answer**: {answer}\n\n"

        for i, agent_context in enumerate(agent_contexts):
            agent_name = f"Agent {i + 1}"
            initial_answer = agent_context[1]["content"]
            updated_answer = agent_context[-1]["content"]

            output += f"### {agent_name}:\n"
            output += f"- **Initial Answer**: {initial_answer}\n"
            output += f"- **Updated Answer**: {updated_answer}\n\n"

        output += f"### Final Consensus:\n"
        output += f"- **Correct Answer**: {answer}\n\n"
        output += "-" * 50 + "\n\n"

    return output


def main(num_agents, num_rounds, num_evals, model_name, *args):
    agents = num_agents
    rounds = num_rounds

    tasks = glob(str(BASE_DIR / "data" / "test" / "*.csv"))

    dfs = [pd.read_csv(task) for task in tasks]

    random.seed(0)
    response_dict = {}

    for i in range(num_evals):
        df = random.choice(dfs)
        ix = len(df)
        idx = random.randint(0, ix - 1)

        question, answer = parse_question_answer(df, idx)

        agent_contexts = [
            [{"role": "user", "content": question}] for _ in range(agents)
        ]

        for round in range(rounds):
            for i, agent_context in enumerate(agent_contexts):

                if round != 0:
                    agent_contexts_other = agent_contexts[:i] + agent_contexts[i + 1 :]
                    message = construct_message(
                        agent_contexts_other, question, 2 * round - 1
                    )
                    agent_context.append(message)

                completion = generate_answer(agent_context, model_name)
                assistant_message = construct_assistant_message(completion)
                agent_context.append(assistant_message)

        response_dict[question] = (agent_contexts, answer)

    # Store results in a JSON file
    json.dump(response_dict, open(f"mmlu_{agents}_{rounds}.json", "w"))

    # Format the output for better readability
    formatted_output = format_output(response_dict)

    print(formatted_output)
    return formatted_output
