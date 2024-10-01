from openai import OpenAI
import json
import numpy as np
import time
import pickle
from tqdm import tqdm
import weave  # Import Weave
import wandb

client = OpenAI()
OPENAI_MODEL = "gpt-4o-mini"

# Initialize the Weave project
weave.init('multiagent_debate_math')

def parse_bullets(sentence):
    bullets_preprocess = sentence.split("\n")
    bullets = []

    for bullet in bullets_preprocess:
        try:
            idx = bullet.find(next(filter(str.isalpha, bullet)))
        except:
            continue

        bullet = bullet[idx:]

        if len(bullet) != 0:
            bullets.append(bullet)

    return bullets

# Weave will track the inputs, outputs, and code of this function
@weave.op()
def generate_answer(answer_context):
    try:
        completion = client.chat.completions.create(
            model=OPENAI_MODEL, messages=answer_context, n=1
        )
    except:
        print("retrying due to an error......")
        time.sleep(20)
        return generate_answer(answer_context)

    return completion

# Weave will track this function as well
@weave.op()
def construct_message(agents, question, idx):

    # Use introspection in the case in which there are no other agents.
    if len(agents) == 0:
        return {
            "role": "user",
            "content": "Can you verify that your answer is correct? Please reiterate your answer, making sure to state your answer at the end of the response.",
        }

    prefix_string = "These are the recent/updated opinions from other agents: "

    for agent in agents:
        agent_response = agent[idx]["content"]
        response = "\n\n One agent response: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    prefix_string = (
        prefix_string
        + "\n\n Use these opinions carefully as additional advice, can you provide an updated answer? Make sure to state your answer at the end of the response."
    )
    return {"role": "user", "content": prefix_string}

# Weave will track this function as well
@weave.op()
def construct_assistant_message(completion):
    content = completion.choices[0].message.content
    return {"role": "assistant", "content": content}

def parse_answer(sentence):
    parts = sentence.split(" ")

    for part in parts[::-1]:
        try:
            answer = float(part)
            return answer
        except:
            continue

def most_frequent(List):
    counter = 0
    num = List[0]

    for i in List:
        current_frequency = List.count(i)
        if current_frequency > counter:
            counter = current_frequency
            num = i

    return num

if __name__ == "__main__":
    
    answer = parse_answer(
        "My answer is the same as the other agents and AI language model: the result of 12+28*19+6 is 550."
    )

    agents = 2
    rounds = 3
    np.random.seed(0)

    # evaluation_round = 100
    evaluation_round = 1
    scores = []

    wandb.init(
        # Set the project where this run will be logged
        project="llm-multiagent_debate_math",
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        # Track hyperparameters and run metadata
        config={
        "agents": agents,
        "rounds": rounds,
        "evaluation_rounds": evaluation_round,
        })


    generated_description = {}

    for round in tqdm(range(evaluation_round)):
        a, b, c, d, e, f = np.random.randint(0, 30, size=6)

        answer = a + b * c + d - e * f
        agent_contexts = [
            [
                {
                    "role": "user",
                    "content": f"What is the result of {a}+{b}*{c}+{d}-{e}*{f}? Make sure to state your answer at the end of the response.",
                }
            ]
            for agent in range(agents)
        ]

        content = agent_contexts[0][0]["content"]
        question_prompt = f"We seek to find the result of {a}+{b}*{c}+{d}-{e}*{f}?"

        for round in range(rounds):
            for i, agent_context in enumerate(agent_contexts):

                if round != 0:
                    agent_contexts_other = agent_contexts[:i] + agent_contexts[i + 1 :]
                    message = construct_message(
                        agent_contexts_other, question_prompt, 2 * round - 1
                    )
                    agent_context.append(message)

                    print("message: ", message)

                completion = generate_answer(agent_context)

                assistant_message = construct_assistant_message(completion)
                agent_context.append(assistant_message)
                print(completion)

        text_answers = []

        for agent_context in agent_contexts:
            text_answer = agent_context[-1]["content"]
            text_answer = text_answer.replace(",", ".")
            text_answer = parse_answer(text_answer)

            if text_answer is None:
                continue

            text_answers.append(text_answer)

        generated_description[(a, b, c, d, e, f)] = (agent_contexts, answer)

        try:
            text_answer = most_frequent(text_answers)
            if text_answer == answer:
                scores.append(1)
            else:
                scores.append(0)
        except:
            continue

        print("performance:", np.mean(scores), np.std(scores) / (len(scores) ** 0.5))

    # Optionally, you can log the final performance to WandB
    wandb.log({"performance_mean": np.mean(scores), "performance_std": np.std(scores)})

    wandb.finish()
