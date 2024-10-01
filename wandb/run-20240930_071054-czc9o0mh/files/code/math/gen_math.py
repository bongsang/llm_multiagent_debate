import numpy as np
import time
from tqdm import tqdm
import weave
import wandb
from openai import OpenAI

client = OpenAI()  # Ensure your API key is set in the environment or pass it here
OPENAI_MODEL = "gpt-4o-mini"

# Initialize Weave project
weave.init('llm-multiagent-debate-math')

@weave.op()
def generate_answer(answer_context):
    try:
        completion = client.chat.completions.create(
            model=OPENAI_MODEL, messages=answer_context, n=1
        )
    except Exception as e:
        print(f"Error occurred: {e}\nRetrying after a short delay...")
        time.sleep(20)
        return generate_answer(answer_context)
    return completion

@weave.op()
def construct_message(agents, idx):
    # Use introspection in the case in which there are no other agents.
    if len(agents) == 0:
        return {
            "role": "user",
            "content": (
                "Can you verify that your answer is correct? Please reiterate your "
                "answer, making sure to state your answer at the end of the response."
            ),
        }

    prefix_string = "These are the recent/updated opinions from other agents:"

    for agent in agents:
        agent_response = agent[idx]["content"]
        response = f"\n\nOne agent response: ```{agent_response}```"
        prefix_string += response

    prefix_string += (
        "\n\nUse these opinions carefully as additional advice. Can you provide an "
        "updated answer? Make sure to state your answer at the end of the response."
    )
    return {"role": "user", "content": prefix_string}

@weave.op()
def construct_assistant_message(completion):
    content = completion.choices[0].message.content
    return {"role": "assistant", "content": content}

def parse_answer(sentence):
    parts = sentence.split(" ")
    for part in reversed(parts):
        try:
            answer = float(part)
            return answer
        except ValueError:
            continue
    return None

def most_frequent(lst):
    if not lst:
        return None
    return max(set(lst), key=lst.count)

if __name__ == "__main__":
    agents = 2
    rounds = 3
    np.random.seed(0)

    evaluation_rounds = 1  # Set this to the desired number of evaluation rounds
    scores = []

    # Initialize WandB
    wandb.init(
        project="llm-multiagent-debate-math",
        config={
            "agents": agents,
            "rounds": rounds,
            "evaluation_rounds": evaluation_rounds,
        },
    )

    generated_description = {}

    for eval_round in tqdm(range(evaluation_rounds)):
        a, b, c, d, e, f = np.random.randint(0, 30, size=6)

        correct_answer = a + b * c + d - e * f
        initial_question = {
            "role": "user",
            "content": (
                f"What is the result of {a}+{b}*{c}+{d}-{e}*{f}? "
                "Make sure to state your answer at the end of the response."
            ),
        }

        agent_contexts = [[initial_question] for _ in range(agents)]

        for round_num in range(rounds):
            for i, agent_context in enumerate(agent_contexts):
                if round_num != 0:
                    # Prepare messages from other agents
                    agent_contexts_other = agent_contexts[:i] + agent_contexts[i + 1 :]
                    message = construct_message(agent_contexts_other, 2 * round_num - 1)
                    agent_context.append(message)

                completion = generate_answer(agent_context)
                assistant_message = construct_assistant_message(completion)
                agent_context.append(assistant_message)

        text_answers = []

        for agent_context in agent_contexts:
            text_answer = agent_context[-1]["content"].replace(",", ".")
            parsed_answer = parse_answer(text_answer)
            if parsed_answer is not None:
                text_answers.append(parsed_answer)

        generated_description[(a, b, c, d, e, f)] = (agent_contexts, correct_answer)

        final_answer = most_frequent(text_answers)
        if final_answer == correct_answer:
            scores.append(1)
        else:
            scores.append(0)

        performance_mean = np.mean(scores)
        performance_std = np.std(scores) / (len(scores) ** 0.5) if len(scores) > 1 else 0
        print(f"Round {eval_round + 1} Performance:", performance_mean, performance_std)

        # Log performance metrics to WandB
        wandb.log(
            {
                "evaluation_round": eval_round + 1,
                "performance_mean": performance_mean,
                "performance_std": performance_std,
            }
        )

    # Finish the WandB run
    wandb.finish()
