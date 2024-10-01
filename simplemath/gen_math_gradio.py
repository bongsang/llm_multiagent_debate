import pandas as pd
from openai import OpenAI
import numpy as np
import time
from tqdm import tqdm

client = OpenAI()


def generate_answer(answer_context, model_name):
    try:
        completion = client.chat.completions.create(
            model=model_name, messages=answer_context, n=1
        )
    except:
        print("retrying due to an error......")
        time.sleep(20)
        return generate_answer(answer_context)
    return completion


def construct_message(agents, idx):
    """
    Construct the message for an agent by incorporating the answers from the other agents.
    Agents receive the responses from all previous agents in the current round.
    """
    if len(agents) == 0:
        # First agent (or no previous agent responses) - introspective check
        return {
            "role": "user",
            "content": "Can you verify that your answer is correct. Please reiterate your answer, making sure to state your answer at the end of the response.",
        }

    # Combine responses from previous agents
    prefix_string = "These are the recent/updated opinions from other agents:"

    for agent in agents:
        agent_response = agent[idx]["content"]
        response = f"\n\nOne agent response: ```{agent_response}```"
        prefix_string += response

    # Ask the agent to update their answer considering previous responses
    prefix_string += "\n\nPlease provide an updated answer, considering the above inputs. State your final answer at the end."
    return {"role": "user", "content": prefix_string}


def construct_assistant_message(completion, previous_answer=None):
    """
    Construct the assistant message and decide whether to apologize or confirm.
    """
    content = completion.choices[0].message.content

    # Check if the previous answer is the same as the new one
    if previous_answer is not None and previous_answer.strip() == content.strip():
        return {
            "role": "assistant",
            "content": f"My previous answer was correct. The result remains: {content}",
        }

    # If the answers differ, provide the new calculation
    return {"role": "assistant", "content": content}


def parse_answer(sentence):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Extracting the answer number from the given sentence. You must return only the number.",
            },
            {
                "role": "user",
                "content": sentence,
            },
        ],
        n=1,
    )
    answer_str = response.choices[0].message.content
    return float(answer_str)


def most_frequent(List):
    counter = 0
    num = List[0]
    for i in List:
        current_frequency = List.count(i)
        if current_frequency > counter:
            counter = current_frequency
            num = i
    return num


def gradio_display(aggregated_data):
    # Create a DataFrame from the aggregated data
    df = pd.DataFrame(aggregated_data)

    # Apply styling to differentiate agents and evaluations
    styler = df.style.applymap(
        lambda val: "color: green" if "assistant" in val else "color: blue",
        subset=["Role"],
    )
    return styler


def main(num_agents, num_rounds, num_evals, model_name):
    agents = num_agents
    rounds = num_rounds
    np.random.seed(0)

    evaluation_round = num_evals
    scores = []
    aggregated_data = []

    for eval_idx in tqdm(range(evaluation_round)):
        # Generate random math question with 6 numbers
        a, b, c, d, e, f = np.random.randint(0, 30, size=6)
        answer = a + b * c + d - e * f  # Ground truth answer
        question_prompt = f"What is the result of {a}+{b}*{c}+{d}-{e}*{f}?"

        # Initialize agent contexts with the math question
        agent_contexts = [
            [{"role": "user", "content": question_prompt}] for _ in range(agents)
        ]

        previous_answer = None  # Track the previous answer for comparison

        for round_idx in range(rounds):
            for i, agent_context in enumerate(agent_contexts):
                if round_idx != 0:
                    # Collect responses from all previous agents in the current round
                    previous_agents = agent_contexts[:i]
                    # Construct message based on previous agents' responses
                    message = construct_message(previous_agents, 2 * round_idx - 1)
                    agent_context.append(message)

                # Generate assistant's response
                completion = generate_answer(agent_context, model_name)
                assistant_message = construct_assistant_message(
                    completion, previous_answer
                )
                agent_context.append(assistant_message)

                # Update previous answer for the next round's comparison
                previous_answer = assistant_message["content"]

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

        # Evaluate performance
        text_answers = []
        for agent_context in agent_contexts:
            text_answer = agent_context[-1]["content"]
            parsed_answer = parse_answer(text_answer)
            if parsed_answer is not None:
                text_answers.append(parsed_answer)

        try:
            most_frequent_answer = most_frequent(text_answers)
            if most_frequent_answer == answer:
                scores.append(1)
            else:
                scores.append(0)
        except Exception as e:
            print(f"Error in answer evaluation during round {eval_idx + 1}: {e}")
            continue

        print(
            f"Performance after evaluation {eval_idx + 1}: {np.mean(scores):.4f} (std error: {np.std(scores) / (len(scores) ** 0.5):.4f})"
        )

    # Final performance summary
    final_performance = np.mean(scores)
    print(f"Final performance over all evaluations: {final_performance:.4f}")

    # Return the display of the aggregated data from all evaluations
    print("Aggregated data:", aggregated_data)
    return gradio_display(aggregated_data)
