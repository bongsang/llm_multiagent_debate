import json
from openai import OpenAI
import random
from tqdm import tqdm
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
client = OpenAI()


def parse_bullets(sentence):
    bullets_preprocess = sentence.split("\n")
    bullets = []

    for bullet in bullets_preprocess:
        try:
            idx = bullet.find(next(filter(str.isalpha, bullet)))
        except:
            continue

        bullet = bullet[idx:].strip()  # Ensure no extra spaces

        if len(bullet) != 0:
            bullets.append(bullet)

    return bullets


def filter_people(person):
    people = person.split("(")[0]
    return people.strip()


def construct_message(agents, idx, person, final=False):
    prefix_string = (
        f"Here are some bullet point biographies of {person} given by other agents: "
    )

    if len(agents) == 0:
        return {
            "role": "user",
            "content": f"Closely examine your biography of {person} and provide an updated bullet point biography.",
        }

    for i, agent in enumerate(agents):
        agent_response = agent[idx]["content"]
        response = f"\n\n Agent response: ```{agent_response}```"
        prefix_string += response

    if final:
        prefix_string += f"\n\n Closely examine your biography of {person} and provide an updated bullet point biography."
    else:
        prefix_string += f"\n\n Using these other biographies of {person} as additional advice, what is your updated bullet point biography of {person}?"

    return {"role": "user", "content": prefix_string}


def construct_assistant_message(completion):
    content = completion.choices[0].message.content
    return {"role": "assistant", "content": content}


def format_for_dataframe(generated_description):
    # Convert the generated_description dictionary into a list of dictionaries suitable for a DataFrame
    rows = []
    for person, agent_contexts in generated_description.items():
        for agent_idx, contexts in enumerate(agent_contexts):
            for message in contexts:
                rows.append(
                    {
                        "Person": person,
                        "Agent": f"Agent {agent_idx + 1}",
                        "Role": message["role"],
                        "Content": message["content"],
                    }
                )
    return pd.DataFrame(rows)


def main(num_agents, num_rounds, num_evals, model_name, selected_person=None):
    with open(BASE_DIR / "article.json", "r") as f:
        data = json.load(f)

    if selected_person:
        people = [
            selected_person
        ]  # If a specific person is selected, use only that person
    else:
        people = sorted(data.keys())
        people = [filter_people(person) for person in people]
        random.seed(1)
        random.shuffle(people)

    agents = num_agents
    rounds = num_rounds
    generated_description = {}

    for person in tqdm(people[:num_evals]):
        agent_contexts = [
            [
                {
                    "role": "user",
                    "content": f"Give a bullet point biography of {person}, highlighting their contributions and achievements as a computer scientist, with each fact separated by a new line character.",
                }
            ]
            for agent in range(agents)
        ]

        for round in range(rounds):
            for i, agent_context in enumerate(agent_contexts):
                if round != 0:
                    agent_contexts_other = agent_contexts[:i] + agent_contexts[i + 1 :]

                    if round == (rounds - 1):
                        message = construct_message(
                            agent_contexts_other,
                            2 * round - 1,
                            person=person,
                            final=True,
                        )
                    else:
                        message = construct_message(
                            agent_contexts_other,
                            2 * round - 1,
                            person=person,
                            final=False,
                        )

                    agent_context.append(message)

                completion = client.chat.completions.create(
                    model=model_name, messages=agent_context, n=1
                )

                assistant_message = construct_assistant_message(completion)
                agent_context.append(assistant_message)

            bullets = parse_bullets(completion.choices[0].message.content)

            if (
                len(bullets) == 1
            ):  # Stop if only one bullet point is generated (indicates low info)
                break

        generated_description[person] = agent_contexts

    json.dump(generated_description, open(f"biography_{agents}_{rounds}.json", "w"))

    # Convert the generated_description to DataFrame
    return format_for_dataframe(generated_description)
