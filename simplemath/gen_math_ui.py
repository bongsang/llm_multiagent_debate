import numpy as np
import pandas as pd
import time
import openai
import gradio as gr
import re

# Sample questions from the paper
sample_questions = [
    "What is the result of 10+20*23+3-11*18?",
    "What is the result of 3+7*9+19-21*18?",
    "What is the result of 4+23*6+24-24*12?",
    "What is the result of 8+14*15+20-3*26?",
]

# List of available models
available_models = [
    "gpt-3.5-turbo",
    "gpt-4",
    "gpt-4-turbo",
    "GPT-4o",
    "GPT-4o-mini",
]

def generate_answer(answer_context, model_name):
    try:
        completion = openai.chat.completions.create(
            model=model_name, messages=answer_context, n=1
        )
    except Exception as e:
        print(f"Error occurred: {e}\nRetrying after a short delay...")
        time.sleep(20)
        return generate_answer(answer_context, model_name)
    return completion

def construct_message(agents, idx):
    # Use introspection if there are no other agents.
    if len(agents) == 0:
        return {
            "role": "user",
            "content": (
                "Can you verify that your answer is correct. Please reiterate your answer, "
                "making sure to state your answer at the end of the response."
            ),
        }

    prefix_string = "These are the recent/updated opinions from other agents:"

    for agent in agents:
        agent_response = agent[idx]["content"]
        response = f"\n\n One agent response: ```{agent_response}```"
        prefix_string += response

    prefix_string += (
        "\n\n Use these opinions carefully as additional advice, can you provide an updated answer? "
        "Make sure to state your answer at the end of the response."
    )
    return {"role": "user", "content": prefix_string}

def construct_assistant_message(completion):
    content = completion.choices[0].message.content
    return {"role": "assistant", "content": content}

def parse_answer(sentence):
    parts = sentence.replace(",", ".").split()
    for part in reversed(parts):
        try:
            # Remove any trailing punctuation
            part = part.rstrip(".")
            answer = float(part)
            return answer
        except ValueError:
            continue
    return None

def calculate_correct_answer(problem_statement):
    # Extract the expression from the problem statement
    match = re.search(r'What is the result of (.+?)\?', problem_statement)
    if match:
        expression = match.group(1)
        try:
            correct_answer = eval(expression)
            return correct_answer
        except Exception as e:
            print(f"Error evaluating expression: {e}")
            return None
    return None

def run_simulation(api_key, agents, rounds, model_name, question_option, custom_question):
    openai.api_key = api_key
    agents = int(agents)
    rounds = int(rounds)

    # Select the question
    if question_option == "Custom":
        problem_statement = custom_question.strip()
        if not problem_statement:
            return (
                "Please enter a custom question.",
                None,
                None
            )
    else:
        problem_statement = question_option

    correct_answer = calculate_correct_answer(problem_statement)

    initial_question = {
        "role": "user",
        "content": problem_statement + " Make sure to state your answer at the end of the response."
    }

    agent_contexts = [[initial_question] for _ in range(agents)]
    agent_answers = [[] for _ in range(agents)]  # To store each agent's answers per round

    for round_num in range(rounds):
        for i, agent_context in enumerate(agent_contexts):
            if round_num != 0:
                # Prepare messages from other agents
                agent_contexts_other = agent_contexts[:i] + agent_contexts[i + 1 :]
                message = construct_message(agent_contexts_other, -1)
                agent_context.append(message)
            else:
                # In the first round, we don't provide other agents' responses
                message = None

            completion = generate_answer(agent_context, model_name)
            assistant_message = construct_assistant_message(completion)
            agent_context.append(assistant_message)

            # Parse and store the agent's answer for this round
            text_answer = assistant_message["content"]
            parsed_answer = parse_answer(text_answer)
            agent_answers[i].append(parsed_answer)

    # Collect the agents' final answers
    final_answers = [answers[-1] if answers else "No answer" for answers in agent_answers]

    # Build a result dictionary
    result = {
        "problem": problem_statement,
        "correct_answer": correct_answer,
        "agents_chat_history": agent_contexts,
        "agents_answers": agent_answers,
        "final_answers": final_answers,
    }

    return display_results(result, agents, rounds)

def display_results(result, agents, rounds):
    problem_md = f"### Problem:\n{result['problem']}\n\n"
    problem_md += f"**Correct Answer:** {result['correct_answer']}\n\n"

    # Build agents' answer histories DataFrame
    answer_history_df = pd.DataFrame(result['agents_answers']).T
    answer_history_df.columns = [f"Agent {i+1}" for i in range(agents)]
    answer_history_df.index = [f"Round {i+1}" for i in range(rounds)]

    # Build agents' chat histories as HTML
    chat_histories_html = ""
    for idx, agent_history in enumerate(result['agents_chat_history']):
        chat_histories_html += f"<h3>Agent {idx+1} Chat History:</h3>"
        chat_histories_html += "<div style='margin-left: 20px;'>"
        for message in agent_history:
            role = message["role"].capitalize()
            content = message["content"].replace('\n', '<br>')
            chat_histories_html += f"<b>{role}:</b> {content}<br><br>"
        chat_histories_html += "</div>"

    return problem_md, answer_history_df, chat_histories_html

def main(api_key, agents, rounds, model_name, question_option, custom_question):
    return run_simulation(api_key, agents, rounds, model_name, question_option, custom_question)
