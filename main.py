import pandas as pd
import gradio as gr
import json
from simplemath import gen_math_gradio
from gsm import gen_gsm_gradio
from mmlu import gen_mmlu_gradio
from biography import gen_conversation_gradio
from pathlib import Path
from glob import glob

BASE_DIR = Path(__file__).resolve().parent

# List of available models
available_models = [
    "gpt-3.5-turbo",
    "gpt-4",
    "gpt-4-turbo",
    "gpt-4o",
    "gpt-4o-mini",
]


def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]


# Load GSM questions from test.jsonl for dropdown
gsm_questions = [
    q["question"] for q in read_jsonl(BASE_DIR / "gsm" / "data" / "test.jsonl")
]


# Function to get CSV file names and map to categories for MMLU
def get_mmlu_categories():
    csv_files = glob(str(BASE_DIR / "mmlu" / "data" / "test" / "*.csv"))
    categories = [
        Path(f).stem for f in csv_files
    ]  # Get the file name without extension
    return categories, csv_files


# Function to load questions from a selected CSV file
def load_questions_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    questions = df.iloc[
        :, 0
    ].tolist()  # Assuming the first column contains the questions
    return questions


# Load MMLU categories
mmlu_categories, mmlu_csv_files = get_mmlu_categories()


def load_people_from_article():
    with open(BASE_DIR / "biography" / "article.json", "r") as f:
        data = json.load(f)
    return list(data.keys())  # Return the names of people from the keys


people_list = load_people_from_article()

# Build the Gradio interface using Tabs
with gr.Blocks() as demo:
    gr.Markdown(
        "# Multiagent Debate WebUI"
    )
    with gr.Tabs():
        with gr.Tab("Math"):
            with gr.Row():
                # Sliders for input configurations
                agents_slider = gr.Slider(
                    minimum=1, maximum=5, step=1, value=2, label="Number of Agents"
                )
                rounds_slider = gr.Slider(
                    minimum=1, maximum=5, step=1, value=2, label="Number of Rounds"
                )
                evals_slider = gr.Slider(
                    minimum=1,
                    maximum=100,
                    step=1,
                    value=1,
                    label="Number of Evaluations",
                )

            # Dropdown for model selection
            model_dropdown = gr.Dropdown(
                choices=available_models,
                value=available_models[0],
                label="Select OpenAI Model",
            )

            # Dataframe output for agent interaction history
            answer_history = gr.Dataframe(
                headers=["Evaluation", "Agent", "Role", "Content"],
                label="Agent Interaction History",
                interactive=False,  # Ensure it's non-interactive to allow styling
            )

            # Submit button to trigger the function
            submit_button = gr.Button("Submit")

            # Link the button click to the main function from gen_math_gradio
            submit_button.click(
                fn=gen_math_gradio.main,
                inputs=[agents_slider, rounds_slider, evals_slider, model_dropdown],
                outputs=[answer_history],
            )

        with gr.Tab("GSM"):
            with gr.Row():
                # Sliders for GSM input configurations
                gsm_agents_slider = gr.Slider(
                    minimum=1, maximum=5, step=1, value=2, label="Number of Agents"
                )
                gsm_rounds_slider = gr.Slider(
                    minimum=1, maximum=5, step=1, value=2, label="Number of Rounds"
                )
                gsm_evals_slider = gr.Slider(
                    minimum=1,
                    maximum=100,
                    step=1,
                    value=1,
                    label="Number of Evaluations",
                )

            # Dropdown for GSM model selection
            gsm_model_dropdown = gr.Dropdown(
                choices=available_models,
                value=available_models[0],
                label="Select OpenAI Model",
            )

            # New Dropdown for selecting GSM questions from test.jsonl
            gsm_question_dropdown = gr.Dropdown(
                choices=gsm_questions,
                label="Select GSM Question",
            )

            # Dataframe output for GSM agent interaction history
            gsm_answer_history = gr.Dataframe(
                headers=["Evaluation", "Agent", "Role", "Content"],
                label="GSM Agent Interaction History",
                interactive=False,
            )

            # Submit button to trigger the function for GSM
            gsm_submit_button = gr.Button("Submit")

            # Link the button click to the main function from gen_gsm_gradio
            gsm_submit_button.click(
                fn=gen_gsm_gradio.main,
                inputs=[
                    gsm_agents_slider,
                    gsm_rounds_slider,
                    gsm_evals_slider,
                    gsm_model_dropdown,
                    gsm_question_dropdown,
                ],
                outputs=[gsm_answer_history],
            )

        # MMLU Tab for Massive Multitask Language Understanding
        with gr.Tab("MMLU"):
            with gr.Row():
                # Sliders for MMLU input configurations
                mmlu_agents_slider = gr.Slider(
                    minimum=1, maximum=5, step=1, value=2, label="Number of Agents"
                )
                mmlu_rounds_slider = gr.Slider(
                    minimum=1, maximum=5, step=1, value=2, label="Number of Rounds"
                )
                mmlu_evals_slider = gr.Slider(
                    minimum=1,
                    maximum=100,
                    step=1,
                    value=1,
                    label="Number of Evaluations",
                )

            # Dropdown for MMLU model selection
            mmlu_model_dropdown = gr.Dropdown(
                choices=available_models,
                value=available_models[0],
                label="Select OpenAI Model",
            )

            # Dropdown for MMLU category (CSV file names)
            mmlu_category_dropdown = gr.Dropdown(
                choices=mmlu_categories,
                label="Select MMLU Category",
            )

            # Dropdown for MMLU questions (this will be populated dynamically)
            mmlu_question_dropdown = gr.Dropdown(
                choices=[],
                label="Select MMLU Question",
            )

            # Function to update questions when a category is selected
            def update_questions(selected_category):
                csv_file = mmlu_csv_files[mmlu_categories.index(selected_category)]
                questions = load_questions_from_csv(csv_file)
                return gr.update(
                    choices=questions
                )  # Use gr.update to update the dropdown

            # Update the question dropdown when a category is selected
            mmlu_category_dropdown.change(
                fn=update_questions,
                inputs=[mmlu_category_dropdown],
                outputs=[mmlu_question_dropdown],
            )

            # Textbox output for MMLU agent interaction history (Changed to a Textbox for better readability)
            mmlu_answer_history = gr.Textbox(
                label="MMLU Agent Interaction History",
                lines=10,
                interactive=False,
            )

            # Submit button to trigger the function for MMLU
            mmlu_submit_button = gr.Button("Submit")

            # Link the button click to the main function from gen_mmlu_gradio
            mmlu_submit_button.click(
                fn=gen_mmlu_gradio.main,
                inputs=[
                    mmlu_agents_slider,
                    mmlu_rounds_slider,
                    mmlu_evals_slider,
                    mmlu_model_dropdown,
                    mmlu_category_dropdown,  # Added category dropdown input
                    mmlu_question_dropdown,  # Added question dropdown input
                ],
                outputs=[mmlu_answer_history],
            )

        # Gradio UI for Biography
        with gr.Tab("Biography"):
            with gr.Row():
                # Sliders for Biography input configurations
                bio_agents_slider = gr.Slider(
                    minimum=1, maximum=5, step=1, value=2, label="Number of Agents"
                )
                bio_rounds_slider = gr.Slider(
                    minimum=1, maximum=5, step=1, value=2, label="Number of Rounds"
                )
                bio_evals_slider = gr.Slider(
                    minimum=1,
                    maximum=40,
                    step=1,
                    value=1,
                    label="Number of Evaluations",
                )

            # Dropdown for biography model selection
            bio_model_dropdown = gr.Dropdown(
                choices=available_models,
                value=available_models[0],
                label="Select OpenAI Model",
            )

            # New Dropdown for selecting a person from article.json
            bio_people_dropdown = gr.Dropdown(
                choices=people_list,  # Populate dropdown with the names from article.json
                label="Select Person for Biography",
            )

            # Dataframe output for biography agent interaction history
            bio_answer_history = gr.Dataframe(
                headers=["Person", "Agent", "Role", "Content"],
                label="Biography Agent Interaction History",
                interactive=False,
            )

            # Submit button to trigger the function for Biography
            bio_submit_button = gr.Button("Submit")

            # Link the button click to the main function from get_conversation_gradio
            bio_submit_button.click(
                fn=gen_conversation_gradio.main,
                inputs=[
                    bio_agents_slider,
                    bio_rounds_slider,
                    bio_evals_slider,
                    bio_model_dropdown,
                    bio_people_dropdown,  # Pass the selected person to the main function
                ],
                outputs=[bio_answer_history],
            )


demo.launch()
