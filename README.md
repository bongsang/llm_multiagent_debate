# Multiagent Debate WebUI

### This is a Web UI implementation of the paper "Improving Factuality and Reasoning in Language Models through Multiagent Debate", designed to make it easier to use and understand.

I aimed to keep the original code consistent with the content of the paper. Therefore, I only updated the OpenAI API code and added bridge functions for communication with the Gradio UI.

```bash
python main.py
Running on local URL:  http://127.0.0.1:7860
```

### WebUI
**Main**
---
![WebUI Main](docs/screenshot1.png)

- You can choose a LLM model

![Model](docs/screenshot6.png)

**Simple Math Demo**
---
![Math](docs/screenshot2.png)

**Graduate Student Math Demo**
---
![GSM](docs/screenshot3.png)

**Massive Multitask Language Understanding Demo**
---
![MMLU](docs/screenshot4.png)

**Biography Demo**
---
![BIO](docs/screenshot5.png)


---
# Improving Factuality and Reasoning in Language Models through Multiagent Debate

### [Project Page](https://composable-models.github.io/llm_debate/) | [Paper](https://arxiv.org/abs/2305.14325) 

[Yilun Du](https://yilundu.github.io/),
[Shuang Li](https://people.csail.mit.edu/lishuang/),
[Antonio Torralba](https://groups.csail.mit.edu/vision/torralbalab),
[Joshua B. Tenenbaum](https://scholar.google.com/citations?user=rRJ9wTJMUB8C&hl=en),
[Igor Mordatch](https://scholar.google.com/citations?user=Vzr1RukAAAAJ&hl=en)

This is a preliminary implementation of the paper "Improving Factuality and Reasoning in Language Models through Multiagent Debate". More tasks and settings will be released soon. 
You may see some additional debate logs [here](https://www.dropbox.com/sh/6kq5ixfnf4zqk09/AABezsYsBhgg1IQAZ12yQ43_a?dl=0).

Also, check out gauss5930's awesome implementation of multiagent debate on opensource LLMs [here](https://github.com/gauss5930/LLM-Agora)!

## Running experiments

The code for running arithmetic, GSM, biographies, and MMLU tasks may be found in the following subfolders

* ./math/ contains code for running math
* ./gsm/ contains code for running gsm
* ./biography/ contains code for running biographies
* ./mmlu/ contains code for running mmlu results.

**Math:**

To generate and evaluated answer for Math problems through multiagent debate, cd into the math directory and run:
	`python gen_math.py`
	
**Grade School Math:**

To generate answers for Grade School Math problems through multiagent debate, cd into the gsm directory and run:
	`python gen_gsm.py`

To evaluate the generated results of Grade School Math problems:
	`python eval_gsm.py`
	
You can download the GSM dataset [here](https://github.com/openai/grade-school-math)


**Biography:**

To generate answers for Biography problems through multiagent debate, cd into the biography directory and run:
	`python gen_conversation.py`

To evaluate the generated results for Biography problems:
	`python eval_conversation.py`
	
**MMLU:**

To generate answers for MMLU through multiagent debate, cd into the MMLU directory and run:
	`python gen_mmlu.py`

To evaluate the generated results of MMLU:
	`python eval_mmlu.py`
	
You can download the MMLU dataset [here](https://github.com/hendrycks/test)

If you would like to cite the paper, here is a bibtex file:
```
@article{du2023improving,
  title={Improving Factuality and Reasoning in Language Models through Multiagent Debate},
  author={Du, Yilun and Li, Shuang and Torralba, Antonio and Tenenbaum, Joshua B and Mordatch, Igor},
  journal={arXiv preprint arXiv:2305.14325},
  year={2023}
}
```
