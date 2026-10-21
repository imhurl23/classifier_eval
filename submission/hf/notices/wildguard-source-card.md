---
license: odc-by
tags:
- safe
- safety
- jailbreak
- ai-safety
- llm
- lm
- moderation
- classification
- refusal
task_categories:
- text-classification
language:
- en
size_categories:
- 10K<n<100K
configs:
- config_name: wildguardtrain
  data_files:
  - split: train
    path: train/wildguard_train.parquet
- config_name: wildguardtest
  data_files:
  - split: test
    path: test/wildguard_test.parquet
extra_gated_prompt: >-
  Access to this dataset is automatically granted upon accepting the [AI2 Responsible Use Guidelines](https://allenai.org/responsible-use.pdf), and completing
  all fields below
extra_gated_fields:
  Your full name: text
  Organization or entity you are affiliated with: text
  State or country you are located in: text
  Contact email: text
  Please describe your intended use of the low risk artifact(s): text
  I understand that this dataset is a research artifact that may contain or produce unfiltered, toxic, or harmful material: checkbox
  I agree to use this dataset for research purposes in accordance with the AI2 Responsible Use Guidelines: checkbox
  I agree that AI2 may use my information as described in the Privacy Policy: checkbox
  I certify that the information I have provided is true and accurate: checkbox
---

# Dataset Card for WildGuardMix




## Disclaimer:

The data includes examples that might be disturbing, harmful or upsetting. It includes a range of harmful topics such as discriminatory language and discussions
about abuse, violence, self-harm, sexual content, misinformation among other high-risk categories. The main goal of this data is for advancing research in building safe LLMs.
It is recommended not to train a LLM exclusively on the harmful examples. 


![image/png](https://cdn-uploads.huggingface.co/production/uploads/625888c2c964680d85d70bff/N-bHkrpknEG_PWvjfds2u.png)

## Dataset Summary

WildGuardMix consists of two splits, WildGuardTrain and WildGuardTest. Here's the breakdown of WildGuardMix:

**WildGuardTrain** used to train WildGuard:
- Data Size: corresponds of 86,759 examples, of which 48,783 are prompt-only and 37,976 contain a prompt and response.
- Data types: synthetic data (87%), in-the-wild user-LLLM interactions (11%), and existing annotator-written data (2%).
- Prompts types: vanilla and adversarial that cover both harmful and benign scenarios.
- Response generations: for the synthetic adversarial and vanilla prompts, we generate matched refusal and compliance responses using a suite of LLMs.
- Labels: for prompt harmfulness, response harmfulness, and response refusal are obtained via GPT-4.
- Data audit:
  - Filtering: to filter responses created through open LMs by assigning labels for each of our three target tasks using GPT-4 and recategorizing items that fail to match intended labels.
  - Human annotation: audit the quality of GPT-4 labels by sampling 500 items and collecting human annotations. 92%, 82%, and 95% agreement of items for prompt harm, response harm, and refusal labels, respectively.

**WildGuardTest** used to evaluate safety classifiers. 
- Data size: contains 1,725 items for prompt harm, response harm, and response refusal classification tasks. 55% are vanilla prompts and 45% are adversarial.
- Similar to WildGuardTrain, the data consists of vanilla and adversarial synthetic data and in-the-wild user-LLLM interactions covering both benign and harmful scenarios.
- Labels:
  - collect annotations from three independent annotators for each prompt-response pair on prompt harmfulness, response refusal, and response harmfulness.
- Labels quality:
  - Fleiss Kappa scores are 0.55, 0.72, and 0.50 for the three tasks, indicating moderate to substantial agreement.
  - Run a prompted GPT-4 classifier on the dataset and manually inspect items on which the output mismatches the chosen annotator label, to further audit the ground-truth labels.

Please check the paper for further details on data construction: [WildGuard: Open One-stop Moderation Tools for Safety Risks, Jailbreaks, and Refusals of LLMs](https://arxiv.org/abs/2406.18495).


## Usage

```python
from datasets import load_dataset

# Load the wildguardtrain dataset
dataset = load_dataset("allenai/wildguardmix", "wildguardtrain")

# Load the wildguardtest dataset
dataset = load_dataset("allenai/wildguardmix", "wildguardtest")
```

## Dataset Details

The dataset contains the following columns:

- `prompt`: str, indicates the user request.
- `adversarial`: bool, indicates whether the prompt is adversarial or not.
- `response`: str, or None for prompt-only items in WildGuardTrain.
- `prompt_harm_label`: str ("harmful" or "unharmful"), or None for items lacking annotator agreement for `prompt_harm_label`. It is possible that other labels, such as `response_harm_label`, is not None but `prompt_harm_label` is None.
- `response_harm_label`: str ("harmful" or "unharmful"), or None for prompt-only items in WildGuardTrain and items lacking annotator agreement for `response_harm_label`. It is possible that other labels, such as `prompt_harm_label`, is not None but `response_harm_label` is None.
- `response_refusal_label`: str ("refusal" or "compliance"), or None for prompt-only items in WildGuardTrain and items lacking annotator agreement for `response_refusal_label`. It is possible that other labels, such as `prompt_harm_label`, is not None but `response_refusal_label` is None.
- `subcategory`: str, indicates the fine-grained risk category of the prompt. 

Additionally, we provide columns of prompt_harm_agreement, response_harm_agreement, and response_refusal_agreement for WildGuardTest which show whether each label is obtained with two-way or three-way inter-annotator agreement.


## Citation

```
@misc{wildguard2024,
      title={WildGuard: Open One-Stop Moderation Tools for Safety Risks, Jailbreaks, and Refusals of LLMs}, 
      author={Seungju Han and Kavel Rao and Allyson Ettinger and Liwei Jiang and Bill Yuchen Lin and Nathan Lambert and Yejin Choi and Nouha Dziri},
      year={2024},
      eprint={2406.18495},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.18495}, 
}
```