---
license: cc-by-4.0
task_categories:
- text-generation
language:
- en
tags:
- dialog safety
- dialog moderation
- LLM safety
- topic control
pretty_name: CantTalkAboutThis Topic Control Dataset
size_categories:
- 1K<n<10K
configs:
  - config_name: mixtral
    data_files:
        - split: train
          path: canttalkaboutthis_topic_control_mixtral.jsonl
        - split: test
          path: canttalkaboutthis_topic_control_human_test_set.jsonl
---

# CantTalkAboutThis Topic Control Dataset

## Dataset Details

### Dataset Description

The CantTalkAboutThis dataset is designed to train language models to maintain topical focus during task-oriented dialogues. It includes synthetic dialogues across nine domains (e.g., health, banking, travel) and incorporates distractor turns to test and improve the model's ability to be resilient to distractors. Fine-tuning models on this dataset enhances their ability to maintain topical coherence and improves alignment for both instruction-following and safety tasks.


- **Language(s) (NLP):** English
- **License:** CC-BY-4.0

### Dataset Sources

- **Repository:** [Link](https://github.com/makeshn/topic_following)
- **Paper:** [Link](https://arxiv.org/abs/2404.03820)
- **Demo:** [NVIDIA AI Playground](https://build.nvidia.com/nvidia/llama-3_1-nemoguard-8b-topic-control)

## Uses


### Direct Use

This dataset is intended for training and fine-tuning language models to maintain topical relevance in dialogues, useful for creating task-oriented bots. Broadly, the inteded use cases are:

- Training language models to recognize sensitive topics
- Developing topic control mechanisms in conversational AI
- Evaluating AI systems' ability to handle restricted content appropriately

### Out-of-Scope Use

This dataset should not be used to train systems for harmful, unethical, or malicious purposes. This dataset should not be used for:

- Training models to generate harmful or inappropriate content
- Bypassing content moderation systems

This dataset should not be used for:
- Training models to generate harmful or inappropriate content
- Bypassing content moderation systems
- Creating adversarial examples to test system vulnerabilities

## Dataset Structure

The dataset includes 1080 dialogues, with each conversation containing distractor turns. Scenarios are categorized into nine domains - health, banking, travel, education, finance, insurance, legal, real estate, and computer troubleshooting. The various fields in the dataset are:

- `domain`: The domain of the conversation
- `scenario`: The specific scenario or task being discussed
- `system_instruction`: The dialogue policy given to the model and it is usually a complex set of instructions on topics allowed and not allowed.
- `conversation`: The full conversation, including both the main topic and distractor turns
- `distractors`: List of distractor turns. This includes a bot turn from the conversation and the distractor turn from the user that should be included in the conversation as a response to the bot's turn.
- `conversation_with_distractors`: The conversation with the distractor turns included.

### Curation Rationale

The dataset is created to address a gap in existing alignment datasets for topic control. Language models are often trained to be as helpful as possible, which can lead to them straying from the intended topic of the conversation. This dataset is designed to test the ability of language models to maintain topical focus during dialogues and to help train guardrail models to detect when a langauge model is straying from the intended topic.

### Source Data

The dataset is created using apipeline to synthetically generate conversations and distractors. This pipline is described in the accompanying [paper](https://arxiv.org/abs/2404.03820).

This version of the dataset is the commercially friendly version and was generated using the [Mixtral-8x7B-Instruct](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) model. We additionally provide an evaluation dataset that is human annotated and includes more complex, realistic distractors that can be used to evaluate the performance of models.


#### Personal and Sensitive Information

The dataset does not contain any personal or sensitive information. The data is synthetically generated and is not expected to contain any real world data that is of sensitive nature.

## Bias, Risks, and Limitations

* Biases: The dataset is synthetic, which may lead to limitations in generalizability.
* Risks: Distractors in the dataset are simpler than real-world off-topic deviations, requiring additional human annotations for robustness. The guardrail models trained on this dataset are not expected to be able to detect all off-topic deviations. 

### Recommendations

Users should be made aware of the risks, biases and limitations of the dataset.

## Citation

**BibTeX:**

```bibtex
@inproceedings{sreedhar2024canttalkaboutthis,
  title={CantTalkAboutThis: Aligning Language Models to Stay on Topic in Dialogues},
  author={Sreedhar, Makesh and Rebedea, Traian and Ghosh, Shaona and Zeng, Jiaqi and Parisien, Christopher},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP 2024},
  pages={12232--12252},
  year={2024},
  organization={Association for Computational Linguistics}
}
```

## Dataset Card Authors

* Makesh Sreedhar
* Traian Rebedea

## Dataset Card Contact

* Makesh Sreedhar {makeshn@nvidia.com}
* Traian Rebedea {trebedea@nvidia.com}