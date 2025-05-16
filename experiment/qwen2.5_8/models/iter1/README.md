---
base_model: qwen2_0.5B_instruct_model_sft
tags:
- alignment-handbook
- generated_from_trainer
datasets:
- qwen_datasets/iter0
library_name: peft
model-index:
- name: outputs
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# outputs

This model is a fine-tuned version of [qwen2_0.5B_instruct_model_sft](https://huggingface.co/qwen2_0.5B_instruct_model_sft) on the qwen_datasets/iter0 dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-07
- train_batch_size: 1
- eval_batch_size: 1
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 3
- mixed_precision_training: Native AMP

### Training results



### Framework versions

- Transformers 4.37.0
- Pytorch 2.1.2+cu121
- Datasets 2.14.6
- Tokenizers 0.15.2
## Training procedure


### Framework versions


- PEFT 0.6.1
