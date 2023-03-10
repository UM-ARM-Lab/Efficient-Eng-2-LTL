---
layout: default
---

## Intro

To make robots accessible to a broad audience, it is critical to endow them with the ability to take universal modes of communication, like commands given in natural language, and extract a concrete desired task specification, defined using a formal language like linear temporal logic (LTL).

In this paper, We present a learning-based approach to translate from natural language commands to LTL specifications with very limited human-labeled training data by leveraging Large Language Models (LLMs). Our model can translate natural language commands at 75% accuracy with about 12 annotations and when given full training data, achieves state-of-the-art performance. We also show how its outputs can be used to plan long-horizon, multi-stage tasks on a 12D quadrotor.

<iframe src="https://drive.google.com/file/d/14Sy5y76YglZ6X3Y3ZZBZZiMGBA9gME9G/preview" width="640" height="360" allow="autoplay"></iframe>

## Approach

![Semantic Parsing](https://i.imgur.com/EZEtGOZ.jpg)

Given a predefined set of possible LTL formulas and atomic
propositions, and up to one natural language annotation for
each formula, we first translate these pre-defined formulas to
(structured) English and then use the paraphrasing abilities of modern LLMs to synthesize a large corpus of diverse natural language commands.

Given this corpus, we use the data to fine-tune an LLM. Here, we explore two variants, where for training labels we use 1) raw LTL formulas, or 2) a canonical form of the LTL formulas (an intermediate representation that is more similar to English).
At evaluation time, we enforce the LLM’s output to be syntactically correct via constrained decoding.

## Result

We evaluate our methods on three datasets, each associated with a different task and environment. We show that our model can translate natural language commands at 75% accuracy with about 12 annotations and when given full training data, constantly obtain state-of-the-art performance. In the paper, we also present comprehensive ablation studies that validate the necessity of each design decision we make.

### Results in low-data regimes

In this setup, models are trained with only 12 human annotations at most, and evaluated on the entire original dataset. Our model significantly advances the state-of-the-art in this regime, achieving 75% average accuracy.

| Model architecture       | Training data | Test data   | Drone Dataset | Cleanup Dataset | Pick Dataset |
| ------------------------ | ------------- | ----------- | ------------- | --------------- | ------------ |
| RNN                      | synthetic     | full golden | 22.41         | 52.54           | 32.39        |
| CopyNet                  | synthetic     | full golden | 36.41         | 53.40           | 40.36        |
| BART-FT-Raw (ours)       | synthetic     | full golden | **69.39**     | **78.00**       | **81.45**    |
| BART-FT-Canonical (ours) | synthetic     | full golden | 68.38         | 77.90           | 78.23        |

### Results in standard data regimes

In this setup, we follow the settings of previous works, where models are evaluated by five-fold cross-validation on the entire dataset. Our model consistently outperforms the state-of-the-art in this regime, with about 1% accuracy improvement on average.

| Model architecture       | Training data | Test data  | Drone Dataset | Cleanup Dataset | Pick Dataset |
| ------------------------ | ------------- | ---------- | ------------- | --------------- | ------------ |
| RNN                      | 4/5 golden    | 1/5 golden | 87.18         | 95.51           | 93.78        |
| CopyNet                  | 4/5 golden    | 1/5 golden | 88.97         | 95.47           | 93.14        |
| BART-FT-Raw (ours)       | 4/5 golden    | 1/5 golden | 90.78         | **97.84**       | **95.97**    |
| BART-FT-Canonical (ours) | 4/5 golden    | 1/5 golden | **90.86**     | 97.81           | 95.70        |

### Demo

![Demo](https://i.imgur.com/ynNTrpf.png)

Finally, we also show how its outputs can be used to plan long-horizon, multi-stage tasks on a 12D quadrotor in simulation.

## Cite

If you find this work useful, please cite our paper:

```
TO BE RELEASED
```
