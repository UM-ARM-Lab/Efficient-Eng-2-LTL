# Data-Efficient Learning of Natural Language to Linear Temporal Logic Translators for Robot Task Specification

![Demo](https://i.imgur.com/ynNTrpf.png)

[[Homepage](https://um-arm-lab.github.io/Efficient-Eng-2-LTL/)] [[Paper](https://arxiv.org/abs/2303.08006)] [[Video](https://drive.google.com/file/d/14Sy5y76YglZ6X3Y3ZZBZZiMGBA9gME9G/view?usp=sharing)] [[Poster](https://drive.google.com/file/d/1j0aZoROb1EKC0oRYYBSwBIx4Xp8ElowN/view?usp=sharing)]

> The associated repo for paper "Data-Efficient Learning of Natural Language to Linear Temporal Logic Translators for Robot Task Specification".

## Repo Structure

- Root
  - datasets
    - [drone-planning](https://arxiv.org/abs/1905.12096)
    - [clean-up](http://www.roboticsproceedings.org/rss14/p67.html)
    - [pick-and-place](http://www.roboticsproceedings.org/rss14/p67.html)
  - augmentation
    - paraphrase with GPT-3
  - run
    - train the models
    - inference with constrained decoding

The constrained decoding inference code is based on: [microsoft/semantic_parsing_with_constrained_lm](https://github.com/microsoft/semantic_parsing_with_constrained_lm)

## Reproduce the Results

### Environment Setup

Install the dependencies:

```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113 # make sure the version is compatible with your cuda version
pip install transformers
pip install datasets
pip install jsons appdirs blobfile cached-property httpx  typer whoosh more_itertools
pip install --upgrade protobuf==3.20.0
```

Download BART-large model:

```bash
python ./run/semantic_parsing_with_constrained_lm/finetune/download_huggingface_lms.py
```

### Prepare the Dataset

The processed dataset (with augmentation from LLM) in already included in the repo. This step is only needed if you want to reprocess the dataset.

To actually process the raw dataset, you can follow the steps below:

1. Pre-process: In each of the three dataset folders, run all cells in "preprocess.ipynb" to generate the processed dataset. （the annotation result is included in the notebook).
2. Augmentation: For each of the three datasets, run all commands in "augment.ipynb" to generate the augmented dataset. Note that this step requires a GPT-3 API key.
3. Move to training folder: You then need to reformat the dataset and move it to the `run/semantic_parsing_with_constrained_lm/domains/ltl/data` folder. A script will be provided later to help you automate this process.

### Train

In our paper, we use the [BART-large model](https://huggingface.co/facebook/bart-large) because it is efficient to fine-tune on a single GPU. Our proposed method can be easily applied to other potentially stronger language models like [T5-XXL](https://arxiv.org/abs/1910.10683) or [GPT-3](https://arxiv.org/abs/2005.14165).

```sh
export PRETRAINED_MODEL_DIR=facebook/bart-large
export TRAINED_MODEL_DIR=trained_models/

DOMAIN=TODO

python -m semantic_parsing_with_constrained_lm.finetune.lm_finetune \
        --config-name semantic_parsing_with_constrained_lm.finetune.configs.emnlp_train_config \
        --exp-names ltl_pick-${DOMAIN}_utterance
```

Here DOMAIN determines which experiment to run.
DOMAIN: {dataset_name}-{experiment_name}

- dataset_name: {drone, cleanup, pick}
- experiment_name:
  - syn-aug: synthetic with augmentation
  - syn: synthetic without augmentation
  - golden-cross0-split{0,1,2,3,4}: golden dataset with cross-validation

### Inference

```sh
export PRETRAINED_MODEL_DIR=facebook/bart-large
export TRAINED_MODEL_DIR=trained_models/

DOMAIN=TODO

python -m semantic_parsing_with_constrained_lm.run_exp \
--config-name semantic_parsing_with_constrained_lm.configs.ltl_config \
--log-dir logs/ \
--model Bart \
--eval-split test-full \
--exp-names "ltl_Bart_test-full_${DOMAIN}_constrained_utterance_train-0"
```

The domain name is the same as the training step.

## Cite
To appear at ICRA 2023
```bibtex
@article{pan2023data,
  title={Data-Efficient Learning of Natural Language to Linear Temporal Logic Translators for Robot Task Specification},
  author={Pan, Jiayi and Chou, Glen and Berenson, Dmitry},
  journal={arXiv preprint arXiv:2303.08006},
  year={2023}
}
```
