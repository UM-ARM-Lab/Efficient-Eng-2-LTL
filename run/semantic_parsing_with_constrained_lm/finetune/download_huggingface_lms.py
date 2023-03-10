# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from pathlib import Path

from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    RobertaTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
)

CLAMP_PRETRAINED_MODEL_DIR = Path("huggingface_models/")


def save_model_and_tokenizer(model, tokenizer, save_dir: Path) -> None:
    save_dir.mkdir(exist_ok=True, parents=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)


def main():
    # T5
    # Bart
    for model_id, huggingface_model_id in [
        ("bart-large", "facebook/bart-large"),
    ]:
        print(f"Downloading {model_id} ...")
        model = BartForConditionalGeneration.from_pretrained(huggingface_model_id)
        tokenizer = BartTokenizer.from_pretrained(huggingface_model_id)
        save_model_and_tokenizer(
            model, tokenizer, CLAMP_PRETRAINED_MODEL_DIR / model_id
        )


if __name__ == "__main__":
    main()
