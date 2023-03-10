# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from pathlib import Path
from typing import Any, Callable, Dict, Tuple, cast

from typing_extensions import Literal

from semantic_parsing_with_constrained_lm.configs.lib.common import make_semantic_parser
from semantic_parsing_with_constrained_lm.datum import Datum
from semantic_parsing_with_constrained_lm.decoding.partial_parse import (
    PartialParse,
    StartsWithSpacePartialParse,
)

from semantic_parsing_with_constrained_lm.domains.ltl import LTLOutputType, LTLPieces

from semantic_parsing_with_constrained_lm.eval import TopKExactMatch
from semantic_parsing_with_constrained_lm.lm import TRAINED_MODEL_DIR, AutoregressiveModel, ClientType
from semantic_parsing_with_constrained_lm.lm_bart import Seq2SeqBart
from semantic_parsing_with_constrained_lm.lm_openai_gpt3 import IncrementalOpenAIGPT3
from semantic_parsing_with_constrained_lm.paths import DOMAINS_DIR
from semantic_parsing_with_constrained_lm.run_exp import EvalSplit, Experiment
from semantic_parsing_with_constrained_lm.train_model_setup import BartModelConfig

from semantic_parsing_with_constrained_lm.domains.benchclamp_data_setup import LTL_DOMAINS


def build_config(_log_dir, **_kwargs) -> Dict[str, Callable[[], Experiment]]:
    # top10 decoding

    BEAM_SIZE = 10
    eval_split = _kwargs["eval_split"]
    model = _kwargs["model"]
    use_gpt3 = model == ClientType.GPT3

    all_pieces: Dict[Tuple[str, LTLOutputType], LTLPieces] = {}
    max_steps_by_config: Dict[Tuple[str, LTLOutputType], int] = {}

    def create_exp(
        problem_type: Literal[
            "constrained", "unconstrained-beam", "unconstrained-greedy"
        ],
        output_type: LTLOutputType,
        domain: str,
        train_size: int,
    ):
        """
        This part initializes the language model with given configs
        """
        lm: AutoregressiveModel

        if model == ClientType.GPT3:
            lm = IncrementalOpenAIGPT3()
        elif model == ClientType.BART:
            model_loc = f"{TRAINED_MODEL_DIR}/ltl_{domain}_{output_type}/checkpoint-10000/"
            bart_model_config = BartModelConfig(
                model_id="Bart", model_loc=Path(model_loc)
            )
            bart_model, clamp_tokenizer, _ = bart_model_config.setup_model()
            lm = Seq2SeqBart(
                # Model location set to match lm_finetune.py
                pretrained_model_dir=model_loc,
                model=bart_model,
                clamp_tokenizer=clamp_tokenizer,
            )
        else:
            raise ValueError(model)

        """
        This part collects all possible outputs for trie decoding
        """
        pieces = all_pieces.get((domain, output_type))
        if not pieces:
            pieces = all_pieces[domain, output_type] = LTLPieces.from_dir(
                lm.tokenizer,
                DOMAINS_DIR / "ltl/data",
                domain,
                is_dev=eval_split in (EvalSplit.DevFull, EvalSplit.DevSubset),
                k=BEAM_SIZE,
                output_type=output_type,
                simplify_logical_forms=True,
                # TODO: Set prefix_with_space properly by inspecting `lm`
                prefix_with_space=True,
            )

        """
        This sets the max length of language model output
        """
        max_steps = max_steps_by_config.get((domain, output_type))
        if max_steps is None:
            max_steps = max_steps_by_config[domain, output_type] = (
                max(
                    len(lm.tokenizer.tokenize(" " + canon))
                    for canon in pieces.denotation_metric.canonical_to_denotation
                )
                + 3  # +3 to be safe
            )

        """
        This collects the training data (used for selecting and constructing GPT-3 prompts on-the-fly)
        """
        train_data = pieces.train_data[:train_size]
        if eval_split == EvalSplit.TrainSubset:
            test_data = pieces.train_data[-100:]
        elif eval_split in (EvalSplit.TestFull, EvalSplit.DevFull):
            test_data = pieces.test_data
        elif eval_split in (EvalSplit.TestSubset, EvalSplit.DevSubset):
            test_data = pieces.test_data[:100]
        else:
            raise ValueError(eval_split)

        """
        This creates the partial parse builder (for different types of decoding)
        """
        partial_parse_builder: Callable[[Datum], PartialParse]
        if problem_type == "constrained":
            partial_parse_builder = pieces.partial_parse_builder  # type: ignore
            beam_size = BEAM_SIZE
        elif problem_type.startswith("unconstrained"):
            partial_parse = StartsWithSpacePartialParse(lm.tokenizer)
            def partial_parse_builder(_): return partial_parse
            if problem_type == "unconstrained-beam":
                beam_size = BEAM_SIZE
            elif problem_type == "unconstrained-greedy":
                beam_size = 1
            else:
                raise ValueError(problem_type)
        else:
            raise ValueError(f"{problem_type} not allowed")

        parser = make_semantic_parser(
            train_data,
            # Erase type of `lm` since we're not passing any other arguments
            # that have the HS type variable.
            # TODO: Think if there's a better way to do this
            cast(AutoregressiveModel[Any], lm),
            use_gpt3,
            max_steps,
            beam_size,
            partial_parse_builder,
            lambda _datum: max_steps,
        )

        return Experiment(
            model=parser,
            client=lm,
            metrics={
                "exact_match": TopKExactMatch(beam_size),
                "denotation": pieces.denotation_metric,
            },
            test_data=test_data,
        )

    def add_exp_to_dict(
        exps_dict: Dict[str, Callable[[], Experiment]],
        problem_type: Literal[
            "constrained", "unconstrained-beam", "unconstrained-greedy"
        ],
        output_type: LTLOutputType,
        domain: str,
        train_size: int,
    ):
        exp_name = f"ltl_{model}_{eval_split}_{domain}_{problem_type}_{output_type}_train-{train_size}"
        exps_dict[exp_name] = lambda: create_exp(
            problem_type, output_type, domain, train_size
        )

    result: Dict[str, Callable[[], Experiment]] = {}
    if eval_split == EvalSplit.TestFull:
        for domain in LTL_DOMAINS:
            add_exp_to_dict(result, "constrained",
                            LTLOutputType.Utterance, domain, 0)
            if not use_gpt3:
                add_exp_to_dict(
                    result, "constrained", LTLOutputType.MeaningRepresentation, domain, 0
                )
                add_exp_to_dict(
                    result, "unconstrained-greedy", LTLOutputType.Utterance, domain, 0
                )
                add_exp_to_dict(
                    result,
                    "unconstrained-greedy",
                    LTLOutputType.MeaningRepresentation,
                    domain,
                    0,
                )
    # elif eval_split == EvalSplit.TestSubset and use_gpt3:
    #     for domain in LTL_DOMAINS:
    #         add_exp_to_dict(result, "constrained",
    #                         LTLOutputType.Utterance, domain, 20)
    #         add_exp_to_dict(result, "constrained",
    #                         LTLOutputType.Utterance, domain, 200)
    #         add_exp_to_dict(
    #             result, "constrained", LTLOutputType.MeaningRepresentation, domain, 200
    #         )
    #         add_exp_to_dict(
    #             result, "unconstrained-greedy", LTLOutputType.Utterance, domain, 200
    #         )
    #         add_exp_to_dict(
    #             result,
    #             "unconstrained-greedy",
    #             LTLOutputType.MeaningRepresentation,
    #             domain,
    #             200,
    #         )
    else:
        raise Exception(f"{eval_split} not supported")

    return result
