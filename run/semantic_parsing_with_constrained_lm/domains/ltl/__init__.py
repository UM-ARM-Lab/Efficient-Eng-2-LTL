# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List

from blobfile import BlobFile

from semantic_parsing_with_constrained_lm.util.trie import Trie
from semantic_parsing_with_constrained_lm.util.types import StrPath
from semantic_parsing_with_constrained_lm.datum import Datum, FullDatum
from semantic_parsing_with_constrained_lm.decoding.trie_partial_parse import TriePartialParse

from semantic_parsing_with_constrained_lm.eval import TopKExactMatch
from semantic_parsing_with_constrained_lm.tokenization import ClampTokenizer

# NOTE: get rid of the catflow/dataflow dependency
from appdirs import user_cache_dir
CACHE_DIR = user_cache_dir("semantic_parsing_as_constrained_lm")


class LTLOutputType(str, Enum):
    Utterance = "utterance"
    MeaningRepresentation = "meaningRepresentation"


@dataclass
class TopKDenotationMatch(TopKExactMatch[FullDatum]):
    canonical_to_denotation: Dict[str, str]

    def _is_correct(self, pred: str, datum: FullDatum) -> bool:
        target = datum.canonical
        pred_denotation = self.canonical_to_denotation.get(pred)
        target_denotation = self.canonical_to_denotation.get(target, None)
        if pred_denotation is None and target_denotation is None:
            return pred == target
        else:
            return pred_denotation == target_denotation


@dataclass
class LTLPieces:
    train_data: List[FullDatum]
    test_data: List[FullDatum]
    partial_parse_builder: Callable[[Datum], TriePartialParse]
    denotation_metric: TopKDenotationMatch
    max_length: int

    @staticmethod
    def from_dir(
        tokenizer: ClampTokenizer,
        root_dir: StrPath,
        domain: str,
        is_dev: bool,
        k: int,
        output_type: LTLOutputType = LTLOutputType.Utterance,
        simplify_logical_forms=False,
        prefix_with_space=False,
    ) -> "LTLPieces":
        data_pieces = LTLDataPieces.from_dir(
            root_dir, domain, is_dev, output_type, simplify_logical_forms
        )
        decoder_pieces = LTLDecoderPieces.create(
            data_pieces, tokenizer, k, prefix_with_space
        )

        return LTLPieces(
            data_pieces.train_data,
            data_pieces.test_data,
            # https://github.com/python/mypy/issues/5485
            decoder_pieces.partial_parse_builder,  # type: ignore
            decoder_pieces.denotation_metric,
            decoder_pieces.max_length,
        )


@dataclass
class LTLDataPieces:
    train_data: List[FullDatum]
    test_data: List[FullDatum]
    target_output_to_denotation: Dict[str, str]

    @staticmethod
    def from_dir(
        root_dir: StrPath,
        domain: str,
        is_dev: bool,
        output_type: LTLOutputType = LTLOutputType.MeaningRepresentation,
        simplify_logical_forms: bool = False,
    ) -> "LTLDataPieces":
        with BlobFile(str(root_dir) + f"/{domain}.canonical.json") as bf:
            canonical_data = json.load(bf)

        if output_type == LTLOutputType.Utterance:
            target_output_to_denotation = {
                k: "DO NOT NEED" for k, v in canonical_data.items()
            }
            datum_key = "canonical"
        else:
            raise ValueError(output_type)

        train_data, test_data = [
            [
                FullDatum(
                    dialogue_id=f"{dataset_name}-{i}",
                    turn_part_index=None,
                    agent_context=None,
                    natural=d["natural"],
                    canonical=d[datum_key]
                    if simplify_logical_forms
                    else d[datum_key],
                )
                for i, line in enumerate(
                    BlobFile(path, streaming=False, cache_dir=CACHE_DIR)
                )
                for d in [json.loads(line)]
            ]
            for dataset_name, path in (
                (
                    "train",
                    f"{root_dir}/{domain}.train.jsonl",
                ),
                ("eval", f"{root_dir}/{domain}.test.jsonl"),
            )
        ]

        return LTLDataPieces(train_data, test_data, target_output_to_denotation)


@dataclass
class LTLDecoderPieces:
    data_pieces: LTLDataPieces
    partial_parse_builder: Callable[[Datum], TriePartialParse]
    denotation_metric: TopKDenotationMatch
    max_length: int

    @staticmethod
    def create(
        data_pieces: LTLDataPieces,
        tokenizer: ClampTokenizer,
        k: int,
        prefix_with_space: bool = False,
    ) -> "LTLDecoderPieces":
        if prefix_with_space:
            canonical_trie = Trie(
                tokenizer.encode(" " + canon)
                for canon in data_pieces.target_output_to_denotation
            )
        else:
            canonical_trie = Trie(
                tokenizer.encode(canon)
                for canon in data_pieces.target_output_to_denotation
            )

        def partial_parse_builder(_): return TriePartialParse(canonical_trie)

        denotation_metric = TopKDenotationMatch(
            k, data_pieces.target_output_to_denotation
        )
        max_length = max(len(x) for x in canonical_trie)

        return LTLDecoderPieces(
            data_pieces, partial_parse_builder, denotation_metric, max_length
        )
