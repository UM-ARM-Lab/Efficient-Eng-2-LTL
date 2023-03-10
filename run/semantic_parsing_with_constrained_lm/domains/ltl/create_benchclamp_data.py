# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from semantic_parsing_with_constrained_lm.datum import BenchClampDatum
from semantic_parsing_with_constrained_lm.domains.benchclamp_data_setup import (
    LTL_DOMAINS,
    BenchClampDataset,
)
from semantic_parsing_with_constrained_lm.domains.create_benchclamp_splits import create_benchclamp_splits
from semantic_parsing_with_constrained_lm.domains.ltl import LTLOutputType, LTLDataPieces
from semantic_parsing_with_constrained_lm.paths import (
    BENCH_CLAMP_PROCESSED_DATA_DIR,
    LTL_DATA_DIR,
)


def main():
    for domain in LTL_DOMAINS:
        # domains in this case are "train-evaluation splits"

        """
        Naming convention for ltl data files:
        - ltl_data.json: contains the canonical data
            - denotion looks like something weird, more like program result
            - used for building the trie constrained decoder
            - either ```canonical``` or ```formula```
        - ltl.dev.jsonl
        - ltl.test.jsonl
        - ltl.train_with_dev.jsonl
        - ltl.train_without_dev.jsonl
        """
        ltl_pieces = LTLDataPieces.from_dir(
            LTL_DATA_DIR,
            is_dev=True,
            domain=domain,
            output_type=LTLOutputType.MeaningRepresentation,
            simplify_logical_forms=True,
        )
        train_data = ltl_pieces.train_data
        dev_data = ltl_pieces.test_data

        ltl_pieces = LTLDataPieces.from_dir(
            LTL_DATA_DIR,
            is_dev=False,
            domain=domain,
            output_type=LTLOutputType.MeaningRepresentation,
            simplify_logical_forms=True,
        )
        test_data = ltl_pieces.test_data

        train_benchclamp_data = []
        dev_benchclamp_data = []
        test_benchclamp_data = []

        for data, benchclamp_data in [
            (train_data, train_benchclamp_data),
            (dev_data, dev_benchclamp_data),
            (test_data, test_benchclamp_data),
        ]:
            for datum in data:
                benchclamp_data.append(
                    BenchClampDatum(
                        dialogue_id=datum.dialogue_id,
                        turn_part_index=datum.turn_part_index,
                        utterance=datum.natural,
                        plan=datum.canonical,
                    )
                )

        create_benchclamp_splits(
            train_benchclamp_data,
            dev_benchclamp_data,
            test_benchclamp_data,
            BENCH_CLAMP_PROCESSED_DATA_DIR / BenchClampDataset.LTL.value / domain,
        )


if __name__ == "__main__":
    main()
