import argparse

from scripts.dev.anima_rum_xpred_train import resolve_max_train_steps


def test_resolve_max_train_steps_from_epochs():
    args = argparse.Namespace(
        max_train_steps=None,
        train_batch_size=2,
        gradient_accumulation_steps=1,
        num_train_epochs=1.0,
    )

    assert resolve_max_train_steps(args, cache_sample_count=5) == 3


def test_resolve_max_train_steps_explicit_override_wins():
    args = argparse.Namespace(
        max_train_steps=7,
        train_batch_size=2,
        gradient_accumulation_steps=1,
        num_train_epochs=1.0,
    )

    assert resolve_max_train_steps(args, cache_sample_count=5) == 7
