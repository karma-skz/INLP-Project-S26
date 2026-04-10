"""
src/dataset/emotion_dataset.py
================================
Construct a tightly controlled prompt set for emotion-direction experiments.

The goal is to compare affirmed emotion statements against their negated
counterparts while keeping lexical and syntactic variation limited and
balanced across categories.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd


PAIR_TEMPLATES: list[tuple[str, str, str]] = [
    ("plain_feel", "I feel {lexeme}", "I do not feel {lexeme}"),
    ("right_now", "Right now I feel {lexeme}", "Right now I do not feel {lexeme}"),
    ("today", "Today I feel {lexeme}", "Today I do not feel {lexeme}"),
    ("moment", "At the moment I feel {lexeme}", "At the moment I do not feel {lexeme}"),
    ("cause", "This makes me feel {lexeme}", "This does not make me feel {lexeme}"),
    ("lately", "Lately I have felt {lexeme}", "Lately I have not felt {lexeme}"),
]

DEFAULT_EMOTION_LEXEMES: dict[str, list[str]] = {
    "joy": ["happy", "joyful", "glad", "cheerful"],
    "sadness": ["sad", "unhappy", "gloomy", "miserable"],
    "anger": ["angry", "mad", "furious", "annoyed"],
    "fear": ["afraid", "scared", "fearful", "anxious"],
}

DEFAULT_NEUTRAL_LEXEMES: list[str] = ["calm", "okay", "neutral", "steady"]

OPPOSITE_EMOTIONS: dict[str, str] = {
    "joy": "sadness",
    "sadness": "joy",
}


@dataclass(frozen=True)
class EmotionPromptPair:
    pair_id: int
    emotion: str
    lexeme: str
    template_id: int
    template_name: str
    affirmed_prompt: str
    negated_prompt: str
    opposite_emotion: str | None = None


@dataclass(frozen=True)
class EmotionPromptExample:
    example_id: int
    emotion: str
    prompt_kind: str
    lexeme: str
    template_id: int
    template_name: str
    prompt: str
    pair_id: int | None = None
    opposite_emotion: str | None = None


@dataclass(frozen=True)
class EmotionPromptDataset:
    pairs: list[EmotionPromptPair]
    neutral_examples: list[EmotionPromptExample]
    examples: list[EmotionPromptExample]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([example.__dict__ for example in self.examples])


def _single_token_filter(
    lexemes: Iterable[str],
    model=None,
    skip_multi_token_lexemes: bool = True,
) -> list[str]:
    cleaned = [lexeme.strip() for lexeme in lexemes if str(lexeme).strip()]
    if model is None or not skip_multi_token_lexemes:
        return cleaned

    valid: list[str] = []
    for lexeme in cleaned:
        try:
            model.to_single_token(f" {lexeme}")
        except Exception:
            continue
        valid.append(lexeme)
    return valid


def build_emotion_prompt_dataset(
    model=None,
    emotion_lexemes: Optional[dict[str, list[str]]] = None,
    neutral_lexemes: Optional[list[str]] = None,
    templates: Optional[list[tuple[str, str, str]]] = None,
    skip_multi_token_lexemes: bool = True,
    verbose: bool = True,
) -> EmotionPromptDataset:
    """
    Build a balanced emotion dataset with affirmed, negated, and neutral prompts.

    Parameters
    ----------
    model:
        Optional TransformerLens model used to keep only single-token lexemes.
    emotion_lexemes:
        Mapping from emotion label to candidate surface forms.
    neutral_lexemes:
        Surface forms for neutral-control prompts.
    templates:
        Prompt templates of the form ``(name, affirmed, negated)``.
    skip_multi_token_lexemes:
        When ``True`` and ``model`` is provided, retain only lexemes that map
        to a single token for that model.
    verbose:
        Print filtering and dataset counts.
    """
    emotion_lexemes = emotion_lexemes or DEFAULT_EMOTION_LEXEMES
    neutral_lexemes = neutral_lexemes or DEFAULT_NEUTRAL_LEXEMES
    templates = templates or PAIR_TEMPLATES

    filtered_emotions: dict[str, list[str]] = {}
    for emotion, lexemes in emotion_lexemes.items():
        valid = _single_token_filter(
            lexemes,
            model=model,
            skip_multi_token_lexemes=skip_multi_token_lexemes,
        )
        if not valid:
            raise ValueError(
                f"No valid lexemes left for emotion '{emotion}'. "
                "Add more candidates or disable single-token filtering."
            )
        filtered_emotions[emotion] = valid

    filtered_neutral = _single_token_filter(
        neutral_lexemes,
        model=model,
        skip_multi_token_lexemes=skip_multi_token_lexemes,
    )
    if not filtered_neutral:
        raise ValueError("No valid neutral lexemes left after filtering.")

    pairs: list[EmotionPromptPair] = []
    examples: list[EmotionPromptExample] = []
    neutral_examples: list[EmotionPromptExample] = []
    pair_id = 0
    example_id = 0

    for emotion, lexemes in filtered_emotions.items():
        for template_id, (template_name, affirmed_template, negated_template) in enumerate(templates):
            for lexeme in lexemes:
                pair = EmotionPromptPair(
                    pair_id=pair_id,
                    emotion=emotion,
                    lexeme=lexeme,
                    template_id=template_id,
                    template_name=template_name,
                    affirmed_prompt=affirmed_template.format(lexeme=lexeme),
                    negated_prompt=negated_template.format(lexeme=lexeme),
                    opposite_emotion=OPPOSITE_EMOTIONS.get(emotion),
                )
                pairs.append(pair)
                examples.append(
                    EmotionPromptExample(
                        example_id=example_id,
                        emotion=emotion,
                        prompt_kind="affirmed",
                        lexeme=lexeme,
                        template_id=template_id,
                        template_name=template_name,
                        prompt=pair.affirmed_prompt,
                        pair_id=pair_id,
                        opposite_emotion=pair.opposite_emotion,
                    )
                )
                example_id += 1
                examples.append(
                    EmotionPromptExample(
                        example_id=example_id,
                        emotion=emotion,
                        prompt_kind="negated",
                        lexeme=lexeme,
                        template_id=template_id,
                        template_name=template_name,
                        prompt=pair.negated_prompt,
                        pair_id=pair_id,
                        opposite_emotion=pair.opposite_emotion,
                    )
                )
                example_id += 1
                pair_id += 1

    for template_id, (template_name, affirmed_template, _) in enumerate(templates):
        for lexeme in filtered_neutral:
            neutral = EmotionPromptExample(
                example_id=example_id,
                emotion="neutral",
                prompt_kind="neutral",
                lexeme=lexeme,
                template_id=template_id,
                template_name=template_name,
                prompt=affirmed_template.format(lexeme=lexeme),
            )
            neutral_examples.append(neutral)
            examples.append(neutral)
            example_id += 1

    if verbose:
        print("Built emotion-negation prompt dataset:")
        for emotion, lexemes in filtered_emotions.items():
            print(
                f"  {emotion:<8} -> {len(lexemes):>2} lexemes x {len(templates)} templates "
                f"= {len(lexemes) * len(templates):>3} pairs"
            )
        print(
            f"  neutral  -> {len(filtered_neutral):>2} lexemes x {len(templates)} templates "
            f"= {len(filtered_neutral) * len(templates):>3} controls"
        )
        print(f"  total prompt pairs: {len(pairs)}")
        print(f"  total prompt examples: {len(examples)}")

    return EmotionPromptDataset(
        pairs=pairs,
        neutral_examples=neutral_examples,
        examples=examples,
    )
