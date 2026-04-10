"""
src/analysis/emotion_directions.py
====================================
Directional linear analysis for emotion/negation prompt pairs.

This module keeps the experiment separate from the repository's existing
"DLA" meaning of direct logit attribution. Here, DLA refers to the
Anthropic-style mean-difference direction in activation space.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

from src.dataset import EmotionPromptDataset


LINEARITY_BASE_PROMPTS: dict[str, list[str]] = {
    "affirmed_prefix": [
        "I feel",
        "Right now I feel",
        "Today I feel",
        "At the moment I feel",
    ],
    "negated_prefix": [
        "I do not feel",
        "Right now I do not feel",
        "Today I do not feel",
        "At the moment I do not feel",
    ],
}


@dataclass(frozen=True)
class EmotionDirectionResult:
    metadata: pd.DataFrame
    summary: pd.DataFrame
    opposite_summary: pd.DataFrame
    negation_sensitivity: pd.DataFrame
    pca: pd.DataFrame
    linearity_curves: pd.DataFrame
    linearity_summary: pd.DataFrame
    direction_vectors: dict[tuple[str, int], np.ndarray]
    mean_vectors: dict[tuple[str, str, int], np.ndarray]
    activations: dict[int, np.ndarray]
    reference_layers: dict[str, int]


def select_reference_layers(n_layers: int) -> dict[str, int]:
    return {
        "early": max(0, n_layers // 4),
        "middle": n_layers // 2,
        "late": n_layers - 1,
    }


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denom <= 0:
        return float("nan")
    return float(np.dot(vec_a, vec_b) / denom)


def _mean_pairwise_cosine(vectors: list[np.ndarray]) -> float:
    valid = [vector for vector in vectors if np.linalg.norm(vector) > 0]
    if len(valid) < 2:
        return float("nan")

    cosines = [_cosine_similarity(vec_a, vec_b) for vec_a, vec_b in combinations(valid, 2)]
    finite = [value for value in cosines if np.isfinite(value)]
    if not finite:
        return float("nan")
    return float(np.mean(finite))


def _standardize_features(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (x_train - mean) / std, (x_test - mean) / std


def ridge_probe_accuracy(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    l2_penalty: float = 1e-2,
) -> float:
    """
    Fit a ridge-regularized linear classifier and return held-out accuracy.
    """
    if len(x_train) == 0 or len(x_test) == 0:
        return float("nan")

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return float("nan")

    x_train_s, x_test_s = _standardize_features(x_train, x_test)
    x_train_aug = np.concatenate([x_train_s, np.ones((len(x_train_s), 1))], axis=1)
    x_test_aug = np.concatenate([x_test_s, np.ones((len(x_test_s), 1))], axis=1)

    targets = np.where(y_train.astype(int) == 1, 1.0, -1.0)
    reg = l2_penalty * np.eye(x_train_aug.shape[1])
    reg[-1, -1] = 0.0

    weights = np.linalg.solve(x_train_aug.T @ x_train_aug + reg, x_train_aug.T @ targets)
    logits = x_test_aug @ weights
    predictions = (logits >= 0).astype(int)
    return float((predictions == y_test.astype(int)).mean())


def _pca_projection(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = x - x.mean(axis=0, keepdims=True)
    if centered.shape[0] < 2:
        coords = np.zeros((centered.shape[0], 2), dtype=float)
        explained = np.array([0.0, 0.0], dtype=float)
        return coords, explained

    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:2].T
    coords = centered @ basis
    variance = singular_values**2
    total_var = float(variance.sum()) if variance.size else 0.0
    if total_var <= 0:
        explained = np.array([0.0, 0.0], dtype=float)
    else:
        explained = variance[:2] / total_var
    return coords, explained


def _build_lexeme_map(metadata: pd.DataFrame) -> dict[str, list[str]]:
    lexeme_map: dict[str, list[str]] = {}
    for emotion, group in metadata.groupby("emotion"):
        lexeme_map[emotion] = sorted(group["lexeme"].dropna().astype(str).unique().tolist())
    return lexeme_map


@torch.no_grad()
def extract_residual_stream_representations(
    model: HookedTransformer,
    dataset: EmotionPromptDataset,
    layers: Optional[list[int]] = None,
    representation: str = "final_token",
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[int, np.ndarray], dict[str, int]]:
    """
    Extract residual-stream representations from ``hook_resid_post``.
    """
    if representation not in {"final_token", "mean_pool"}:
        raise ValueError("representation must be one of {'final_token', 'mean_pool'}")

    n_layers = model.cfg.n_layers
    layers = sorted(set(layers if layers is not None else list(range(n_layers))))
    hook_names = {f"blocks.{layer}.hook_resid_post" for layer in layers}

    metadata_rows: list[dict] = []
    activations: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}

    iterator = tqdm(dataset.examples, desc="Emotion activations") if verbose else dataset.examples
    for example in iterator:
        _, cache = model.run_with_cache(
            example.prompt,
            names_filter=lambda name: name in hook_names,
        )

        token_length = int(cache["hook_embed"].shape[1]) if "hook_embed" in cache else None
        metadata_rows.append(
            {
                "example_id": example.example_id,
                "emotion": example.emotion,
                "prompt_kind": example.prompt_kind,
                "lexeme": example.lexeme,
                "template_id": example.template_id,
                "template_name": example.template_name,
                "prompt": example.prompt,
                "pair_id": example.pair_id,
                "opposite_emotion": example.opposite_emotion,
                "token_length": token_length,
            }
        )

        for layer in layers:
            resid = cache[f"blocks.{layer}.hook_resid_post"][0]
            if representation == "final_token":
                vector = resid[-1, :]
            else:
                vector = resid.mean(dim=0)
            activations[layer].append(vector.detach().float().cpu().numpy())

    stacked = {layer: np.vstack(values).astype(np.float32) for layer, values in activations.items()}
    metadata = pd.DataFrame(metadata_rows)
    reference_layers = select_reference_layers(n_layers)
    return metadata, stacked, reference_layers


def _template_stability(
    metadata: pd.DataFrame,
    activations: np.ndarray,
    emotion: str,
) -> float:
    vectors: list[np.ndarray] = []
    emotion_mask = metadata["emotion"] == emotion
    for template_id in sorted(metadata.loc[emotion_mask, "template_id"].unique().tolist()):
        affirmed = (
            emotion_mask
            & (metadata["prompt_kind"] == "affirmed")
            & (metadata["template_id"] == template_id)
        ).values
        negated = (
            emotion_mask
            & (metadata["prompt_kind"] == "negated")
            & (metadata["template_id"] == template_id)
        ).values
        if affirmed.sum() == 0 or negated.sum() == 0:
            continue
        vectors.append(activations[affirmed].mean(axis=0) - activations[negated].mean(axis=0))
    return _mean_pairwise_cosine(vectors)


def _lexeme_stability(
    metadata: pd.DataFrame,
    activations: np.ndarray,
    emotion: str,
) -> float:
    vectors: list[np.ndarray] = []
    emotion_mask = metadata["emotion"] == emotion
    for lexeme in sorted(metadata.loc[emotion_mask, "lexeme"].unique().tolist()):
        affirmed = (
            emotion_mask
            & (metadata["prompt_kind"] == "affirmed")
            & (metadata["lexeme"] == lexeme)
        ).values
        negated = (
            emotion_mask
            & (metadata["prompt_kind"] == "negated")
            & (metadata["lexeme"] == lexeme)
        ).values
        if affirmed.sum() == 0 or negated.sum() == 0:
            continue
        vectors.append(activations[affirmed].mean(axis=0) - activations[negated].mean(axis=0))
    return _mean_pairwise_cosine(vectors)


def _symmetry_cosine(
    metadata: pd.DataFrame,
    activations: np.ndarray,
    emotion: str,
) -> float:
    """
    Compare affirmed-minus-negated on one template split with the reverse
    direction on the complementary split. Using disjoint splits keeps the
    check non-trivial.
    """
    emotion_mask = metadata["emotion"] == emotion
    affirmed_train = (
        emotion_mask
        & (metadata["prompt_kind"] == "affirmed")
        & ((metadata["template_id"] % 2) == 0)
    ).values
    negated_train = (
        emotion_mask
        & (metadata["prompt_kind"] == "negated")
        & ((metadata["template_id"] % 2) == 0)
    ).values
    affirmed_test = (
        emotion_mask
        & (metadata["prompt_kind"] == "affirmed")
        & ((metadata["template_id"] % 2) == 1)
    ).values
    negated_test = (
        emotion_mask
        & (metadata["prompt_kind"] == "negated")
        & ((metadata["template_id"] % 2) == 1)
    ).values

    if min(affirmed_train.sum(), negated_train.sum(), affirmed_test.sum(), negated_test.sum()) == 0:
        return float("nan")

    forward = activations[affirmed_train].mean(axis=0) - activations[negated_train].mean(axis=0)
    reverse = activations[negated_test].mean(axis=0) - activations[affirmed_test].mean(axis=0)
    return _cosine_similarity(forward, -reverse)


def _linear_probe_accuracy(
    metadata: pd.DataFrame,
    activations: np.ndarray,
    emotion: str,
) -> float:
    affirmed_train = (
        (metadata["emotion"] == emotion)
        & (metadata["prompt_kind"] == "affirmed")
        & ((metadata["template_id"] % 2) == 0)
    ).values
    negated_train = (
        (metadata["emotion"] == emotion)
        & (metadata["prompt_kind"] == "negated")
        & ((metadata["template_id"] % 2) == 0)
    ).values
    affirmed_test = (
        (metadata["emotion"] == emotion)
        & (metadata["prompt_kind"] == "affirmed")
        & ((metadata["template_id"] % 2) == 1)
    ).values
    negated_test = (
        (metadata["emotion"] == emotion)
        & (metadata["prompt_kind"] == "negated")
        & ((metadata["template_id"] % 2) == 1)
    ).values

    x_train = np.vstack([activations[affirmed_train], activations[negated_train]])
    y_train = np.array([1] * int(affirmed_train.sum()) + [0] * int(negated_train.sum()))
    x_test = np.vstack([activations[affirmed_test], activations[negated_test]])
    y_test = np.array([1] * int(affirmed_test.sum()) + [0] * int(negated_test.sum()))
    return ridge_probe_accuracy(x_train, y_train, x_test, y_test)


def _projection_gap(
    activations: np.ndarray,
    affirmed_mask: np.ndarray,
    negated_mask: np.ndarray,
    direction: np.ndarray,
) -> tuple[float, float, float]:
    norm = np.linalg.norm(direction)
    if norm <= 0:
        return float("nan"), float("nan"), float("nan")
    unit_direction = direction / norm
    affirmed_proj = activations[affirmed_mask] @ unit_direction
    negated_proj = activations[negated_mask] @ unit_direction
    return (
        float(affirmed_proj.mean()),
        float(negated_proj.mean()),
        float(affirmed_proj.mean() - negated_proj.mean()),
    )


def _fit_linearity(alpha_values: np.ndarray, margins: np.ndarray) -> tuple[float, float, float]:
    if len(alpha_values) < 2 or np.allclose(margins, margins[0]):
        return float("nan"), float("nan"), float("nan")
    slope, intercept = np.polyfit(alpha_values, margins, deg=1)
    predicted = slope * alpha_values + intercept
    ss_res = float(np.sum((margins - predicted) ** 2))
    ss_tot = float(np.sum((margins - margins.mean()) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return float(slope), float(intercept), float(r2)


def _make_residual_add_hook(direction: torch.Tensor):
    def hook_fn(value: torch.Tensor, hook) -> torch.Tensor:
        value[:, -1, :] = value[:, -1, :] + direction
        return value

    return hook_fn


@torch.no_grad()
def _direction_linearity_sweep(
    model: HookedTransformer,
    emotion: str,
    layer: int,
    direction: np.ndarray,
    activations: np.ndarray,
    emotion_token_ids: list[int],
    control_token_ids: list[int],
    alpha_values: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    curve_rows: list[dict] = []
    summary_rows: list[dict] = []

    norm = np.linalg.norm(direction)
    if norm <= 0:
        return pd.DataFrame(), pd.DataFrame()

    unit_direction = direction / norm
    projection_scale = float(np.std(activations @ unit_direction))
    if not np.isfinite(projection_scale) or projection_scale <= 1e-6:
        projection_scale = 1.0

    device = next(model.parameters()).device
    hook_name = f"blocks.{layer}.hook_resid_post"

    for base_condition, prompts in LINEARITY_BASE_PROMPTS.items():
        margins_for_condition: list[float] = []
        for alpha in alpha_values:
            emotion_masses: list[float] = []
            control_masses: list[float] = []
            scaled_direction = torch.tensor(
                unit_direction * (alpha * projection_scale),
                dtype=torch.float32,
                device=device,
            )

            for prompt in prompts:
                logits = model.run_with_hooks(
                    prompt,
                    fwd_hooks=[(hook_name, _make_residual_add_hook(scaled_direction))],
                )
                probs = torch.softmax(logits[0, -1, :], dim=-1)
                emotion_mass = float(probs[emotion_token_ids].sum().item())
                control_mass = float(probs[control_token_ids].sum().item())
                emotion_masses.append(emotion_mass)
                control_masses.append(control_mass)

            margin = float(np.mean(emotion_masses) - np.mean(control_masses))
            margins_for_condition.append(margin)
            curve_rows.append(
                {
                    "emotion": emotion,
                    "layer": layer,
                    "base_condition": base_condition,
                    "alpha": float(alpha),
                    "emotion_mass": float(np.mean(emotion_masses)),
                    "control_mass": float(np.mean(control_masses)),
                    "margin": margin,
                }
            )

        margin_array = np.asarray(margins_for_condition, dtype=float)
        alpha_array = np.asarray(alpha_values, dtype=float)
        slope, intercept, r2 = _fit_linearity(alpha_array, margin_array)
        summary_rows.append(
            {
                "emotion": emotion,
                "layer": layer,
                "base_condition": base_condition,
                "slope": slope,
                "intercept": intercept,
                "linearity_r2": r2,
                "start_margin": float(margin_array[0]),
                "end_margin": float(margin_array[-1]),
            }
        )

    return pd.DataFrame(curve_rows), pd.DataFrame(summary_rows)


@torch.no_grad()
def analyze_emotion_negation(
    model: HookedTransformer,
    dataset: EmotionPromptDataset,
    layers: Optional[list[int]] = None,
    representation: str = "final_token",
    alpha_values: Optional[list[float]] = None,
    verbose: bool = True,
) -> EmotionDirectionResult:
    metadata, layer_activations, reference_layers = extract_residual_stream_representations(
        model,
        dataset,
        layers=layers,
        representation=representation,
        verbose=verbose,
    )
    alpha_values = alpha_values or [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]

    emotion_labels = sorted(label for label in metadata["emotion"].unique().tolist() if label != "neutral")
    lexeme_map = _build_lexeme_map(metadata)

    mean_vectors: dict[tuple[str, str, int], np.ndarray] = {}
    direction_vectors: dict[tuple[str, int], np.ndarray] = {}
    summary_rows: list[dict] = []
    opposite_rows: list[dict] = []
    negation_rows: list[dict] = []

    for layer, activations in layer_activations.items():
        neutral_mask = (metadata["emotion"] == "neutral").values
        neutral_mean = activations[neutral_mask].mean(axis=0) if neutral_mask.any() else None

        for emotion in emotion_labels:
            affirmed_mask = (
                (metadata["emotion"] == emotion)
                & (metadata["prompt_kind"] == "affirmed")
            ).values
            negated_mask = (
                (metadata["emotion"] == emotion)
                & (metadata["prompt_kind"] == "negated")
            ).values

            if affirmed_mask.sum() == 0 or negated_mask.sum() == 0:
                continue

            mu_affirmed = activations[affirmed_mask].mean(axis=0)
            mu_negated = activations[negated_mask].mean(axis=0)
            direction = mu_affirmed - mu_negated

            mean_vectors[(emotion, "affirmed", layer)] = mu_affirmed
            mean_vectors[(emotion, "negated", layer)] = mu_negated
            direction_vectors[(emotion, layer)] = direction

            affirmed_proj, negated_proj, projection_gap = _projection_gap(
                activations=activations,
                affirmed_mask=affirmed_mask,
                negated_mask=negated_mask,
                direction=direction,
            )

            summary_rows.append(
                {
                    "emotion": emotion,
                    "layer": layer,
                    "n_affirmed": int(affirmed_mask.sum()),
                    "n_negated": int(negated_mask.sum()),
                    "direction_norm": float(np.linalg.norm(direction)),
                    "symmetry_cosine": _symmetry_cosine(metadata, activations, emotion),
                    "template_stability_cosine": _template_stability(metadata, activations, emotion),
                    "lexeme_stability_cosine": _lexeme_stability(metadata, activations, emotion),
                    "probe_accuracy": _linear_probe_accuracy(metadata, activations, emotion),
                    "affirmed_projection_mean": affirmed_proj,
                    "negated_projection_mean": negated_proj,
                    "projection_gap": projection_gap,
                }
            )

            opposite_emotion = next(
                (
                    value
                    for value in metadata.loc[metadata["emotion"] == emotion, "opposite_emotion"].dropna().unique().tolist()
                ),
                None,
            )
            opposite_mean = None
            if opposite_emotion:
                opposite_affirmed_mask = (
                    (metadata["emotion"] == opposite_emotion)
                    & (metadata["prompt_kind"] == "affirmed")
                ).values
                if opposite_affirmed_mask.any():
                    opposite_mean = activations[opposite_affirmed_mask].mean(axis=0)
                    mean_vectors[(opposite_emotion, "affirmed", layer)] = opposite_mean
                    opposite_direction = mu_affirmed - opposite_mean
                    opposite_rows.append(
                        {
                            "emotion": emotion,
                            "opposite_emotion": opposite_emotion,
                            "layer": layer,
                            "contrast_norm": float(np.linalg.norm(opposite_direction)),
                            "negation_vs_opposite_cosine": _cosine_similarity(direction, opposite_direction),
                        }
                    )

            negated_vectors = activations[negated_mask]
            dist_to_neutral = float("nan")
            cos_to_neutral = float("nan")
            dist_to_opposite = float("nan")
            cos_to_opposite = float("nan")
            closer_to_neutral_rate = float("nan")
            if neutral_mean is not None:
                dist_to_neutral = float(
                    np.linalg.norm(negated_vectors - neutral_mean[None, :], axis=1).mean()
                )
                cos_to_neutral = float(
                    np.nanmean([_cosine_similarity(vector, neutral_mean) for vector in negated_vectors])
                )
            if opposite_mean is not None:
                dist_to_opposite = float(
                    np.linalg.norm(negated_vectors - opposite_mean[None, :], axis=1).mean()
                )
                cos_to_opposite = float(
                    np.nanmean([_cosine_similarity(vector, opposite_mean) for vector in negated_vectors])
                )
                neutral_distances = np.linalg.norm(negated_vectors - neutral_mean[None, :], axis=1)
                opposite_distances = np.linalg.norm(negated_vectors - opposite_mean[None, :], axis=1)
                closer_to_neutral_rate = float((neutral_distances < opposite_distances).mean())

            negation_rows.append(
                {
                    "emotion": emotion,
                    "layer": layer,
                    "opposite_emotion": opposite_emotion,
                    "distance_to_neutral": dist_to_neutral,
                    "distance_to_opposite": dist_to_opposite,
                    "cosine_to_neutral": cos_to_neutral,
                    "cosine_to_opposite": cos_to_opposite,
                    "closer_to_neutral_rate": closer_to_neutral_rate,
                }
            )

    summary_df = pd.DataFrame(summary_rows).sort_values(["emotion", "layer"]).reset_index(drop=True)
    opposite_df = pd.DataFrame(opposite_rows).sort_values(["emotion", "layer"]).reset_index(drop=True)
    negation_df = pd.DataFrame(negation_rows).sort_values(["emotion", "layer"]).reset_index(drop=True)

    # PCA on early/middle/late layers for visualization.
    pca_rows: list[dict] = []
    for layer_band, layer in reference_layers.items():
        if layer not in layer_activations:
            continue
        coords, explained = _pca_projection(layer_activations[layer])
        layer_df = metadata.copy()
        layer_df["layer"] = layer
        layer_df["layer_band"] = layer_band
        layer_df["pc1"] = coords[:, 0]
        layer_df["pc2"] = coords[:, 1]
        layer_df["pc1_explained"] = float(explained[0]) if len(explained) > 0 else float("nan")
        layer_df["pc2_explained"] = float(explained[1]) if len(explained) > 1 else float("nan")
        pca_rows.append(layer_df)
    pca_df = pd.concat(pca_rows, ignore_index=True) if pca_rows else pd.DataFrame()

    # Linearity / causal direction injection on the strongest layer per emotion.
    emotion_token_ids = {
        emotion: [model.to_single_token(f" {lexeme}") for lexeme in lexeme_map.get(emotion, [])]
        for emotion in emotion_labels
    }
    neutral_token_ids = [model.to_single_token(f" {lexeme}") for lexeme in lexeme_map.get("neutral", [])]

    linearity_curve_frames: list[pd.DataFrame] = []
    linearity_summary_frames: list[pd.DataFrame] = []
    for emotion in emotion_labels:
        emotion_rows = summary_df[summary_df["emotion"] == emotion]
        if emotion_rows.empty:
            continue
        peak_row = emotion_rows.sort_values("direction_norm", ascending=False).iloc[0]
        peak_layer = int(peak_row["layer"])
        direction = direction_vectors[(emotion, peak_layer)]

        opposite_rows_for_emotion = opposite_df[opposite_df["emotion"] == emotion]
        if not opposite_rows_for_emotion.empty:
            opposite_emotion = str(opposite_rows_for_emotion.iloc[0]["opposite_emotion"])
            control_token_ids = emotion_token_ids.get(opposite_emotion, neutral_token_ids)
            control_label = opposite_emotion
        else:
            control_token_ids = neutral_token_ids
            control_label = "neutral"

        curve_df, summary_curve_df = _direction_linearity_sweep(
            model=model,
            emotion=emotion,
            layer=peak_layer,
            direction=direction,
            activations=layer_activations[peak_layer],
            emotion_token_ids=emotion_token_ids[emotion],
            control_token_ids=control_token_ids,
            alpha_values=alpha_values,
        )
        if not curve_df.empty:
            curve_df["contrast_label"] = control_label
            linearity_curve_frames.append(curve_df)
        if not summary_curve_df.empty:
            summary_curve_df["contrast_label"] = control_label
            linearity_summary_frames.append(summary_curve_df)

    linearity_curves = (
        pd.concat(linearity_curve_frames, ignore_index=True)
        if linearity_curve_frames
        else pd.DataFrame()
    )
    linearity_summary = (
        pd.concat(linearity_summary_frames, ignore_index=True)
        if linearity_summary_frames
        else pd.DataFrame()
    )

    return EmotionDirectionResult(
        metadata=metadata,
        summary=summary_df,
        opposite_summary=opposite_df,
        negation_sensitivity=negation_df,
        pca=pca_df,
        linearity_curves=linearity_curves,
        linearity_summary=linearity_summary,
        direction_vectors=direction_vectors,
        mean_vectors=mean_vectors,
        activations=layer_activations,
        reference_layers=reference_layers,
    )
