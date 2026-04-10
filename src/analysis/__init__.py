from .per_head import compute_head_dla_batch, per_head_dla, select_top_heads, top_inhibition_heads
from .amplification import amplify_heads, amplification_sweep, dataset_amplification_experiment
from .emotion_directions import (
    EmotionDirectionResult,
    analyze_emotion_negation,
    extract_residual_stream_representations,
    ridge_probe_accuracy,
    select_reference_layers,
)
from .patching import activation_patching_scan, patched_prompt_metrics, dataset_activation_patching_experiment
