"""
src/analysis/causal_tracing.py
==============================
Unified module for causal interventions (Activation Patching, Amplification)
and calculating the Compositional Interference Score (CIS).
"""

try:
    import torch
except ImportError:
    torch = None
from functools import partial

class CausalInterventions:
    """
    Houses standardized TransformerLens hooks for causal tracing.
    """
    @staticmethod
    def patch_activation_hook(activation, hook, source_activation=None):
        """Standard hook to replace an activation with a source activation."""
        if source_activation is not None:
            return source_activation
        return activation

    @staticmethod
    def amplify_activation_hook(activation, hook, scale=2.0):
        """Standard hook to artificially amplify an activation."""
        return activation * scale

class CISMetric:
    """
    Implements the Compositional Interference Score (CIS) using actual DLA differences.
    """
    @staticmethod
    def compute_dla(model, tokens, layer, head, target_token_id):
        """Computes Direct Logit Attribution for a specific head."""
        # A full robust implementation computes the residual stream contribution
        # and projects it through the unembedding matrix.
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: f"blocks.{layer}.attn.hook_result" in n)
        head_output = cache[f"blocks.{layer}.attn.hook_result"][0, -1, head, :]
        
        # Project through unembedding (W_U)
        w_u = model.W_U[:, target_token_id]
        dla = torch.dot(head_output, w_u).item()
        return dla

    @staticmethod
    def compute_cis(model, p_tokens, not_not_p_tokens, target_token_id, layer, head):
        """
        CIS = (DLA of inhibition head on single negation context) / 
              (DLA of that SAME head on double negation context)
        
        If CIS > 1, the second negator actively attenuates the first's inhibition circuit.
        """
        # Note: In a true experiment, we would first find the inhibition head on a ~P prompt.
        # Here we assume we are evaluating a known inhibition head on P (or ~P) vs ~~P.
        
        # Assume p_tokens is actually the single negation ~P in this context
        base_inhibition = CISMetric.compute_dla(model, p_tokens, layer, head, target_token_id)
        comp_inhibition = CISMetric.compute_dla(model, not_not_p_tokens, layer, head, target_token_id)
        
        if comp_inhibition == 0:
            return float('inf')
            
        return base_inhibition / comp_inhibition
