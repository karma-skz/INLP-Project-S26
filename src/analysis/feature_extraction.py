"""
src/analysis/feature_extraction.py
===================================
Module for extracting interpretable features from the residual stream using
Sparse Autoencoders (SAEs) and Latent Truth-Value Toggling (LTVT) probes.
"""

import numpy as np
try:
    from sklearn.linear_model import LogisticRegression
except ImportError:
    LogisticRegression = None

class LatentTruthProbe:
    """
    Trains logistic regression probes on intermediate representations to track the binary
    truth state (P vs ~P) and calculate Latent Truth-Value Toggling (LTVT).
    """
    def __init__(self, d_model: int):
        self.d_model = d_model
        self.probes = {} 

    def train_probes(self, acts_p: dict, acts_not_p: dict):
        """
        Trains a Logistic Regression model at each layer to separate P from ~P.
        acts_p: dict mapping layer_idx -> numpy array of activations for P (shape: [N, d_model])
        acts_not_p: dict mapping layer_idx -> numpy array of activations for ~P (shape: [N, d_model])
        """
        if LogisticRegression is None:
            print("WARNING: scikit-learn not installed. Probe training requires sklearn.")
            return

        for layer in acts_p.keys():
            if layer not in acts_not_p:
                continue
            
            X_p = acts_p[layer]
            X_not_p = acts_not_p[layer]
            
            X = np.vstack([X_p, X_not_p])
            y = np.concatenate([np.ones(len(X_p)), np.zeros(len(X_not_p))])
            
            clf = LogisticRegression(solver='liblinear', max_iter=1000)
            clf.fit(X, y)
            
            # Store the trained classifier
            self.probes[layer] = clf
            
    def compute_ltvt(self, target_acts: dict) -> dict:
        """
        Calculates the probability mass towards P (1.0) vs ~P (0.0) for target acts.
        target_acts: dict mapping layer_idx -> numpy array of shape [1, d_model]
        """
        ltvt_scores = {}
        for layer, act in target_acts.items():
            if layer in self.probes:
                clf = self.probes[layer]
                # Return the probability of class 1 (P)
                prob = clf.predict_proba(act.reshape(1, -1))[0, 1]
                ltvt_scores[layer] = prob
        return ltvt_scores

class SAETracker:
    """
    Integrates with `sae_lens` to extract specific sparse features from the residual stream.
    """
    def __init__(self, model_name: str, sae_release: str = "gpt2-small-res-jb"):
        self.model_name = model_name
        self.sae_release = sae_release
        self.saes = {}
        
    def load_saes(self, layers: list):
        """Loads SAEs from HuggingFace via sae_lens."""
        try:
            from sae_lens import SAE
            print(f"Loading SAEs for layers {layers}...")
            for layer in layers:
                sae, _, _ = SAE.from_pretrained(
                    release=self.sae_release, 
                    sae_id=f"blocks.{layer}.hook_resid_post"
                )
                self.saes[layer] = sae
        except ImportError:
            print("WARNING: sae_lens not installed. SAE tracking disabled.")

    def extract_features(self, hidden_states: dict) -> dict:
        """
        Encodes dense hidden states into sparse feature activations.
        hidden_states: dict layer -> torch.Tensor of shape [batch, pos, d_model]
        """
        features = {}
        for layer, acts in hidden_states.items():
            if layer in self.saes:
                sae = self.saes[layer]
                # Encode the activations
                sparse_acts = sae.encode(acts)
                features[layer] = sparse_acts
            else:
                features[layer] = None
        return features
