import matplotlib.pyplot as plt
import numpy as np

def plot_ltvt_trajectory(ltvt_scores: dict, title: str, output_path: str):
    """
    Plots Latent Truth-Value Toggling across layers.
    """
    layers = sorted(list(ltvt_scores.keys()))
    scores = [ltvt_scores[l] for l in layers]
    
    plt.figure(figsize=(10, 6))
    plt.plot(layers, scores, marker='o', linestyle='-')
    plt.axhline(0, color='red', linestyle='--', label="Decision Boundary")
    plt.title(title)
    plt.xlabel("Transformer Layer")
    plt.ylabel("Latent Truth-Value (P vs ~P)")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    print(f"Saved LTVT trajectory plot to {output_path}")

if __name__ == "__main__":
    # Dummy data
    dummy_scores = {i: np.sin(i / 3.0) for i in range(32)}
    plot_ltvt_trajectory(dummy_scores, "Sample LTVT Trajectory", "sample_ltvt.png")
