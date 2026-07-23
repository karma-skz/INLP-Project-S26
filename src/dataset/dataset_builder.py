"""
src/dataset/dataset_builder.py
==============================
Consolidated module for all dataset generation, fetching, and processing.
Combines parity generation, rule-based reasoning negation, and dataset fetching.
"""

import os
import json
import urllib.request
import subprocess
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Iterator, List, Optional

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class PromptPair:
    case_id: int
    subject: str
    positive_prompt: str
    negated_prompt: str
    target_token: str
    target_false: str

@dataclass
class ParityTriplet:
    P: str
    not_P: str
    not_not_P: str
    target: str

# ---------------------------------------------------------------------------
# Dataset Downloader
# ---------------------------------------------------------------------------
class DatasetDownloader:
    @staticmethod
    def download_all():
        base_dir = Path(__file__).parent.parent.parent
        data_dir = base_dir / "data"
        factual_dir = data_dir / "factual"
        reasoning_dir = data_dir / "reasoning"
        
        factual_dir.mkdir(parents=True, exist_ok=True)
        reasoning_dir.mkdir(parents=True, exist_ok=True)
        
        DatasetDownloader._download_counterfact(factual_dir)
        DatasetDownloader._download_gsm8k(reasoning_dir)
        DatasetDownloader._download_negated_lama(factual_dir)
        DatasetDownloader._download_condaqa(reasoning_dir)
        DatasetDownloader._download_thunder_nubench(factual_dir)
        print("Dataset download phase complete.")

    @staticmethod
    def _download_counterfact(output_dir: Path):
        print("Downloading CounterFact...")
        url = "https://rome.baulab.info/data/dsets/counterfact.json"
        output_path = output_dir / "counterfact.json"
        if output_path.exists():
            print("CounterFact already exists. Skipping.")
            return
        try:
            urllib.request.urlretrieve(url, output_path)
            print(f"Successfully saved CounterFact to {output_path}")
        except Exception as e:
            print(f"Failed to download CounterFact: {e}")

    @staticmethod
    def _download_gsm8k(output_dir: Path):
        print("Downloading GSM8K...")
        try:
            from datasets import load_dataset
            dataset = load_dataset("gsm8k", "main")
            output_path = output_dir / "gsm8k_train.jsonl"
            dataset["train"].to_json(output_path)
            print(f"Successfully saved GSM8K to {output_path}")
        except ImportError:
            print("HuggingFace 'datasets' library not installed. Skipping GSM8K.")
        except Exception as e:
            print(f"Failed to download GSM8K: {e}")

    @staticmethod
    def _download_negated_lama(output_dir: Path):
        print("Downloading Negated LAMA...")
        url = "https://dl.fbaipublicfiles.com/LAMA/negated_data.tar.gz"
        tar_path = output_dir / "negated_data.tar.gz"
        if (output_dir / "negated_data").exists():
            print("Negated LAMA already exists. Skipping.")
            return
        try:
            urllib.request.urlretrieve(url, tar_path)
            subprocess.run(["tar", "-xzvf", str(tar_path), "-C", str(output_dir)], check=True, capture_output=True)
            os.remove(tar_path)
            print(f"Successfully downloaded Negated LAMA to {output_dir}")
        except Exception as e:
            print(f"Failed to download Negated LAMA: {e}")

    @staticmethod
    def _download_condaqa(output_dir: Path):
        print("Downloading CondaQA...")
        target_dir = output_dir / "CondaQA"
        if target_dir.exists():
            print("CondaQA already exists. Skipping.")
            return
        try:
            subprocess.run(["git", "clone", "https://github.com/AbhilashaRavichander/CondaQA.git", str(target_dir)], check=True, capture_output=True)
            print(f"Successfully cloned CondaQA to {target_dir}")
        except Exception as e:
            print(f"Failed to clone CondaQA: {e}")

    @staticmethod
    def _download_thunder_nubench(output_dir: Path):
        print("Downloading Thunder-NUBench...")
        target_dir = output_dir / "SNU_Thunder-NUBench"
        if target_dir.exists():
            print("Thunder-NUBench already exists. Skipping.")
            return
        try:
            subprocess.run(["git", "clone", "https://huggingface.co/datasets/thunder-research-group/SNU_Thunder-NUBench", str(target_dir)], check=True, capture_output=True)
            print(f"Successfully cloned Thunder-NUBench to {target_dir}")
        except Exception as e:
            print(f"Failed to clone Thunder-NUBench: {e}")

# ---------------------------------------------------------------------------
# Generator Modules
# ---------------------------------------------------------------------------
class ParityGenerator:
    """Generates Factual Triplet datasets for DNE evaluation."""
    def generate_factual_triplets(self, base_facts: List[dict]) -> List[ParityTriplet]:
        triplets = []
        for fact in base_facts:
            subject = fact.get("subject", "")
            relation = fact.get("relation", "")
            target = fact.get("target", "")
            if not subject or not relation or not target:
                continue
            
            p = f"The {relation} of {subject} is"
            not_p = f"The {relation} of {subject} is not"
            not_not_p = f"The {relation} of {subject} is not not"
            
            triplets.append(ParityTriplet(p, not_p, not_not_p, target))
        return triplets

class ReasoningNegator:
    """Inverts logical constraints in reasoning problems."""
    def negate_question(self, question: str) -> str:
        """
        Robust syntactic heuristic for logical negation.
        Note: For full academic rigor, an LLM pass or strict Dependency Parsing (spaCy)
        should be utilized. This rule-based matcher serves as an offline approximation.
        """
        # Replace main verbs with negated forms
        replacements = [
            (r"\bis\b", "is not"),
            (r"\bare\b", "are not"),
            (r"\bhas\b", "does not have"),
            (r"\bhave\b", "do not have"),
            (r"\bwill\b", "will not"),
            (r"\bcan\b", "cannot"),
            (r"\bmust\b", "must not")
        ]
        negated_q = question
        for pattern, replacement in replacements:
            # Only negate the first occurrence to avoid destroying the entire semantic structure
            negated_q = re.sub(pattern, replacement, negated_q, count=1)
            if negated_q != question:
                break
        return negated_q

# ---------------------------------------------------------------------------
# Loader Functions
# ---------------------------------------------------------------------------
def load_counterfact(split: str = "train", max_samples: Optional[int] = None) -> List[PromptPair]:
    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError:
        print("HuggingFace 'datasets' not installed. Cannot load CF dynamically.")
        return []

    raw = hf_load_dataset("NeelNanda/counterfact-tracing", split=split)
    pairs = []
    
    for raw_idx, entry in enumerate(raw):
        if max_samples and len(pairs) >= max_samples:
            break
            
        case_id = int(entry.get("case_id", raw_idx))
        subject = entry.get("subject", "")
        positive_prompt = entry.get("prompt", "").strip()
        
        raw_true = entry.get("target_true", "")
        if isinstance(raw_true, dict):
            raw_true = raw_true.get("str", "")
        target_token = " " + str(raw_true).strip()
        
        raw_false = entry.get("target_false", "")
        if isinstance(raw_false, dict):
            raw_false = raw_false.get("str", "")
        target_false = " " + str(raw_false).strip()
        
        if not positive_prompt or not target_token.strip():
            continue
            
        negated_prompt = positive_prompt + " not"
        pairs.append(PromptPair(case_id, subject, positive_prompt, negated_prompt, target_token, target_false))
        
    return pairs

def main():
    print("Dataset Builder initialized.")

if __name__ == "__main__":
    main()
