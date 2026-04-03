"""Quick project readiness checks for local workflow."""

from __future__ import annotations

import os


def check(path: str) -> tuple[bool, str]:
    ok = os.path.exists(path)
    return ok, path


def main() -> None:
    required = [
        "config.yaml",
        "requirements.txt",
        "data/ECL (1).csv",
        "data/processed/tabular.csv",
        "data/processed/graph.pt",
        "data/raw/mda_texts",
        "src/train.py",
        "src/evaluate.py",
        "app/streamlit_app.py",
    ]
    optional = [
        "data/processed/text_embeddings.pt",
        "checkpoints/local_lite/model.joblib",
        "checkpoints/multimodal/best_model.pt",
    ]

    print("=== Required ===")
    missing = 0
    for p in required:
        ok, name = check(p)
        if ok:
            print(f"[OK]   {name}")
        else:
            print(f"[MISS] {name}")
            missing += 1

    print("\n=== Optional ===")
    for p in optional:
        ok, name = check(p)
        print(f"[{'OK' if ok else 'MISS'}] {name}")

    print("\n=== Summary ===")
    if missing == 0:
        print("[READY] Core project files are in place.")
    else:
        print(f"[ACTION] Missing required items: {missing}")


if __name__ == "__main__":
    main()
