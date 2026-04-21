# Clinical NLP with Biomedical Text Data

**Course Project 3 — Multi-Class Text Classification on Medical Questions**

---

## Project Description

This project implements a natural language processing (NLP) text classification algorithm applied to biomedical data. Specifically, we formulate a multiple-choice medical question answering (MCQA) task as a multi-class text classification problem.

Each question is paired with four candidate answer choices, and the model is trained to classify which answer is correct. This is achieved by evaluating each (question, answer choice) pair and selecting the most probable class among four possible labels (A, B, C, D).

Although this task is framed as question answering, it is fundamentally a supervised multi-class classification problem, where each answer choice represents a distinct class. The model learns to map input text to one of these discrete labels, satisfying the definition of a text classification NLP algorithm.

This approach aligns with standard NLP classification frameworks while extending them to a more complex and clinically relevant setting involving medical reasoning.

---

## Project Overview

We fine-tune a pretrained transformer model (DistilBERT or BERT) on the **MedMCQA** dataset — a large-scale, multi-subject medical multiple-choice question answering benchmark. The task is treated as 4-class classification: given a question and four answer options (A, B, C, D), the model predicts the correct option.

The pipeline is fully modular:

```
project_root/
│
├── data/               ← Downloaded/cached dataset (git-ignored)
├── outputs/            ← Model checkpoints, logs, predictions (git-ignored)
├── src/
│   ├── data.py         ← Dataset loading & preprocessing (James Garner)
│   ├── model.py        ← Model & tokenizer initialisation (Pascual Jahuey)
│   ├── train.py        ← Training pipeline (Pascual Jahuey)
│   ├── evaluate.py     ← Evaluation & error analysis (Riley Bendure)
│   ├── utils.py        ← Shared utilities
│   └── main.py         ← End-to-end runner (Pascual Jahuey)
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Team Roles

| Member | Responsibilities |
|---|---|
| **Carolina Horey** | Introduction, Literature Review, Clinical Framing, Discussion |
| **James Garner** | Dataset loading, Preprocessing pipeline, Label validation, Data documentation (`src/data.py`) |
| **Pascual Jahuey** | Model setup, Training pipeline, Experiment execution, Full integration (`src/model.py`, `src/train.py`, `src/main.py`) |
| **Riley Bendure** | Evaluation, Accuracy metrics, Error analysis, Figures/tables, README polish (`src/evaluate.py`) |

---

## Clinical Relevance

Medical question answering is a critical capability for clinical decision support systems. The MedMCQA dataset is derived from real entrance exam questions spanning 21 medical subjects (anatomy, pharmacology, pathology, etc.). Models that can correctly answer such questions can:

- Assist medical students in exam preparation
- Support clinical reasoning aids
- Serve as a baseline for more advanced clinical NLP systems (e.g., diagnosis, triage)

Treating MCQA as text classification is a pragmatic approach that leverages well-established transformer fine-tuning recipes while remaining interpretable: each answer choice is scored independently against the question.

---

## Dataset Description

**MedMCQA** ([openlifescienceai/medmcqa](https://huggingface.co/datasets/openlifescienceai/medmcqa)) is a large-scale open-domain multi-choice QA dataset designed to address real-world medical entrance exam questions.

- **Total questions**: ~187,000 (train) + ~6,100 (validation)
- **Answer choices**: 4 options (A, B, C, D)
- **Subjects**: 21 medical subjects
- **We use**: 5,000 training + 1,000 validation samples (configurable)

Each example contains:
- `question` — the medical question text
- `opa`, `opb`, `opc`, `opd` — four candidate answer strings
- `cop` — correct option index (0-based: 0=A, 1=B, 2=C, 3=D)
- `subject_name` — medical subject category

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/Pjahuey/Clinical-NLP-with-Biomedical-Text-Data.git
cd Clinical-NLP-with-Biomedical-Text-Data
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU users**: ensure a CUDA-compatible version of PyTorch is installed.
> Visit [pytorch.org](https://pytorch.org) for platform-specific instructions.

---

## How to Run

Run the full pipeline with default settings (DistilBERT, 3 epochs, 5000 train / 1000 val):

```bash
python -m src.main
```

### Common options

```bash
# Use BERT instead of DistilBERT
python -m src.main --model bert-base-uncased

# Faster run for testing (2 epochs, smaller dataset)
python -m src.main --epochs 2 --train_size 1000 --val_size 200

# Custom output directory
python -m src.main --output_dir my_outputs

# Full option list
python -m src.main --help
```

### CLI arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | `distilbert-base-uncased` | HuggingFace model name or shorthand (`distilbert`/`bert`) |
| `--epochs` | `3` | Number of training epochs |
| `--batch_size` | `8` | Per-device batch size |
| `--learning_rate` | `2e-5` | AdamW learning rate |
| `--train_size` | `5000` | Training samples to use |
| `--val_size` | `1000` | Validation samples to use |
| `--max_length` | `128` | Max token length per (question, option) pair |
| `--output_dir` | `outputs` | Directory for all saved outputs |
| `--seed` | `42` | Random seed for reproducibility |

---

## Outputs

After a successful run the following are saved to `outputs/` (or `--output_dir`):

| File / Folder | Contents |
|---|---|
| `model/` | Trained model weights + tokenizer |
| `logs/` | TensorBoard-compatible training logs |
| `predictions.csv` | Per-example predictions vs. ground truth |
| `metrics.json` | Overall accuracy |
| `correct_examples.csv` | 5 correctly classified examples |
| `incorrect_examples.csv` | 5 misclassified examples |
| `subject_accuracy.csv` | Per-subject accuracy breakdown (if available) |

---

## Results Summary

> Results will be populated after running the full pipeline.

| Model | Train Size | Val Size | Epochs | Val Accuracy |
|---|---|---|---|---|
| distilbert-base-uncased | 5000 | 1000 | 3 | TBD |
| bert-base-uncased | 5000 | 1000 | 3 | TBD |

Random baseline: **25.0%** (4-class uniform random).

---

## Reproducibility Notes

- Random seed is set globally for Python, NumPy, and PyTorch via `src/utils.set_seed()`.
- All dataset subsets use deterministic slicing (`dataset.select(range(N))`).
- Model weights are initialised from a fixed pretrained checkpoint.
- Training uses the Hugging Face `Trainer` API with `seed` passed to `TrainingArguments`.
- To reproduce exactly, use the same environment pinned in `requirements.txt`.

---

## Report Structure

1. **Abstract**
2. **Introduction** — Carolina Horey
3. **Literature Review** — Carolina Horey
4. **Methods & Data** — James Garner
5. **Training & Experiments** — Pascual Jahuey
6. **Results & Evaluation** — Riley Bendure
7. **Discussion** — Full team
8. **References**

---

## License

This project is for educational purposes. The MedMCQA dataset is distributed under its own license; see [Hugging Face dataset page](https://huggingface.co/datasets/openlifescienceai/medmcqa) for details.

