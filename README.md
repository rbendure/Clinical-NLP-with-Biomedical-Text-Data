# Clinical NLP with Biomedical Text Data

**Course Project 3 — Multi-Class Text Classification on Medical Questions**

## Project Description
This project implements a natural language processing (NLP) text classification algorithm applied to biomedical data. Specifically, we formulate a multiple-choice medical question answering (MCQA) task as a multi-class text classification problem.

Each question is paired with four candidate answer choices, and the model is trained to classify which answer is correct. This is achieved by evaluating each (question, answer choice) pair and selecting the most probable class among four possible labels (A, B, C, D).

Although this task is framed as question answering, it is fundamentally a supervised multi-class classification problem, where each answer choice represents a distinct class. The model learns to map input text to one of these discrete labels, satisfying the definition of a text classification NLP algorithm.

This approach aligns with standard NLP classification frameworks while extending them to a more complex and clinically relevant setting involving medical reasoning.

## Assignment Compliance
- This repository implements a **supervised multi-class text classification NLP algorithm**.
- The biomedical text source is **MedMCQA** (`openlifescienceai/medmcqa`).
- The target labels/classes are **A, B, C, D** (4-class classification).
- The repository includes data loading, preprocessing/tokenization, training, validation, evaluation, error analysis, and reproducibility artifacts.

## Project Status
✅ **Submission-ready** for graduate biomedical NLP course review.

## Project Overview
We fine-tune pretrained transformer models (`distilbert-base-uncased` and `bert-base-uncased`) on **MedMCQA**. The task is treated as 4-class classification: given a question and four answer options (A, B, C, D), the model predicts the correct option.

## Project Structure
```text
project_root/
├── figures/                     # Generated plots (subject/model comparison)
├── outputs/                     # Run artifacts (config, metrics, predictions, etc.)
├── reports/
│   └── final_report.md          # Course report draft in markdown
├── src/
│   ├── data.py                  # Dataset loading & preprocessing (James Garner)
│   ├── model.py                 # Model & tokenizer initialization (Pascual Jahuey)
│   ├── train.py                 # Training pipeline + Trainer metrics
│   ├── evaluate.py              # Evaluation, error analysis, plots (Riley Bendure)
│   ├── utils.py                 # Shared utilities
│   └── main.py                  # End-to-end runner + model comparison
├── run_experiment.sh            # Quick run helper (Linux/macOS)
├── run_experiment.bat           # Quick run helper (Windows)
├── requirements.txt
└── README.md
```

## Team Roles
| Member | Responsibilities |
|---|---|
| **Carolina Horey** | Introduction, Literature Review, Clinical Framing, Discussion |
| **James Garner** | Dataset loading, preprocessing pipeline, label validation, data documentation (`src/data.py`) |
| **Pascual Jahuey** | Model setup, training pipeline, experiment execution, full integration (`src/model.py`, `src/train.py`, `src/main.py`) |
| **Riley Bendure** | Evaluation, metrics, error analysis, figures/tables, README polish (`src/evaluate.py`) |

## Clinical Context
MedMCQA covers 21 medical subjects (anatomy, pathology, pharmacology, surgery, etc.). This project explores biomedical reasoning in a structured classification setting useful for educational decision support and benchmark-oriented clinical NLP evaluation.

## Dataset Description
**MedMCQA**: https://huggingface.co/datasets/openlifescienceai/medmcqa

- Task: 4-way medical multiple-choice classification
- Labels: A, B, C, D
- Fields used: `question`, `opa`, `opb`, `opc`, `opd`, `cop`, `subject_name`
- Default subset: 5,000 training + 1,000 validation samples (configurable)

## Setup Instructions
> Recommended Python version: **Python 3.10+** (tested with modern 3.10/3.11 environments).

### 1) Clone repository
```bash
git clone https://github.com/Pjahuey/Clinical-NLP-with-Biomedical-Text-Data.git
cd Clinical-NLP-with-Biomedical-Text-Data
```

### 2) Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate                    # Linux/macOS
venv\Scripts\activate.bat                 # Windows (Command Prompt)
# or: .\venv\Scripts\Activate.ps1         # Windows PowerShell
```

### 3) Install dependencies
```bash
pip install -r requirements.txt
```

## Quick Start
Default single-model run:
```bash
python -m src.main
```

Or with helper scripts:
```bash
./run_experiment.sh
# Windows:
run_experiment.bat
```

## Usage
### Single model
```bash
python -m src.main --model distilbert-base-uncased
python -m src.main --model bert-base-uncased
```

### Compare both required models in one run
```bash
python -m src.main --compare_models
```

### Faster smoke run
```bash
python -m src.main --epochs 1 --train_size 500 --val_size 200
```

### CLI Arguments
| Argument | Type | Default | Description |
|---|---|---|---|
| `--model` | `str` | `distilbert-base-uncased` | Single model name/alias or comma-separated model list |
| `--compare_models` | `flag` | `False` | Runs `distilbert-base-uncased` and `bert-base-uncased` together |
| `--epochs` | `int` | `3` | Number of training epochs |
| `--batch_size` | `int` | `8` | Per-device training batch size |
| `--eval_batch_size` | `int` | `16` | Per-device evaluation batch size |
| `--learning_rate` | `float` | `2e-5` | AdamW learning rate |
| `--train_size` | `int` | `5000` | Training subset size |
| `--val_size` | `int` | `1000` | Validation subset size |
| `--max_length` | `int` | `128` | Max token length per (question, option) pair |
| `--output_dir` | `str` | `outputs` | Root directory for artifacts |
| `--figure_dir` | `str` | `figures` | Directory for generated figures |
| `--seed` | `int` | `42` | Random seed |

## Outputs
After each run, artifacts are automatically saved:

| File / Folder | Description |
|---|---|
| `outputs/config.json` | Full run configuration |
| `outputs/metrics.json` | Accuracy, macro precision/recall/F1 |
| `outputs/predictions.csv` | Per-example predictions |
| `outputs/correct_examples.csv` | Sample correct predictions |
| `outputs/incorrect_examples.csv` | Sample incorrect predictions |
| `outputs/subject_accuracy.csv` | Subject-wise accuracy (if available) |
| `outputs/model/` | Trained model + tokenizer |
| `figures/subject_accuracy.png` | Subject-wise accuracy bar chart (if available) |

When running two models:

| File / Folder | Description |
|---|---|
| `outputs/model_comparison.csv` | Side-by-side model metrics summary |
| `outputs/readme_placeholders.json` | Auto-generated values for README result placeholders |
| `figures/model_comparison_accuracy.png` | Model comparison bar chart |

## Results Summary (placeholders)
These placeholders are auto-populated in `outputs/readme_placeholders.json` after comparison runs.

| Model | Train Size | Val Size | Epochs | Val Accuracy |
|---|---:|---:|---:|---:|
| distilbert-base-uncased | 5000 | 1000 | 3 | `{{DISTILBERT_VAL_ACCURACY}}` |
| bert-base-uncased | 5000 | 1000 | 3 | `{{BERT_VAL_ACCURACY}}` |

Best model placeholder: `{{BEST_MODEL}}`

Random baseline (4-class uniform): **25.0%**

## Reproducibility Notes
- Deterministic seeds are set for Python, NumPy, and PyTorch.
- Subset selection uses deterministic indexing (`select(range(N))`).
- Run settings are saved to `config.json` for each execution.
- Output and figure directories are created automatically.
- Input validation catches unsupported model names and invalid subset sizes with clear errors.

## Limitations
- Current training defaults use subsets for practical runtime.
- This repository version emphasizes transformer models and does not include a full RNN/LSTM baseline implementation.
- Performance is benchmark-oriented and does not imply clinical deployment readiness.
- Additional external validation is needed before any real-world clinical use.

## References
1. Devlin, J., et al. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (2019).
2. Sanh, V., et al. DistilBERT, a distilled version of BERT (2019).
3. Pal, A., et al. MedMCQA: A Large-scale Multi-Subject Multi-Choice Dataset for Medical QA (2022).
4. Wolf, T., et al. Transformers: State-of-the-Art Natural Language Processing (2020).
5. Hugging Face Datasets: MedMCQA card — https://huggingface.co/datasets/openlifescienceai/medmcqa

## License
This project is for educational purposes. MedMCQA is distributed under its own license; see the dataset card for details.
