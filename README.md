# Clinical NLP with Biomedical Text Data
**Course Project 3 — Multi-Class Text Classification on Medical Questions**

## Project Description
This project implements a natural language processing (NLP) text classification pipeline applied to biomedical data. We formulate a multiple-choice medical question answering (MCQA) task as a multi-class text classification problem.

Each question is paired with four candidate answer choices, and the model is trained to classify which answer is correct by evaluating each (question, answer choice) pair and selecting the most probable class among four labels (A, B, C, D).

Three modeling approaches are compared: a bidirectional LSTM baseline trained from scratch, and two pretrained transformer models (DistilBERT and BERT), all evaluated on the same dataset splits.

## Assignment Compliance
- Implements a **supervised multi-class text classification NLP algorithm**
- Biomedical text source: **MedMCQA** (`openlifescienceai/medmcqa`)
- Target labels/classes: **A, B, C, D** (4-class classification)
- Includes: data loading, preprocessing/tokenization, training, validation, evaluation, error analysis, EDA, confusion matrix, training curves, and reproducibility artifacts
- Includes both a **required RNN/LSTM baseline** and **pretrained transformer models**

## Project Status
✅ **Submission-ready** for graduate biomedical NLP course review.

## Project Structure
```text
project_root/
├── figures/                        # Generated plots per model
│   ├── lstm/
│   │   ├── eda/                    # EDA figures
│   │   ├── confusion_matrix.png
│   │   ├── error_breakdown.png
│   │   ├── subject_accuracy.png
│   │   └── training_curves.png
│   ├── distilbert-base-uncased/
│   │   ├── eda/
│   │   ├── confusion_matrix.png
│   │   ├── error_breakdown.png
│   │   ├── subject_accuracy.png
│   │   └── training_curves.png
│   ├── bert-base-uncased/
│   │   ├── eda/
│   │   ├── confusion_matrix.png
│   │   ├── error_breakdown.png
│   │   ├── subject_accuracy.png
│   │   └── training_curves.png
│   └── model_comparison.png
├── outputs/                        # Run artifacts per model
├── reports/
│   └── final_report.md
├── src/
│   ├── data.py                     # Dataset loading & preprocessing (James Garner)
│   ├── model.py                    # Model & tokenizer initialization (Pascual Jahuey)
│   ├── train.py                    # Training pipeline (Pascual Jahuey)
│   ├── evaluate.py                 # Evaluation, error analysis, plots (Riley Bendure)
│   ├── eda.py                      # Exploratory data analysis (Riley Bendure)
│   ├── lstm_model.py               # BiLSTM baseline model (Riley Bendure)
│   ├── tokenization_report.py      # Tokenization documentation (Riley Bendure)
│   ├── utils.py                    # Shared utilities
│   └── main.py                     # End-to-end runner + model comparison (Pascual Jahuey)
├── requirements.txt
└── README.md
```

## Team Roles
| Member | Responsibilities |
|---|---|
| **Carolina Horey** | Introduction, Literature Review, Clinical Framing, Discussion |
| **James Garner** | Dataset loading, preprocessing pipeline, label validation, data documentation (`src/data.py`) |
| **Pascual Jahuey** | Model setup, training pipeline, experiment execution, full integration (`src/model.py`, `src/train.py`, `src/main.py`) |
| **Riley Bendure** | LSTM baseline, evaluation, EDA, error analysis, figures/tables, tokenization docs, README polish (`src/evaluate.py`, `src/lstm_model.py`,`src/eda.py`) |

## Neural Modeling Approches

LSTM Baseline (required)
Architecture: Bidirectional LSTM, 2 layers, hidden size 256, embedding dim 128
Tokenizer: BERT WordPiece tokenizer (vocab size 30,522) — shared with transformer models
Embeddings: Learned from scratch via nn.Embedding
Unknown tokens: Handled by BERT's [UNK] token (id=100)
Padding: Packed sequences used during forward pass to ignore padding
Prediction head: Linear(hidden_dim * 2, 1) per choice, mean-pooled LSTM output

Pretrained Transformer Models (Required)
DistilBERT: distilbert-base-uncased — compressed BERT, 6 layers, 66M parameters
BERT: bert-base-uncased — 12 layers, 110M parameters
Tokenizer: WordPiece, padding="max_length", truncation=True, max_length=128
Fine-tuning: AutoModelForMultipleChoice with AdamW optimizer

Hyperparameters (All Models)
PARAMETER       VALUE
Learning rate   2e-5
Batch size      8
Epochs          3
Max sequence length     128
Warmup ratio    0.1
Weight decay    0.01
Seed            42

## Clinical Context
MedMCQA covers 21 medical subjects (anatomy, pathology, pharmacology, surgery, etc.). This project explores biomedical reasoning in a structured classification setting useful for educational decision support and benchmark-oriented clinical NLP evaluation.

## Dataset Description
**MedMCQA**: https://huggingface.co/datasets/openlifescienceai/medmcqa

- Task: 4-way medical multiple-choice classification
- Labels: A, B, C, D
- Fields used: `question`, `opa`, `opb`, `opc`, `opd`, `cop`, `subject_name`
- Default subset: 5,000 training + 1,000 validation samples (configurable)
- Average question length: 12.7 words

## Setup Instructions
> Recommended Python version: **Python 3.12** 

### 1) Clone repository
```bash
git clone https://github.com/rbendure/Clinical-NLP-with-Biomedical-Text-Data.git
cd Clinical-NLP-with-Biomedical-Text-Data
```

### 2) Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate                    # Linux/macOS
venv\Scripts\activate.bat                  # Windows (Command Prompt)
# or: .\venv\Scripts\activate.bat         # Windows PowerShell
```

### 3) Install dependencies
```bash
pip install -r requirements.txt
```

## Quick Start
# Run all three models
```bash
python -m src.main --compare_models
```
# Single model
```bash
python -m src.main --model lstm
python -m src.main --model distilbert
python -m src.main --model bert
```
# Smoke test
```bash
python -m src.main --compare_models --epochs 1 --train_size 100 --val_size 50
```

Or with helper scripts:
```bash
./run_experiment.sh
# Windows:
run_experiment.bat
```


### CLI Arguments
| Argument | Type | Default | Description |
|---|---|---|---|
| `--model` | `str` | `distilbert-base-uncased` | Single model name/alias or comma-separated model list |
| `--compare_models` | `flag` | `False` | Runs `distilbert-base-uncased` and `bert-base-uncased` and `lstm` together |
| `--epochs` | `int` | `3` | Number of training epochs |
| `--batch_size` | `int` | `8` | Per-device training batch size |
| `--eval_batch_size` | `int` | `16` | Per-device evaluation batch size |
| `--learning_rate` | `float` | `2e-5` | AdamW learning rate |
| `--train_size` | `int` | `5000` | Training subset size |
| `--val_size` | `int` | `1000` | Validation subset size |
| `--max_length` | `int` | `128` | Max token length per (question, option) pair |
| `--output_dir` | `str` | `outputs` | Output Directory |
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
| `outputs/model_comparison.csv` | Side by side model metrics |
| `outputs/model_comparison.png` | Model comaprison bar chart |
| `figures/<model>/confusion_matrix.png` | Confusion matrix per model |
| `figures/<model>/training_curves.png` | Train/val loss curves per model |
| `figures/<model>/error_breakdown.png` | Correct vs incorrect counts |
| `figures/<model>/subject_accuracy.png` | Subject-wise accuracy |
| `figures/<model>/eda` | EDA figures |

When running two models:

| File / Folder | Description |
|---|---|
| `outputs/model_comparison.csv` | Side-by-side model metrics summary |
| `outputs/readme_placeholders.json` | Optional placeholder map for external automation |
| `figures/model_comparison.png` | Model comparison bar chart with 25% random baseline |

## Results Summary
| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---:|---:|---:|---:|
| BERT-base-uncased | 30.60% | 30.97% | 31.18% | 30.48% |
| DistilBERT-base-uncased | 28.30% | 28.49% | 29.14% | 28.29% |
| LSTM (BiLSTM baseline) | 26.10% | 26.48% | 26.36% | 25.92% |
| Random baseline | 25.00% | — | — | — |
Best model: **distilbert-base-uncased**

Random baseline (4-class uniform): **25.0%**

### Figure 1: Model Comparison
All three models perform above the 25% random baseline

BERT outperforms DistilBERT by ~2.3%, consistent with its larger capacity

The LSTM baseline (26.10%) sits just above random chance, demonstrating the significant advantage of pretrained contextual representations for medical text

Low overall accuracy reflects the difficulty of medical reasoning with only 5,000 training examples and 3 epochs


## Error Analysis
Most errors involve semantically similar options, negation questions ("which is NOT true"), long question stems, and rare medical terminology where shallow lexical overlap is insufficient for correct reasoning.

| Category | Subject | Question (shortened) | True | Predicted |
|---|---|---|---|---|
| Correct | Anatomy | Which nerve supplies the deltoid muscle? | C | C |
| Correct | Pharmacology | First-line treatment for anaphylaxis is: | A | A |
| Correct | Pathology | Reed-Sternberg cells are associated with: | B | B |
| Incorrect | Biochemistry | Rate-limiting enzyme of glycolysis is: | A | C |
| Incorrect | Physiology | Primary determinant of pulse pressure is: | D | B |
| Incorrect | Microbiology | Most common cause of lobar pneumonia is: | B | A |

## Reproducibility Notes
- Deterministic seeds are set for Python, NumPy, and PyTorch.
- Subset selection uses deterministic indexing (`select(range(N))`).
- Run settings are saved to `config.json` for each execution.
- Output and figure directories are created automatically.

## Limitations
- Low accuracy reflects small training subset (5,000 examples) and limited epochs
- LSTM baseline lacks pretrained medical knowledge — performs near random chance
- Performance is benchmark-oriented and does not imply clinical deployment readiness
- Domain-specific models (BioClinicalBERT, PubMedBERT) may improve results significantly

## References
1. Devlin, J., et al. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (2019).
2. Sanh, V., et al. DistilBERT, a distilled version of BERT (2019).
3. Pal, A., et al. MedMCQA: A Large-scale Multi-Subject Multi-Choice Dataset for Medical QA (2022).
4. Wolf, T., et al. Transformers: State-of-the-Art Natural Language Processing (2020).
5. Hugging Face Datasets: MedMCQA card — https://huggingface.co/datasets/openlifescienceai/medmcqa

## License
This project is for educational purposes. MedMCQA is distributed under its own license; see the dataset card for details.
