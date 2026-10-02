# Individual contributions

Copy these rows into the official Google Sheets tracker, replacing "Member N" with names. Keep entries
factual: what you built, which experiments you ran (IDs from `experiments/`), and which report and demo
parts you wrote.

| Member | Model(s) trained | Experiments | Code / notebooks | Report sections | Demo segment |
|---|---|---|---|---|---|
| Member 1 | TF-IDF + LR; fastText-style bag | BL-01 – BL-06, FT-01 – FT-02 | `01_eda`, `02_baselines`, `src/data.py`, `src/text.py` | Introduction; Dataset & EDA | Problem, data, baselines |
| Member 2 | BiLSTM/BiGRU + attention | RNN-01 – RNN-07 | `04_birnn_attention`, `src/models.py` (BiRNN), `src/datasets.py` | Related Work; Methodology (RNN) | Literature + BiRNN |
| Member 3 | TextCNN | CNN-01 – CNN-06 | `03_textcnn`, `src/train.py`, `src/evaluate.py`, `06` (comparison) | Evaluation metrics; Results & Discussion | CNN, metrics, results |
| Member 4 | XLM-R / AfroXLMR / AfriBERTa | TR-01 – TR-05 | `05_transformer`, `src/pipeline.py`, `06` (error analysis), README | Error Analysis & Limitations; Conclusion | Transformer, errors, conclusion |

Shared by everyone: reviewing each other's notebooks, writing the `observation` column for your own
experiments, proofreading the whole report, and being able to explain any part of the project in the demo.
