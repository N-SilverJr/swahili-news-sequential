# Data

The Zindi files are **not** committed (redistribution is not allowed).

1. Download `Train.csv`, `Test.csv` and `SampleSubmission.csv` from
   <https://zindi.africa/competitions/swahili-news-classification-challenge/data>.
2. For Colab: put them in a Google Drive folder named `swahili_news` (each notebook copies them here).
   For local runs: put them directly in this `data/` folder.

`splits.csv` **is** committed. It is the shared stratified 70/15/15 split created by
`notebooks/01_eda.ipynb` (seed 42), so all five approaches are evaluated on exactly the same articles.
