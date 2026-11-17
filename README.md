# NewsLens

Final project for COMP6140001 Data Mining (Group 4).

NewsLens sorts English news articles into four desks: World, Sports, Business, and Sci/Tech. We follow the KDD process on the AG News dataset, compare seven models, and turn the best one into a small Streamlit app. The case study is a fictional news site called WartaHub that wants to stop sorting articles by hand.

The app does not just give a category. It also gives a confidence score. If the score is high enough the article is filed automatically, and if not it goes to an editor. Our question was: which model works best, and how many articles can we file automatically while still being right at least 95% of the time?

## Team

| Name | Student ID |
|---|---|
| Adrian Marcello Budiman | 2802407300 |
| Hans Ewaldo Kristiawan | 2802406443 |
| Christian Verrell | 2802407080 |
| Raynald Arvan Lim | 2802404583 |
| Kenneth Sean Ternadi | 2802422245 |

## Folder structure

```
news-lens/
├── notebooks/
│   ├── newslens.ipynb      # EDA, preprocessing, TF-IDF, 6 classical models, evaluation, routing
│   └── distilbert.ipynb    # DistilBERT fine-tuning (needs a GPU)
├── models/                 # saved models used by the app
├── results/                # CSV results exported from the notebooks
├── samples/                # example CSV files for the batch feature
├── app.py                  # Streamlit app
├── preprocessing.py        # clean_text(), same function as in the notebooks
└── requirements.txt
```

## Dataset

AG News (Zhang, Zhao & LeCun, 2015), loaded from Hugging Face: https://huggingface.co/datasets/fancyzhx/ag_news

- 120,000 training articles and 7,600 test articles
- 4 balanced classes (30,000 per class in the training set)
- each article is a title plus a short description

Cleaned dataset (Google Drive): **[add link here]**

## What we did

1. **Data selection & EDA.** Class balance, text length, common words, and noise. We found that news-agency tags are not spread evenly (Reuters shows up in 18.5% of Business articles but only 4.8% of Sports articles) and that URLs appear almost only in Business and Sci/Tech. A model could learn these instead of the actual topic, so we removed them.
2. **Preprocessing.** `clean_text()` fixes HTML entities, removes HTML tags, URLs, emails and agency tags (Reuters, AP, AFP), then lowercases the text. After cleaning we removed 578 duplicates, 70 groups of identical texts with different labels, and 53 training articles that also appear in the test set. The test set was not changed.
3. **Transformation.** 90/10 stratified split into fit and validation sets (107,369 / 11,930). TF-IDF with 50,000 features (unigrams and bigrams), fitted on the fit set only. DistilBERT uses its own tokenizer with a max length of 128 tokens.
4. **Data mining.** Multinomial NB, Logistic Regression, Linear SVC, KNN, Decision Tree, Random Forest, and DistilBERT. Hyperparameters were tuned on the validation set. The test set was only used once at the end.
5. **Evaluation.** Macro-F1, accuracy, precision, recall, confusion matrix, log-loss, ROC-AUC, model size, speed, and confidence-based routing.

## Results (test set)

| Model | Macro-F1 | Auto-filed | Accuracy on auto-filed | Size |
|---|---|---|---|---|
| DistilBERT | 0.938 | 97.4% | 94.9% | 255 MB |
| Linear SVC | 0.918 | - | - | 1.5 MB |
| Linear SVC (Calibrated) | 0.916 | 90.2% | 95.0% | 4.6 MB |
| Logistic Regression | 0.912 | 87.7% | 94.9% | 1.5 MB |
| Multinomial NB | 0.906 | 86.8% | 94.8% | 3.1 MB |
| KNN | 0.905 | 86.8% | 94.7% | 33 MB |
| Random Forest | 0.886 | 75.0% | 95.4% | 471 MB |
| Decision Tree | 0.812 | - | - | 2.1 MB |

The thresholds were picked on the validation set and then used as is on the test set. Linear SVC has no probability output, so it can't be used for routing (that is why we also made a calibrated version). Decision Tree never got to 95% accuracy at any threshold.

DistilBERT has the best Macro-F1, but it is about 170 times bigger than Linear SVC and needs a GPU to be fast. Our pick for WartaHub is **Linear SVC (Calibrated) with a 0.70 threshold**: it files about 90% of articles automatically at 95% accuracy and runs fine on a normal laptop.

Full numbers are in `results/final_comparison.csv`.

## Running the notebooks

Both notebooks were made in Google Colab.

1. Run `notebooks/newslens.ipynb` first (Runtime > Run all). It saves the models and CSV files to Google Drive. The Random Forest tuning takes a while (around 30-45 minutes in total).
2. Then run `notebooks/distilbert.ipynb` with a T4 GPU. It reads `final_summary.csv` from the first notebook and adds DistilBERT to the final table.

## Running the app

Tested with Python 3.13 on Windows.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

If you use PowerShell, activate with `.\.venv\Scripts\Activate.ps1` instead.

Two model files are too big for GitHub (Random Forest and DistilBERT), so they are not in this repo. Download them from Google Drive and put them in the `models/` folder:

- `random_forest.joblib`: **[add link here]**
- `distilbert-agnews/` (whole folder): **[add link here]**

The app still works without them. Only those two options in the model list will show a "model file not found" message.

**In the app you can:**
- paste one article and see the predicted desk, the confidence score, and whether it would be auto-filed or sent to an editor
- upload a CSV (column `text`, or `title` + `description`) to classify many articles at once and download the results. If the CSV has a `label` column, the app also shows accuracy and a confusion matrix. There are example files in `samples/`.

## Limitations

- AG News is from 2004 and English only, so newer topics and other languages may not work well.
- Each article gets exactly one desk, even when a story fits two (for example tech company earnings).
- A threshold chosen on validation data is not guaranteed on new data. Logistic Regression, NB and KNN all went slightly under 95% on the test set.

## Links

- Presentation slides: **[add link here]**
- Demo video: **[add link here]**
