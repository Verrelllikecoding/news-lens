import json
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch

from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from preprocessing import clean_text


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
SAMPLE_DIR = BASE_DIR / "samples"

MODELS = {
    "Linear SVC (Calibrated)": "linear_svc_calibrated.joblib",
    "Linear SVC": "linear_svc.joblib",
    "Logistic Regression": "logistic_regression.joblib",
    "Multinomial NB": "multinomial_nb.joblib",
    "K-Nearest Neighbors": "k-nearest_neighbors.joblib",
    "Decision Tree": "decision_tree.joblib",
    "Random Forest": "random_forest.joblib",
    "DistilBERT": None
}

# Same max_length as the DistilBERT training notebook
MAX_LENGTH = 128

# Used when a model has no validation-selected threshold
DEFAULT_THRESHOLD = 0.70

# Batch limits
MAX_BATCH_ROWS = 5000
DISTILBERT_BATCH_SIZE = 32

CATEGORIES = ["World", "Sports", "Business", "Sci/Tech"]

ICONS = {
    "World": "🌍",
    "Sports": "🏆",
    "Business": "💼",
    "Sci/Tech": "🔬",
}

CATEGORY_COLORS = {
    "World": "#2F6FED",
    "Sports": "#16A34A",
    "Business": "#E08A00",
    "Sci/Tech": "#7C3AED",
}

STATUS_COLORS = {
    "AUTO-FILED": "#16A34A",
    "EDITOR REVIEW": "#E08A00",
    "NO SCORE": "#64758F",
}

# Accepted spellings in the optional label column of a batch CSV
LABEL_ALIASES = {
    "0": "World", "world": "World",
    "1": "Sports", "sports": "Sports", "sport": "Sports",
    "2": "Business", "business": "Business",
    "3": "Sci/Tech", "sci/tech": "Sci/Tech", "scitech": "Sci/Tech", "sci-tech": "Sci/Tech",
    "science and technology": "Sci/Tech", "science & technology": "Sci/Tech",
}

EXAMPLES = {
    "🏆 Sports": (
        "The team secured a dramatic victory after scoring "
        "a last-minute goal in the championship final."
    ),
    "🌍 World": (
        "World leaders gathered for an international summit "
        "to discuss global security and climate change."
    ),
    "💼 Business": (
        "The stock market rose sharply as major technology "
        "companies reported stronger quarterly earnings."
    ),
    "🔬 Sci/Tech": (
        "Scientists have developed a new artificial intelligence "
        "system that can improve medical diagnosis."
    ),
}

st.set_page_config(
    page_title="NewsLens",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"], .stApp, button, input, textarea {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }

    .stApp {
        background:
            radial-gradient(circle at 0% 0%, rgba(52, 120, 246, 0.10), transparent 35%),
            radial-gradient(circle at 100% 0%, rgba(124, 58, 237, 0.08), transparent 30%),
            #F5F8FF;
        color: #26364F;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }

    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(10px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #EAF1FF 0%, #F3F0FF 100%);
        border-right: 1px solid #D5E1F2;
    }

    .brand-title {
        color: #173D75;
        font-size: 28px;
        font-weight: 800;
        letter-spacing: 1.5px;
    }

    .brand-subtitle {
        color: #64758F;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1.5px;
    }

    .snapshot {
        background: #FFFFFF;
        border: 1px solid #DCE6F4;
        border-radius: 14px;
        padding: 14px 16px;
        margin-top: 6px;
    }

    .snapshot-row {
        display: flex;
        justify-content: space-between;
        padding: 6px 0;
        border-bottom: 1px dashed #E3EAF5;
        font-size: 13px;
        color: #64758F;
    }

    .snapshot-row:last-child {
        border-bottom: none;
    }

    .snapshot-row b {
        color: #193B70;
    }

    /* Hero */
    .hero {
        background: linear-gradient(120deg, #173D75 0%, #3478F6 55%, #7C3AED 100%);
        border-radius: 22px;
        padding: 34px 38px;
        color: #FFFFFF;
        box-shadow: 0 14px 40px rgba(36, 89, 166, 0.25);
        position: relative;
        overflow: hidden;
        animation: fadeUp 0.5s ease;
    }

    .hero::after {
        content: "";
        position: absolute;
        right: -60px;
        top: -60px;
        width: 260px;
        height: 260px;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.08);
    }

    .hero-eyebrow {
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 2px;
        opacity: 0.85;
    }

    .hero-title {
        font-size: 44px;
        font-weight: 800;
        margin: 6px 0 4px 0;
        line-height: 1.1;
    }

    .hero-subtitle {
        font-size: 16px;
        opacity: 0.92;
        max-width: 680px;
    }

    .hero-chips {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 20px;
    }

    .hero-chip {
        background: rgba(255, 255, 255, 0.16);
        border: 1px solid rgba(255, 255, 255, 0.28);
        border-radius: 999px;
        padding: 7px 14px;
        font-size: 13px;
        font-weight: 600;
        backdrop-filter: blur(4px);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #FFFFFF;
        border: 1px solid #DCE6F4;
        border-radius: 14px;
        padding: 6px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 10px 18px;
        font-weight: 700;
        color: #64758F;
    }

    .stTabs [aria-selected="true"] {
        background: #3478F6 !important;
        color: #FFFFFF !important;
    }

    .stTabs [aria-selected="true"] p {
        color: #FFFFFF !important;
    }

    .stTabs [data-baseweb="tab-highlight"],
    .stTabs [data-baseweb="tab-border"] {
        display: none;
    }

    h1, h2, h3 {
        color: #193B70 !important;
        font-weight: 800 !important;
    }

    /* Buttons */
    .stButton > button,
    .stDownloadButton > button {
        border-radius: 12px !important;
        font-weight: 700 !important;
        min-height: 46px;
        transition: all 0.2s ease;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(52, 120, 246, 0.22);
    }

    /* Inputs */
    .stTextArea textarea {
        background: #FFFFFF !important;
        border-radius: 14px !important;
        font-size: 15px !important;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: #FFFFFF;
        border: 2px dashed #9DB8E8;
        border-radius: 16px;
    }

    /* Metrics */
    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #DCE6F4;
        border-radius: 14px;
        padding: 14px 16px;
        box-shadow: 0 3px 12px rgba(35, 65, 110, 0.05);
    }

    [data-testid="stMetricValue"] {
        color: #2459A6 !important;
        font-weight: 800 !important;
    }

    /* Result card */
    .result-card {
        background: #FFFFFF;
        border: 1px solid #DCE6F4;
        border-left: 8px solid var(--accent);
        border-radius: 18px;
        padding: 24px 26px;
        box-shadow: 0 8px 24px rgba(35, 65, 110, 0.08);
        animation: fadeUp 0.4s ease;
        height: 100%;
    }

    .result-label {
        color: #64758F;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1.5px;
    }

    .result-category {
        color: var(--accent);
        font-size: 38px;
        font-weight: 800;
        margin-top: 6px;
    }

    .result-detail {
        color: #64758F;
        font-size: 14px;
        margin-top: 6px;
    }

    .status-pill {
        display: inline-block;
        margin-top: 14px;
        padding: 6px 14px;
        border-radius: 999px;
        font-size: 13px;
        font-weight: 800;
        letter-spacing: 0.8px;
        color: #FFFFFF;
        background: var(--status);
    }

    /* Confidence gauge */
    .gauge-card {
        background: #FFFFFF;
        border: 1px solid #DCE6F4;
        border-radius: 18px;
        padding: 24px 26px;
        box-shadow: 0 8px 24px rgba(35, 65, 110, 0.08);
        animation: fadeUp 0.5s ease;
        height: 100%;
    }

    .gauge-value {
        font-size: 38px;
        font-weight: 800;
        color: #193B70;
        margin-top: 6px;
    }

    .gauge-track {
        position: relative;
        height: 14px;
        background: #E8EEF8;
        border-radius: 999px;
        margin-top: 14px;
        overflow: visible;
    }

    .gauge-fill {
        height: 14px;
        border-radius: 999px;
        background: linear-gradient(90deg, #3478F6, #7C3AED);
    }

    .gauge-marker {
        position: absolute;
        top: -6px;
        width: 3px;
        height: 26px;
        background: #E08A00;
        border-radius: 2px;
    }

    .gauge-caption {
        color: #64758F;
        font-size: 13px;
        margin-top: 12px;
    }

    .section-note {
        color: #64758F;
        font-size: 14px;
    }

    [data-testid="stExpander"] {
        background: #FFFFFF;
        border: 1px solid #DCE6F4;
        border-radius: 14px;
    }

    hr {
        border-color: #D5E1F2;
    }

    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_vectorizer():
    return joblib.load(MODEL_DIR / "vectorizer.joblib")


@st.cache_resource
def load_model(filename):
    return joblib.load(MODEL_DIR / filename)


@st.cache_resource
def load_label_mapping():
    with open(MODEL_DIR / "label_mapping.json", "r") as file:
        mapping = json.load(file)

    return {int(key): value for key, value in mapping.items()}


@st.cache_resource
def load_comparison():
    # Final evaluation table from the notebooks (DistilBERT notebook adds every model)
    for folder in [RESULTS_DIR, MODEL_DIR]:
        for filename in ["final_comparison.csv", "final_summary.csv"]:
            path = folder / filename

            if path.exists():
                return pd.read_csv(path)

    return None


@st.cache_resource
def load_thresholds():
    # Thresholds chosen on the validation set in the notebooks
    # (lowest threshold that keeps auto-filed accuracy >= 95%)
    comparison = load_comparison()

    for folder in [RESULTS_DIR, MODEL_DIR]:
        if comparison is None and (folder / "routing_results.csv").exists():
            comparison = pd.read_csv(folder / "routing_results.csv")

    if comparison is None or "Threshold (from validation)" not in comparison.columns:
        return {}

    routing = comparison.dropna(subset=["Threshold (from validation)"])

    return dict(zip(
        routing["Model"],
        routing["Threshold (from validation)"]
    ))


@st.cache_resource
def load_distilbert():
    distilbert_dir = MODEL_DIR / "distilbert-agnews"

    tokenizer = AutoTokenizer.from_pretrained(
        distilbert_dir,
        local_files_only=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        distilbert_dir,
        local_files_only=True
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)
    model.eval()

    return tokenizer, model, device


def predict_texts(texts, model_name, progress=None):
    # Every model was trained on clean_text output, so all models use it
    label_mapping = load_label_mapping()
    cleaned_texts = [clean_text(text) for text in texts]

    if model_name == "DistilBERT":
        tokenizer, model, device = load_distilbert()
        probability_batches = []

        # Predict in small batches so long CSV files do not run out of memory
        for start in range(0, len(cleaned_texts), DISTILBERT_BATCH_SIZE):
            batch = cleaned_texts[start:start + DISTILBERT_BATCH_SIZE]

            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH
            )

            inputs = {
                key: value.to(device)
                for key, value in inputs.items()
            }

            with torch.no_grad():
                outputs = model(**inputs)

            probability_batches.append(
                torch.softmax(outputs.logits, dim=1).cpu().numpy()
            )

            if progress is not None:
                progress(min(start + DISTILBERT_BATCH_SIZE, len(cleaned_texts)) / len(cleaned_texts))

        probability_values = np.vstack(probability_batches)
        class_ids = list(range(probability_values.shape[1]))
        predictions = probability_values.argmax(axis=1)

    else:
        vectorizer = load_vectorizer()
        model = load_model(MODELS[model_name])

        article_vectors = vectorizer.transform(cleaned_texts)
        predictions = model.predict(article_vectors)

        probability_values = None
        class_ids = [int(class_id) for class_id in model.classes_]

        if hasattr(model, "predict_proba"):
            probability_values = model.predict_proba(article_vectors)

        if progress is not None:
            progress(1.0)

    categories = [label_mapping[int(prediction)] for prediction in predictions]

    probabilities = None
    confidence = None

    if probability_values is not None:
        probabilities = pd.DataFrame(
            probability_values,
            columns=[label_mapping[class_id] for class_id in class_ids]
        )
        confidence = probability_values.max(axis=1)

    return {
        "categories": categories,
        "confidence": confidence,
        "probabilities": probabilities,
        "cleaned_texts": cleaned_texts
    }


def classify_news(article, model_name):
    output = predict_texts([article], model_name)

    probabilities = None

    if output["probabilities"] is not None:
        probabilities = output["probabilities"].iloc[0].to_dict()

    return {
        "category": output["categories"][0],
        "cleaned_text": output["cleaned_texts"][0],
        "probabilities": probabilities,
        "model": model_name
    }


def editorial_status(confidence, threshold):
    if confidence is None or pd.isna(confidence):
        return "NO SCORE"

    return "AUTO-FILED" if confidence >= threshold else "EDITOR REVIEW"


def read_batch_csv(uploaded_file):
    # 1. Read the file (UTF-8 first, then a common Windows encoding)
    try:
        data = pd.read_csv(uploaded_file)
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        data = pd.read_csv(uploaded_file, encoding="latin-1")
    except pd.errors.EmptyDataError:
        raise ValueError("The CSV file is empty.")
    except pd.errors.ParserError:
        raise ValueError("The file could not be read as a CSV. Please check the separator and quotes.")

    if data.empty:
        raise ValueError("The CSV file has no rows.")

    # 2. Find the article text (text, or title + description like the proposal)
    columns = {column.strip().lower(): column for column in data.columns}

    if "text" in columns:
        texts = data[columns["text"]]
    elif "title" in columns and "description" in columns:
        texts = data[columns["title"]].fillna("").astype(str) + " " + data[columns["description"]].fillna("").astype(str)
    else:
        raise ValueError(
            "Missing article column. Use a column named `text`, or both `title` and `description`. "
            f"Columns found: {', '.join(map(str, data.columns))}"
        )

    texts = texts.fillna("").astype(str).str.strip()

    # 3. Skip empty articles
    valid_rows = texts != ""
    skipped_rows = int((~valid_rows).sum())

    data = data[valid_rows].reset_index(drop=True)
    texts = texts[valid_rows].reset_index(drop=True)

    if data.empty:
        raise ValueError("Every article in the file is empty.")

    if len(data) > MAX_BATCH_ROWS:
        raise ValueError(f"The file has {len(data):,} articles. The limit is {MAX_BATCH_ROWS:,} per upload.")

    # 4. Optional true labels (0-3 or category names) for evaluation
    labels = None

    if "label" in columns:
        labels = data[columns["label"]].astype(str).str.strip().str.lower().map(LABEL_ALIASES)

        if labels.isna().all():
            labels = None

    return data, texts, labels, skipped_rows


def probability_chart(probabilities):
    chart_data = pd.DataFrame({
        "Category": list(probabilities.keys()),
        "Score": [value * 100 for value in probabilities.values()]
    })

    bars = alt.Chart(chart_data).mark_bar(cornerRadiusEnd=8, height=26).encode(
        x=alt.X("Score:Q", scale=alt.Scale(domain=[0, 100]), title="Score (%)"),
        y=alt.Y("Category:N", sort="-x", title=None),
        color=alt.Color(
            "Category:N",
            scale=alt.Scale(domain=list(CATEGORY_COLORS), range=list(CATEGORY_COLORS.values())),
            legend=None
        ),
        tooltip=["Category", alt.Tooltip("Score:Q", format=".2f")]
    )

    labels = bars.mark_text(align="left", dx=6, fontWeight="bold", color="#193B70").encode(
        text=alt.Text("Score:Q", format=".1f")
    )

    return (bars + labels).properties(height=200)


def donut_chart(data, field, colors, title):
    counts = data[field].value_counts().reset_index()
    counts.columns = [field, "Articles"]

    # Only show legend entries that appear in this batch
    present = [name for name in colors if name in set(counts[field])]

    return alt.Chart(counts).mark_arc(innerRadius=60, cornerRadius=4).encode(
        theta="Articles:Q",
        color=alt.Color(
            f"{field}:N",
            scale=alt.Scale(domain=present, range=[colors[name] for name in present]),
            legend=alt.Legend(orient="bottom", title=None)
        ),
        tooltip=[field, "Articles"]
    ).properties(height=280, title=title)


def confusion_chart(true_labels, predicted_labels):
    table = pd.crosstab(
        pd.Categorical(true_labels, categories=CATEGORIES),
        pd.Categorical(predicted_labels, categories=CATEGORIES),
        dropna=False
    )

    long_table = table.stack().reset_index()
    long_table.columns = ["Actual", "Predicted", "Articles"]

    base = alt.Chart(long_table).encode(
        x=alt.X("Predicted:N", sort=CATEGORIES),
        y=alt.Y("Actual:N", sort=CATEGORIES)
    )

    heat = base.mark_rect(cornerRadius=6).encode(
        color=alt.Color("Articles:Q", scale=alt.Scale(scheme="blues"), legend=None),
        tooltip=["Actual", "Predicted", "Articles"]
    )

    text = base.mark_text(fontWeight="bold", fontSize=14).encode(
        text="Articles:Q",
        color=alt.condition(alt.datum.Articles > long_table["Articles"].max() / 2, alt.value("white"), alt.value("#193B70"))
    )

    return (heat + text).properties(height=300, title="Actual vs Predicted")


def show_model_error(error):
    if isinstance(error, FileNotFoundError):
        st.error(
            f"Model file not found: {Path(str(error.filename)).name}. "
            "Please check your models folder."
        )
    else:
        st.error(f"An error occurred: {error}")


comparison = load_comparison()
thresholds = load_thresholds()


with st.sidebar:
    st.markdown(
        '<div class="brand-title">📰 NEWSLENS</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="brand-subtitle">NEWS CLASSIFICATION SYSTEM</div>',
        unsafe_allow_html=True
    )

    st.divider()

    st.subheader("⚙️ Model Settings")

    selected_model = st.selectbox(
        "Choose Model",
        list(MODELS.keys())
    )

    recommended_threshold = float(thresholds.get(selected_model, DEFAULT_THRESHOLD))

    # One slider per model, so each model starts at its own validation threshold
    threshold = st.slider(
        "Editorial Review Threshold",
        min_value=0.50,
        max_value=1.00,
        value=recommended_threshold,
        step=0.05,
        key=f"threshold_{selected_model}",
        help="Default value is the threshold chosen on the validation set "
             "(lowest threshold with at least 95% accuracy on auto-filed articles)."
    )

    if selected_model in thresholds:
        st.caption(f"✅ Recommended threshold for {selected_model}: **{recommended_threshold:.2f}**")

    # Small snapshot of the selected model from the notebook results
    if comparison is not None and selected_model in set(comparison["Model"]):
        row = comparison[comparison["Model"] == selected_model].iloc[0]

        snapshot_rows = [("Test Macro-F1", f"{row['Macro-F1']:.4f}")]

        if "Test Coverage" in row and pd.notna(row["Test Coverage"]):
            snapshot_rows.append(("Auto-filed (test)", f"{row['Test Coverage'] * 100:.1f}%"))

        if "Model Size (MB)" in row and pd.notna(row["Model Size (MB)"]):
            snapshot_rows.append(("Model size", f"{row['Model Size (MB)']:.1f} MB"))

        rows_html = "".join(
            f'<div class="snapshot-row"><span>{name}</span><b>{value}</b></div>'
            for name, value in snapshot_rows
        )

        st.markdown("**📌 Model Snapshot**")
        st.markdown(f'<div class="snapshot">{rows_html}</div>', unsafe_allow_html=True)

    st.divider()

    st.caption(
        "Powered by TF-IDF, Machine Learning & DistilBERT"
    )


# Hero banner
hero_chips = [f"🧠 {len(MODELS)} models", "🗂️ 4 editorial desks", "📚 AG News · KDD"]

if comparison is not None:
    best = comparison.sort_values("Macro-F1", ascending=False).iloc[0]
    hero_chips.append(f"🏅 Best Macro-F1 {best['Macro-F1']:.3f} ({best['Model']})")

chips_html = "".join(f'<span class="hero-chip">{chip}</span>' for chip in hero_chips)

st.markdown(
    f"""
    <div class="hero">
        <div class="hero-eyebrow">WARTAHUB · AUTOMATIC DESK ROUTING</div>
        <div class="hero-title">📰 NewsLens</div>
        <div class="hero-subtitle">
            Classify English news into World, Sports, Business, or Sci/Tech.
            Confident articles are auto-filed, uncertain ones go to an editor.
        </div>
        <div class="hero-chips">{chips_html}</div>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")

single_tab, batch_tab = st.tabs(["✍️ Single Article", "📄 Batch CSV"])


with single_tab:
    st.subheader("Enter Your News")

    st.markdown(
        '<div class="section-note">Paste an English news article or start with one of the examples.</div>',
        unsafe_allow_html=True
    )

    example_columns = st.columns(4)

    for index, (name, example) in enumerate(EXAMPLES.items()):
        with example_columns[index]:
            if st.button(name, width="stretch", key=f"example_{index}"):
                st.session_state["article"] = example
                st.session_state.pop("result", None)
                st.rerun()

    article = st.text_area(
        "News Article",
        key="article",
        height=170,
        max_chars=2000,
        placeholder="Paste your news article here...",
        label_visibility="collapsed"
    )

    if st.button(
        "🔎 Classify News",
        type="primary",
        width="stretch",
        key="classify_single"
    ):
        if not article.strip():
            st.warning("Please enter a news article first.")

        else:
            try:
                with st.spinner("Analyzing your news article..."):
                    st.session_state["result"] = classify_news(article, selected_model)

            except Exception as error:
                show_model_error(error)

    result = st.session_state.get("result")

    if result and result["model"] == selected_model:

        category = result["category"]
        icon = ICONS.get(category, "📰")
        accent = CATEGORY_COLORS.get(category, "#3478F6")
        probabilities = result["probabilities"]

        confidence = probabilities[category] if probabilities is not None else None
        status = editorial_status(confidence, threshold)

        st.write("")

        card_column, gauge_column = st.columns([1.15, 1])

        with card_column:
            st.markdown(
                f"""
                <div class="result-card" style="--accent: {accent}; --status: {STATUS_COLORS[status]};">
                    <div class="result-label">PREDICTED DESK</div>
                    <div class="result-category">{icon} {category}</div>
                    <div class="result-detail">Model: {result["model"]}</div>
                    <span class="status-pill">{status}</span>
                </div>
                """,
                unsafe_allow_html=True
            )

        with gauge_column:
            if confidence is not None:
                st.markdown(
                    f"""
                    <div class="gauge-card">
                        <div class="result-label">CONFIDENCE</div>
                        <div class="gauge-value">{confidence * 100:.2f}%</div>
                        <div class="gauge-track">
                            <div class="gauge-fill" style="width: {confidence * 100:.1f}%;"></div>
                            <div class="gauge-marker" style="left: calc({threshold * 100:.1f}% - 1px);"></div>
                        </div>
                        <div class="gauge-caption">
                            Orange line = review threshold ({threshold:.2f}).
                            {"Above it, so the article is filed automatically." if status == "AUTO-FILED" else "Below it, so an editor should check this article."}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    """
                    <div class="gauge-card">
                        <div class="result-label">CONFIDENCE</div>
                        <div class="gauge-value">—</div>
                        <div class="gauge-caption">
                            This model does not support predict_proba, so a confidence score
                            and editorial routing are not available. Try <b>Linear SVC (Calibrated)</b>.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        if probabilities is not None:
            st.write("")
            st.subheader("📊 Category Scores")
            st.altair_chart(probability_chart(probabilities), width="stretch")

            st.caption(
                "Note: Scores are model probability outputs, "
                "not guaranteed measures of correctness. "
                "DistilBERT scores are generated using softmax."
            )

        with st.expander("🧹 View Preprocessed Article"):
            st.write(result["cleaned_text"])

            if result["model"] == "DistilBERT":
                st.caption(
                    "DistilBERT receives this cleaned text and processes it "
                    f"with its own WordPiece tokenizer (max {MAX_LENGTH} tokens)."
                )


with batch_tab:
    st.subheader("Classify Many Articles at Once")

    st.markdown(
        f"""
        <div class="section-note">
        Upload a CSV with a <b>text</b> column (or <b>title</b> + <b>description</b>).
        An optional <b>label</b> column (0–3 or World / Sports / Business / Sci/Tech) adds an accuracy check.
        Up to {MAX_BATCH_ROWS:,} articles per file. The model and threshold come from the sidebar.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    sample_columns = st.columns(2)

    for column, (filename, label) in zip(
        sample_columns,
        [
            ("sample_batch_labeled.csv", "⬇️ Sample CSV (with labels)"),
            ("sample_batch.csv", "⬇️ Sample CSV (text only)")
        ]
    ):
        sample_path = SAMPLE_DIR / filename

        if sample_path.exists():
            with column:
                st.download_button(
                    label,
                    data=sample_path.read_bytes(),
                    file_name=filename,
                    mime="text/csv",
                    width="stretch",
                    key=f"sample_{filename}"
                )

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        key="batch_file"
    )

    if uploaded_file is not None:
        try:
            batch_data, batch_texts, batch_labels, skipped_rows = read_batch_csv(uploaded_file)

        except ValueError as error:
            st.error(f"❌ {error}")
            batch_data = None

        if batch_data is not None:
            info_message = f"📄 **{len(batch_data):,} article{'s' if len(batch_data) > 1 else ''}** ready"

            if skipped_rows:
                info_message += f" · {skipped_rows:,} empty rows skipped"

            if batch_labels is not None:
                info_message += f" · labels found for {int(batch_labels.notna().sum()):,} rows"

            st.info(info_message)

            if st.button(
                f"🚀 Classify {len(batch_data):,} Article{'s' if len(batch_data) > 1 else ''} with {selected_model}",
                type="primary",
                width="stretch",
                key="classify_batch"
            ):
                progress_bar = st.progress(0.0, text="Classifying articles...")

                try:
                    output = predict_texts(
                        batch_texts.tolist(),
                        selected_model,
                        progress=lambda value: progress_bar.progress(value, text=f"Classifying articles... {value * 100:.0f}%")
                    )

                    st.session_state["batch_result"] = {
                        "file": uploaded_file.name,
                        "model": selected_model,
                        "data": batch_data,
                        "labels": batch_labels,
                        "categories": output["categories"],
                        "confidence": output["confidence"]
                    }

                except Exception as error:
                    show_model_error(error)

                progress_bar.empty()

    batch_result = st.session_state.get("batch_result")

    if (
        uploaded_file is not None
        and batch_result is not None
        and batch_result["file"] == uploaded_file.name
        and batch_result["model"] == selected_model
    ):
        # Status is recalculated here, so moving the slider updates the table immediately
        result_data = batch_result["data"].copy()
        result_data["Predicted Category"] = batch_result["categories"]

        if batch_result["confidence"] is not None:
            result_data["Confidence"] = np.round(batch_result["confidence"], 4)
            result_data["Editorial Status"] = [
                editorial_status(value, threshold) for value in batch_result["confidence"]
            ]
        else:
            result_data["Confidence"] = np.nan
            result_data["Editorial Status"] = "NO SCORE"

        total = len(result_data)
        auto_filed = int((result_data["Editorial Status"] == "AUTO-FILED").sum())
        editor_review = int((result_data["Editorial Status"] == "EDITOR REVIEW").sum())

        st.divider()
        st.subheader("📊 Batch Result")

        metric_columns = st.columns(4)

        metric_columns[0].metric("Articles", f"{total:,}")

        if batch_result["confidence"] is not None:
            metric_columns[1].metric("Auto-filed", f"{auto_filed:,}", f"{auto_filed / total * 100:.1f}%", delta_color="off", delta_arrow="off")
            metric_columns[2].metric("Editor Review", f"{editor_review:,}", f"{editor_review / total * 100:.1f}%", delta_color="off", delta_arrow="off")
            metric_columns[3].metric("Avg. Confidence", f"{np.mean(batch_result['confidence']) * 100:.1f}%")
        else:
            metric_columns[1].metric("Auto-filed", "—")
            metric_columns[2].metric("Editor Review", "—")
            metric_columns[3].metric("Avg. Confidence", "—")
            st.info("This model does not support predict_proba, so routing is not available. Try Linear SVC (Calibrated).")

        chart_columns = st.columns(2)

        with chart_columns[0]:
            st.altair_chart(
                donut_chart(result_data, "Predicted Category", CATEGORY_COLORS, "Predicted desks"),
                width="stretch"
            )

        with chart_columns[1]:
            if batch_result["confidence"] is not None:
                st.altair_chart(
                    donut_chart(result_data, "Editorial Status", STATUS_COLORS, "Editorial routing"),
                    width="stretch"
                )

        # Accuracy check when the CSV contains true labels
        labels = batch_result["labels"]

        if labels is not None:
            labeled = labels.notna()
            true_labels = labels[labeled]
            predicted_labels = result_data.loc[labeled, "Predicted Category"]

            st.subheader("🎯 Accuracy Check")

            check_columns = st.columns(3)
            check_columns[0].metric("Accuracy", f"{accuracy_score(true_labels, predicted_labels) * 100:.2f}%")
            # Only average over desks that appear in the uploaded labels
            present_labels = [category for category in CATEGORIES if category in set(true_labels)]

            check_columns[1].metric("Macro-F1", f"{f1_score(true_labels, predicted_labels, average='macro', labels=present_labels, zero_division=0):.4f}")

            auto_mask = labeled & (result_data["Editorial Status"] == "AUTO-FILED")

            if auto_mask.any():
                check_columns[2].metric(
                    "Accuracy (auto-filed only)",
                    f"{accuracy_score(labels[auto_mask], result_data.loc[auto_mask, 'Predicted Category']) * 100:.2f}%"
                )

            st.altair_chart(confusion_chart(true_labels, predicted_labels), width="stretch")

            result_data["Correct"] = np.where(
                labeled,
                labels == result_data["Predicted Category"],
                None
            )

        st.subheader("📋 Predictions")

        status_filter = st.segmented_control(
            "Show",
            ["All", "AUTO-FILED", "EDITOR REVIEW"],
            default="All",
            key="batch_filter"
        )

        shown_data = result_data

        if status_filter in ["AUTO-FILED", "EDITOR REVIEW"]:
            shown_data = result_data[result_data["Editorial Status"] == status_filter]

        # Prediction columns first, the original CSV columns after them
        result_columns = ["Predicted Category", "Confidence", "Editorial Status", "Correct"]
        column_order = [column for column in result_columns if column in shown_data.columns]
        column_order += [column for column in shown_data.columns if column not in result_columns]

        st.dataframe(
            shown_data,
            width="stretch",
            hide_index=True,
            column_order=column_order,
            column_config={
                "Confidence": st.column_config.ProgressColumn(
                    "Confidence",
                    min_value=0.0,
                    max_value=1.0,
                    format="percent"
                )
            }
        )

        st.download_button(
            "⬇️ Download Predictions (CSV)",
            data=result_data.to_csv(index=False).encode("utf-8"),
            file_name=f"newslens_predictions_{selected_model.lower().replace(' ', '_').replace('(', '').replace(')', '')}.csv",
            mime="text/csv",
            type="primary",
            width="stretch",
            key="download_batch"
        )


st.divider()

st.caption(
    "NewsLens · AG News Classification Project"
)
