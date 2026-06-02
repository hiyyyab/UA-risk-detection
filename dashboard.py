import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score
)

st.set_page_config(
    page_title="Diabetes Risk Prediction — ML Dashboard",
    layout="wide"
)

FEATURES = [
    'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
    'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
    'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
    'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
]

MODEL_COLORS = {
    "Logistic Regression": "#636EFA",
    "Random Forest": "#EF553B",
    "Gradient Boosting": "#00CC96",
    "Gaussian NB": "#AB63FA"
}


@st.cache_data
def load_data():
    return pd.read_csv("d_dset.csv")


@st.cache_data
def train_and_evaluate():
    data = load_data()
    X = data.drop(columns=["Diabetes_binary"])
    y = data["Diabetes_binary"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model_defs = {
        "Logistic Regression": (LogisticRegression(max_iter=10000), True),
        "Random Forest": (RandomForestClassifier(n_estimators=100, random_state=42), False),
        "Gradient Boosting": (GradientBoostingClassifier(n_estimators=100, random_state=42), False),
        "Gaussian NB": (GaussianNB(), False),
    }

    results = {}
    for name, (model, use_scaled) in model_defs.items():
        Xtr = X_train_scaled if use_scaled else X_train.values
        Xte = X_test_scaled if use_scaled else X_test.values

        model.fit(Xtr, y_train)
        y_pred = model.predict(Xte)
        probs = model.predict_proba(Xte)

        fpr, tpr, _ = roc_curve(y_test, probs[:, 1])

        fi = None
        if hasattr(model, "feature_importances_"):
            fi = model.feature_importances_
        elif hasattr(model, "coef_"):
            fi = np.abs(model.coef_[0])

        results[name] = {
            "y_pred": y_pred,
            "probs": probs,
            "y_test": y_test.values,
            "fpr": fpr,
            "tpr": tpr,
            "roc_auc": float(auc(fpr, tpr)),
            "feature_importance": fi,
        }

    return results, y_test.values


def apply_deferral(probs, y_test, threshold):
    final_preds = []
    for p in probs:
        if max(p) < threshold:
            final_preds.append("DEFER")
        else:
            final_preds.append(int(np.argmax(p)))

    final_preds = np.array(final_preds, dtype=object)
    mask = final_preds != "DEFER"
    y_after = y_test[mask]
    pred_after = final_preds[mask].astype(int)
    return y_after, pred_after, int(np.sum(~mask))


def get_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
    }


# ── Header ────────────────────────────────────────────────────────────────────
st.title("Diabetes Risk Prediction — ML Dashboard")
st.write(
    """
    Comparing four machine learning classifiers on the CDC Diabetes Health Indicators dataset
    (~70,000 patient records). Each model is evaluated before and after applying a
    **deferral mechanism** — low-confidence predictions are withheld to simulate referral
    to a specialist, improving precision on the cases the model does decide.
    Use the tabs below to explore the dataset, compare model accuracy, tune the deferral
    threshold interactively, and inspect which features drive each model's decisions.
    """
)

with st.spinner("Training models — this only runs once and is then cached..."):
    results, y_test = train_and_evaluate()

data = load_data()

tab1, tab2, tab3, tab4 = st.tabs([
    "Dataset Overview",
    "Model Performance",
    "Deferral Analysis",
    "Feature Importance",
])


# ── Tab 1: Dataset Overview ────────────────────────────────────────────────────
with tab1:
    st.header("Dataset Overview")
    st.markdown(
        """
        The dataset comes from the CDC's Behavioral Risk Factor Surveillance System (BRFSS).
        Each row is one survey respondent; the target column `Diabetes_binary` indicates
        whether that person has been diagnosed with diabetes or pre-diabetes.
        """
    )

    n_pos = int(data["Diabetes_binary"].sum())
    n_neg = int(len(data) - n_pos)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{len(data):,}")
    c2.metric("Features", len(FEATURES))
    c3.metric("Diabetic (1)", f"{n_pos:,}")
    c4.metric("Non-Diabetic (0)", f"{n_neg:,}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Class Distribution")
        fig_pie = px.pie(
            values=[n_neg, n_pos],
            names=["Non-Diabetic", "Diabetic"],
            color_discrete_sequence=["#636EFA", "#EF553B"],
            hole=0.4,
        )
        fig_pie.update_layout(height=360)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.subheader("Feature Distribution by Outcome")
        selected_feat = st.selectbox("Select a feature", FEATURES)
        fig_hist = px.histogram(
            data,
            x=selected_feat,
            color=data["Diabetes_binary"].map({0.0: "Non-Diabetic", 1.0: "Diabetic"}),
            barmode="overlay",
            color_discrete_map={"Non-Diabetic": "#636EFA", "Diabetic": "#EF553B"},
            labels={"color": "Outcome"},
            opacity=0.7,
        )
        fig_hist.update_layout(height=360, legend_title="Outcome")
        st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("Feature Correlation Heatmap")
    st.markdown(
        "Darker red cells indicate a stronger positive correlation with diabetes; "
        "darker blue cells indicate negative correlation."
    )
    corr = data.corr()
    fig_corr = px.imshow(
        corr,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
        text_auto=".2f",
    )
    fig_corr.update_layout(height=560)
    st.plotly_chart(fig_corr, use_container_width=True)

    with st.expander("Raw Data Preview"):
        st.dataframe(data.head(100), use_container_width=True)


# ── Tab 2: Model Performance ───────────────────────────────────────────────────
with tab2:
    st.header("Model Performance — Before Deferral")
    st.markdown(
        """
        All four models are trained on 80 % of the data and evaluated on the held-out 20 %.
        Below you can compare accuracy, precision, recall, F1, and AUC side-by-side.
        """
    )

    metric_rows = []
    for name, res in results.items():
        m = get_metrics(res["y_test"], res["y_pred"])
        m["Model"] = name
        m["AUC"] = res["roc_auc"]
        metric_rows.append(m)

    df_metrics = pd.DataFrame(metric_rows).set_index("Model")
    st.dataframe(df_metrics.style.format("{:.4f}"), use_container_width=True)

    metrics_melted = (
        df_metrics.drop(columns=["AUC"])
        .reset_index()
        .melt(id_vars="Model", var_name="Metric", value_name="Score")
    )
    fig_bar = px.bar(
        metrics_melted,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        color_discrete_map=MODEL_COLORS,
        range_y=[0, 1],
        title="Metric Comparison Across Models",
    )
    fig_bar.update_layout(height=420)
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("ROC Curves")
    fig_roc = go.Figure()
    fig_roc.add_shape(
        type="line", x0=0, y0=0, x1=1, y1=1,
        line=dict(dash="dash", color="gray")
    )
    for name, res in results.items():
        fig_roc.add_trace(go.Scatter(
            x=res["fpr"],
            y=res["tpr"],
            mode="lines",
            name=f"{name} (AUC = {res['roc_auc']:.3f})",
            line=dict(color=MODEL_COLORS[name], width=2),
        ))
    fig_roc.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=460,
        legend=dict(x=0.55, y=0.1),
    )
    st.plotly_chart(fig_roc, use_container_width=True)

    st.subheader("Confusion Matrices")
    cols = st.columns(4)
    for i, (name, res) in enumerate(results.items()):
        cm = confusion_matrix(res["y_test"], res["y_pred"])
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual"),
            x=["No Diabetes", "Diabetes"],
            y=["No Diabetes", "Diabetes"],
            color_continuous_scale="Blues",
            text_auto=True,
        )
        fig_cm.update_layout(title=name, height=300, coloraxis_showscale=False)
        cols[i].plotly_chart(fig_cm, use_container_width=True)


# ── Tab 3: Deferral Analysis ───────────────────────────────────────────────────
with tab3:
    st.header("Deferral Analysis")
    st.markdown(
        """
        When a model's maximum class probability falls below the **confidence threshold**,
        the prediction is *deferred* — representing cases a clinician should review rather
        than relying on the model alone. Raising the threshold defers more cases but
        typically improves precision and recall on the cases that are decided.
        """
    )

    threshold = st.slider("Confidence Threshold", 0.50, 0.99, 0.70, 0.01)
    selected_model = st.selectbox("Select Model", list(results.keys()))

    res = results[selected_model]
    y_after, pred_after, n_deferred = apply_deferral(res["probs"], res["y_test"], threshold)
    deferral_rate = n_deferred / len(res["y_test"]) * 100

    m_before = get_metrics(res["y_test"], res["y_pred"])
    m_after = (
        get_metrics(y_after, pred_after)
        if len(y_after) > 0
        else {k: 0.0 for k in m_before}
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Deferred Samples", f"{n_deferred:,}", f"{deferral_rate:.1f}% of test set")
    c2.metric("Samples Decided", f"{len(y_after):,}")
    c3.metric(
        "F1 After Deferral",
        f"{m_after['F1 Score']:.4f}",
        f"{m_after['F1 Score'] - m_before['F1 Score']:+.4f}",
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Before Deferral")
        for k, v in m_before.items():
            st.metric(k, f"{v:.4f}")
    with col2:
        st.subheader(f"After Deferral (threshold = {threshold})")
        for k, v in m_after.items():
            st.metric(k, f"{v:.4f}", f"{v - m_before[k]:+.4f}")

    compare_df = pd.DataFrame({
        "Metric": list(m_before.keys()) * 2,
        "Score": list(m_before.values()) + list(m_after.values()),
        "Phase": ["Before Deferral"] * 4 + ["After Deferral"] * 4,
    })
    fig_compare = px.bar(
        compare_df,
        x="Metric",
        y="Score",
        color="Phase",
        barmode="group",
        color_discrete_map={"Before Deferral": "#636EFA", "After Deferral": "#EF553B"},
        range_y=[0, 1],
        title=f"{selected_model}: Before vs. After Deferral",
    )
    fig_compare.update_layout(height=380)
    st.plotly_chart(fig_compare, use_container_width=True)

    st.subheader(f"Deferral Rate Across All Models (threshold = {threshold})")
    defer_rows = []
    for name, r in results.items():
        _, _, nd = apply_deferral(r["probs"], r["y_test"], threshold)
        defer_rows.append({"Model": name, "Deferred (%)": nd / len(r["y_test"]) * 100})

    fig_defer = px.bar(
        pd.DataFrame(defer_rows),
        x="Model",
        y="Deferred (%)",
        color="Model",
        color_discrete_map=MODEL_COLORS,
        range_y=[0, 100],
    )
    fig_defer.update_layout(showlegend=False, height=360)
    st.plotly_chart(fig_defer, use_container_width=True)


# ── Tab 4: Feature Importance ──────────────────────────────────────────────────
with tab4:
    st.header("Feature Importance")
    st.markdown(
        """
        Feature importance reveals which patient attributes most influence each classifier.
        For **Random Forest** and **Gradient Boosting** this is the mean decrease in impurity.
        For **Logistic Regression** it is the magnitude of the learned coefficient (after scaling).
        *Gaussian NB does not produce a single importance score and is excluded here.*
        """
    )

    models_with_fi = {
        name: res for name, res in results.items()
        if res["feature_importance"] is not None
    }

    cols = st.columns(len(models_with_fi))
    for i, (name, res) in enumerate(models_with_fi.items()):
        fi_df = pd.DataFrame({
            "Feature": FEATURES,
            "Importance": res["feature_importance"],
        }).sort_values("Importance", ascending=True)

        fig_fi = px.bar(
            fi_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title=name,
            color_discrete_sequence=[MODEL_COLORS[name]],
        )
        fig_fi.update_layout(height=540, showlegend=False)
        cols[i].plotly_chart(fig_fi, use_container_width=True)

    st.subheader("Top 10 Features — Average Importance Across Models")
    avg_fi = np.zeros(len(FEATURES))
    for res in models_with_fi.values():
        fi = res["feature_importance"]
        avg_fi += fi / fi.sum()
    avg_fi /= len(models_with_fi)

    top_df = (
        pd.DataFrame({"Feature": FEATURES, "Avg Importance": avg_fi})
        .sort_values("Avg Importance", ascending=False)
        .head(10)
    )

    fig_top = px.bar(
        top_df,
        x="Feature",
        y="Avg Importance",
        color="Avg Importance",
        color_continuous_scale="Viridis",
        title="Average feature importance (normalized) across LR, RF, and GB",
    )
    fig_top.update_layout(height=420)
    st.plotly_chart(fig_top, use_container_width=True)
