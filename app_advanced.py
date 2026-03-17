import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from statistics import NormalDist
import json
import joblib
import os
import re
import openai
try:
    from openai import OpenAI
except Exception:
    OpenAI = None
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense

# Optional hardcoded credentials (not recommended for shared/public repos).
OPENAI_API_KEY_HARDCODED = "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"
OPENAI_MODEL_HARDCODED = "gpt-4o-mini"

# ===========================
# PAGE CONFIGURATION
# ===========================

st.set_page_config(
    page_title="Carbon Emissions Research Platform",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional, modern look
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500&display=swap');

    :root {
        --ink-900: #0f172a;
        --ink-700: #334155;
        --ink-500: #64748b;
        --sky-600: #0284c7;
        --sky-500: #0ea5e9;
        --teal-400: #2dd4bf;
        --panel: rgba(255, 255, 255, 0.9);
        --panel-border: rgba(148, 163, 184, 0.28);
    }

    html, body, [class*="css"] {
        font-family: 'Sora', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(80rem 50rem at 6% -5%, rgba(2, 132, 199, 0.16), transparent 55%),
            radial-gradient(70rem 40rem at 95% 0%, rgba(45, 212, 191, 0.13), transparent 60%),
            linear-gradient(180deg, #f7fbff 0%, #eef5fb 55%, #f8fbff 100%);
        color: var(--ink-900);
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b2947 0%, #12345a 58%, #0f2f52 100%);
        border-right: 1px solid rgba(148, 163, 184, 0.25);
    }

    section[data-testid="stSidebar"] * {
        color: #e2edf8;
    }

    /* Main Header */
    .main-header {
        background: linear-gradient(120deg, #0f3460 0%, #0a4f83 55%, #0284c7 100%);
        color: white;
        text-align: center;
        padding: 48px 22px;
        border-radius: 18px;
        box-shadow: 0 24px 60px rgba(15, 52, 96, 0.28);
        margin-bottom: 20px;
        letter-spacing: 0.01em;
    }

    .subtitle {
        color: #35506a;
        font-size: 1.05rem;
        text-align: center;
        margin: 0 0 26px 0;
        font-weight: 500;
    }

    .section-header {
        color: #0a4f83;
        font-size: 1.5rem;
        font-weight: 800;
        margin-top: 34px;
        margin-bottom: 20px;
        padding-bottom: 12px;
        border-bottom: 2px solid rgba(14, 165, 233, 0.35);
    }

    .insight-box {
        background: linear-gradient(135deg, #f0f9ff 0%, #e6fffb 100%);
        padding: 18px;
        border: 1px solid rgba(14, 165, 233, 0.28);
        border-left: 5px solid var(--sky-500);
        border-radius: 12px;
        margin: 14px 0;
        box-shadow: 0 8px 26px rgba(14, 165, 233, 0.16);
    }

    .success-box {
        background: linear-gradient(135deg, #edfff7 0%, #eafff0 100%);
        padding: 18px;
        border: 1px solid rgba(34, 197, 94, 0.25);
        border-left: 5px solid #22c55e;
        border-radius: 12px;
        margin: 14px 0;
        box-shadow: 0 8px 24px rgba(34, 197, 94, 0.14);
    }

    .warning-box {
        background: linear-gradient(135deg, #fff9eb 0%, #fffef2 100%);
        padding: 18px;
        border: 1px solid rgba(251, 146, 60, 0.28);
        border-left: 5px solid #f97316;
        border-radius: 12px;
        margin: 14px 0;
    }

    div[data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--panel-border);
        border-radius: 14px;
        padding: 14px 12px;
        box-shadow: 0 10px 26px rgba(15, 23, 42, 0.06);
    }

    div[data-baseweb="select"], div[data-baseweb="input"] {
        border-radius: 12px;
    }

    .stButton > button {
        border-radius: 12px;
        font-weight: 700;
        border: 1px solid rgba(2, 132, 199, 0.35);
        box-shadow: 0 10px 24px rgba(2, 132, 199, 0.18);
    }

    .stDataFrame {
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid rgba(148, 163, 184, 0.24);
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.07);
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #5b7088;
        font-size: 0.9em;
        margin-top: 50px;
        padding-top: 25px;
        border-top: 1px solid rgba(148, 163, 184, 0.32);
        font-weight: 500;
    }
    
    h3 {
        color: #0a395f;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>🌍 Carbon Emissions Analytics Platform</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Research-Grade Carbon Forecasting with Scenario Intelligence, Uncertainty Quantification, and Explainability</p>", unsafe_allow_html=True)

BASE_FEATURE_ORDER = [
    'population', 'gdp', 'coal_co2', 'oil_co2',
    'gas_co2', 'methane', 'nitrous_oxide', 'primary_energy_consumption'
]

# ===========================
# LOAD MODELS & DATA
# ===========================

@st.cache_resource
def load_regression_models():
    models = {}
    load_errors = []

    model_files = {
        'gradient_boosting': 'models/regression_gradient_boosting.pkl',
        'xgboost': 'models/regression_xgboost.pkl'
    }

    for model_name, model_path in model_files.items():
        try:
            models[model_name] = joblib.load(model_path)
        except Exception as e:
            load_errors.append(f"Failed to load {model_name} from {model_path}: {e}")

    return models, load_errors

@st.cache_resource
def load_deep_learning_models():
    dl_models = {}
    load_errors = []

    class DenseCompat(Dense):
        """Compatibility Dense layer for models saved with extra config keys."""

        @classmethod
        def from_config(cls, config):
            cfg = dict(config)
            cfg.pop('quantization_config', None)
            return super().from_config(cfg)

    model_files = {
        'dnn': 'models/dnn_model.h5',
        'lstm': 'models/lstm_model.h5',
        'gru': 'models/gru_model.h5',
        'autoencoder': 'models/autoencoder_model.h5'
    }

    for model_name, model_path in model_files.items():
        try:
            dl_models[model_name] = load_model(model_path, compile=False)
        except Exception as e:
            try:
                dl_models[model_name] = load_model(
                    model_path,
                    compile=False,
                    custom_objects={'Dense': DenseCompat}
                )
            except Exception as compat_error:
                load_errors.append(f"Failed to load {model_name} from {model_path}: {compat_error}")

    return dl_models, load_errors

@st.cache_resource
def load_scaler():
    try:
        return joblib.load('models/scaler_regression.pkl'), None
    except Exception as e:
        return None, f"Failed to load scaler from models/scaler_regression.pkl: {e}"

@st.cache_resource
def load_feature_info():
    try:
        with open('models/feature_info.json', 'r') as f:
            return json.load(f), None
    except Exception as e:
        return None, f"Failed to load feature info from models/feature_info.json: {e}"

@st.cache_resource
def load_metadata():
    try:
        with open('models/metadata.json', 'r') as f:
            return json.load(f), None
    except Exception as e:
        return None, f"Failed to load metadata from models/metadata.json: {e}"

@st.cache_data
def load_country_profiles():
    try:
        df = pd.read_csv('owid-co2-data.csv', usecols=['country', 'year', 'co2'] + BASE_FEATURE_ORDER)
        numeric_cols = ['year', 'co2'] + BASE_FEATURE_ORDER
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        trained_countries = set()
        if os.path.exists('models/metadata.json'):
            with open('models/metadata.json', 'r') as f:
                metadata = json.load(f)
            trained_countries = set(metadata.get('countries_trained', []))

        df = df.dropna(subset=BASE_FEATURE_ORDER)
        if df.empty:
            return {}, [], "No country profiles contain all required features."

        latest_profiles = df.sort_values(['country', 'year']).groupby('country', as_index=False).tail(1)

        profiles = {}
        for _, row in latest_profiles.iterrows():
            profiles[row['country']] = {
                'year': int(row['year']),
                'features': {feature: float(row[feature]) for feature in BASE_FEATURE_ORDER},
                'co2': float(row['co2']) if pd.notna(row['co2']) else None,
                'is_trained': row['country'] in trained_countries
            }

        countries = sorted(profiles.keys())
        return profiles, countries, None
    except Exception as e:
        return {}, [], f"Failed to load country profiles from owid-co2-data.csv: {e}"

@st.cache_resource
def load_all_performance():
    try:
        return pd.read_csv('models/all_models_performance.csv'), None
    except Exception as e:
        return None, f"Failed to load model performance from models/all_models_performance.csv: {e}"

# Load resources
regression_models, regression_load_errors = load_regression_models()
dl_models, dl_load_errors = load_deep_learning_models()
scaler, scaler_load_error = load_scaler()
feature_info, feature_info_error = load_feature_info()
metadata, metadata_error = load_metadata()
country_profiles, available_countries, country_profiles_error = load_country_profiles()
all_perf_df, all_perf_error = load_all_performance()

load_issues = []
load_issues.extend(regression_load_errors)
load_issues.extend(dl_load_errors)
if scaler_load_error:
    load_issues.append(scaler_load_error)
if feature_info_error:
    load_issues.append(feature_info_error)
if metadata_error:
    load_issues.append(metadata_error)
if country_profiles_error:
    load_issues.append(country_profiles_error)
if all_perf_error:
    load_issues.append(all_perf_error)

if not regression_models:
    st.error("❌ Models not found! Please check models/ directory.")
    st.stop()

if load_issues:
    st.caption("Some optional resources are unavailable. Core prediction features remain active.")
    with st.expander("Show loading details"):
        for issue in load_issues:
            st.write(f"- {issue}")

# ===========================
# HELPER FUNCTIONS
# ===========================

FEATURE_ORDER = BASE_FEATURE_ORDER.copy()

MODEL_LABELS = {
    'gradient_boosting': 'Gradient Boosting',
    'xgboost': 'XGBoost',
    'dnn': 'Deep Neural Network',
    'lstm': 'LSTM',
    'gru': 'GRU',
    'autoencoder': 'Autoencoder'
}

MODEL_SHORT_NAMES = {
    'gradient_boosting': 'GB',
    'xgboost': 'XGB',
    'dnn': 'DNN',
    'lstm': 'LSTM',
    'gru': 'GRU',
    'autoencoder': 'AE'
}

DEFAULT_FEATURES = {
    'population': 100000000,
    'gdp': 2500000000000,
    'coal_co2': 500,
    'oil_co2': 300,
    'gas_co2': 200,
    'methane': 50,
    'nitrous_oxide': 30,
    'primary_energy_consumption': 100
}

PREDICTION_STATE_DEFAULTS = {
    'last_prediction_mean': None,
    'advisor_text': "",
    'advisor_source': "",
    'advisor_error': "",
    'advisor_chat_history': [],
    'last_predictions': None,
    'last_uncertainty': None,
    'last_prediction_errors': []
}

def clear_prediction_state():
    for key, default_value in PREDICTION_STATE_DEFAULTS.items():
        if isinstance(default_value, list):
            st.session_state[key] = default_value.copy()
        else:
            st.session_state[key] = default_value

def apply_country_profile(country_name):
    profile = country_profiles.get(country_name)
    if not profile:
        return

    st.session_state.features = profile['features'].copy()
    st.session_state.active_country = country_name
    st.session_state.active_country_year = profile.get('year')
    st.session_state.active_country_actual_co2 = profile.get('co2')
    st.session_state.active_country_trained = profile.get('is_trained', False)
    clear_prediction_state()

def handle_country_change():
    selected_country = st.session_state.get('selected_country')
    if selected_country == 'Custom':
        st.session_state.active_country = None
        st.session_state.active_country_year = None
        st.session_state.active_country_actual_co2 = None
        st.session_state.active_country_trained = False
        clear_prediction_state()
        return

    apply_country_profile(selected_country)

def calculate_prediction_uncertainty(predictions_list, confidence_level=0.95):
    """Calculate configurable confidence intervals for predictions."""
    predictions_array = np.array(predictions_list)
    mean = np.mean(predictions_array)
    std = np.std(predictions_array)
    confidence_level = min(max(float(confidence_level), 0.5), 0.999)
    z_score = NormalDist().inv_cdf((1 + confidence_level) / 2)
    return {
        'mean': mean,
        'std': std,
        'confidence_level': confidence_level,
        'ci_lower': mean - z_score * std,
        'ci_upper': mean + z_score * std,
        'ci_95_lower': mean - 1.96 * std,
        'ci_95_upper': mean + 1.96 * std,
        'cv': (std / mean * 100) if mean > 0 else 0
    }

def get_model_priority_order(perf_df):
    """Return model short names ordered by observed R2 (high to low)."""
    if perf_df is None or perf_df.empty or 'Model' not in perf_df.columns or 'R² Score' not in perf_df.columns:
        return []

    model_to_short = {
        'Gradient Boosting': 'GB',
        'XGBoost': 'XGB',
        'DNN': 'DNN',
        'LSTM': 'LSTM',
        'GRU': 'GRU',
        'Autoencoder': 'AE'
    }

    ordered_models = perf_df.sort_values('R² Score', ascending=False)['Model'].tolist()
    priority = []
    for model_name in ordered_models:
        short_name = model_to_short.get(model_name)
        if short_name and short_name not in priority:
            priority.append(short_name)
    return priority

def aggregate_predictions(predictions, method='Mean', trim_fraction=0.1):
    """Aggregate multiple model predictions using expert-configurable strategies."""
    values = np.array(list(predictions.values()), dtype=float)
    if len(values) == 0:
        return None

    method_l = (method or 'Mean').lower()
    if method_l == 'median':
        return float(np.median(values))
    if method_l == 'trimmed mean':
        if len(values) < 3:
            return float(np.mean(values))
        trim_fraction = min(max(float(trim_fraction), 0.0), 0.4)
        trim_n = int(len(values) * trim_fraction)
        if trim_n == 0 or len(values) - (2 * trim_n) <= 0:
            return float(np.mean(values))
        sorted_values = np.sort(values)
        return float(np.mean(sorted_values[trim_n:len(values)-trim_n]))
    return float(np.mean(values))

def get_available_model_keys(reg_models, deep_models):
    """Return available model keys in display order."""
    ordered = ['gradient_boosting', 'xgboost', 'dnn', 'lstm', 'gru', 'autoencoder']
    available = []
    for model_key in ordered:
        if model_key in reg_models or model_key in deep_models:
            available.append(model_key)
    return available

def generate_predictions(X_scaled, selected_model_keys, reg_models, deep_models):
    """Generate predictions for selected models with per-model error capture."""
    predictions = {}
    errors = []

    for model_key in selected_model_keys:
        if model_key in reg_models:
            try:
                predictions[MODEL_SHORT_NAMES[model_key]] = float(reg_models[model_key].predict(X_scaled)[0])
            except Exception as e:
                errors.append(f"{MODEL_LABELS[model_key]} failed: {e}")
        elif model_key in deep_models:
            try:
                pred = deep_models[model_key].predict(X_scaled, verbose=0)
                predictions[MODEL_SHORT_NAMES[model_key]] = float(np.ravel(pred)[0])
            except Exception as e:
                errors.append(f"{MODEL_LABELS[model_key]} failed: {e}")

    return predictions, errors

def validate_feature_ranges(features, info):
    """Return out-of-range warnings when feature metadata is available."""
    if not info or 'feature_ranges' not in info:
        return []

    warnings_list = []
    ranges = info['feature_ranges']
    for feature in FEATURE_ORDER:
        if feature not in ranges:
            continue

        min_v = ranges[feature].get('min')
        max_v = ranges[feature].get('max')
        value = float(features[feature])

        if min_v is not None and value < float(min_v):
            warnings_list.append(
                f"{feature.replace('_', ' ').title()} is below training range ({value:,.2f} < {float(min_v):,.2f})."
            )
        if max_v is not None and value > float(max_v):
            warnings_list.append(
                f"{feature.replace('_', ' ').title()} is above training range ({value:,.2f} > {float(max_v):,.2f})."
            )

    return warnings_list

def get_primary_explainability_model(models_dict):
    """Pick a stable traditional model for explainability charts."""
    if 'xgboost' in models_dict:
        return 'xgboost', models_dict['xgboost']
    if 'gradient_boosting' in models_dict:
        return 'gradient_boosting', models_dict['gradient_boosting']
    return None, None

def get_model_feature_importance(model, feature_names):
    """Extract normalized feature importance from fitted model attributes."""
    importance_values = None

    if hasattr(model, 'feature_importances_'):
        importance_values = np.array(model.feature_importances_)
    elif hasattr(model, 'coef_'):
        importance_values = np.abs(np.ravel(model.coef_))

    if importance_values is None or len(importance_values) != len(feature_names):
        return None

    total = float(np.sum(importance_values))
    if total > 0:
        importance_values = importance_values / total

    return pd.DataFrame({
        'Feature': [f.replace('_', ' ').title() for f in feature_names],
        'Importance': importance_values
    }).sort_values('Importance', ascending=True)

def get_emissions_mix_dataframe(features):
    """Build a ranked emissions mix table for the current scenario."""
    driver_labels = {
        'coal_co2': 'Coal Carbon',
        'oil_co2': 'Oil Carbon',
        'gas_co2': 'Gas Carbon',
        'methane': 'Methane',
        'nitrous_oxide': 'Nitrous Oxide'
    }
    rows = []
    total = sum(float(features.get(feature, 0)) for feature in driver_labels)

    for feature, label in driver_labels.items():
        value = float(features.get(feature, 0))
        rows.append({
            'Driver': label,
            'Value (MT)': value,
            'Share (%)': (value / total * 100) if total > 0 else 0.0
        })

    return pd.DataFrame(rows).sort_values('Value (MT)', ascending=False).reset_index(drop=True)

def get_scenario_intelligence(features, predicted_co2_mt, actual_co2_mt=None):
    """Compute business-style scenario metrics for the current prediction."""
    predicted_value = max(float(predicted_co2_mt), 0.0)
    population = max(float(features.get('population', 0)), 1.0)
    gdp = float(features.get('gdp', 0))
    energy_consumption = float(features.get('primary_energy_consumption', 0))
    mix_df = get_emissions_mix_dataframe(features)

    top_driver = mix_df.iloc[0]['Driver'] if not mix_df.empty else 'N/A'
    top_share = float(mix_df.iloc[0]['Share (%)']) if not mix_df.empty else 0.0

    delta_absolute = None
    delta_percent = None
    if actual_co2_mt is not None:
        delta_absolute = predicted_value - float(actual_co2_mt)
        if float(actual_co2_mt) != 0:
            delta_percent = (delta_absolute / float(actual_co2_mt)) * 100

    return {
        'predicted_per_capita': predicted_value / population,
        'predicted_per_billion_gdp': (predicted_value / (gdp / 1_000_000_000)) if gdp > 0 else None,
        'predicted_per_ej': (predicted_value / energy_consumption) if energy_consumption > 0 else None,
        'top_driver': top_driver,
        'top_driver_share': top_share,
        'delta_absolute': delta_absolute,
        'delta_percent': delta_percent,
        'mix_df': mix_df
    }

def get_research_overview(metadata_obj, performance_df, profiles):
    """Build concise research-facing summary metrics."""
    best_model = "N/A"
    best_r2 = None
    if performance_df is not None and not performance_df.empty:
        best_row = performance_df.sort_values('R² Score', ascending=False).iloc[0]
        best_model = str(best_row['Model'])
        best_r2 = float(best_row['R² Score'])

    trained_country_count = len(metadata_obj.get('countries_trained', [])) if metadata_obj else 0
    available_country_count = len(profiles)
    feature_count = len(FEATURE_ORDER)

    return {
        'best_model': best_model,
        'best_r2': best_r2,
        'trained_country_count': trained_country_count,
        'available_country_count': available_country_count,
        'feature_count': feature_count,
        'lookback_lstm': metadata_obj.get('lookback_lstm') if metadata_obj else None,
        'train_test_split': metadata_obj.get('train_test_split') if metadata_obj else None,
        'training_date': metadata_obj.get('training_date') if metadata_obj else None
    }

def render_stat_card(label, value, meta):
    st.markdown(
        f"""
        <div class='stat-card'>
            <div class='stat-label'>{label}</div>
            <div class='stat-value'>{value}</div>
            <div class='stat-meta'>{meta}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def get_feature_range_summary(info):
    """Create short feature range summaries for research UI."""
    if not info or 'feature_ranges' not in info:
        return []

    summaries = []
    for feature in FEATURE_ORDER:
        feature_meta = info['feature_ranges'].get(feature)
        if not feature_meta:
            continue
        summaries.append(
            f"{feature.replace('_', ' ').title()}: {float(feature_meta.get('min', 0)):,.2f} to {float(feature_meta.get('max', 0)):,.2f}"
        )
    return summaries

def get_rule_based_reduction_suggestions(features):
    """Fallback advisor when LLM is unavailable."""
    energy_features = ['coal_co2', 'oil_co2', 'gas_co2', 'methane', 'nitrous_oxide']
    ranked = sorted(energy_features, key=lambda x: float(features.get(x, 0)), reverse=True)
    target_map = {
        0: "18-25%",
        1: "12-18%",
        2: "8-12%"
    }

    suggestions = []
    for rank_idx, feat in enumerate(ranked[:3]):
        value = float(features.get(feat, 0))
        label = feat.replace('_', ' ').title()
        reduction_target = target_map.get(rank_idx, "8-12%")
        if feat == 'coal_co2':
            action = "shift power generation from coal to renewables"
        elif feat == 'oil_co2':
            action = "reduce oil-heavy transport and improve EV adoption"
        elif feat == 'gas_co2':
            action = "improve industrial efficiency and reduce gas leakage"
        elif feat == 'methane':
            action = "tighten methane leak detection and capture"
        else:
            action = "optimize fertilizer and industrial process controls"

        suggestions.append(
            f"- Prioritize {label} ({value:,.1f}): {action}; target reduction: {reduction_target} in the next 12 months."
        )

    suggestions.append("- Improve primary energy efficiency with grid modernization and efficiency standards; target 6-10% reduction.")
    suggestions.append("- Track progress quarterly and keep combined reduction target at 15-22% across the top two drivers.")
    return "\n".join(suggestions)

def run_openai_chat(api_key, model_name, system_prompt, user_prompt, temperature=0.3, max_tokens=450):
    """Run chat completion across new and legacy OpenAI SDK interfaces."""
    if OpenAI is not None:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return (response.choices[0].message.content or "").strip()

    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message['content'].strip()

def classify_openai_error(err_text):
    """Return user-friendly reason for known OpenAI API failures."""
    text = (err_text or "").lower()
    if "insufficient_quota" in text or "exceeded your current quota" in text or "error code: 429" in text:
        return "OpenAI quota exceeded (429 insufficient_quota)."
    if "invalid_api_key" in text or "incorrect api key" in text:
        return "Invalid OpenAI API key."
    if "model" in text and "not found" in text:
        return "Configured model is unavailable for this API key."
    return err_text

def get_rule_based_followup_response(user_question, features):
    """Provide a practical follow-up answer when LLM is unavailable."""
    q = (user_question or "").lower()
    emission_drivers = ['coal_co2', 'oil_co2', 'gas_co2', 'methane', 'nitrous_oxide']
    ranked = sorted(emission_drivers, key=lambda k: float(features.get(k, 0)), reverse=True)

    driver_aliases = {
        'coal_co2': ['coal', 'coal co2'],
        'oil_co2': ['oil', 'oil co2'],
        'gas_co2': ['gas', 'gas co2', 'natural gas'],
        'methane': ['methane', 'ch4'],
        'nitrous_oxide': ['nitrous', 'nitrous oxide', 'n2o']
    }

    selected_driver = None
    for driver, aliases in driver_aliases.items():
        if any(alias in q for alias in aliases):
            selected_driver = driver
            break

    if selected_driver is None:
        selected_driver = ranked[0]

    pct_match = re.search(r"(\d+(?:\.\d+)?)\s*%", q)
    months_match = re.search(r"(\d+)\s*(?:month|months|mo|mos)\b", q)

    target_percent = float(pct_match.group(1)) if pct_match else None
    target_months = int(months_match.group(1)) if months_match else (12 if '12 month' in q or '1 year' in q else None)

    if target_percent is not None and target_months is not None:
        driver_value = float(features.get(selected_driver, 0))
        reduction = driver_value * (target_percent / 100.0)
        driver_label = selected_driver.replace('_', ' ').title()

        if target_percent <= 12:
            feasibility = "realistic"
        elif target_percent <= 18:
            feasibility = "possible but stretch"
        else:
            feasibility = "aggressive and requires strong policy support"

        second_driver = ranked[1] if len(ranked) > 1 else ranked[0]
        second_label = second_driver.replace('_', ' ').title()

        return (
            f"Yes, a {target_percent:.1f}% reduction in {driver_label} over {target_months} months is {feasibility} if phased properly.\n"
            f"- Target: reduce {driver_label} by about {reduction:,.1f} from the current level ({driver_value:,.1f}).\n"
            "- Months 1-3: capture quick wins through efficiency and dispatch optimization.\n"
            "- Months 4-8: shift fuel mix and scale operational controls to sustain monthly cuts.\n"
            f"- Months 9-{target_months}: lock in policy/compliance measures and monitor against monthly checkpoints.\n"
            f"- Next quarter sequencing: after stabilizing {driver_label}, start reductions in {second_label}.\n"
            "- Keep a 1-2% buffer to absorb seasonal demand variability."
        )

    top_driver = ranked[0].replace('_', ' ').title()
    return (
        "LLM follow-up is currently unavailable. "
        f"Start with your top driver ({top_driver}) and apply the recommended percentage range first, "
        "then sequence the second-largest source for the next quarter."
    )

def get_llm_reduction_suggestions(features, predicted_co2_mt):
    """Generate reduction plan with OpenAI; fallback to deterministic rules."""
    api_key = os.getenv('OPENAI_API_KEY', '').strip() or OPENAI_API_KEY_HARDCODED.strip()
    if not api_key:
        return get_rule_based_reduction_suggestions(features), "rule-based", "OPENAI_API_KEY is missing"

    try:
        model_name = os.getenv('OPENAI_MODEL', '').strip() or OPENAI_MODEL_HARDCODED
        prompt = f"""
You are a climate policy and carbon optimization advisor.

Current country profile:
- population: {features['population']}
- gdp: {features['gdp']}
- coal_co2: {features['coal_co2']}
- oil_co2: {features['oil_co2']}
- gas_co2: {features['gas_co2']}
- methane: {features['methane']}
- nitrous_oxide: {features['nitrous_oxide']}
- primary_energy_consumption: {features['primary_energy_consumption']}
- predicted_total_carbon_mt: {predicted_co2_mt:.2f}

Task:
1. Identify top 3 usage drivers to reduce first.
2. Give specific actions for each driver.
3. For each driver, include a percentage reduction target range for 12 months.
4. Estimate relative impact (high/medium/low) for each action.
4. End with a 90-day action plan in 3 bullets.

Keep response concise and practical.
Output each driver line in this format:
- <driver>: <action> | target reduction: <x-y%> | impact: <high/medium/low>
"""

        content = run_openai_chat(
            api_key=api_key,
            model_name=model_name,
            system_prompt="You provide practical carbon reduction recommendations.",
            user_prompt=prompt,
            temperature=0.3,
            max_tokens=450
        )
        return content, f"openai:{model_name}", ""
    except Exception as e:
        return get_rule_based_reduction_suggestions(features), "rule-based", str(e)

def get_followup_advisor_response(user_question, features, predicted_co2_mt, current_plan):
    """Answer user follow-up questions about reduction strategy."""
    api_key = os.getenv('OPENAI_API_KEY', '').strip() or OPENAI_API_KEY_HARDCODED.strip()
    if not api_key:
        return (
            get_rule_based_followup_response(user_question, features),
            "rule-based",
            "OPENAI_API_KEY is missing"
        )

    try:
        model_name = os.getenv('OPENAI_MODEL', '').strip() or OPENAI_MODEL_HARDCODED

        prompt = f"""
You are a practical carbon reduction advisor.

Current profile:
- population: {features['population']}
- gdp: {features['gdp']}
- coal_co2: {features['coal_co2']}
- oil_co2: {features['oil_co2']}
- gas_co2: {features['gas_co2']}
- methane: {features['methane']}
- nitrous_oxide: {features['nitrous_oxide']}
- primary_energy_consumption: {features['primary_energy_consumption']}
- predicted_total_carbon_mt: {predicted_co2_mt:.2f}

Current reduction plan:
{current_plan}

User follow-up question:
{user_question}

Answer in 4-8 bullet points. Include concrete actions and percentage targets when relevant.
"""

        content = run_openai_chat(
            api_key=api_key,
            model_name=model_name,
            system_prompt="You answer climate strategy follow-up questions with practical steps.",
            user_prompt=prompt,
            temperature=0.3,
            max_tokens=350
        )
        return content, f"openai:{model_name}", ""
    except Exception as e:
        return (
            get_rule_based_followup_response(user_question, features),
            "rule-based",
            classify_openai_error(str(e))
        )

# ===========================
# SIDEBAR - CONFIGURATION
# ===========================

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.divider()

    st.markdown("**📋 Experience Mode**")
    mode = st.radio(
        "Select Mode:",
        options=['Quick Predict', 'Model Explainability'],
        help="Choose the right mode for your use case",
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown("**🤖 Prediction Models**")
    available_model_keys = get_available_model_keys(regression_models, dl_models)
    selected_model_keys = st.multiselect(
        "Choose models:",
        options=available_model_keys,
        default=available_model_keys,
        format_func=lambda k: MODEL_LABELS.get(k, k),
        help="These selected models will be used for quick and batch predictions.",
        label_visibility="collapsed"
    )

    st.markdown("**📊 Output Options**")
    show_confidence_band = st.checkbox("Show confidence interval band", value=True)

    st.divider()

    st.markdown("""
    ### 📌 Feature Ranges
    - **Population**: 0 - 1.4B
    - **GDP**: $0 - $100T
    - **Coal Carbon**: 0 - 10K MT
    - **Oil Carbon**: 0 - 10K MT
    - **Gas Carbon**: 0 - 10K MT
    """)

# ===========================
# MAIN CONTENT
# ===========================

if mode == 'Quick Predict':
    st.markdown("<h2 class='section-header'>⚡ Quick Prediction Engine</h2>", unsafe_allow_html=True)
    
    if 'features' not in st.session_state:
        st.session_state.features = DEFAULT_FEATURES.copy()
    for state_key, default_value in PREDICTION_STATE_DEFAULTS.items():
        if state_key not in st.session_state:
            if isinstance(default_value, list):
                st.session_state[state_key] = default_value.copy()
            else:
                st.session_state[state_key] = default_value
    if 'active_country' not in st.session_state:
        st.session_state.active_country = None
    if 'active_country_year' not in st.session_state:
        st.session_state.active_country_year = None
    if 'active_country_actual_co2' not in st.session_state:
        st.session_state.active_country_actual_co2 = None
    if 'active_country_trained' not in st.session_state:
        st.session_state.active_country_trained = False
    country_options = ['Custom'] + available_countries
    if 'selected_country' not in st.session_state:
        st.session_state.selected_country = 'Custom'
    if st.session_state.selected_country not in country_options:
        st.session_state.selected_country = 'Custom'

    st.markdown("<h3 style='color: #0066ff; font-weight: 700;'>🌐 Country Profile</h3>", unsafe_allow_html=True)
    st.selectbox(
        "Select country profile",
        options=country_options,
        key='selected_country',
        on_change=handle_country_change,
        help="Choose a trained country to auto-fill the latest available profile, or keep Custom for manual inputs."
    )

    if available_countries:
        st.caption(f"Country presets are available for {len(available_countries)} countries with complete feature data.")

    if st.session_state.active_country:
        profile_parts = [f"Using latest available profile for {st.session_state.active_country}"]
        if st.session_state.active_country_year is not None:
            profile_parts.append(f"year {int(st.session_state.active_country_year)}")
        if st.session_state.active_country_actual_co2 is not None:
            profile_parts.append(f"historical carbon {st.session_state.active_country_actual_co2:,.1f} MT")
        st.info(" | ".join(profile_parts))
        if not st.session_state.active_country_trained:
            st.caption("Generalization mode: this country was outside the original training set, so confidence should be interpreted more cautiously.")
    else:
        st.caption("Custom scenario mode: enter values manually for prediction.")

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h3 style='color: #00d4ff; font-weight: 700;'>📍 Location & Economy</h3>", unsafe_allow_html=True)
        st.session_state.features['population'] = st.number_input("Population", min_value=0, value=int(st.session_state.features['population']), step=1000000)
        st.session_state.features['gdp'] = st.number_input("GDP (USD)", min_value=0, value=int(st.session_state.features['gdp']), step=100000000000)
        
        st.markdown("<h3 style='color: #00d4ff; font-weight: 700;'>⛽ Energy Sources (MT Carbon)</h3>", unsafe_allow_html=True)
        st.session_state.features['coal_co2'] = st.number_input("Coal Carbon", min_value=0.0, value=float(st.session_state.features['coal_co2']), step=10.0)
        st.session_state.features['oil_co2'] = st.number_input("Oil Carbon", min_value=0.0, value=float(st.session_state.features['oil_co2']), step=10.0)
        st.session_state.features['gas_co2'] = st.number_input("Gas Carbon", min_value=0.0, value=float(st.session_state.features['gas_co2']), step=10.0)
    
    with col2:
        st.markdown("<h3 style='color: #00ff66; font-weight: 700;'>💨 Other Emissions</h3>", unsafe_allow_html=True)
        st.session_state.features['methane'] = st.number_input("Methane", min_value=0.0, value=float(st.session_state.features['methane']), step=5.0)
        st.session_state.features['nitrous_oxide'] = st.number_input("Nitrous Oxide", min_value=0.0, value=float(st.session_state.features['nitrous_oxide']), step=5.0)
        
        st.markdown("<h3 style='color: #ff9900; font-weight: 700;'>⚡ Energy Consumption</h3>", unsafe_allow_html=True)
        st.session_state.features['primary_energy_consumption'] = st.number_input("Primary Energy (EJ)", min_value=0.0, value=float(st.session_state.features['primary_energy_consumption']), step=5.0)
        
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🚀 Predict Carbon Emissions", use_container_width=True, type="primary", key="predict_btn")

    feature_warnings = validate_feature_ranges(st.session_state.features, feature_info)
    if feature_warnings:
        st.warning("⚠️ Some inputs are outside training ranges.")
        with st.expander("Show range warnings"):
            for warn_msg in feature_warnings:
                st.write(f"- {warn_msg}")
    
    st.divider()
    
    if predict_btn:
        try:
            if scaler is None:
                st.error("Scaler is not available, so prediction cannot run.")
                st.stop()

            if not selected_model_keys:
                st.error("Select at least one model from the sidebar to run predictions.")
                st.stop()

            X_input = np.array([st.session_state.features[f] for f in FEATURE_ORDER])
            X_scaled = scaler.transform(X_input.reshape(1, -1))
            
            # Get predictions
            predictions, prediction_errors = generate_predictions(
                X_scaled,
                selected_model_keys,
                regression_models,
                dl_models
            )
            
            if predictions:
                st.session_state.last_predictions = predictions
                st.session_state.last_prediction_errors = prediction_errors
                st.markdown("<div class='success-box'><span style='color: #00ff66; font-weight: 700; font-size: 1.1em;'>✅ Prediction successful!</span></div>", unsafe_allow_html=True)
                if prediction_errors:
                    st.info("Some model outputs were skipped due to runtime errors.")
                    with st.expander("Show prediction warnings"):
                        for err in prediction_errors:
                            st.write(f"- {err}")
                
                # Calculate uncertainty
                pred_values = list(predictions.values())
                uncertainty = calculate_prediction_uncertainty(pred_values, confidence_level=0.95)
                st.session_state.last_uncertainty = uncertainty
                final_prediction = float(uncertainty['mean'])
                st.session_state.last_prediction_mean = float(final_prediction)
                
                # Display main carbon prediction
                prediction_scope = st.session_state.active_country or 'Custom Scenario'
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #0066ff 0%, #00d4ff 100%); padding: 25px; border-radius: 15px; margin: 20px 0; box-shadow: 0 8px 20px rgba(0, 102, 255, 0.3);'>
                    <div style='color: white; text-align: center;'>
                        <p style='font-size: 0.9em; margin: 0 0 10px 0; opacity: 0.9;'>🌍 Predicted Carbon Emissions for {prediction_scope}</p>
                        <h2 style='color: #00ff66; font-size: 2.5em; margin: 0; font-weight: 800;'>{final_prediction:,.0f} MT</h2>
                        <p style='font-size: 0.85em; margin: 8px 0 0 0; opacity: 0.9;'>Average across all models</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Display individual model predictions
                st.markdown("<h4 style='color: #00d4ff; font-weight: 700; margin-top: 20px;'>Individual Model Carbon Predictions:</h4>", unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                for idx, (model, value) in enumerate(sorted(predictions.items(), key=lambda x: x[1], reverse=True)):
                    with [col1, col2, col3][idx % 3]:
                        st.metric(model, f"{value:,.0f} MT")
                
                # Display metrics
                st.markdown("<h3 style='color: #00d4ff; margin-top: 30px; font-weight: 700;'>📊 Prediction Results</h3>", unsafe_allow_html=True)
                metric_cols = st.columns(5)
                
                metrics_data = [
                    ("🎯 Final", f"{final_prediction:,.0f} MT"),
                    ("📊 Std Dev", f"{uncertainty['std']:,.0f}"),
                    ("📈 CV", f"{uncertainty['cv']:.1f}%"),
                    (f"📉 {int(uncertainty['confidence_level'] * 100)}% Lower Carbon", f"{uncertainty['ci_lower']:,.0f} MT"),
                    (f"📈 {int(uncertainty['confidence_level'] * 100)}% Upper Carbon", f"{uncertainty['ci_upper']:,.0f} MT")
                ]
                
                for col, (label, value) in zip(metric_cols, metrics_data):
                    with col:
                        st.metric(label, value)
                
                # Insights
                st.markdown("<h3 style='color: #ff9900; font-weight: 700; margin-top: 30px;'>💡 Prediction Insights</h3>", unsafe_allow_html=True)
                col_i1, col_i2, col_i3 = st.columns(3)
                
                with col_i1:
                    agreement = 100 - uncertainty['cv']
                    st.metric("Model Agreement", f"{agreement:.1f}%")
                
                with col_i2:
                    selected_pred = max(predictions.values()) if predictions else None
                    if selected_pred is not None:
                        st.metric("Peak Model Output", f"{selected_pred:,.0f} MT")
                    else:
                        st.metric("Peak Model Output", "N/A")
                
                with col_i3:
                    if uncertainty['cv'] < 10:
                        risk = "🟢 Low Risk"
                    elif uncertainty['cv'] < 20:
                        risk = "🟡 Medium Risk"
                    else:
                        risk = "🔴 High Risk"
                    st.metric("Confidence", risk)

                scenario_intel = get_scenario_intelligence(
                    st.session_state.features,
                    final_prediction,
                    st.session_state.active_country_actual_co2
                )

                st.markdown("<h3 style='color: #0f3460; margin-top: 30px; font-weight: 700;'>🧠 Scenario Intelligence</h3>", unsafe_allow_html=True)
                intel_cols = st.columns(4)

                with intel_cols[0]:
                    if scenario_intel['delta_absolute'] is not None:
                        st.metric(
                            "Vs Latest Actual Carbon",
                            f"{scenario_intel['delta_absolute']:,.0f} MT",
                            delta=f"{scenario_intel['delta_percent']:.1f}%" if scenario_intel['delta_percent'] is not None else None
                        )
                    else:
                        st.metric("Vs Latest Actual Carbon", "N/A")

                with intel_cols[1]:
                    st.metric("Carbon / Capita", f"{scenario_intel['predicted_per_capita']:.4f} MT")

                with intel_cols[2]:
                    if scenario_intel['predicted_per_billion_gdp'] is not None:
                        st.metric("Carbon / $1B GDP", f"{scenario_intel['predicted_per_billion_gdp']:.2f} MT")
                    else:
                        st.metric("Carbon / $1B GDP", "N/A")

                with intel_cols[3]:
                    st.metric("Primary Driver", scenario_intel['top_driver'], delta=f"{scenario_intel['top_driver_share']:.1f}% share")

                mix_df = scenario_intel['mix_df'].copy()
                mix_df['Value (MT)'] = mix_df['Value (MT)'].map(lambda value: f"{value:,.2f}")
                mix_df['Share (%)'] = mix_df['Share (%)'].map(lambda value: f"{value:.1f}%")
                st.dataframe(mix_df, use_container_width=True, hide_index=True)

                top_actions = scenario_intel['mix_df'].head(3)
                if not top_actions.empty:
                    st.markdown("""
                    <div class='insight-box'>
                    <b style='color: #0f3460;'>Priority Action Queue</b>
                    </div>
                    """, unsafe_allow_html=True)
                    for idx, row in top_actions.iterrows():
                        st.write(f"{idx + 1}. {row['Driver']} drives {row['Share (%)']:.1f}% of the current emissions mix and should be targeted first.")
            else:
                st.warning("No predictions were generated with the selected models.")
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

    if (not predict_btn) and st.session_state.last_predictions and st.session_state.last_uncertainty:
        st.markdown("<h3 style='color: #0066ff; font-weight: 700;'>📌 Last Prediction Snapshot</h3>", unsafe_allow_html=True)
        st.caption("Showing the most recent prediction results. Click Predict Carbon Emissions to refresh.")

        last_predictions = st.session_state.last_predictions
        last_uncertainty = st.session_state.last_uncertainty

        if st.session_state.last_prediction_errors:
            with st.expander("Show prediction warnings"):
                for err in st.session_state.last_prediction_errors:
                    st.write(f"- {err}")

        snap_cols = st.columns(3)
        sorted_snap_preds = sorted(last_predictions.items(), key=lambda x: x[1], reverse=True)
        for idx, (model, value) in enumerate(sorted_snap_preds):
            with snap_cols[idx % 3]:
                st.metric(model, f"{value:,.0f} MT")

        metric_cols = st.columns(5)
        snap_metrics = [
            ("🎯 Final", f"{st.session_state.last_prediction_mean:,.0f} MT"),
            ("📊 Std Dev", f"{last_uncertainty['std']:,.0f}"),
            ("📈 CV", f"{last_uncertainty['cv']:.1f}%"),
            (f"📉 {int(last_uncertainty.get('confidence_level', 0.95) * 100)}% Lower Carbon", f"{last_uncertainty.get('ci_lower', last_uncertainty['ci_95_lower']):,.0f} MT"),
            (f"📈 {int(last_uncertainty.get('confidence_level', 0.95) * 100)}% Upper Carbon", f"{last_uncertainty.get('ci_upper', last_uncertainty['ci_95_upper']):,.0f} MT")
        ]
        for col, (label, value) in zip(metric_cols, snap_metrics):
            with col:
                st.metric(label, value)

    if st.session_state.last_prediction_mean is not None:
        st.divider()
        st.markdown("<h3 style='color: #0f3460; font-weight: 700;'>🤖 AI Carbon Reduction Advisor</h3>", unsafe_allow_html=True)
        st.caption("Get targeted suggestions on which usage areas to decrease to reduce carbon output.")

        if st.button("Generate Reduction Suggestions", type="primary", use_container_width=True, key="ai_advisor_btn"):
            with st.spinner("Analyzing current profile and generating recommendations..."):
                advice_text, advisor_source, advisor_error = get_llm_reduction_suggestions(
                    st.session_state.features,
                    st.session_state.last_prediction_mean
                )
            st.session_state.advisor_text = advice_text
            st.session_state.advisor_source = advisor_source
            st.session_state.advisor_error = advisor_error

        if st.session_state.advisor_text:
            if st.session_state.advisor_source.startswith("openai:"):
                st.success(f"Suggestions generated using {st.session_state.advisor_source}.")
            else:
                st.info("OPENAI_API_KEY not configured or API unavailable. Showing rule-based recommendations.")
                if st.session_state.advisor_error:
                    with st.expander("Advisor fallback reason"):
                        st.write(st.session_state.advisor_error)

            st.markdown("""
            <div class='insight-box'>
            <b style='color: #0f3460;'>Recommended Reduction Priorities</b>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(st.session_state.advisor_text)

            st.markdown("### 💬 Ask Follow-up Question")
            with st.form("advisor_followup_form", clear_on_submit=True):
                user_followup = st.text_input(
                    "Ask about implementation, timeline, sector-specific actions, or budget trade-offs:",
                    key="advisor_followup_input"
                )
                ask_advisor_btn = st.form_submit_button("Ask Advisor", use_container_width=True)

            if ask_advisor_btn:
                if user_followup.strip():
                    with st.spinner("Generating follow-up answer..."):
                        followup_answer, followup_source, followup_error = get_followup_advisor_response(
                            user_followup.strip(),
                            st.session_state.features,
                            st.session_state.last_prediction_mean,
                            st.session_state.advisor_text
                        )
                    st.session_state.advisor_chat_history.append({
                        'question': user_followup.strip(),
                        'answer': followup_answer,
                        'source': followup_source,
                        'error': followup_error
                    })
                else:
                    st.warning("Please enter a follow-up question.")

            if st.session_state.advisor_chat_history:
                st.markdown("### 🗣️ Advisor Conversation")
                for idx, item in enumerate(st.session_state.advisor_chat_history, start=1):
                    st.markdown(f"**You {idx}:** {item['question']}")
                    st.markdown(f"**Advisor {idx}:** {item['answer']}")
                    if item['source'].startswith("openai:"):
                        st.caption(f"Source: {item['source']}")
                    elif item['error']:
                        with st.expander(f"Follow-up fallback reason {idx}"):
                            st.write(item['error'])

elif mode == 'Model Explainability':
    st.markdown("<h2 class='section-header'>🤖 Model Explainability</h2>", unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🔍 Feature Importance", "📈 Sensitivity Analysis"])
    
    with tab1:
        st.markdown("<h3 style='color: #00d4ff; font-weight: 700;'>Feature Importance</h3>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class='insight-box'>
        <span style='color: #00d4ff; font-weight: 600;'>Shows which input features have the biggest impact on carbon predictions.</span>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔍 Calculate Importance", type="primary", use_container_width=True, key="feature_importance"):
            with st.spinner("Analyzing..."):
                model_name, explain_model = get_primary_explainability_model(regression_models)
                if explain_model is None:
                    st.error("No compatible regression model is available for feature importance.")
                else:
                    importance_df = get_model_feature_importance(explain_model, FEATURE_ORDER)
                    if importance_df is None:
                        st.error("Selected model does not expose feature importance attributes.")
                    else:
                        fig_imp = px.bar(
                            importance_df,
                            x='Importance',
                            y='Feature',
                            orientation='h',
                            color='Importance',
                            color_continuous_scale='Viridis',
                            text='Importance'
                        )
                        fig_imp.update_traces(texttemplate='%{text:.1%}', textposition='outside')
                        fig_imp.update_layout(
                            height=400,
                            template='plotly_white',
                            showlegend=False,
                            title=f"Feature Importance ({model_name})"
                        )
                        st.plotly_chart(fig_imp, use_container_width=True)
    
    with tab2:
        st.markdown("<h3 style='color: #ff9900; font-weight: 700;'>Sensitivity Analysis</h3>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class='insight-box'>
        <span style='color: #ff9900; font-weight: 600;'>Analyze how carbon changes when you vary each feature.</span>
        </div>
        """, unsafe_allow_html=True)
        
        col_s1, col_s2 = st.columns(2)
        
        with col_s1:
            feature_to_analyze = st.selectbox(
                "📊 Select feature:",
                options=FEATURE_ORDER,
                format_func=lambda x: x.replace('_', ' ').title()
            )
        
        with col_s2:
            variation_range = st.slider("📈 Variation Range (%)", 10, 100, 50, step=10)
        
        if st.button("▶️ Run Analysis", type="primary", use_container_width=True, key="sensitivity_run"):
            model_name, explain_model = get_primary_explainability_model(regression_models)
            if explain_model is None or scaler is None:
                st.error("Sensitivity analysis requires a loaded regression model and scaler.")
            else:
                base_features = st.session_state.get('features', DEFAULT_FEATURES.copy())

                base_input = np.array([base_features[f] for f in FEATURE_ORDER], dtype=float)
                feature_idx = FEATURE_ORDER.index(feature_to_analyze)
                base_value = base_input[feature_idx]

                lower = base_value * (1 - variation_range / 100)
                upper = base_value * (1 + variation_range / 100)
                variations = np.linspace(lower, upper, 20)
                predictions = []

                for var in variations:
                    modified_input = base_input.copy()
                    modified_input[feature_idx] = var
                    modified_scaled = scaler.transform(modified_input.reshape(1, -1))
                    predictions.append(float(explain_model.predict(modified_scaled)[0]))

                fig_sens = go.Figure()
                fig_sens.add_trace(go.Scatter(
                    x=variations,
                    y=predictions,
                    mode='lines+markers',
                    name='Prediction',
                    line=dict(color='#667eea', width=3),
                    marker=dict(size=8)
                ))
                fig_sens.add_vline(
                    x=base_value,
                    line_dash="dash",
                    line_color="#ef4444",
                    line_width=2,
                    annotation_text="Current"
                )

                fig_sens.update_layout(
                    title=f"Sensitivity to {feature_to_analyze.replace('_', ' ').title()} ({model_name})",
                    xaxis_title="Feature Value",
                    yaxis_title="Predicted Carbon (MT)",
                    height=400,
                    template='plotly_white'
                )
                st.plotly_chart(fig_sens, use_container_width=True)

# Footer
st.divider()
st.markdown("""
<div class='footer'>
<p><b>🌍 Carbon Analytics Platform</b> | Enterprise ML System<br>
<small>© 2026 | Production Ready</small></p>
</div>
""", unsafe_allow_html=True)
