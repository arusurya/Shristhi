# UI framework for building the interactive dashboard
import streamlit as st

# Data manipulation and numerical operations
import pandas as pd
import numpy as np

# Minor randomness used in simulation
import random

# Time utilities for timestamps and slot calculations
from datetime import datetime, timedelta, time

# Visualization libraries for interactive charts
import plotly.express as px
import plotly.graph_objects as go

# ML models for prediction and anomaly detection
from sklearn.ensemble import RandomForestRegressor, IsolationForest

# Train-test split for model validation
from sklearn.model_selection import train_test_split

# Metrics to evaluate model performance
from sklearn.metrics import mean_absolute_error, r2_score

# Suppress warnings to keep UI clean
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

# configure dashboard layout and appearance
st.set_page_config(
    page_title="Smart Campus Mess Intelligence",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────

# styling for cards, alerts, and progress bars (UI only)
st.markdown("""
<style>
    .metric-card {
        background: #f0f2f6;
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
        margin-bottom: 8px;
    }
    .metric-card h2 { margin: 0; font-size: 2rem; }
    .metric-card p  { margin: 0; color: #555; font-size: 0.85rem; }

    .prob-bar-container {
        background: #e8ecef;
        border-radius: 20px;
        height: 28px;
        overflow: hidden;
        margin: 6px 0;
    }
    .prob-bar {
        height: 100%;
        border-radius: 20px;
        display: flex;
        align-items: center;
        padding-left: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        color: white;
        transition: width 0.4s ease;
    }
    .status-green  { background: linear-gradient(90deg, #28a745, #5cb85c); }
    .status-yellow { background: linear-gradient(90deg, #ffc107, #ffca2c); color: #333; }
    .status-red    { background: linear-gradient(90deg, #dc3545, #ff6b6b); }

    .alert-box {
        padding: 14px 18px;
        border-radius: 10px;
        margin: 10px 0;
        font-size: 0.9rem;
    }
    .alert-info    { background: #d1ecf1; border-left: 4px solid #0c5460; color: #0c5460; }
    .alert-warning { background: #fff3cd; border-left: 4px solid #856404; color: #533f03; }
    .alert-danger  { background: #f8d7da; border-left: 4px solid #721c24; color: #721c24; }
    .alert-success { background: #d4edda; border-left: 4px solid #155724; color: #155724; }
    .cold-start-banner {
        background: linear-gradient(135deg, #e3f2fd, #f3e5f5);
        border-left: 4px solid #7b1fa2;
        border-radius: 10px;
        padding: 14px 18px;
        margin: 10px 0;
        font-size: 0.9rem;
        color: #4a148c;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SNACKS DATA HANDLING
# ─────────────────────────────────────────────

# extract snacks data; if missing, estimate using lunch pattern
def snacks_data_summary(df: pd.DataFrame) -> dict:

    # aggregate daily counts
    daily = df.groupby(["date", "meal_slot"]).size().reset_index(name="students")

    # filter snacks rows
    snacks_rows = daily[daily["meal_slot"] == "Snacks"]

    # if snacks data exists → use it
    if not snacks_rows.empty:
        return {
            "days": len(snacks_rows),
            "mean": snacks_rows["students"].mean(),
            "available": True
        }

    # fallback: estimate snacks using lunch data
    lunch_rows = daily[daily["meal_slot"] == "Lunch"]

    if lunch_rows.empty:
        assumed_mean = 120
    else:
        assumed_mean = lunch_rows["students"].mean() * 0.28

    return {
        "days": 14,
        "mean": assumed_mean,
        "available": True
    }
    # ─────────────────────────────────────────────
# DATA SIMULATION
# ─────────────────────────────────────────────

# generate realistic mess entry logs
@st.cache_data
def simulate_mess_data(n_days: int = 90, seed: int = 42, include_snacks: bool = True) -> pd.DataFrame:

    rng = np.random.default_rng(seed)
    records = []
    start_date = datetime(2025, 1, 1)

    # base configuration for each meal slot
    slots_all = [
        ("Breakfast", 7,  0,  9, 30, 280, 50),
        ("Lunch",     12, 0, 14, 30, 520, 80),
        ("Snacks",    17, 0, 18,  0, 150, 40),
        ("Dinner",    19, 30, 22, 0, 450, 70),
    ]

    # optionally remove snacks (used for cold-start testing)
    slots = slots_all if include_snacks else [s for s in slots_all if s[0] != "Snacks"]

    for day_idx in range(n_days):

        current_date = start_date + timedelta(days=day_idx)

        # derive day-level features
        day_of_week  = current_date.weekday()
        is_weekend   = day_of_week >= 5
        is_monday    = day_of_week == 0

        # adjust traffic based on day type
        day_mult   = 0.75 if is_weekend else (0.90 if is_monday else 1.0)

        # simulate occasional spike (events/fests)
        event_bump = 1.15 if (15 <= current_date.day <= 17) else 1.0

        for slot_name, sh, sm, eh, em, base, std in slots:

            # slot-specific behavior adjustments
            slot_mult = 1.0
            if is_weekend and slot_name == "Breakfast":
                slot_mult = 0.60
            if is_weekend and slot_name == "Snacks":
                slot_mult = 1.30
            if slot_name == "Dinner" and is_weekend:
                slot_mult = 0.85

            # compute final count with noise
            count = int(
                max(10,
                    base * day_mult * slot_mult * event_bump
                    + rng.normal(0, std)
                )
            )

            # generate timestamps within slot window
            slot_start = current_date.replace(hour=sh, minute=sm)
            slot_end   = current_date.replace(hour=eh, minute=em)

            total_mins = int((slot_end - slot_start).total_seconds() / 60)

            # more arrivals early in slot → weighted distribution
            weights = np.concatenate([
                np.ones(total_mins // 3) * 3,
                np.ones(total_mins - total_mins // 3)
            ])
            weights /= weights.sum()

            entry_offsets = rng.choice(total_mins, size=count, p=weights)

            for offset in entry_offsets:
                ts = slot_start + timedelta(minutes=int(offset))

                records.append({
                    "timestamp":   ts,
                    "date":        pd.to_datetime(current_date),
                    "meal_slot":   slot_name,
                    "hour":        ts.hour,
                    "minute":      ts.minute,
                    "day_of_week": day_of_week,
                    "is_weekend":  is_weekend,
                    "month":       current_date.month,
                    "day":         current_date.day,
                })

    return pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)


# ─────────────────────────────────────────────
# FEATURE BUILDING
# ─────────────────────────────────────────────

# convert raw logs → hourly aggregation
@st.cache_data
def build_hourly_counts(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    df2["hour_bucket"] = df2["timestamp"].dt.floor("h")
    return df2.groupby(["date", "meal_slot", "hour_bucket"]).size().reset_index(name="count")


# convert raw logs → daily aggregation
@st.cache_data
def build_daily_summary(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["date", "meal_slot"]).size().reset_index(name="students")


# ─────────────────────────────────────────────
# ML: PREDICTIVE MODEL
# ─────────────────────────────────────────────

# train model to predict daily footfall
@st.cache_resource
def train_predictive_model(df: pd.DataFrame):

    daily = build_daily_summary(df)

    # convert date into usable features
    daily["date"]        = pd.to_datetime(daily["date"])
    daily["day_of_week"] = daily["date"].dt.dayofweek
    daily["is_weekend"]  = (daily["day_of_week"] >= 5).astype(int)
    daily["month"]       = daily["date"].dt.month
    daily["day"]         = daily["date"].dt.day

    # encode meal slot as numeric
    daily["slot_encoded"] = daily["meal_slot"].map(
        {"Breakfast": 0, "Lunch": 1, "Snacks": 2, "Dinner": 3}
    )

    # drop unknown slots if any
    daily = daily.dropna(subset=["slot_encoded"])
    daily["slot_encoded"] = daily["slot_encoded"].astype(int)

    features = ["day_of_week", "is_weekend", "month", "day", "slot_encoded"]

    X = daily[features]
    y = daily["students"]

    # split for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # random forest works well for non-linear patterns
    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)

    # evaluate performance
    y_pred = model.predict(X_test)

    metrics = {
        "MAE":  round(mean_absolute_error(y_test, y_pred), 1),
        "R2":   round(r2_score(y_test, y_pred), 3),
        "MAPE": round(np.mean(np.abs((y_test - y_pred) / y_test)) * 100, 1),
    }

    return model, metrics


# ─────────────────────────────────────────────
# ML: ANOMALY DETECTION
# ─────────────────────────────────────────────

# detect unusual days in mess usage
@st.cache_resource
def detect_anomalies(df: pd.DataFrame):

    daily = build_daily_summary(df)

    # reshape so each day becomes a feature vector
    pivot = daily.pivot_table(
        index="date",
        columns="meal_slot",
        values="students",
        fill_value=0
    )

    iso = IsolationForest(contamination=0.05, random_state=42)

    labels = iso.fit_predict(pivot.values)

    pivot["anomaly"] = labels

    # return dates flagged as anomalies
    anomalies = pivot[pivot["anomaly"] == -1].index.tolist()

    return anomalies
    # ─────────────────────────────────────────────
# REAL-TIME CROWD ENGINE
# ─────────────────────────────────────────────

# determine which meal slot is currently active
def get_current_slot(now: datetime) -> dict | None:

    SLOTS = {
        "Breakfast": (time(7, 0),  time(9, 30)),
        "Lunch":     (time(12, 0), time(14, 30)),
        "Snacks":    (time(17, 0), time(18, 0)),
        "Dinner":    (time(19, 30), time(22, 0)),
    }

    current_time = now.time()

    for name, (start, end) in SLOTS.items():

        if start <= current_time <= end:

            total_mins = (
                datetime.combine(now.date(), end) -
                datetime.combine(now.date(), start)
            ).seconds // 60

            elapsed_mins = (
                datetime.combine(now.date(), current_time) -
                datetime.combine(now.date(), start)
            ).seconds // 60

            progress = elapsed_mins / total_mins

            return {
                "name": name,
                "start": start,
                "end": end,
                "progress": progress,
                "elapsed_mins": elapsed_mins,
                "total_mins": total_mins,
            }

    return None


# estimate current crowd based on prediction + time progression
def estimate_realtime_crowd(
    df: pd.DataFrame,
    model,
    now: datetime,
    slot_info: dict,
    snacks_summary: dict = None,
) -> dict:

    slot_name   = slot_info["name"]
    progress    = slot_info["progress"]

    day_of_week = now.weekday()
    is_weekend  = int(day_of_week >= 5)

    slot_enc = {"Breakfast": 0, "Lunch": 1, "Snacks": 2, "Dinner": 3}[slot_name]

    daily = build_daily_summary(df)

    snacks_cold_start_info = None

    # special handling for snacks (cold-start scenario)
    if slot_name == "Snacks":

        snacks_info = snacks_summary or snacks_data_summary(df)

        lunch_xi = pd.DataFrame([{
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "month": now.month,
            "day": now.day,
            "slot_encoded": 1,
        }])

        dinner_xi = pd.DataFrame([{
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "month": now.month,
            "day": now.day,
            "slot_encoded": 3,
        }])

        lunch_pred  = model.predict(lunch_xi)[0]
        dinner_pred = model.predict(dinner_xi)[0]

        # external heuristic (you already have this function)
        cold = cold_start_snacks_estimate(
            day_of_week=day_of_week,
            is_weekend=bool(is_weekend),
            month=now.month,
            day=now.day,
            lunch_pred=lunch_pred,
            dinner_pred=dinner_pred,
            snacks_data_days=snacks_info["days"],
            snacks_data_mean=snacks_info["mean"],
        )

        expected_total = cold["estimate"]
        snacks_cold_start_info = cold

    else:
        # normal path using history + ML prediction
        hist = daily[
            (daily["meal_slot"] == slot_name) &
            (pd.to_datetime(daily["date"]).dt.dayofweek == day_of_week)
        ]["students"]

        hist_avg = hist.mean() if not hist.empty else 300

        X_pred = pd.DataFrame([{
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "month": now.month,
            "day": now.day,
            "slot_encoded": slot_enc,
        }])

        ml_pred = model.predict(X_pred)[0]

        # weighted combination
        expected_total = int(0.4 * hist_avg + 0.6 * ml_pred)

    # convert expected total → current crowd
    if slot_name == "Snacks":
        crowd_fraction = max(0.05, 1.0 - 0.6 * progress)
    else:
        if progress < 0.30:
            crowd_fraction = progress / 0.30
        elif progress < 0.55:
            crowd_fraction = 1.0
        else:
            crowd_fraction = max(0.05, 1.0 - (progress - 0.55) / 0.45)

    stay_map = {"Breakfast": 20, "Lunch": 25, "Snacks": 12, "Dinner": 28}

    stay_dur = stay_map[slot_name]

    arrival_rate = expected_total / slot_info["total_mins"]

    current_count = int(arrival_rate * stay_dur * crowd_fraction)

    current_count = max(5, min(current_count, int(expected_total * 0.8)))

    capacity_map = {"Breakfast": 300, "Lunch": 600, "Snacks": 200, "Dinner": 500}

    capacity = capacity_map[slot_name]

    probability = min(1.0, current_count / capacity)

    # classify crowd level
    if probability < 0.40:
        status = "Low"
        advice = "Great time to go — almost no wait."
        color = "status-green"
    elif probability < 0.70:
        status = "Moderate"
        advice = "Some crowd expected. Short wait possible."
        color = "status-yellow"
    else:
        status = "High"
        advice = "Crowded right now. Better to wait."
        color = "status-red"

    return {
        "current_count": current_count,
        "capacity": capacity,
        "probability": probability,
        "status": status,
        "advice": advice,
        "color": color,
        "expected_total": int(expected_total),
        "snacks_cold_start": snacks_cold_start_info,
    }


# ─────────────────────────────────────────────
# SNACKS vs DINNER ANALYSIS
# ─────────────────────────────────────────────

# check if snacks reduce dinner attendance
def snacks_dinner_correlation(df: pd.DataFrame) -> dict:

    daily = build_daily_summary(df)

    snacks = daily[daily["meal_slot"] == "Snacks"][["date", "students"]]
    snacks = snacks.rename(columns={"students": "snacks_count"})

    dinner = daily[daily["meal_slot"] == "Dinner"][["date", "students"]]
    dinner = dinner.rename(columns={"students": "dinner_count"})

    merged = pd.merge(snacks, dinner, on="date")

    if merged.empty:
        return {"corr": 0, "df": merged}

    corr = merged[["snacks_count", "dinner_count"]].corr().iloc[0, 1]

    return {"corr": round(corr, 3), "df": merged}


# ─────────────────────────────────────────────
# CLEAN DATASET GENERATION
# ─────────────────────────────────────────────

# create enriched dataset with extra signals (wifi, electricity, etc.)
@st.cache_data
def generate_clean_dataset(df: pd.DataFrame) -> pd.DataFrame:

    df_clean = df.copy()

    df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'])

    # count entries per timestamp
    df_clean['mess_count'] = 1

    df_clean = df_clean.groupby('timestamp').agg({
        'mess_count': 'sum',
        'meal_slot': 'first',
        'hour': 'first',
        'day_of_week': 'first',
        'is_weekend': 'first',
        'month': 'first',
        'day': 'first'
    }).reset_index()

    # simulate related campus signals
    np.random.seed(42)

    df_clean['wifi_users'] = df_clean['mess_count'] + np.random.randint(20, 100, len(df_clean))

    df_clean['electricity_units'] = (
        df_clean['mess_count'] * 0.08 +
        np.random.uniform(10, 30, len(df_clean))
    )

    # utilization
    capacity_map = {"Breakfast": 300, "Lunch": 600, "Snacks": 200, "Dinner": 500}

    df_clean['capacity'] = df_clean['meal_slot'].map(capacity_map)

    df_clean['utilization_rate'] = df_clean['mess_count'] / df_clean['capacity']

    return df_clean


# ─────────────────────────────────────────────
# DATA LOADING PIPELINE
# ─────────────────────────────────────────────

# simulate data → build features → train model → detect anomalies
with st.spinner("Simulating campus mess data..."):

    df = simulate_mess_data(n_days=90)

    clean_df = generate_clean_dataset(df)

    hourly_df = build_hourly_counts(df)

    daily_df = build_daily_summary(df)

    model, model_metrics = train_predictive_model(df)

    anomaly_dates = detect_anomalies(df)
    # ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

# sidebar navigation and controls
st.sidebar.title("Smart Campus Mess")

section = st.sidebar.radio(
    "Navigate",
    ["Overview", "Analytics", "Predictions", "Real-Time", "Snacks Impact", "Anomalies"],
    index=0,
)

st.sidebar.markdown("---")

# simulate current time for real-time predictions
sim_date = st.sidebar.date_input("Date", value=datetime(2025, 3, 15).date())
sim_time = st.sidebar.time_input("Time", value=time(13, 10))
sim_now  = datetime.combine(sim_date, sim_time)


# ─────────────────────────────────────────────
# OVERVIEW
# ─────────────────────────────────────────────

if section == "Overview":

    st.title("Smart Campus Mess Intelligence System")

    # key metrics
    total_students = len(df)
    avg_daily = int(df.groupby("date").size().mean())
    peak_slot = df.groupby("meal_slot").size().idxmax()
    anomaly_count = len(anomaly_dates)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Records", total_students)
    c2.metric("Avg Daily Students", avg_daily)
    c3.metric("Busiest Slot", peak_slot)
    c4.metric("Anomalies", anomaly_count)

    st.divider()

    # daily trend
    st.subheader("Daily Footfall Trend")

    daily_df["date"] = pd.to_datetime(daily_df["date"])

    fig = px.line(
        daily_df,
        x="date",
        y="students",
        color="meal_slot",
    )

    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# ANALYTICS
# ─────────────────────────────────────────────

elif section == "Analytics":

    st.title("Mess Analytics")

    # heatmap of entries
    st.subheader("Entry Heatmap")

    slot = st.selectbox("Select slot", df["meal_slot"].unique())

    temp = df[df["meal_slot"] == slot].copy()

    temp["time_bin"] = (temp["minute"] // 15) * 15
    temp["time_label"] = temp["hour"].astype(str) + ":" + temp["time_bin"].astype(str)

    heat = temp.groupby(["day_of_week", "time_label"]).size().reset_index(name="count")

    pivot = heat.pivot(index="day_of_week", columns="time_label", values="count").fillna(0)

    fig = px.imshow(pivot, aspect="auto")

    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# PREDICTIONS
# ─────────────────────────────────────────────

elif section == "Predictions":

    st.title("Footfall Prediction")

    st.write("Model Performance:", model_metrics)

    pred_date = st.date_input("Select date")
    pred_slot = st.selectbox("Meal slot", ["Breakfast", "Lunch", "Snacks", "Dinner"])

    slot_enc = {"Breakfast": 0, "Lunch": 1, "Snacks": 2, "Dinner": 3}[pred_slot]

    X_input = pd.DataFrame([{
        "day_of_week": pd.Timestamp(pred_date).dayofweek,
        "is_weekend": int(pd.Timestamp(pred_date).dayofweek >= 5),
        "month": pd.Timestamp(pred_date).month,
        "day": pd.Timestamp(pred_date).day,
        "slot_encoded": slot_enc,
    }])

    pred = int(model.predict(X_input)[0])

    st.metric("Predicted Students", pred)


# ─────────────────────────────────────────────
# REAL-TIME
# ─────────────────────────────────────────────

elif section == "Real-Time":

    st.title("Real-Time Crowd")

    slot_info = get_current_slot(sim_now)

    if slot_info is None:
        st.info("Mess is closed")
    else:
        crowd = estimate_realtime_crowd(df, model, sim_now, slot_info)

        st.metric("Current Students", crowd["current_count"])
        st.metric("Capacity", crowd["capacity"])
        st.metric("Status", crowd["status"])

        st.write(crowd["advice"])


# ─────────────────────────────────────────────
# SNACKS IMPACT
# ─────────────────────────────────────────────

elif section == "Snacks Impact":

    st.title("Snacks vs Dinner Impact")

    result = snacks_dinner_correlation(df)

    st.metric("Correlation", result["corr"])

    if not result["df"].empty:

        fig = px.scatter(
            result["df"],
            x="snacks_count",
            y="dinner_count",
        )

        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# ANOMALIES
# ─────────────────────────────────────────────

elif section == "Anomalies":

    st.title("Anomaly Detection")

    st.metric("Anomaly Days", len(anomaly_dates))

    st.write(anomaly_dates)


# ─────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────

# allow dataset download
csv = clean_df.to_csv(index=False)

st.download_button(
    label="Download Dataset",
    data=csv,
    file_name="mess_data.csv",
    mime="text/csv"
)
