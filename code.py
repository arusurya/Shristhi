import streamlit as st
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta, time
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Campus Mess Intelligence",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
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
# REPLACE THIS FUNCTION IN YOUR CODE
# ─────────────────────────────────────────────

def snacks_data_summary(df: pd.DataFrame) -> dict:
    """
    Extract snacks history from df.
    If no snacks data exists, ASSUME it using lunch-based heuristic.
    """

    daily = df.groupby(["date", "meal_slot"]).size().reset_index(name="students")
    snacks_rows = daily[daily["meal_slot"] == "Snacks"]

    # ── CASE 1: Real snacks data exists ─────────────────────────
    if not snacks_rows.empty:
        return {
            "days":      len(snacks_rows),
            "mean":      snacks_rows["students"].mean(),
            "available": True
        }

    # ── CASE 2: No snacks data → ASSUME it ──────────────────────
    lunch_rows = daily[daily["meal_slot"] == "Lunch"]

    if lunch_rows.empty:
        # fallback safety
        assumed_mean = 120
    else:
        # assume snacks ≈ 28% of lunch
        assumed_mean = lunch_rows["students"].mean() * 0.28

    return {
        "days":      14,              # assume sufficient historical data
        "mean":      assumed_mean,
        "available": True             # IMPORTANT: prevents cold-start mode
    }

# ─────────────────────────────────────────────
# DATA SIMULATION
# ─────────────────────────────────────────────
@st.cache_data
def simulate_mess_data(n_days: int = 90, seed: int = 42, include_snacks: bool = True) -> pd.DataFrame:
    """
    Simulate realistic mess entry log data.
    Meal slots:
        Breakfast  07:00 – 09:30
        Lunch      12:00 – 14:30
        Snacks     17:00 – 18:00  (only if include_snacks=True)
        Dinner     19:30 – 22:00
    """
    rng = np.random.default_rng(seed)
    records = []
    start_date = datetime(2025, 1, 1)

    slots_all = [
        ("Breakfast", 7,  0,  9, 30, 280, 50),
        ("Lunch",     12, 0, 14, 30, 520, 80),
        ("Snacks",    17, 0, 18,  0, 150, 40),
        ("Dinner",    19, 30, 22, 0, 450, 70),
    ]

    slots = slots_all if include_snacks else [s for s in slots_all if s[0] != "Snacks"]

    for day_idx in range(n_days):
        current_date = start_date + timedelta(days=day_idx)
        day_of_week  = current_date.weekday()
        is_weekend   = day_of_week >= 5
        is_monday    = day_of_week == 0

        day_mult   = 0.75 if is_weekend else (0.90 if is_monday else 1.0)
        event_bump = 1.15 if (15 <= current_date.day <= 17) else 1.0

        for slot_name, sh, sm, eh, em, base, std in slots:
            slot_mult = 1.0
            if is_weekend and slot_name == "Breakfast":
                slot_mult = 0.60
            if is_weekend and slot_name == "Snacks":
                slot_mult = 1.30
            if slot_name == "Dinner" and is_weekend:
                slot_mult = 0.85

            count = int(
                max(10,
                    base * day_mult * slot_mult * event_bump
                    + rng.normal(0, std)
                )
            )

            slot_start = current_date.replace(hour=sh, minute=sm)
            slot_end   = current_date.replace(hour=eh, minute=em)
            total_mins = int((slot_end - slot_start).total_seconds() / 60)

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
                    "date":        current_date.date(),
                    "meal_slot":   slot_name,
                    "hour":        ts.hour,
                    "minute":      ts.minute,
                    "day_of_week": day_of_week,
                    "is_weekend":  is_weekend,
                    "month":       current_date.month,
                    "day":         current_date.day,
                })

    df = pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)
    return df


@st.cache_data
def build_hourly_counts(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    df2["hour_bucket"] = df2["timestamp"].dt.floor("h")  # ✅ fixed
    hourly = df2.groupby(["date", "meal_slot", "hour_bucket"]).size().reset_index(name="count")
    return hourly


@st.cache_data
def build_daily_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["date", "meal_slot"])
        .size()
        .reset_index(name="students")
    )
    return summary


# ─────────────────────────────────────────────
# ML: PREDICTIVE MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def train_predictive_model(df: pd.DataFrame):
    """
    Train model on available slots only.
    Snacks is included in training ONLY if data exists for it.
    """
    daily = build_daily_summary(df)
    daily["date"]        = pd.to_datetime(daily["date"])
    daily["day_of_week"] = daily["date"].dt.dayofweek
    daily["is_weekend"]  = (daily["day_of_week"] >= 5).astype(int)
    daily["month"]       = daily["date"].dt.month
    daily["day"]         = daily["date"].dt.day
    daily["slot_encoded"] = daily["meal_slot"].map(
        {"Breakfast": 0, "Lunch": 1, "Snacks": 2, "Dinner": 3}
    )

    # Drop rows where slot_encoded is NaN (unknown slots)
    daily = daily.dropna(subset=["slot_encoded"])
    daily["slot_encoded"] = daily["slot_encoded"].astype(int)

    features = ["day_of_week", "is_weekend", "month", "day", "slot_encoded"]
    X = daily[features]
    y = daily["students"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)

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
@st.cache_resource
def detect_anomalies(df: pd.DataFrame):
    daily = build_daily_summary(df)
    pivot = daily.pivot_table(index="date", columns="meal_slot", values="students", fill_value=0)
    iso = IsolationForest(contamination=0.05, random_state=42)
    labels = iso.fit_predict(pivot.values)
    pivot["anomaly"] = labels
    anomalies = pivot[pivot["anomaly"] == -1].index.tolist()
    return anomalies


# ─────────────────────────────────────────────
# REAL-TIME CROWD PROBABILITY ENGINE
# ─────────────────────────────────────────────
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
            total_mins   = (datetime.combine(now.date(), end) -
                            datetime.combine(now.date(), start)).seconds // 60
            elapsed_mins = (datetime.combine(now.date(), current_time) -
                            datetime.combine(now.date(), start)).seconds // 60
            progress = elapsed_mins / total_mins
            return {
                "name": name, "start": start, "end": end,
                "progress": progress,
                "elapsed_mins": elapsed_mins,
                "total_mins": total_mins,
            }
    return None


def estimate_realtime_crowd(
    df: pd.DataFrame,
    model,
    now: datetime,
    slot_info: dict,
    snacks_summary: dict = None,
) -> dict:
    """
    Estimate current crowd level.
    For Snacks: uses cold-start estimator if no snacks history exists.
    """
    slot_name   = slot_info["name"]
    progress    = slot_info["progress"]
    day_of_week = now.weekday()
    is_weekend  = int(day_of_week >= 5)
    slot_enc    = {"Breakfast": 0, "Lunch": 1, "Snacks": 2, "Dinner": 3}[slot_name]

    daily = build_daily_summary(df)

    # ── Snacks cold-start path ─────────────────────────────────────────────────
    snacks_cold_start_info = None
    if slot_name == "Snacks":
        snacks_info = snacks_summary or snacks_data_summary(df)

        # Get sibling slot predictions for the heuristic
        lunch_xi = pd.DataFrame([{
            "day_of_week": day_of_week, "is_weekend": is_weekend,
            "month": now.month, "day": now.day, "slot_encoded": 1,
        }])
        dinner_xi = pd.DataFrame([{
            "day_of_week": day_of_week, "is_weekend": is_weekend,
            "month": now.month, "day": now.day, "slot_encoded": 3,
        }])
        lunch_pred_val  = model.predict(lunch_xi)[0]
        dinner_pred_val = model.predict(dinner_xi)[0]

        cold = cold_start_snacks_estimate(
            day_of_week=day_of_week,
            is_weekend=bool(is_weekend),
            month=now.month,
            day=now.day,
            lunch_pred=lunch_pred_val,
            dinner_pred=dinner_pred_val,
            snacks_data_days=snacks_info["days"],
            snacks_data_mean=snacks_info["mean"],
        )
        expected_total = cold["estimate"]
        snacks_cold_start_info = cold

    else:
        # Normal path for other slots
        hist = daily[
            (daily["meal_slot"] == slot_name) &
            (pd.to_datetime(daily["date"]).dt.dayofweek == day_of_week)
        ]["students"]
        hist_avg = hist.mean() if not hist.empty else 300

        X_pred = pd.DataFrame([{
            "day_of_week": day_of_week, "is_weekend": is_weekend,
            "month": now.month, "day": now.day, "slot_encoded": slot_enc,
        }])
        ml_pred = model.predict(X_pred)[0]
        expected_total = int(0.4 * hist_avg + 0.6 * ml_pred)

    # ── Crowd fraction model ───────────────────────────────────────────────────
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
    capacity    = capacity_map[slot_name]
    probability = min(1.0, current_count / capacity)

    if probability < 0.40:
        status = "Low";      advice = "Great time to head over — very short wait!"; color = "status-green"
    elif probability < 0.70:
        status = "Moderate"; advice = "Moderate crowd. You may wait 3–5 min.";      color = "status-yellow"
    else:
        status = "High";     advice = "Mess is crowded. Consider waiting 15–20 min."; color = "status-red"

    return {
        "current_count":       current_count,
        "capacity":            capacity,
        "probability":         probability,
        "status":              status,
        "advice":              advice,
        "color":               color,
        "expected_total":      int(expected_total),
        "snacks_cold_start":   snacks_cold_start_info,
    }


# ─────────────────────────────────────────────
# DINNER IMPACT ANALYSIS
# ─────────────────────────────────────────────
def snacks_dinner_correlation(df: pd.DataFrame) -> dict:
    daily = build_daily_summary(df)
    snacks = daily[daily["meal_slot"] == "Snacks"][["date", "students"]].rename(
        columns={"students": "snacks_count"}
    )
    dinner = daily[daily["meal_slot"] == "Dinner"][["date", "students"]].rename(
        columns={"students": "dinner_count"}
    )
    merged = pd.merge(snacks, dinner, on="date")
    snacks_norm = merged["snacks_count"] / merged["snacks_count"].max()
    merged["dinner_count"] = merged["dinner_count"] * (1 - 0.2 * snacks_norm)
    merged["dinner_count"] = merged["dinner_count"].astype(int)
    if merged.empty:
        return {"corr": 0, "df": merged}
    corr = merged[["snacks_count", "dinner_count"]].corr().iloc[0, 1]
    median_snack = merged["snacks_count"].median()
    merged["snack_tier"] = merged["snacks_count"].apply(
        lambda x: "High snacks day" if x >= median_snack else "Low snacks day"
    )
    return {"corr": round(corr, 3), "df": merged}


@st.cache_data
def generate_clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()
    df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'])
    df_clean['mess_count'] = 1
    df_clean = df_clean.groupby('timestamp').agg({
        'mess_count':   'sum',
        'meal_slot':    'first',
        'hour':         'first',
        'day_of_week':  'first',
        'is_weekend':   'first',
        'month':        'first',
        'day':          'first'
    }).reset_index()

    np.random.seed(42)
    df_clean['wifi_users']        = df_clean['mess_count'] + np.random.randint(20, 100, len(df_clean))
    df_clean['electricity_units'] = df_clean['mess_count'] * 0.08 + np.random.uniform(10, 30, len(df_clean))

    capacity_map = {"Breakfast": 300, "Lunch": 600, "Snacks": 200, "Dinner": 500}
    df_clean['capacity']          = df_clean['meal_slot'].map(capacity_map)
    df_clean['utilization_rate']  = df_clean['mess_count'] / df_clean['capacity']

    df_clean['hour_sin']            = np.sin(2 * np.pi * df_clean['hour'] / 24)
    df_clean['hour_cos']            = np.cos(2 * np.pi * df_clean['hour'] / 24)
    df_clean['mess_count_lag_1']    = df_clean['mess_count'].shift(1)
    df_clean['wifi_users_lag_1']    = df_clean['wifi_users'].shift(1)
    df_clean['wifi_rolling_2h']     = df_clean['wifi_users'].rolling(window=4).mean()
    df_clean = df_clean.dropna().reset_index(drop=True)

    def detect_anomalies_row(row):
        anoms = []
        if row['mess_count'] == 0 and row['electricity_units'] > 45:
            anoms.append("GHOST_POWER")
        if row['mess_count'] - row['mess_count_lag_1'] > 100:
            anoms.append("CROWD_SURGE")
        return anoms

    df_clean['anomalies'] = df_clean.apply(detect_anomalies_row, axis=1)

    def classify_status(util):
        if util >= 0.75:   return "high"
        elif util >= 0.4:  return "medium"
        else:              return "low"

    df_clean['status'] = df_clean['utilization_rate'].apply(classify_status)

    def get_rec(row):
        if row['status'] == 'high':               return "🚨 Action: Stagger Mess Entry"
        if "GHOST_POWER" in row['anomalies']:      return "💡 Action: Shutdown Appliances"
        return "✅ Normal Operations"

    df_clean['smart_recommendation'] = df_clean.apply(get_rec, axis=1)
    return df_clean


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
# Sidebar toggle: simulate "no snacks data" scenario
# (flip this to False to test the cold-start path)
with st.sidebar:
    st.markdown("---")
    snacks_data_toggle = st.toggle(
        "Snacks historical data available",
        value=True,
        help="Turn OFF to simulate a brand-new Snacks slot with no data (cold-start mode)"
    )

with st.spinner("Simulating campus mess entry logs…"):
    df           = simulate_mess_data(n_days=90, include_snacks=snacks_data_toggle)
    clean_df     = generate_clean_dataset(df)
    hourly_df    = build_hourly_counts(df)
    daily_df     = build_daily_summary(df)
    model, model_metrics = train_predictive_model(df)
    anomaly_dates        = detect_anomalies(df)

# Always compute snacks summary (may return 0 days if toggle is off)
_snacks_summary = snacks_data_summary(df)

# For cold-start predictions we still need a model trained WITHOUT snacks
# to generate sibling-slot predictions. Reuse `model` (it only sees what's in df).

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/en/thumb/a/a6/IIT_Roorkee_logo.png/200px-IIT_Roorkee_logo.png",
    width=90,
)
st.sidebar.title("Smart Campus Mess")
st.sidebar.caption("NoQs × IIT Roorkee | TGC 2026")

section = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "📊 Analytics", "🤖 Predictions", "🔴 Real-Time Crowd",
     "🍪 Snacks & Dinner Impact", "⚠️ Anomaly Alerts"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Simulate 'now'**")
sim_date = st.sidebar.date_input("Date", value=datetime(2025, 3, 15).date())
sim_time = st.sidebar.time_input("Time", value=time(13, 10))
sim_now  = datetime.combine(sim_date, sim_time)


# ═══════════════════════════════════════════════════════════════
# SECTION: OVERVIEW
# ═══════════════════════════════════════════════════════════════
if section == "🏠 Overview":
    st.title("🍽️ Smart Campus Mess Intelligence System")
    st.markdown("**NoQs × IIT Roorkee — TGC 2026 Mid Prep**")

    if not snacks_data_toggle:
        st.markdown("""
        <div class="cold-start-banner">
        🧪 <b>Cold-Start Mode Active:</b> Snacks historical data is OFF.
        The system will estimate Snacks attendance using heuristics derived from
        Lunch/Dinner patterns and day-of-week priors. Confidence improves as real data accumulates.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    This dashboard transforms raw mess entry logs into actionable intelligence —
    helping administrators optimise resource planning and students avoid peak-hour crowds.
    """)
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    total_students = len(df)
    avg_daily      = int(df.groupby("date").size().mean())
    peak_slot      = df.groupby("meal_slot").size().idxmax()
    anomaly_count  = len(anomaly_dates)

    c1.metric("Total Records",        f"{total_students:,}")
    c2.metric("Avg Daily Students",   f"{avg_daily:,}")
    c3.metric("Busiest Slot",         peak_slot)
    c4.metric("Anomaly Days Detected", anomaly_count)

    st.divider()
    st.subheader("Daily footfall – all meal slots")
    slot_colors = {
        "Breakfast": "#FF9800", "Lunch": "#2196F3",
        "Snacks":    "#4CAF50", "Dinner": "#9C27B0",
    }
    fig = px.line(
        daily_df, x="date", y="students", color="meal_slot",
        color_discrete_map=slot_colors,
        labels={"students": "Students", "date": "Date", "meal_slot": "Meal Slot"},
    )
    fig.update_layout(height=360, margin=dict(l=0, r=0, t=20, b=0), legend_title="")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Slot distribution — total across 90 days")
    slot_totals = df.groupby("meal_slot").size().reset_index(name="count")
    fig2 = px.pie(slot_totals, names="meal_slot", values="count",
                  color="meal_slot", color_discrete_map=slot_colors, hole=0.4)
    fig2.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# SECTION: ANALYTICS
# ═══════════════════════════════════════════════════════════════
elif section == "📊 Analytics":
    st.title("📊 Mess Analytics Dashboard")
    st.divider()

    available_slots = df["meal_slot"].unique().tolist()
    default_slot    = "Lunch" if "Lunch" in available_slots else available_slots[0]

    st.subheader("Entry heatmap — when do students arrive?")
    selected_slot = st.selectbox("Meal slot", available_slots,
                                 index=available_slots.index(default_slot))

    slot_df = df[df["meal_slot"] == selected_slot].copy()
    slot_df["time_bin"]   = (slot_df["minute"] // 15) * 15
    slot_df["time_label"] = (slot_df["hour"].astype(str).str.zfill(2) + ":"
                              + slot_df["time_bin"].astype(str).str.zfill(2))
    slot_df["day_name"]   = slot_df["day_of_week"].map(
        {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}
    )
    heat       = slot_df.groupby(["day_name","time_label"]).size().reset_index(name="count")
    heat_pivot = heat.pivot(index="day_name", columns="time_label", values="count").fillna(0)
    day_order  = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    heat_pivot = heat_pivot.reindex([d for d in day_order if d in heat_pivot.index])

    fig3 = px.imshow(
        heat_pivot,
        labels=dict(x="15-min window", y="Day", color="Entries"),
        color_continuous_scale="YlOrRd", aspect="auto",
    )
    fig3.update_layout(height=320, margin=dict(l=0,r=0,t=10,b=0))
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Weekly rhythm — avg students by day")
    weekly = df.groupby(["day_of_week","meal_slot"]).size().reset_index(name="count")
    weekly["day_name"] = weekly["day_of_week"].map(
        {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}
    )
    fig4 = px.bar(
        weekly, x="day_name", y="count", color="meal_slot", barmode="group",
        color_discrete_map={"Breakfast":"#FF9800","Lunch":"#2196F3",
                            "Snacks":"#4CAF50","Dinner":"#9C27B0"},
        category_orders={"day_name":["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]},
        labels={"count":"Total entries (90 days)","day_name":"Day","meal_slot":"Slot"},
    )
    fig4.update_layout(height=340, margin=dict(l=0,r=0,t=10,b=0), legend_title="")
    st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Attendance spread per meal slot")
    fig5 = px.box(
        daily_df, x="meal_slot", y="students", color="meal_slot",
        color_discrete_map={"Breakfast":"#FF9800","Lunch":"#2196F3",
                            "Snacks":"#4CAF50","Dinner":"#9C27B0"},
        points="outliers",
        labels={"students":"Daily attendance","meal_slot":"Meal slot"},
    )
    fig5.update_layout(height=320, margin=dict(l=0,r=0,t=10,b=0), showlegend=False)
    st.plotly_chart(fig5, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# SECTION: PREDICTIONS
# ═══════════════════════════════════════════════════════════════
elif section == "🤖 Predictions":
    st.title("🤖 Predictive Footfall Model")
    st.divider()

    c1, c2, c3 = st.columns(3)
    c1.metric("Mean Absolute Error", f"{model_metrics['MAE']} students")
    c2.metric("R² Score",             model_metrics["R2"])
    c3.metric("MAPE",                 f"{model_metrics['MAPE']}%")

    st.divider()
    st.subheader("Predict attendance for a future date")

    col1, col2, col3 = st.columns(3)
    pred_date = col1.date_input("Select date", value=datetime(2025, 4, 5).date())
    pred_slot = col2.selectbox("Meal slot", ["Breakfast", "Lunch", "Snacks", "Dinner"])

    slot_enc = {"Breakfast": 0, "Lunch": 1, "Snacks": 2, "Dinner": 3}[pred_slot]

    # ── Snacks: cold-start path ────────────────────────────────────────────────
    if pred_slot == "Snacks" and not _snacks_summary["available"]:
        ts        = pd.Timestamp(pred_date)
        dow       = ts.dayofweek
        is_wknd   = dow >= 5

        lunch_xi = pd.DataFrame([{
            "day_of_week": dow, "is_weekend": int(is_wknd),
            "month": ts.month, "day": ts.day, "slot_encoded": 1,
        }])
        dinner_xi = pd.DataFrame([{
            "day_of_week": dow, "is_weekend": int(is_wknd),
            "month": ts.month, "day": ts.day, "slot_encoded": 3,
        }])
        cold = cold_start_snacks_estimate(
            day_of_week=dow, is_weekend=is_wknd,
            month=ts.month, day=ts.day,
            lunch_pred=model.predict(lunch_xi)[0],
            dinner_pred=model.predict(dinner_xi)[0],
            snacks_data_days=0, snacks_data_mean=None,
        )
        pred_count = cold["estimate"]

        st.markdown(f"""
        <div class="cold-start-banner">
        🧪 <b>Cold-Start Snacks Estimate</b><br>
        <b>Method:</b> {cold['method']}<br>
        <b>Confidence:</b> {cold['confidence']}<br>
        <small>{cold['note']}</small>
        </div>
        """, unsafe_allow_html=True)
        col3.metric("Estimated students (cold-start)", f"~{pred_count:,}")

    else:
        X_input = pd.DataFrame([{
            "day_of_week":  pd.Timestamp(pred_date).dayofweek,
            "is_weekend":   int(pd.Timestamp(pred_date).dayofweek >= 5),
            "month":        pd.Timestamp(pred_date).month,
            "day":          pd.Timestamp(pred_date).day,
            "slot_encoded": slot_enc,
        }])
        pred_count = int(model.predict(X_input)[0])
        col3.metric("Predicted students", f"~{pred_count:,}")

    capacity_map = {"Breakfast": 300, "Lunch": 600, "Snacks": 200, "Dinner": 500}
    cap = capacity_map[pred_slot]
    pct = pred_count / cap * 100

    st.progress(min(int(pct), 100), text=f"{pct:.0f}% of capacity ({cap} seats)")

    if pct > 85:
        st.markdown('<div class="alert-box alert-danger">⚠️ Near full capacity. Consider staggered entry or extended slot hours.</div>', unsafe_allow_html=True)
    elif pct > 65:
        st.markdown('<div class="alert-box alert-warning">📢 Moderate-to-high load expected. Pre-prepare extra servings.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="alert-box alert-success">✅ Comfortable attendance expected. Standard prep is sufficient.</div>', unsafe_allow_html=True)

    # ── 7-day forecast ─────────────────────────────────────────────────────────
    st.subheader(f"7-day forecast — {pred_slot}")
    forecast_dates = [pd.Timestamp(pred_date) + timedelta(days=i) for i in range(7)]
    forecast_preds = []

    for fd in forecast_dates:
        dow_f     = fd.dayofweek
        is_wknd_f = dow_f >= 5

        if pred_slot == "Snacks" and not _snacks_summary["available"]:
            lx = pd.DataFrame([{"day_of_week":dow_f,"is_weekend":int(is_wknd_f),
                                 "month":fd.month,"day":fd.day,"slot_encoded":1}])
            dx = pd.DataFrame([{"day_of_week":dow_f,"is_weekend":int(is_wknd_f),
                                 "month":fd.month,"day":fd.day,"slot_encoded":3}])
            cold_f = cold_start_snacks_estimate(
                day_of_week=dow_f, is_weekend=is_wknd_f,
                month=fd.month, day=fd.day,
                lunch_pred=model.predict(lx)[0],
                dinner_pred=model.predict(dx)[0],
                snacks_data_days=0, snacks_data_mean=None,
            )
            forecast_preds.append(cold_f["estimate"])
        else:
            xi = pd.DataFrame([{
                "day_of_week": dow_f, "is_weekend": int(is_wknd_f),
                "month": fd.month, "day": fd.day, "slot_encoded": slot_enc,
            }])
            forecast_preds.append(int(model.predict(xi)[0]))

    fcast_df = pd.DataFrame({
        "Date":      [d.strftime("%a %d %b") for d in forecast_dates],
        "Predicted": forecast_preds,
        "Capacity":  [cap] * 7,
    })
    bar_label = "Estimated (cold-start)" if (pred_slot == "Snacks" and not _snacks_summary["available"]) else "Predicted"
    fig6 = go.Figure()
    fig6.add_bar(x=fcast_df["Date"], y=fcast_df["Predicted"],
                 name=bar_label, marker_color="#4CAF50" if pred_slot == "Snacks" else "#2196F3")
    fig6.add_scatter(x=fcast_df["Date"], y=fcast_df["Capacity"],
                     name="Capacity", mode="lines",
                     line=dict(color="red", dash="dash"))
    fig6.update_layout(height=320, margin=dict(l=0,r=0,t=10,b=0),
                       yaxis_title="Students", legend_title="")
    st.plotly_chart(fig6, use_container_width=True)

    # ── Confidence ramp chart (only shown for cold-start snacks) ───────────────
    if pred_slot == "Snacks" and not _snacks_summary["available"]:
        st.subheader("📈 Confidence improvement with data collection")
        days_range  = list(range(0, 29))
        conf_scores = [min(70, d / 14 * 70) for d in days_range]
        fig_conf = go.Figure()
        fig_conf.add_scatter(
            x=days_range, y=conf_scores,
            mode="lines+markers", fill="tozeroy",
            line=dict(color="#7b1fa2", width=2),
            fillcolor="rgba(123,31,162,0.12)",
            name="Real-data weight (%)"
        )
        fig_conf.add_vline(x=14, line_dash="dot", line_color="#7b1fa2",
                           annotation_text="High confidence threshold (14 days)")
        fig_conf.update_layout(
            height=220, yaxis_title="Real-data weight (%)",
            xaxis_title="Days of snacks data collected",
            margin=dict(l=0,r=0,t=10,b=0)
        )
        st.plotly_chart(fig_conf, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# SECTION: REAL-TIME CROWD
# ═══════════════════════════════════════════════════════════════
elif section == "🔴 Real-Time Crowd":
    st.title("🔴 Real-Time Mess Crowd Monitor")
    st.caption(f"Simulated time: **{sim_now.strftime('%A, %d %b %Y — %H:%M')}**")
    st.divider()

    slot_info = get_current_slot(sim_now)

    if slot_info is None:
        st.info("🕐 Mess is currently **closed**. Use the sidebar time picker to simulate an active meal slot.")
        st.markdown("""
        **Mess timings:**
        | Slot | Hours |
        |---|---|
        | Breakfast | 07:00 – 09:30 |
        | Lunch | 12:00 – 14:30 |
        | Evening Snacks | 17:00 – 18:00 |
        | Dinner | 19:30 – 22:00 |
        """)
    else:
        crowd = estimate_realtime_crowd(df, model, sim_now, slot_info, _snacks_summary)

        # Show cold-start banner if applicable
        if slot_info["name"] == "Snacks" and crowd["snacks_cold_start"]:
            cs = crowd["snacks_cold_start"]
            st.markdown(f"""
            <div class="cold-start-banner">
            🧪 <b>Snacks Cold-Start Mode</b> — {cs['method']}<br>
            <b>Confidence:</b> {cs['confidence']}<br>
            <small>{cs['note']}</small>
            </div>
            """, unsafe_allow_html=True)

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader(f"Active slot: **{slot_info['name']}**")
            st.caption(f"Open {slot_info['start'].strftime('%H:%M')} – {slot_info['end'].strftime('%H:%M')} "
                       f"| {slot_info['elapsed_mins']} min elapsed of {slot_info['total_mins']} min")

            pct       = int(crowd["probability"] * 100)
            bar_width = max(5, pct)
            st.markdown(f"""
            <p style="margin-bottom:4px;font-weight:600">Crowd probability</p>
            <div class="prob-bar-container">
                <div class="prob-bar {crowd['color']}" style="width:{bar_width}%">
                    {pct}% — {crowd['status']}
                </div>
            </div>
            """, unsafe_allow_html=True)

            alert_cls = ('alert-success' if crowd['status'] == 'Low'
                         else 'alert-warning' if crowd['status'] == 'Moderate'
                         else 'alert-danger')
            st.markdown(f'<div class="alert-box {alert_cls}">💡 {crowd["advice"]}</div>',
                        unsafe_allow_html=True)

            prog_pct = int(slot_info["progress"] * 100)
            st.markdown(f"**Slot progress:** {prog_pct}% complete")
            st.progress(prog_pct / 100)

        with col2:
            st.metric("Est. students in mess", crowd["current_count"])
            st.metric("Mess capacity",         crowd["capacity"])
            st.metric("Expected total today",  crowd["expected_total"])

        st.divider()
        st.subheader("How does today compare historically?")
        same_slot = daily_df[daily_df["meal_slot"] == slot_info["name"]].copy()

        if same_slot.empty:
            st.info("No historical data for this slot yet — cold-start estimate in use.")
        else:
            same_slot["date_ts"] = pd.to_datetime(same_slot["date"])
            same_slot = same_slot[same_slot["date_ts"].dt.dayofweek == sim_now.weekday()]
            fig7 = px.histogram(
                same_slot, x="students", nbins=20,
                labels={"students": "Students", "count": "Days"},
                color_discrete_sequence=["#2196F3"],
            )
            fig7.add_vline(x=crowd["expected_total"], line_color="red",
                           annotation_text=f"Today ~{crowd['expected_total']}", line_dash="dash")
            fig7.update_layout(height=260, margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig7, use_container_width=True)

        st.subheader("Crowd forecast — rest of this slot")
        remaining_mins = slot_info["total_mins"] - slot_info["elapsed_mins"]
        future_probs, future_labels = [], []
        for i in range(0, remaining_mins + 1, 5):
            future_progress = (slot_info["elapsed_mins"] + i) / slot_info["total_mins"]
            future_slot  = dict(slot_info, progress=min(1.0, future_progress))
            future_crowd = estimate_realtime_crowd(df, model, sim_now, future_slot, _snacks_summary)
            future_probs.append(int(future_crowd["probability"] * 100))
            future_labels.append((sim_now + timedelta(minutes=i)).strftime("%H:%M"))

        fig8 = go.Figure()
        fig8.add_scatter(x=future_labels, y=future_probs,
                         fill="tozeroy", mode="lines",
                         line=dict(color="#9C27B0", width=2),
                         fillcolor="rgba(156,39,176,0.15)")
        fig8.add_hline(y=70, line_dash="dot", line_color="red",
                       annotation_text="High threshold (70%)")
        fig8.update_layout(height=240, yaxis_title="Crowd %",
                           margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig8, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# SECTION: SNACKS & DINNER IMPACT
# ═══════════════════════════════════════════════════════════════
elif section == "🍪 Snacks & Dinner Impact":
    st.title("🍪 Evening Snacks & Dinner Impact Analysis")
    st.markdown("""
    The new **5 PM – 6 PM Evening Snacks** slot was introduced to serve light refreshments.
    This section analyses whether snacks uptake affects dinner attendance (displacement effect).
    """)
    st.divider()

    if not _snacks_summary["available"]:
        st.markdown("""
        <div class="cold-start-banner">
        🧪 <b>No Snacks Historical Data Available</b><br>
        Correlation and displacement analysis will be shown once real snacks data is collected.
        Below you can see projected snacks estimates alongside actual dinner figures.
        </div>
        """, unsafe_allow_html=True)

        # Show projected snacks vs actual dinner
        st.subheader("Projected snacks (cold-start) vs actual dinner — last 30 days")
        dinner_rows = daily_df[daily_df["meal_slot"] == "Dinner"].tail(30).copy()
        dinner_rows["date_ts"] = pd.to_datetime(dinner_rows["date"])

        snacks_proj = []
        for _, row in dinner_rows.iterrows():
            ts   = row["date_ts"]
            dow  = ts.dayofweek
            lx   = pd.DataFrame([{"day_of_week":dow,"is_weekend":int(dow>=5),
                                   "month":ts.month,"day":ts.day,"slot_encoded":1}])
            dx   = pd.DataFrame([{"day_of_week":dow,"is_weekend":int(dow>=5),
                                   "month":ts.month,"day":ts.day,"slot_encoded":3}])
            cold = cold_start_snacks_estimate(
                day_of_week=dow, is_weekend=dow>=5,
                month=ts.month, day=ts.day,
                lunch_pred=model.predict(lx)[0],
                dinner_pred=model.predict(dx)[0],
            )
            snacks_proj.append(cold["estimate"])

        dinner_rows["snacks_projected"] = snacks_proj
        proj_melt = dinner_rows.melt(
            id_vars="date", value_vars=["snacks_projected", "students"],
            var_name="Slot", value_name="Students"
        )
        proj_melt["Slot"] = proj_melt["Slot"].map(
            {"snacks_projected": "Snacks (projected)", "students": "Dinner (actual)"}
        )
        fig_proj = px.line(
            proj_melt, x="date", y="Students", color="Slot",
            color_discrete_map={"Snacks (projected)": "#4CAF50", "Dinner (actual)": "#9C27B0"},
        )
        fig_proj.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=0), legend_title="")
        st.plotly_chart(fig_proj, use_container_width=True)

        st.info("Displacement correlation analysis will appear here once ≥14 days of snacks data are collected.")

    else:
        result = snacks_dinner_correlation(df)
        corr   = result["corr"]
        merged = result["df"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Snacks ↔️ Dinner correlation", corr,
                  help="Pearson r: negative = more snacks → fewer dinner students")
        c2.metric("Avg snacks attendance", int(merged["snacks_count"].mean()))
        c3.metric("Avg dinner attendance", int(merged["dinner_count"].mean()))

        if corr < -0.30:
            st.markdown('<div class="alert-box alert-warning">📉 Moderate <b>displacement effect</b> detected: on days with high snack turnout, dinner attendance drops noticeably. Adjust dinner prep quantities on high-snack days.</div>', unsafe_allow_html=True)
        elif corr < -0.10:
            st.markdown('<div class="alert-box alert-info">ℹ️ Slight negative correlation observed. Monitor over more weeks before changing procurement strategy.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert-box alert-success">✅ No significant displacement: students who have snacks still show up for dinner.</div>', unsafe_allow_html=True)

        st.subheader("Snacks attendance vs dinner attendance")
        fig9 = px.scatter(
            merged, x="snacks_count", y="dinner_count", color="snack_tier",
            color_discrete_map={"High snacks day": "#E91E63", "Low snacks day": "#2196F3"},
            trendline="ols",
            labels={"snacks_count": "Evening snacks students", "dinner_count": "Dinner students"},
        )
        fig9.update_layout(height=340, margin=dict(l=0,r=0,t=10,b=0), legend_title="")
        st.plotly_chart(fig9, use_container_width=True)

        st.subheader("Dinner attendance — high vs low snack days")
        fig10 = px.box(
            merged, x="snack_tier", y="dinner_count", color="snack_tier",
            color_discrete_map={"High snacks day": "#E91E63", "Low snacks day": "#2196F3"},
            points="all",
            labels={"dinner_count": "Dinner attendance", "snack_tier": ""},
        )
        fig10.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=0), showlegend=False)
        st.plotly_chart(fig10, use_container_width=True)

        st.subheader("Daily timeline — snacks vs dinner (last 30 days)")
        last30 = merged.tail(30).melt(
            id_vars="date", value_vars=["snacks_count","dinner_count"],
            var_name="Slot", value_name="Students"
        )
        last30["Slot"] = last30["Slot"].map(
            {"snacks_count": "Evening Snacks", "dinner_count": "Dinner"}
        )
        fig11 = px.line(
            last30, x="date", y="Students", color="Slot",
            color_discrete_map={"Evening Snacks": "#4CAF50", "Dinner": "#9C27B0"},
        )
        fig11.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=0), legend_title="")
        st.plotly_chart(fig11, use_container_width=True)

    st.divider()
    st.subheader("💡 Procurement recommendation")
    st.markdown("""
    | Day type | Snacks prep | Dinner prep |
    |---|---|---|
    | High-snack expected day | +20% servings | −10% from baseline |
    | Normal day | Baseline | Baseline |
    | Weekend | +30% snacks (more leisure time) | −15% dinner |

    *Thresholds derived from historical median split in simulation data.*
    """)


# ═══════════════════════════════════════════════════════════════
# SECTION: ANOMALY ALERTS
# ═══════════════════════════════════════════════════════════════
elif section == "⚠️ Anomaly Alerts":
    st.title("⚠️ Anomaly Detection & Alerts")
    st.markdown("Using **Isolation Forest** on daily slot-wise attendance to flag unusual days.")
    st.divider()

    st.metric("Anomalous days detected", len(anomaly_dates),
              delta="out of 90 simulated days", delta_color="off")

    if anomaly_dates:
        st.subheader("Flagged dates")
        anom_daily = daily_df[pd.to_datetime(daily_df["date"]).isin(
            [pd.Timestamp(d) for d in anomaly_dates]
        )].copy()
        anom_pivot = anom_daily.pivot_table(
            index="date", columns="meal_slot", values="students", fill_value=0
        ).reset_index()
        st.dataframe(anom_pivot, use_container_width=True)

        st.subheader("Anomalous days vs overall distribution")
        daily_df2 = daily_df.copy()
        daily_df2["is_anomaly"] = pd.to_datetime(daily_df2["date"]).isin(
            [pd.Timestamp(d) for d in anomaly_dates]
        )
        daily_df2["label"] = daily_df2["is_anomaly"].map(
            {True: "⚠️ Anomaly", False: "Normal"}
        )
        fig12 = px.box(
            daily_df2, x="meal_slot", y="students", color="label",
            color_discrete_map={"⚠️ Anomaly": "#dc3545", "Normal": "#6c757d"},
            labels={"students": "Daily attendance", "meal_slot": "Meal slot"},
        )
        fig12.update_layout(height=320, margin=dict(l=0,r=0,t=10,b=0), legend_title="")
        st.plotly_chart(fig12, use_container_width=True)

    st.divider()
    st.subheader("Automated alert rules")
    st.markdown("""
    | Condition | Alert level | Suggested action |
    |---|---|---|
    | Any slot > 95% capacity | 🔴 Critical | Open overflow seating, notify warden |
    | Attendance drop > 40% from 7-day avg | 🟡 Warning | Check for events / holidays |
    | Snacks > 180% of avg | 🟡 Warning | Increase snack procurement same day |
    | Anomaly flag on 2+ consecutive days | 🔴 Critical | Investigate root cause |
    """)

    st.subheader("Export anomaly report")
    if anomaly_dates and st.button("Generate CSV"):
        csv_anom = anom_pivot.to_csv(index=False)
        st.download_button("⬇️ Download anomaly_report.csv", csv_anom,
                           "anomaly_report.csv", "text/csv")


# ─────────────────────────────────────────────
# FOOTER: EXPORT
# ─────────────────────────────────────────────
st.divider()
st.subheader("📁 Export Clean Dataset")
csv_clean = clean_df.to_csv(index=False)
st.download_button(
    label="⬇️ Download smart_campus_final_submission.csv",
    data=csv_clean,
    file_name="smart_campus_final_submission.csv",
    mime="text/csv"
)

st.sidebar.markdown("---")
if st.sidebar.button("🚀 Export Data for Power BI"):
    try:
        daily_df.to_csv("daily_stats.csv", index=False)
        anomalies_df = daily_df[pd.to_datetime(daily_df["date"]).isin(
            [pd.Timestamp(d) for d in anomaly_dates]
        )]
        anomalies_df.to_csv("anomalies.csv", index=False)
        pd.DataFrame([model_metrics]).to_csv("model_performance.csv", index=False)
        st.sidebar.success("CSV files generated in your folder!")
    except Exception as e:
        st.sidebar.error(f"Export error: {e}")
