# Reads violation_events.csv (+ snapshots) produced by test_violation_pipeline.py
import json
import time as time_module
import os
import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(page_title="WARDEN - Violation Dashboard", layout="wide")

CSV_PATH = "violation_events.csv"
SNAPSHOT_DIR = "violation_snapshots"


@st.cache_data
def load_events(csv_path):
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    return df

def live_monitor_tab():
    st.subheader("🔴 Live Monitor")
    frame_placeholder = st.empty()
    status_placeholder = st.empty()
    table_placeholder = st.empty()

    refresh = st.checkbox("Auto-refresh (every 2s)", value=True)

    if os.path.exists("live_frame.jpg"):
        frame_placeholder.image("live_frame.jpg", use_container_width=True)
    if os.path.exists("live_status.json"):
        with open("live_status.json") as f:
            status = json.load(f)
        status_placeholder.write(
            f"**Timestamp:** {status['timestamp']}s | "
            f"**Barrier:** {status['barrier_state']} | "
            f"**Active violations this frame:** {status['active_violations']}"
        )
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        table_placeholder.dataframe(df.tail(10), use_container_width=True, hide_index=True)

    if refresh:
        time_module.sleep(2)
        st.rerun()

def main():
    st.title("🚦 WARDEN — Railway Crossing Violation Dashboard")
    tab1, tab2 = st.tabs(["📊 Dashboard", "🔴 Live Monitor"])

    
    with tab1:

        df = load_events(CSV_PATH)

        if df is None or df.empty:
            st.warning(
                f"No violation events found at '{CSV_PATH}'. "
                "Run scripts/test_violation_pipeline.py first to generate it."
            )
            return

    # ---------------- Sidebar filters ----------------
        st.sidebar.header("Filters")

        classes = sorted(df["class"].dropna().unique().tolist())
        selected_classes = st.sidebar.multiselect("Class", classes, default=classes)

        barrier_states = sorted(df["barrier_state"].dropna().unique().tolist())
        selected_states = st.sidebar.multiselect(
        "Barrier state", barrier_states, default=barrier_states
        )

        min_conf = st.sidebar.slider("Minimum confidence", 0.0, 1.0, 0.0, 0.05)
        min_duration = st.sidebar.slider(
        "Minimum duration (s)", 0.0, float(df["duration"].max()), 0.0, 0.1
        )

        filtered = df[
            df["class"].isin(selected_classes)
            & df["barrier_state"].isin(selected_states)
            & (df["best_conf"] >= min_conf)
            & (df["duration"] >= min_duration)
        ].sort_values("start_time")

    # ---------------- Summary stats ----------------
        st.subheader("Summary")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total violations", len(filtered))
        col2.metric("Avg duration (s)", round(filtered["duration"].mean(), 2) if len(filtered) else 0)
        col3.metric("Avg confidence", round(filtered["best_conf"].mean(), 2) if len(filtered) else 0)
        most_common = filtered["class"].mode()[0] if len(filtered) else "-"
        col4.metric("Most common class", most_common)

        st.divider()

    # ---------------- Breakdown by class ----------------
        left, right = st.columns([1, 1])

        with left:
            st.subheader("Violations by class")
            if len(filtered):
                st.bar_chart(filtered["class"].value_counts())
            else:
                st.caption("No data for current filters.")

        with right:
            st.subheader("Timeline")
            if len(filtered):
                timeline_df = filtered[["start_time", "class"]].copy()
                timeline_df["start_time"] = timeline_df["start_time"].round(1)
                st.scatter_chart(timeline_df, x="start_time", y="class")
            else:
                st.caption("No data for current filters.")

        st.divider()

    # ---------------- Event table ----------------
        st.subheader(f"Violation log ({len(filtered)} events)")

        display_cols = [
        "violation_id", "track_id", "class", "barrier_state",
        "start_time", "end_time", "duration", "best_conf",
        ]
        st.dataframe(
            filtered[display_cols].style.format(
            {"start_time": "{:.2f}", "end_time": "{:.2f}",
             "duration": "{:.2f}", "best_conf": "{:.2f}"}
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.divider()

    # ---------------- Evidence viewer ----------------
        st.subheader("Evidence viewer")

        if len(filtered) == 0:
            st.caption("No events to display.")
            return

        options = {
            f"#{row.violation_id} - {row['class']} @ {row.start_time:.1f}s "
            f"(conf {row.best_conf:.2f})": row.violation_id
            for _, row in filtered.iterrows()
        }
        selected_label = st.selectbox("Select a violation to inspect", list(options.keys()))
        selected_id = options[selected_label]
        event_row = filtered[filtered["violation_id"] == selected_id].iloc[0]

        img_col, info_col = st.columns([1, 1])

        with img_col:
            snapshot_path = event_row.get("snapshot_path")
            if isinstance(snapshot_path, str) and os.path.exists(snapshot_path):
                st.image(Image.open(snapshot_path), caption=f"Track #{event_row.track_id}", use_container_width=True)
            else:
                st.warning("No snapshot image available for this event.")

        with info_col:
            st.markdown(f"**Violation ID:** {event_row.violation_id}")
            st.markdown(f"**Track ID:** {event_row.track_id}")
            st.markdown(f"**Class:** {event_row['class']}")
            st.markdown(f"**Barrier state:** {event_row.barrier_state}")
            st.markdown(f"**Start time:** {event_row.start_time:.2f}s")
            st.markdown(f"**End time:** {event_row.end_time:.2f}s")
            st.markdown(f"**Duration:** {event_row.duration:.2f}s")
            st.markdown(f"**Max confidence:** {event_row.best_conf:.2f}")
            st.markdown(f"**Entry / exit frame:** {event_row.entry_frame} / {event_row.exit_frame}")

    with tab2:
        live_monitor_tab()

if __name__ == "__main__":
    main()