import json
import os

import pandas as pd
import streamlit as st
from PIL import Image

from gemini_ocr import read_plate_with_gemini


st.set_page_config(
    page_title="WARDEN - Violation Dashboard",
    layout="wide"
)


CSV_PATH = "violation_events.csv"
SNAPSHOT_DIR = "violation_snapshots"
LIVE_STATUS_PATH = "live_status.json"


CLASS_COLORS = {
    "person": "#e74c3c",
    "two wheeler": "#f39c12",
    "car": "#3498db",
    "cycle": "#2ecc71",
    "rickshaw": "#9b59b6",
}


# ============================================================
# Utility functions
# ============================================================

def _file_mtime(path):
    return os.path.getmtime(path) if os.path.exists(path) else 0


@st.cache_data
def load_events(csv_path, _mtime):
    """
    `_mtime` is part of the cache key.
    Whenever violation_events.csv changes, Streamlit reloads it.
    """

    if not os.path.exists(csv_path):
        return None

    return pd.read_csv(csv_path)


@st.cache_data
def load_image(path, _mtime):
    if not os.path.exists(path):
        return None

    return Image.open(path)


def class_badge(cls):
    color = CLASS_COLORS.get(cls, "#7f8c8d")

    return (
        f'<span style="'
        f'background-color:{color};'
        f'color:white;'
        f'padding:2px 8px;'
        f'border-radius:10px;'
        f'font-size:0.85em">'
        f'{cls}'
        f'</span>'
    )


# ============================================================
# Dashboard tab
# ============================================================

def dashboard_tab():

    df = load_events(
        CSV_PATH,
        _file_mtime(CSV_PATH)
    )

    if df is None or df.empty:

        st.warning(
            f"No violation events found at "
            f"'{CSV_PATH}'. "
            f"Run the violation pipeline first "
            f"to generate it."
        )

        return

    # --------------------------------------------------------
    # Sidebar filters
    # --------------------------------------------------------

    st.sidebar.header("Filters")

    classes = sorted(
        df["class"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_classes = st.sidebar.multiselect(
        "Class",
        classes,
        default=classes
    )

    barrier_states = sorted(
        df["barrier_state"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_states = st.sidebar.multiselect(
        "Barrier state",
        barrier_states,
        default=barrier_states
    )

    min_conf = st.sidebar.slider(
        "Minimum confidence",
        0.0,
        1.0,
        0.0,
        0.05
    )

    min_duration = st.sidebar.slider(
        "Minimum duration (s)",
        0.0,
        float(df["duration"].max()),
        0.0,
        0.1
    )

    filtered = df[
        df["class"].isin(selected_classes)
        & df["barrier_state"].isin(selected_states)
        & (df["best_conf"] >= min_conf)
        & (df["duration"] >= min_duration)
    ].sort_values(
        "start_time",
        ascending=False
    )

    # ========================================================
    # Summary
    # ========================================================

    st.subheader("Summary")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total violations",
        len(filtered)
    )

    col2.metric(
        "Avg duration (s)",
        round(
            filtered["duration"].mean(),
            2
        ) if len(filtered) else 0
    )

    col3.metric(
        "Avg confidence",
        round(
            filtered["best_conf"].mean(),
            2
        ) if len(filtered) else 0
    )

    most_common = (
        filtered["class"].mode()[0]
        if len(filtered)
        else "-"
    )

    col4.metric(
        "Most common class",
        most_common
    )

    st.divider()

    # ========================================================
    # Recent violations gallery
    # ========================================================

    st.subheader("Recent violations")

    recent = filtered.head(6)

    if len(recent):

        cols = st.columns(len(recent))

        for col, (_, row) in zip(
            cols,
            recent.iterrows()
        ):

            with col:

                snap = row.get(
                    "snapshot_path"
                )

                img = (
                    load_image(
                        snap,
                        _file_mtime(snap)
                    )
                    if isinstance(snap, str)
                    else None
                )

                if img:
                    st.image(
                        img,
                        use_container_width=True
                    )
                else:
                    st.caption(
                        "no image"
                    )

                st.markdown(
                    class_badge(row["class"]),
                    unsafe_allow_html=True
                )

                st.caption(
                    f"#{row.violation_id} · "
                    f"{row.start_time:.1f}s · "
                    f"{row.best_conf:.2f}"
                )

    else:

        st.caption(
            "No violations match "
            "the current filters."
        )

    st.divider()

    # ========================================================
    # Breakdown charts
    # ========================================================

    left, right = st.columns([1, 1])

    with left:

        st.subheader(
            "Violations by class"
        )

        if len(filtered):

            st.bar_chart(
                filtered["class"].value_counts()
            )

        else:

            st.caption(
                "No data for current filters."
            )

    with right:

        st.subheader("Timeline")

        if len(filtered):

            timeline_df = filtered[
                ["start_time", "class"]
            ].copy()

            timeline_df["start_time"] = (
                timeline_df["start_time"]
                .round(1)
            )

            st.scatter_chart(
                timeline_df,
                x="start_time",
                y="class"
            )

        else:

            st.caption(
                "No data for current filters."
            )

    st.divider()

    # ========================================================
    # Event table
    # ========================================================

    st.subheader(
        f"Violation log ({len(filtered)} events)"
    )

    display_cols = [
        "violation_id",
        "track_id",
        "class",
        "barrier_state",
        "start_time",
        "end_time",
        "duration",
        "best_conf",
    ]

    st.dataframe(
        filtered[display_cols].style.format(
            {
                "start_time": "{:.2f}",
                "end_time": "{:.2f}",
                "duration": "{:.2f}",
                "best_conf": "{:.2f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # ========================================================
    # Evidence viewer
    # ========================================================

    st.subheader("Evidence viewer")

    if len(filtered) == 0:

        st.caption(
            "No violations available."
        )

    else:

        options = {
            (
                f"#{row.violation_id} - "
                f"{row['class']} @ "
                f"{row.start_time:.1f}s "
                f"(conf {row.best_conf:.2f})"
            ): row.violation_id
            for _, row in filtered.iterrows()
        }

        selected_label = st.selectbox(
            "Select a violation to inspect",
            list(options.keys())
        )

        selected_id = options[
            selected_label
        ]

        event_row = filtered[
            filtered["violation_id"]
            == selected_id
        ].iloc[0]

        img_col, info_col = st.columns(
            [1, 1]
        )

        with img_col:

            snapshot_path = event_row.get(
                "snapshot_path"
            )

            img = (
                load_image(
                    snapshot_path,
                    _file_mtime(snapshot_path)
                )
                if isinstance(
                    snapshot_path,
                    str
                )
                else None
            )

            if img:

                st.image(
                    img,
                    caption=(
                        f"Track "
                        f"#{event_row.track_id}"
                    ),
                    use_container_width=True
                )

            else:

                st.warning(
                    "No snapshot image "
                    "available for this event."
                )

        with info_col:

            st.markdown(
                f"**Violation ID:** "
                f"{event_row.violation_id}"
            )

            st.markdown(
                f"**Track ID:** "
                f"{event_row.track_id}"
            )

            st.markdown(
                f"**Class:** "
                f"{event_row['class']}"
            )

            st.markdown(
                f"**Barrier state:** "
                f"{event_row.barrier_state}"
            )

            st.markdown(
                f"**Start time:** "
                f"{event_row.start_time:.2f}s"
            )

            st.markdown(
                f"**End time:** "
                f"{event_row.end_time:.2f}s"
            )

            st.markdown(
                f"**Duration:** "
                f"{event_row.duration:.2f}s"
            )

            st.markdown(
                f"**Max confidence:** "
                f"{event_row.best_conf:.2f}"
            )

            st.markdown(
                f"**Entry / exit frame:** "
                f"{event_row.entry_frame} / "
                f"{event_row.exit_frame}"
            )

            # ------------------------------------------------
            # Existing OCR result, if available
            # ------------------------------------------------

            if (
                "plate_number" in event_row
                and pd.notna(
                    event_row.get(
                        "plate_number"
                    )
                )
            ):

                st.success(
                    f"**Plate number:** "
                    f"{event_row['plate_number']}"
                )

                if (
                    "plate_confidence"
                    in event_row
                    and pd.notna(
                        event_row.get(
                            "plate_confidence"
                        )
                    )
                ):

                    st.caption(
                        "OCR confidence: "
                        f"{event_row['plate_confidence']}"
                    )


    # ========================================================
    # Gemini OCR
    # ========================================================

    st.divider()

    st.subheader(
        "🔎 License Plate OCR"
    )

    st.caption(
        "Select car violation snapshots and "
        "send only those images to Gemini for "
        "on-demand license plate recognition."
    )

    # --------------------------------------------------------
    # Only cars are eligible for plate OCR
    # --------------------------------------------------------

    car_events = filtered[
        filtered["class"] == "car"
    ].copy()

    if len(car_events) == 0:

        st.info(
            "No car violations available "
            "for OCR with the current filters."
        )

    else:

        ocr_options = {
            (
                f"#{row.violation_id} | "
                f"Track {row.track_id} | "
                f"{row.start_time:.1f}s"
            ): row.violation_id
            for _, row in car_events.iterrows()
        }

        selected_ocr_labels = st.multiselect(
            "Select car violations for OCR",
            options=list(
                ocr_options.keys()
            ),
            help=(
                "Only the selected snapshots "
                "will be sent to Gemini."
            )
        )

        selected_ocr_ids = [
            ocr_options[label]
            for label in selected_ocr_labels
        ]

        if selected_ocr_ids:

            st.caption(
                f"{len(selected_ocr_ids)} "
                f"snapshot(s) selected."
            )

            # ------------------------------------------------
            # Run Gemini OCR
            # ------------------------------------------------

            if st.button(
                "🔍 Run OCR on selected images",
                type="primary"
            ):

                ocr_results = []

                progress = st.progress(
                    0
                )

                status_text = st.empty()

                for i, violation_id in enumerate(
                    selected_ocr_ids
                ):

                    row = car_events[
                        car_events["violation_id"]
                        == violation_id
                    ].iloc[0]

                    snapshot_path = row.get(
                        "snapshot_path"
                    )

                    status_text.write(
                        f"Processing "
                        f"violation #{violation_id}..."
                    )

                    # ----------------------------------------
                    # Check snapshot
                    # ----------------------------------------

                    if (
                        not isinstance(
                            snapshot_path,
                            str
                        )
                        or not os.path.exists(
                            snapshot_path
                        )
                    ):

                        ocr_results.append({
                            "violation_id":
                                violation_id,
                            "plate_number":
                                None,
                            "confidence":
                                "low",
                            "reasoning":
                                "Snapshot not found.",
                        })

                        progress.progress(
                            (i + 1)
                            / len(
                                selected_ocr_ids
                            )
                        )

                        continue

                    # ----------------------------------------
                    # Gemini OCR
                    # ----------------------------------------

                    try:

                        result = (
                            read_plate_with_gemini(
                                snapshot_path
                            )
                        )

                        ocr_results.append({
                            "violation_id":
                                violation_id,
                            "plate_number":
                                result.get(
                                    "plate_text"
                                ),
                            "confidence":
                                result.get(
                                    "confidence",
                                    "low"
                                ),
                            "reasoning":
                                result.get(
                                    "reasoning",
                                    ""
                                ),
                            "valid_format":
                                result.get(
                                    "valid_format",
                                    False
                                ),
                        })

                    except Exception as e:

                        ocr_results.append({
                            "violation_id":
                                violation_id,
                            "plate_number":
                                None,
                            "confidence":
                                "low",
                            "reasoning":
                                f"OCR error: {e}",
                            "valid_format":
                                False,
                        })

                    progress.progress(
                        (i + 1)
                        / len(
                            selected_ocr_ids
                        )
                    )

                status_text.success(
                    "OCR processing completed."
                )

                # Store results in session
                st.session_state[
                    "ocr_results"
                ] = ocr_results

        # ====================================================
        # Display OCR results
        # ====================================================

        if (
            "ocr_results"
            in st.session_state
        ):

            st.divider()

            st.subheader(
                "OCR Results"
            )

            ocr_results = (
                st.session_state[
                    "ocr_results"
                ]
            )

            for result in ocr_results:

                violation_id = (
                    result["violation_id"]
                )

                matching_rows = car_events[
                    car_events["violation_id"]
                    == violation_id
                ]

                if matching_rows.empty:
                    continue

                row = matching_rows.iloc[0]

                result_col1, result_col2 = (
                    st.columns([1, 1])
                )

                # --------------------------------------------
                # Image
                # --------------------------------------------

                with result_col1:

                    snapshot_path = row.get(
                        "snapshot_path"
                    )

                    if (
                        isinstance(
                            snapshot_path,
                            str
                        )
                        and os.path.exists(
                            snapshot_path
                        )
                    ):

                        st.image(
                            snapshot_path,
                            caption=(
                                f"Violation "
                                f"#{violation_id} "
                                f"| Track "
                                f"#{row.track_id}"
                            ),
                            use_container_width=True
                        )

                    else:

                        st.warning(
                            "Snapshot unavailable."
                        )

                # --------------------------------------------
                # OCR information
                # --------------------------------------------

                with result_col2:

                    plate = result.get(
                        "plate_number"
                    )

                    confidence = result.get(
                        "confidence",
                        "low"
                    )

                    reasoning = result.get(
                        "reasoning",
                        ""
                    )

                    valid_format = result.get(
                        "valid_format",
                        False
                    )

                    if plate:

                        st.success(
                            f"### {plate}"
                        )

                        st.write(
                            f"**Confidence:** "
                            f"{confidence}"
                        )

                        st.write(
                            f"**Indian format:** "
                            f"{'Valid' if valid_format else 'Invalid'}"
                        )

                        if reasoning:

                            st.caption(
                                f"Reason: "
                                f"{reasoning}"
                            )

                    else:

                        st.error(
                            "Plate could not be "
                            "reliably read."
                        )

                        st.write(
                            f"**Confidence:** "
                            f"{confidence}"
                        )

                        if reasoning:

                            st.caption(
                                f"Reason: "
                                f"{reasoning}"
                            )

                st.divider()


# ============================================================
# Live monitor
# ============================================================

@st.fragment(run_every=2)
def live_monitor_fragment():

    status_col, table_col = (
        st.columns([1, 2])
    )

    with status_col:

        if os.path.exists(
            LIVE_STATUS_PATH
        ):

            with open(
                LIVE_STATUS_PATH
            ) as f:

                status = json.load(f)

            state = status.get(
                "barrier_state",
                "unknown"
            )

            state_emoji = {
                "open": "🟢",
                "closing": "🟡",
                "closed": "🔴",
                "unknown": "⚪",
            }.get(
                state,
                "⚪"
            )

            st.metric(
                "Barrier state",
                f"{state_emoji} {state}"
            )

            st.metric(
                "Active violations (this frame)",
                status.get(
                    "active_violations",
                    0
                )
            )

            st.caption(
                f"Last update: "
                f"t={status.get('timestamp', 0)}s, "
                f"frame {status.get('frame_idx', 0)}"
            )

        else:

            st.info(
                "Waiting for pipeline to start "
                "writing status..."
            )

    with table_col:

        mtime = _file_mtime(
            CSV_PATH
        )

        df = load_events(
            CSV_PATH,
            mtime
        )

        if (
            df is not None
            and len(df)
        ):

            st.caption(
                "Most recent logged violations"
            )

            recent = (
                df.sort_values(
                    "start_time",
                    ascending=False
                )
                .head(5)
            )

            st.dataframe(
                recent[
                    [
                        "violation_id",
                        "class",
                        "barrier_state",
                        "start_time",
                        "best_conf",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

        else:

            st.caption(
                "No violations logged yet "
                "this session."
            )


def live_monitor_tab():

    st.subheader(
        "🔴 Live Monitor"
    )

    st.caption(
        "Auto-refreshes every 2 seconds "
        "while this tab is open."
    )

    live_monitor_fragment()


# ============================================================
# Main
# ============================================================

def main():

    st.title(
        "🚦 WARDEN — Railway Crossing "
        "Violation Dashboard"
    )

    tab1, tab2 = st.tabs(
        [
            "📊 Dashboard",
            "🔴 Live Monitor",
        ]
    )

    with tab1:
        dashboard_tab()

    with tab2:
        live_monitor_tab()


if __name__ == "__main__":
    main()