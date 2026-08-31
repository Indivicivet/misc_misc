"""Race Distribution Analysis Tool

This script reads a race results CSV file and produces distribution analysis plots.

CSV Schema Requirements:
-----------------------
The input CSV file must adhere to the following schema:
- place: int
    Finishing position of the participant (1, 2, 3, ...). Required.
- bib: str or int
    Bib / race number assigned to the participant.
- name: str
    Full name of the participant.
- gender: str
    Gender category ('M' for Male, 'F' for Female, 'NB' for Non-binary). Required.
- age_category: str
    Age bracket / category (e.g. '20-39', '20-34', '35-44', '0-12', '50-59'). Required.
- division: str
    Combined division string (e.g. 'M 50-59', 'F 20-34').
- finish_time: str
    Finish time string in 'MM:SS' or 'HH:MM:SS' format.
- finish_seconds: float or int
    Finish time converted to total seconds. Required for plotting.
- pace_sec_km: float (optional)
    Pace in seconds per kilometer.
- division_place: int (optional)
    Finishing position within the participant's age/gender division.
- division_total: int (optional)
    Total participants in the participant's division.
- gender_place: int (optional)
    Finishing position within the participant's gender category.
- gender_total: int (optional)
    Total participants in the participant's gender category.
- team: str (optional)
    Running club or team affiliation.
"""

import argparse
from pathlib import Path
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import scipy.stats as stats


def format_seconds_to_mmss(sec, _=None):
    sec = max(0, int(round(sec)))
    minutes = sec // 60
    seconds = sec % 60
    if minutes >= 60:
        hours = minutes // 60
        minutes = minutes % 60
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def get_age_group_marker(age_category_str: str) -> tuple[str, str]:
    cleaned = str(age_category_str).strip()
    digits = [int(val) for val in re.findall(r"\d+", cleaned)]
    if digits:
        min_age = digits[0]
        max_age = digits[1] if len(digits) > 1 else min_age
        if max_age < 20:
            return "^", "Youngsters (<20)"
        if min_age >= 50:
            return "D", "Veterans / Oldies (50+)"
        if min_age >= 35 or min_age >= 40:
            return "s", "Masters (35-49)"
        return "o", "Primary / Open (20-39)"
    return "o", "Open / Other"


def plot_race_analysis(
    df: pd.DataFrame,
    title: str = "Race Distribution Analysis",
    bw_factor: float = 0.18,
):
    df = df.dropna(subset=["place", "finish_seconds"]).copy()
    df["place"] = pd.to_numeric(df["place"], errors="coerce")
    df["finish_seconds"] = pd.to_numeric(df["finish_seconds"], errors="coerce")
    df = df.sort_values("place")

    gender_colors = {
        "M": "#1f77b4",
        "Male": "#1f77b4",
        "F": "#e377c2",
        "Female": "#e377c2",
        "NB": "#7f7f7f",
        "Non-binary": "#7f7f7f",
        "Non-Binary": "#7f7f7f",
    }

    fig, (ax_pos, ax_kde) = plt.subplots(1, 2, figsize=(16, 7))

    # --- Plot 1: Finishing position # (x) vs Finish time (y) ---
    present_markers = {}
    present_genders = {}

    for _, row in df.iterrows():
        raw_gender = str(row.get("gender", "NB")).strip()
        color = gender_colors.get(raw_gender, "#7f7f7f")
        marker, group_label = get_age_group_marker(row.get("age_category", ""))

        present_markers[group_label] = marker
        canonical_gender = (
            "Men"
            if raw_gender in ["M", "Male"]
            else ("Women" if raw_gender in ["F", "Female"] else "Non-binary")
        )
        present_genders[canonical_gender] = color

        ax_pos.scatter(
            row["place"],
            row["finish_seconds"],
            color=color,
            marker=marker,
            s=60,
            alpha=0.85,
            edgecolors="black" if marker == "D" else "none",
            linewidths=0.5,
        )

    ax_pos.set_xlabel("Finishing Position (#)", fontsize=11, fontweight="bold")
    ax_pos.set_ylabel("Finish Time (MM:SS)", fontsize=11, fontweight="bold")
    ax_pos.set_title(
        "Finishing Position vs Finish Time", fontsize=13, fontweight="bold"
    )
    ax_pos.yaxis.set_major_formatter(ticker.FuncFormatter(format_seconds_to_mmss))
    ax_pos.invert_yaxis()
    ax_pos.grid(True, linestyle=":", alpha=0.6)

    gender_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=col,
            markersize=8,
            label=label,
        )
        for label, col in present_genders.items()
    ]
    marker_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=m,
            color="w",
            markerfacecolor="dimgray",
            markersize=8,
            label=label,
        )
        for label, m in present_markers.items()
    ]

    gender_legend = ax_pos.legend(
        handles=gender_handles, loc="upper right", title="Gender", framealpha=0.9
    )
    ax_pos.add_artist(gender_legend)
    ax_pos.legend(
        handles=marker_handles,
        loc="lower left",
        title="Age Category",
        framealpha=0.9,
    )

    # --- Plot 2: Continuous smooth KDE plot for finish time ---
    min_time = df["finish_seconds"].min()
    max_time = df["finish_seconds"].max()
    time_grid = np.linspace(max(0, min_time - 30), max_time + 45, 600)

    # 1. Overall KDE
    if len(df["finish_seconds"]) >= 2:
        overall_kde = stats.gaussian_kde(df["finish_seconds"], bw_method=bw_factor)
        ax_kde.plot(
            time_grid,
            overall_kde(time_grid),
            color="black",
            linewidth=2.5,
            label=f"Overall (n={len(df)})",
        )
        ax_kde.fill_between(
            time_grid, 0, overall_kde(time_grid), color="black", alpha=0.08
        )

    # 2. Men KDE
    men_data = df[df["gender"].isin(["M", "Male"])]["finish_seconds"]
    if len(men_data) >= 3:
        men_kde = stats.gaussian_kde(men_data, bw_method=bw_factor)
        ax_kde.plot(
            time_grid,
            men_kde(time_grid),
            color="#1f77b4",
            linewidth=2.0,
            label=f"Men (n={len(men_data)})",
        )

    # 3. Women KDE
    women_data = df[df["gender"].isin(["F", "Female"])]["finish_seconds"]
    if len(women_data) >= 3:
        women_kde = stats.gaussian_kde(women_data, bw_method=bw_factor)
        ax_kde.plot(
            time_grid,
            women_kde(time_grid),
            color="#e377c2",
            linewidth=2.0,
            label=f"Women (n={len(women_data)})",
        )

    # 4. Non-binary KDE (if present)
    nb_data = df[df["gender"].isin(["NB", "Non-binary", "Non-Binary"])][
        "finish_seconds"
    ]
    if len(nb_data) >= 3:
        nb_kde = stats.gaussian_kde(nb_data, bw_method=bw_factor)
        ax_kde.plot(
            time_grid,
            nb_kde(time_grid),
            color="#7f7f7f",
            linewidth=2.0,
            label=f"Non-binary (n={len(nb_data)})",
        )

    # 5. Primary Male Bracket (20-39 / 20-34)
    male_primary = df[
        df["gender"].isin(["M", "Male"])
        & df["age_category"]
        .astype(str)
        .str.contains(r"20-39|20-34|20-29|30-39|Senior", regex=True, na=False)
    ]["finish_seconds"]
    if len(male_primary) >= 3:
        m_prim_kde = stats.gaussian_kde(male_primary, bw_method=bw_factor)
        ax_kde.plot(
            time_grid,
            m_prim_kde(time_grid),
            color="#1f77b4",
            linestyle="--",
            linewidth=1.8,
            label=f"Men Primary Age Bracket (n={len(male_primary)})",
        )

    # 6. Primary Female Bracket (20-39 / 20-34)
    female_primary = df[
        df["gender"].isin(["F", "Female"])
        & df["age_category"]
        .astype(str)
        .str.contains(r"20-39|20-34|20-29|30-39|Senior", regex=True, na=False)
    ]["finish_seconds"]
    if len(female_primary) >= 3:
        f_prim_kde = stats.gaussian_kde(female_primary, bw_method=bw_factor)
        ax_kde.plot(
            time_grid,
            f_prim_kde(time_grid),
            color="#e377c2",
            linestyle="--",
            linewidth=1.8,
            label=f"Women Primary Age Bracket (n={len(female_primary)})",
        )

    ax_kde.set_xlabel("Finish Time (MM:SS)", fontsize=11, fontweight="bold")
    ax_kde.set_ylabel("Probability Density", fontsize=11, fontweight="bold")
    ax_kde.set_title(
        "Finishing Time Distribution (KDE)", fontsize=13, fontweight="bold"
    )
    ax_kde.xaxis.set_major_formatter(ticker.FuncFormatter(format_seconds_to_mmss))
    ax_kde.set_xlim(max(0, min_time - 30), max_time + 45)
    ax_kde.grid(True, linestyle=":", alpha=0.6)
    ax_kde.legend(loc="upper right", framealpha=0.9)

    plt.suptitle(title, fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Analyze race finishing time distributions from CSV."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="Northstowe Festival of Running 2026 5k.csv",
        help="Path to race results CSV file",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="Northstowe_Festival_of_Running_2026_5k_analysis.png",
        help="Output path for plot image",
    )
    parser.add_argument(
        "--bandwidth",
        "-b",
        type=float,
        default=0.18,
        help="KDE bandwidth factor (default: 0.18)",
    )
    parser.add_argument(
        "--show", action="store_true", help="Display plot interactively"
    )
    args = parser.parse_args()

    csv_file = Path(args.csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    df = pd.read_csv(csv_file)
    fig = plot_race_analysis(
        df,
        title=f"Race Results Distribution: {csv_file.stem}",
        bw_factor=args.bandwidth,
    )

    if args.output:
        fig.savefig(args.output, dpi=300)
        print(f"Analysis plot saved to {args.output}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
