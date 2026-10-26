import datetime
import random
from dateutil.relativedelta import relativedelta
import pandas as pd
import plotly.express as px

# 1. Load data
df = pd.read_csv("roster.csv")

# Ensure required columns exist
df = df.dropna(subset=["member", "join"]).copy()
today = pd.Timestamp(datetime.date.today())
df["join"] = pd.to_datetime(df["join"])
df["leave"] = pd.to_datetime(df["leave"]).fillna(today)

# Fill missing flavour with 'unknown'
df["flavour"] = df["flavour"].fillna("unknown")

# 2. Identify interns: working <= 3 months (~93 days) strictly within June-October (months 6 to 10)
df["duration_days"] = (df["leave"] - df["join"]).dt.days
df["is_intern"] = (
    (df["duration_days"] <= 94)
    & (df["join"].dt.year == df["leave"].dt.year)
    & (df["join"].dt.month >= 6)
    & (df["leave"].dt.month <= 10)
)


def format_duration(start, end):
    rd = relativedelta(end, start)
    if rd.years > 0:
        return f"{rd.years}y {rd.months}m"
    if rd.months > 0:
        return f"{rd.months}m {rd.days}d"
    return f"{(end - start).days}d"


df["duration_label"] = [
    format_duration(start, end) for start, end in zip(df["join"], df["leave"])
]
df["bar_label"] = df["member"] + " " + df["duration_label"]

# 3. Stack top-down into available rows, inserting longest duration first
df = df.sort_values(by=["duration_days", "join"], ascending=[False, True]).reset_index(
    drop=True
)

row_intervals = []
assigned_rows = []
for _, row in df.iterrows():
    join_time = row["join"]
    leave_time = row["leave"]
    placed = False
    for r_idx, intervals in enumerate(row_intervals):
        # Fits if it does not overlap any existing interval on this row
        if all(leave_time <= s or join_time >= e for s, e in intervals):
            intervals.append((join_time, leave_time))
            assigned_rows.append(r_idx)
            placed = True
            break
    if not placed:
        assigned_rows.append(len(row_intervals))
        row_intervals.append([(join_time, leave_time)])

df["row"] = assigned_rows

# 4. Assign random color per unique flavour
random.seed(42)
palette = (
    px.colors.qualitative.Plotly + px.colors.qualitative.D3 + px.colors.qualitative.Set2
)
unique_flavours = list(df["flavour"].unique())
random.shuffle(palette)
color_map = {
    flavour: palette[i % len(palette)] for i, flavour in enumerate(unique_flavours)
}

# 5. Dynamic height based on packed row count (thick bars for clarity and overflow)
num_rows = len(row_intervals)
chart_height = max(500, num_rows * 48)

# 6. Build timeline
fig = px.timeline(
    df,
    x_start="join",
    x_end="leave",
    y="row",
    color="flavour",
    color_discrete_map=color_map,
    pattern_shape="is_intern",
    pattern_shape_map={True: "/", False: ""},
    hover_name="member",
    hover_data={
        "row": False,
        "join": "|%Y-%m-%d",
        "leave": "|%Y-%m-%d",
        "duration_label": True,
        "flavour": True,
        "is_intern": True,
        "duration_days": True,
    },
    title="Roster Timeline",
)

# Intern stripes: 50:50 coverage with a lighter color instead of thin dark stripes
# Clean legend: group by flavour and hide duplicate entries
seen_flavours = set()
for trace in fig.data:
    trace.width = 0.8
    flavour = trace.name.split(",")[0].strip()
    trace.name = flavour
    trace.legendgroup = flavour
    if flavour in seen_flavours:
        trace.showlegend = False
    else:
        trace.showlegend = True
        seen_flavours.add(flavour)

    if trace.marker.pattern and trace.marker.pattern.shape:
        trace.marker.pattern.solidity = 0.5
        trace.marker.pattern.fgcolor = "rgba(255, 255, 255, 0.7)"

# Calculate pure text extents and only stagger text when adjacent labels actually collide
total_span_days = max(1, (df["leave"].max() - df["join"].min()).days)
approx_plot_width_px = 1200
days_per_px = total_span_days / approx_plot_width_px
yshift_amount = 11

annotations = []
for r in df["row"].unique():
    row_indices = df[df["row"] == r].sort_values(by="join").index.tolist()
    text_extents = []
    for idx in row_indices:
        item = df.loc[idx]
        mid = item["join"] + (item["leave"] - item["join"]) / 2
        char_px = 7.0
        text_px = max(len(item["bar_label"]) * char_px, 20.0)
        half_width_days = pd.Timedelta(days=(text_px * days_per_px) / 2)
        text_extents.append((mid - half_width_days, mid + half_width_days))

    # Form connected components of overlapping text extents
    n = len(row_indices)
    clusters = []
    curr_cluster = [0]
    buffer_days = pd.Timedelta(days=5 * days_per_px)
    for i in range(1, n):
        if text_extents[i][0] < text_extents[i - 1][1] + buffer_days:
            curr_cluster.append(i)
        else:
            clusters.append(curr_cluster)
            curr_cluster = [i]
    if curr_cluster:
        clusters.append(curr_cluster)

    shifts = {}
    for cluster in clusters:
        if len(cluster) == 1:
            shifts[cluster[0]] = 0
        else:
            for step, pos in enumerate(cluster):
                shifts[pos] = yshift_amount if step % 2 == 0 else -yshift_amount

    for i, idx in enumerate(row_indices):
        item = df.loc[idx]
        mid_date = item["join"] + (item["leave"] - item["join"]) / 2
        annotations.append(
            dict(
                x=mid_date.strftime("%Y-%m-%d %H:%M:%S"),
                y=int(r),
                text=item["bar_label"],
                showarrow=False,
                yshift=shifts[i],
                font=dict(
                    size=12,
                    color="white",
                    weight="bold",
                    shadow="auto",
                ),
            )
        )

fig.update_layout(
    annotations=annotations,
    height=chart_height,
    xaxis_title="",
    yaxis_title="",
    bargap=0.15,
    legend_title_text="flavour",
    xaxis=dict(rangeslider=dict(visible=True), type="date"),
    yaxis=dict(
        autorange="reversed",
        showticklabels=False,
        showgrid=False,
        zeroline=False,
    ),
)

fig.write_html("timeline.html")
fig.show()
