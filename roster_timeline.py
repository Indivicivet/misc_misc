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
    (df["duration_days"] <= 93)
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

# 3. Stack top-down into available rows
df = df.sort_values(by=["join", "duration_days"], ascending=[True, False]).reset_index(
    drop=True
)

row_ends = []
assigned_rows = []
for _, row in df.iterrows():
    join_time = row["join"]
    leave_time = row["leave"]
    placed = False
    for idx, end_time in enumerate(row_ends):
        if join_time >= end_time:
            row_ends[idx] = leave_time
            assigned_rows.append(idx)
            placed = True
            break
    if not placed:
        assigned_rows.append(len(row_ends))
        row_ends.append(leave_time)

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

# 5. Dynamic height based on packed row count
num_rows = len(row_ends)
chart_height = max(400, num_rows * 32)

# 6. Build timeline
fig = px.timeline(
    df,
    x_start="join",
    x_end="leave",
    y="row",
    text="bar_label",
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
    title="Company Roster Timeline",
)

fig.update_traces(
    textposition="inside",
    insidetextanchor="middle",
    constraintext="inside",
)

fig.update_layout(
    height=chart_height,
    xaxis_title="",
    yaxis_title="",
    bargap=0.2,
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
