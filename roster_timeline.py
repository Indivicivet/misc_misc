import datetime
import random
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

# 3. Sort top to bottom based on time with the company
df = df.sort_values(by="duration_days", ascending=False).reset_index(drop=True)

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

# 5. Dynamic height
row_count = df["member"].nunique()
chart_height = max(600, row_count * 22)

# 6. Build timeline
fig = px.timeline(
    df,
    x_start="join",
    x_end="leave",
    y="member",
    color="flavour",
    color_discrete_map=color_map,
    pattern_shape="is_intern",
    pattern_shape_map={True: "/", False: ""},
    category_orders={"member": df["member"].tolist()},
    hover_name="member",
    hover_data={
        "join": "|%Y-%m-%d",
        "leave": "|%Y-%m-%d",
        "flavour": True,
        "is_intern": True,
        "duration_days": True,
    },
    title="Company Roster Timeline",
)

# Reverse y-axis so the longest tenure appears at the top
fig.update_yaxes(autorange="reversed")
fig.update_layout(
    height=chart_height,
    xaxis_title="",
    yaxis_title="",
    bargap=0.25,
    xaxis=dict(rangeslider=dict(visible=True), type="date"),
)

fig.write_html("timeline.html")
fig.show()
