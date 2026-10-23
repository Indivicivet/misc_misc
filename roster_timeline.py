import datetime
import pandas as pd
import plotly.express as px

# 1. Load data
# Expected columns: person, start_date, end_date, role (optional)
df = pd.read_csv("roster.csv")

# 2. Parse dates; blank end dates imply currently active
df["start_date"] = pd.to_datetime(df["start_date"])
today = pd.Timestamp(datetime.date.today())
df["end_date"] = pd.to_datetime(df["end_date"]).fillna(today)

# 3. Sort chronologically by earliest hire date
first_starts = df.groupby("person")["start_date"].min()
df["first_start"] = df["person"].map(first_starts)
df = df.sort_values(by=["first_start", "start_date"], ascending=[True, True])

# 4. Dynamic height: ~22px per person ensures readable row spacing
row_count = df["person"].nunique()
chart_height = max(600, row_count * 22)

# 5. Build timeline
fig = px.timeline(
    df,
    x_start="start_date",
    x_end="end_date",
    y="person",
    color="role" if "role" in df.columns else None,
    hover_name="person",
    hover_data={
        "start_date": "|%Y-%m-%d",
        "end_date": "|%Y-%m-%d",
        "first_start": False,
    },
    title="Company Roster Timeline",
)

# Put the earliest joiners at the top
fig.update_yaxes(autorange="reversed")
fig.update_layout(
    height=chart_height,
    xaxis_title="",
    yaxis_title="",
    bargap=0.25,
    xaxis=dict(rangeslider=dict(visible=True), type="date"),
)

# Export to a standalone interactive HTML file
fig.write_html("timeline.html")
fig.show()
