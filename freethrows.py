import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import mannwhitneyu
from scipy.stats import gaussian_kde
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Analysis of Follow-Through in Shooting Mechanics",  
    page_icon="🏀", 
    layout="wide"
)

st.title("Hold It! Reaching for Cookies and Made Free Throws")

st.markdown("""
**Follow-through** is the part of shooting mechanics where the ball is released, and the shooting hand extends toward the basket. 
Growing up, I remember learning how to shoot in gym class, where my gym teacher always emphasized *"reaching into the cookie jar"* and holding that position. It always confused me — how could what you do after releasing the ball affect the shot’s outcome?
This analysis examines the impact of follow-through, using data from the **MLSE SPL Open Data Repository**. The dataset includes one participant and data from 125 trials.

The goal of this analysis is to gain more clarity on the following questions: does follow through really matter? Is it consistent amongst trials? Can it tell us important information about the participant's form?
""")
st.divider()

## Initial Observations 

st.header("Initial Observations")
st.write("First, I chose to explore the right wrist movement in the z direction - exploring the time of the wind-up, release and follow through. I've created an interactive example annotating where these points will be.")

participants = pd.read_csv('participants.csv')
duration = pd.read_csv('durations.csv')
df1 = pd.merge(participants, duration, on="trial_id", how="inner")

tracking = pd.read_csv('tracking.csv')
df = pd.merge(df1, tracking, on="trial_id", how="inner")


# Asked Chat-gpt for colourblind friendly colours for accessibility
PRIMARY_COLOR = "#0072B2"
BACKGROUND_COLOR = "#E69F00"  
TEXT_COLOR = "#3E3E3E" 
ACCENT_COLOR = "#009E73" 

color_map = {
    'made': PRIMARY_COLOR,   
    'missed': BACKGROUND_COLOR   
}


trial_id_to_plot = st.text_input("Enter the trial ID (between T0001 - T0125):", "T0002")

trial_data = df[(df['trial_id'] == trial_id_to_plot) & (df['time'] >= 100) & (df['time'] <= 9000)].reset_index(drop=True)
value1 = trial_data['windup_duration'].iloc[0]  
value2 = trial_data['follow_through_duration'].iloc[0]
value3 = trial_data['result'].iloc[0]

st.text(f"Wind-up Duration: {value1} ms | Follow Through Duration: {value2} ms | Result: {value3} ")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=trial_data['time'],
    y=trial_data['R_WRIST_z'],
    mode='lines',
    name="R_WRIST_z Position",
    line=dict(color=PRIMARY_COLOR)
))


fig.add_trace(go.Scatter(
    x=trial_data['windup_start'],
    y=trial_data['windup_height'],
    mode='markers+text',
    name="Wind-up Start",
    marker=dict(color=BACKGROUND_COLOR, size=10)
))
fig.add_trace(go.Scatter(
    x=trial_data['release_time'],
    y=trial_data['release_height'],
    mode='markers+text',
    name="Release Point",
    marker=dict(color=ACCENT_COLOR, size=10)
))
fig.add_trace(go.Scatter(
    x=trial_data['follow_through_time'],
    y=trial_data['follow_through_height'],
    mode='markers+text',
    name="Follow Through Release Point",
    marker=dict(color=BACKGROUND_COLOR, size=10)
))

fig.update_layout(
    title=f"R_WRIST_z Position over Time for Trial {trial_id_to_plot}",
    xaxis_title="Time",
    yaxis_title="R_WRIST_z Position",
    legend_title="Legend",
    font=dict(color=TEXT_COLOR), 
    margin=dict(l=50, r=50, t=50, b=50)
)
st.plotly_chart(fig, use_container_width=True)


follow_through_duration_made = df[df['result'] == 'made']['follow_through_duration'].mean()
follow_through_duration_missed = df[df['result'] == 'missed']['follow_through_duration'].mean()
wind_up_duration_made = df[df['result'] == 'made']['windup_duration'].mean()
wind_up_duration_missed = df[df['result'] == 'missed']['windup_duration'].mean()
release_height_made = df[df['result'] == 'made']['release_height'].mean()
release_height_missed = df[df['result'] == 'missed']['release_height'].mean()

def create_bar_chart(metric, made, missed, metric_title):
    fig = go.Figure()
    max_value = max(made, missed)
    fig.add_trace(go.Bar(
        name="Made",
        x=[metric],
        y=[made],
        marker_color=PRIMARY_COLOR,  
        text=[f"{made:.2f}"],
        textposition="outside"
    ))
    fig.add_trace(go.Bar(
        name="Missed",
        x=[metric],
        y=[missed],
        marker_color=BACKGROUND_COLOR , 
        text=[f"{missed:.2f}"],
        textposition="outside"
    ))
    fig.update_layout(
        title=metric_title,
        yaxis_title="Average Value",
        yaxis_range=[0, max_value * 1.2],
        font=dict(color=TEXT_COLOR),
        showlegend=False,
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig


col1, col2, col3 = st.columns(3)

with col1:
    st.plotly_chart(create_bar_chart("Follow-through Duration", follow_through_duration_made, follow_through_duration_missed, "Follow-through Duration (ms)"), use_container_width=True)

with col2:
    st.plotly_chart(create_bar_chart("Wind-up Duration", wind_up_duration_made, wind_up_duration_missed, "Wind-up Duration (ms)"), use_container_width=True)

with col3:
    st.plotly_chart(create_bar_chart("Release Height", release_height_made, release_height_missed, "Release Height (m)"), use_container_width=True)

st.markdown("""
The wind-up duration and release height indicate the participant is consistent in these categories of their follow through. The follow duration is longer on average for missed
baskets. This is interesting, perhaps made buckets have quicker release, and that is contributing to the success of these attempts.
""")

## Wrist Stability
st.divider()
st.header("Wrist Stability")

st.markdown("""
The graph comparing R_WRIST_z and time reveals noticeable differences in the smoothness of the wrist movement 
from release point to the follow-through end point. For instance, trials like T0001 exhibit a smooth, consistent line, while 
trials like T0123 show more variability with fluctuations. To quantify this, I extracted the values for this segment, 
calculated the rate of change between consecutive points, and then computed the standard deviation of these rates. 
A higher standard deviation reflects greater variability in the line, indicating less consistent wrist movement 
during the follow-through phase.
""")
col1, spacer, col2 = st.columns([1, 0.2, 1])

with col1:
    fig = px.violin(df, 
                    x='result', 
                    y='slope', 
                    color='result', 
                    title="Comparison of Wrist Stability for Made vs Missed Shots",
                    color_discrete_map=color_map,
                    box=True,  
                    points="all")  

    fig.update_layout(
        xaxis_title="Shot Outcome",  
        yaxis_title="Wrist Stability (Follow Through)"  
    )

    st.plotly_chart(fig)

col2.empty()

with col2:
    made_data = df[df['result'] == 'made']['slope']
    missed_data = df[df['result'] == 'missed']['slope']

    u_stat, p_value = mannwhitneyu(made_data, missed_data)
    n1 = len(made_data)
    n2 = len(missed_data)
    z = (u_stat - (n1 * n2) / 2) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)

    r = z / np.sqrt(n1 + n2)

    st.write("\n") 
    st.write("\n")
    st.text(f"U statistic: {round(u_stat, 1)} | p value: {p_value:.1e} | Effect Size: {round(r, 1)}")


  
    st.markdown("""
        We can see that the interquartile range for missed shots is larger than made, which implies there is greater inconsistency with missed shots during this period. 
        A lot of overlap here, though, this may be due to the small sample size, but also due to the fact that there are smaller margins of difference. 
        I also looked at a statistical test to compare the two groups. Because this data is non-parametric, I used a Mann-Whitney U-test. The null hypothesis is that there is no
        difference between the two groups. The U statistic is quite large and the p value is very small (smaller than a significance level of 0.05), so these two groups seem
        statistically different. But I also wanted to check the effect size to determine if these results were to be trusted (magnitude of difference).
        Based on the effect size, it's small, indicating that the practical difference between these two groups is small. But it is negative, implying wrist stability is greater for made shots, which is interesting.
    """)


## Body Symmetry 

st.divider()
st.header("Body Symmetry")
    
st.markdown("""
Next, I wanted to looked at the parts of the body not directly involved during the follow through, was there a difference in made versus missed baskets? For this next visualization, I looked at 
I looked at the symmetry of the body, by calculating the absolute difference between the right and left hip, ankle, eye and ear in the x direction. I wanted to see if there was a statisically significant difference between hip symmetry in 
made versus missed baskets. I used a density plot with kernel density estimation due to the non-parametric nature of the data. Ankle symmetry stands out, as the missed baskets have a higher concentration of 
Across the board (Except for the eyes), the made baskets do not have any data points greater than missed baskets in terms of symmetry in the x direction, this suggests that 
symmetry of the body is important when shooting. For hip symmetry, the peak for made shots is slightly more to the right than missed, this suggests the participant may shoot on a angle, maybe for power? The eye symmetry follows this trend, supporting
this assumption. 
""")

follow_through_data = df[(df['time'] >= df['release_time']) & 
                                  (df['time'] <= df['follow_through_time'])]
df = follow_through_data

df['hip_symmetry_x'] = abs(df['R_HIP_x'] - df['L_HIP_x'])
df['ankle_symmetry_x'] = abs(df['R_ANKLE_x'] - df['L_ANKLE_x'])
df['eye_symmetry_x'] = abs(df['R_EYE_x'] - df['L_EYE_x'])
df['ear_symmetry_x'] = abs(df['R_EAR_x'] - df['L_EAR_x'])

df['shot_outcome'] = df['result'].apply(lambda x: 1 if x == 'made' else 0)


def plot_density_kde_subplot(feature, title, row, col, fig):
    for outcome, group in df.groupby('shot_outcome'):
        kde = gaussian_kde(group[feature])
        x_vals = np.linspace(group[feature].min(), group[feature].max(), 1000)
        y_vals = kde(x_vals)

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='lines',
                name='Made' if outcome == 1 else 'Missed',
                line=dict(width=2, color=PRIMARY_COLOR if outcome == 1 else BACKGROUND_COLOR ),
                showlegend=(row == 1 and col == 1)  
            ),
            row=row,
            col=col
        )

    fig.update_xaxes(title_text=feature, row=row, col=col)
    fig.update_yaxes(title_text="Density", row=row, col=col)

fig = make_subplots(rows=1, cols=4, subplot_titles=[
    "Hip Symmetry X", "Ankle Symmetry X", "Eye Symmetry X", "Ear Symmetry X"
])

plot_density_kde_subplot('hip_symmetry_x', 'Hip Symmetry X', 1, 1, fig)
plot_density_kde_subplot('ankle_symmetry_x', 'Ankle Symmetry X', 1, 2, fig)
plot_density_kde_subplot('eye_symmetry_x', 'Eye Symmetry X', 1, 3, fig)
plot_density_kde_subplot('ear_symmetry_x', 'Ear Symmetry X', 1, 4, fig)

fig.update_layout(
    title_text="Symmetry KDE Density Plots",
    legend_title_text="Shot Outcome",
    legend=dict(
        orientation="h",  
        yanchor="bottom",
        y=1.1,
        xanchor="center",
        x=0.5
    ),
    height=400,
    template="plotly_white"
)


st.plotly_chart(fig, use_container_width=True)


# Pinky Offset

st.divider()
st.header("Follow through Fingers")
st.markdown("""
I also wanted to analyze the finger positioning during the follow-through, as different players have unique follow-throughs, 
often characterized by the curvature of the fingers (e.g., 'gooseneck', 'flat', etc.). To do this, I calculated the offset between the 
wrist and pinky base point during the follow-through, specifically at the moment when the wrist and pinky were closest together in the 
z-direction. Interestingly, I found that made baskets tend to have a larger absolute offset compared to missed baskets. However, most of 
the trials cluster around an offset of -0.15, which suggests that the follow-through position of the fingers is consistent and 
reproducible across attempts, demonstrating that the participant's finger positioning remains stable.
""")

col1, col2 = st.columns(2)

with col1:
    fig1 = px.box(df, x='pinky_offset', y='result', 
                  color='result', 
                  color_discrete_map=color_map,  
                  labels={'pinky_offset': 'Pinky to Wrist Offset', 'result': 'Shot Outcome'},
                  title="Made vs Missed Baskets vs. Pinky to Wrist Offset")
    st.plotly_chart(fig1)


with col2:
    fig2 = px.scatter(df, x='pinky_offset', y='entry_angle', 
                      color='result',  
                      color_discrete_map=color_map,  
                      labels={'pinky_offset': 'Offset (wrist to pinky (R_WRIST_z - R_5THFINGER_z))', 'entry_angle': 'Entry Angle (degrees)', 'result': 'Shot Outcome'},
                      title="Scatter Plot: Offset vs Entry Angle for Made vs Missed Baskets")
    st.plotly_chart(fig2)

# Other 
st.divider()
st.header("Parting Observations")
st.markdown("""
While working on this analysis, I was reflecting on some insights I gathered during the Raptors vs. Celtics game.
Specifically, I observed that some players released their free throws with a straight upward motion (Al Horford), while others exhibited a slight forward movement (Chris Boucher). 
Comparing these observations with the data, it became apparent that the forward motion was consistently present during successful shots, whereas missed baskets 
showed more variance in this regard. For this particular participant, there appears to be less forward motion during successful shots.

Another key moment that stood out during the game was in the fourth quarter, when RJ Barrett had an opportunity for a three-point play but missed the free throw with about 2
 minutes left in regular time. Notably, his feet were much wider than usual — a stance I’ve seen him take before, but this time it seemed more exaggerated.
  Having followed RJ’s performance throughout the season and with Team Canada, I’ve noticed that he tends to shoot with a wider base compared to other players, 
  but this seemed particularly pronounced during this specific game.

Curious to see if this was a factor for the participant in my analysis, I examined the distance between their feet. 
However, the data did not show significant variance in foot positioning between the successful and missed baskets for this participant, 
suggesting that the wider stance didn’t play a major role in shot outcomes here.
""")

col1, col2 = st.columns(2)

with col1:
    fig1 = px.box(df1, 
                     x='result', 
                     y='x_wrist', 
                     color='result', 
                     title="Forward Motion during follow-through (Right Wrist in X)",
                     points='all', 
                     color_discrete_map=color_map  
                    )

    fig1.update_layout(
        xaxis_title="Shot Outcome",  
        yaxis_title="Forward Motion of Right Wrist in X direction",
        margin=dict(l=10, r=10, t=40, b=10) 
    )

    st.plotly_chart(fig1)

with col2:
    fig2 = px.box(df1, 
                     x='result', 
                     y='y_ankle', 
                     color='result', 
                     title="Distance between feet",
                     points='all',  
                     color_discrete_map=color_map  
                    )
    fig2.update_layout(
        xaxis_title="Shot Outcome",  
        yaxis_title="Distance between feet",  
        margin=dict(l=10, r=10, t=40, b=10)  
    )

    st.plotly_chart(fig2)

st.divider()
st.header("Conclusion")
st.markdown("""
This analysis explored the role of follow-through in free throw shooting, focusing on wrist stability, body symmetry, and hand positioning. 
It found that wrist stability was greater for made shots, while more variability was observed in missed shots, suggesting that consistency in wrist movement is important 
for success. Body symmetry, particularly in the hips and eyes, also seemed to play a role, with more consistent symmetry associated with made baskets. 
Finger positioning was stable across trials, with a slight larger offset for made shots. There seems to be less forward motion during the follow through with made baskets.
However, foot positioning did not show significant impact. 

The goal of this analysis was to see if there was a difference in follow through between made and missed free throws. Through the variety of metrics looked at, there was noticeable difference
between the follow through of made and missed baskets. For the future, it would be interesting to get datapoints on the right and left heels. From watching NBA players shoot free throws, many of them
lift up their heel during the free throw. The distance from the ground, along with the stability of the overall body would be interesting to explore.
""")