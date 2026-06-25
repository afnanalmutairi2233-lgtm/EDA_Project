import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ── Theme colors ──
BG        = '#8A9A6A'   # soft olive background
CARD_BG   = '#9AAA78'   # slightly lighter olive
TEXT      = '#1A1A1A'   # near-black text
BEIGE1    = '#E8D5B7'   # light beige
BEIGE2    = '#D4B896'   # medium beige
PINK      = "#D46E6E"   # light pink
GREEN     = '#A8D8A8'   # light green
BLACK     = '#1A1A1A'
MOCHA     = '#6B4226'   # mocha brown for sidebar
PALETTE   = [BEIGE1, PINK, GREEN, BEIGE2, '#C9A96E', '#B5C9A8']

st.set_page_config(page_title="EDA Dashboard", layout="wide")

# ── Global CSS ──
st.markdown(f"""
<style>
    .stApp {{ background-color: {BG}; color: {TEXT}; }}
    section[data-testid="stSidebar"] {{ background-color: #7D6B5A; }}
    section[data-testid="stSidebar"] * {{ color: #F5EDE3 !important; }}
    div[data-baseweb="select"] > div {{
        background-color: #E8C4B8 !important;
        border-color: #C4A090 !important;
        color: #1A1A1A !important;
    }}
    div[data-baseweb="tag"] {{
        background-color: #E8D5B7 !important;
        color: #1A1A1A !important;
        border: none !important;
        border-radius: 4px !important;
    }}
    div[data-baseweb="tag"] span,
    div[data-baseweb="tag"] svg {{
        color: #1A1A1A !important;
        fill: #1A1A1A !important;
    }}
    span[data-baseweb="tag"] {{
        background-color: #E8D5B7 !important;
        color: #1A1A1A !important;
        border: none !important;
    }}
    .stMultiSelect span {{
        background-color: #E8D5B7 !important;
        color: #1A1A1A !important;
    }}
    div[data-baseweb="select"] span {{
        color: #1A1A1A !important;
    }}
    div[data-testid="stSlider"] > div > div > div {{
        background-color: #C47A6A !important;
    }}
    div[data-testid="stSlider"] > div > div > div > div {{
        background-color: #C47A6A !important;
        border-color: #C47A6A !important;
    }}
    h1, h2, h3 {{ color: {BLACK}; }}
    .stMetric {{ background-color: {CARD_BG}; border-radius: 10px; padding: 10px; }}
    .stDataFrame {{ background-color: {CARD_BG}; }}
    div[data-testid="metric-container"] {{
        background-color: #6B8A52;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #4A6335;
    }}
</style>
""", unsafe_allow_html=True)

# ── Matplotlib theme ──
plt.rcParams.update({
    'figure.facecolor': '#9AAA78',
    'axes.facecolor':   '#A8B882',
    'axes.edgecolor':   BLACK,
    'axes.labelcolor':  BLACK,
    'xtick.color':      BLACK,
    'ytick.color':      BLACK,
    'text.color':       BLACK,
    'grid.color':       '#7A8A5A',
    'grid.alpha':       0.3,
})

@st.cache_data
def load_data():
    df = pd.read_csv('datasetEDA.csv', sep=';')
    df.drop(columns=['Location_Lat', 'Location_Long', 'Nationality'], inplace=True, errors='ignore')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d/%m/%y %H:%M')
    df['Hour'] = df['Timestamp'].dt.hour
    df['Adjusted_Temperature'] = np.where(
        df['Weather_Conditions'] == 'Rainy', df['Temperature'] - 5,
        np.where(df['Weather_Conditions'] == 'Cloudy', df['Temperature'] - 2, df['Temperature'])
    )
    df['Adjusted_Waiting_Time'] = np.where(
        df['Transport_Mode'] == 'Bus', df['Waiting_Time_for_Transport'] + 20,
        np.where(df['Transport_Mode'] == 'Train', df['Waiting_Time_for_Transport'] + 10,
        np.where(df['Transport_Mode'] == 'Metro', df['Waiting_Time_for_Transport'] - 10,
                 df['Waiting_Time_for_Transport']))
    )
    df['Crowded_Area'] = (df['Distance_Between_People_m'] < 1).astype(int)
    return df

df = load_data()

# Sidebar
st.sidebar.title(" Hajj Crowd EDA🕌")
st.sidebar.header("🔧 Filters")
weather_options = df['Weather_Conditions'].unique().tolist()
selected_weather = st.sidebar.multiselect("Weather", weather_options, default=weather_options)
transport_options = df['Transport_Mode'].unique().tolist()
selected_transport = st.sidebar.multiselect("Transport Mode", transport_options, default=transport_options)
hour_range = st.sidebar.slider("Hour Range", 0, 23, (0, 23))

df_filtered = df[
    df['Weather_Conditions'].isin(selected_weather) &
    df['Transport_Mode'].isin(selected_transport) &
    df['Hour'].between(hour_range[0], hour_range[1])
]

# Overview
st.header(" Data Overview🔍")
col1, col2, col3 = st.columns(3)
col1.metric("Rows", f"{df_filtered.shape[0]:,}")
col2.metric("Columns", df_filtered.shape[1])
col3.metric("Missing Values", int(df_filtered.isnull().sum().sum()))

with st.expander("View Data"):
    n_rows = st.slider("Number of rows", 5, 50, 10)
    st.dataframe(df_filtered.head(n_rows), use_container_width=True)

with st.expander("Descriptive Statistics"):
    st.dataframe(df_filtered.describe(), use_container_width=True)

st.divider()

# Distributions
st.header("📈 Variable Distributions")
col_left, col_right = st.columns(2)

with col_left:
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df_filtered['Crowd_Density'], bins=30, kde=True, color=PINK, ax=ax)
    ax.set_title('Distribution of Crowd Density')
    st.pyplot(fig)
    plt.close()

with col_right:
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df_filtered['Queue_Time_minutes'], bins=30, kde=True, color=GREEN, ax=ax)
    ax.set_title('Distribution of Queue Time')
    st.pyplot(fig)
    plt.close()

col_left2, col_right2 = st.columns(2)

with col_left2:
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df_filtered['Satisfaction_Rating'], bins=10, color=BEIGE1, ax=ax)
    ax.set_title('Satisfaction Rating')
    st.pyplot(fig)
    plt.close()

with col_right2:
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df_filtered['Security_Checkpoint_Wait_Time'], bins=30, kde=True, color=BEIGE2, ax=ax)
    ax.set_title('Security Checkpoint Wait Time')
    ax.set_xlabel('Wait Time (minutes)')
    st.pyplot(fig)
    plt.close()

st.divider()

# Weather and Transport
st.header("Weather and Transport")
col_left3, col_right3 = st.columns(2)

with col_left3:
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=df_filtered, x='Weather_Conditions', y='Adjusted_Temperature',
                palette=[BEIGE1, PINK, GREEN], ax=ax)
    ax.set_title('Adjusted Temperature by Weather Conditions')
    st.pyplot(fig)
    plt.close()

with col_right3:
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=df_filtered, x='Transport_Mode', y='Adjusted_Waiting_Time',
                palette=PALETTE, ax=ax)
    plt.xticks(rotation=45)
    ax.set_title('Average Waiting Time by Transport Mode')
    ax.set_ylabel('Avg Waiting Time (min)')
    ax.grid(axis='y', alpha=0.3)
    st.pyplot(fig)
    plt.close()

fig, ax = plt.subplots(figsize=(9, 4))
sns.countplot(x='Weather_Conditions', hue='Crowd_Density', data=df_filtered,
              palette=[BEIGE1, PINK, GREEN], ax=ax)
ax.set_title('Crowd Density by Weather')
st.pyplot(fig)
plt.close()

st.divider()

# Time Distribution
st.header(" Time Distribution")
fig, ax = plt.subplots(figsize=(10, 4))
df_filtered['Hour'].value_counts().sort_index().plot(kind='bar', color=PINK, ax=ax)
ax.set_title('Number of Events by Hour')
ax.set_xlabel('Hour')
ax.set_ylabel('Count')
st.pyplot(fig)
plt.close()

st.divider()

# Stress Level
st.header(" Stress Level in Crowded Areas")
cross = pd.crosstab(df_filtered['Crowded_Area'], df_filtered['Stress_Level'], normalize='index') * 100
if set(['Low', 'Medium', 'High']).issubset(cross.columns):
    cross = cross[['Low', 'Medium', 'High']]

fig, ax = plt.subplots(figsize=(8, 4))
cross.plot(kind='bar', stacked=True, color=[GREEN, BEIGE2, PINK], ax=ax)
ax.set_title('Stress Level Distribution by Crowded Area')
ax.set_ylabel('Percentage (%)')
ax.legend(title='Stress Level')
plt.xticks(ticks=[0, 1], labels=['Not Crowded', 'Crowded'], rotation=0)
st.pyplot(fig)
plt.close()

st.caption("Built from EDA_ML.ipynb")

