import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from pathlib import Path

############################
# Data Loading and Preprocessing
############################

def preprocess_real_data(df):
    # Show columns for debugging
    # st.write("Columns in CSV:", df.columns.tolist())

    # Attempt to use dispatch_date directly
    if 'dispatch_date' in df.columns:
        # Try parsing with inference
        df['dispatch_date'] = pd.to_datetime(df['dispatch_date'], errors='coerce')
    else:
        # If dispatch_date is not present, try dispatch_date_time
        if 'dispatch_date_time' in df.columns:
            # Parse and extract just the date portion
            df['dispatch_date_time'] = pd.to_datetime(df['dispatch_date_time'], errors='coerce')
            df['dispatch_date'] = df['dispatch_date_time'].dt.date
            df['dispatch_date'] = pd.to_datetime(df['dispatch_date'], errors='coerce')
        else:
            st.write("No suitable date column found (dispatch_date or dispatch_date_time).")
            return pd.DataFrame()

    # Drop rows with no valid dates
    df = df.dropna(subset=['dispatch_date'])
    if df.empty:
        st.write("No valid dispatch_date entries found after parsing.")
        return pd.DataFrame()

    # Map earliest date to day=1
    unique_dates = sorted(df['dispatch_date'].unique())
    if len(unique_dates) == 0:
        st.write("No unique dates found after parsing.")
        return pd.DataFrame()

    day_map = {date: i+1 for i, date in enumerate(unique_dates)}
    df['day'] = df['dispatch_date'].map(day_map)

    # Ensure hour column exists and is numeric
    if 'hour' not in df.columns:
        st.write("No 'hour' column found. Cannot proceed.")
        return pd.DataFrame()
    df['hour'] = pd.to_numeric(df['hour'], errors='coerce')
    df = df.dropna(subset=['hour'])
    df['hour'] = df['hour'].astype(int)

    # Group by (day, hour) to count the number of crimes
    # Each row is presumably a crime incident, so counting rows gives crimes.
    grouped = df.groupby(['day', 'hour']).size().reset_index(name='crimes')

    return grouped[['day', 'hour', 'crimes']]

def generate_synthetic_data(days=30, base_rate=5, nighttime_factor=2, unemployment_rate=0.05, weather_factor=1.0):
    adjusted_base_rate = base_rate * (1 + unemployment_rate) * weather_factor
    data = []
    for day in range(1, days+1):
        for hour in range(24):
            rate = adjusted_base_rate
            if hour >= 20 or hour < 5:
                rate *= nighttime_factor
            crimes = np.random.poisson(rate)
            data.append({"day": day, "hour": hour, "crimes": crimes})
    df = pd.DataFrame(data)
    return df

def preprocess_synthetic_data(df):
    return df[['day', 'hour', 'crimes']]

def get_data(mode='real', **kwargs):
    current_dir = Path(__file__).parent.resolve()
    csv_path = current_dir / 'data' / 'incidents_part1_part2.csv'

    if mode == 'real':
        df = pd.read_csv(csv_path)
        df = preprocess_real_data(df)
    elif mode == 'synthetic':
        df = generate_synthetic_data(**kwargs)
        df = preprocess_synthetic_data(df)
    return df

############################
# Modeling
############################

def train_predictive_model(df):
    daily_data = df.groupby("day", as_index=False)["crimes"].sum()
    X = daily_data[['day']]
    y = daily_data['crimes']

    if len(daily_data) < 2:
        return None, None, daily_data

    split = int(len(daily_data)*0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    return model, mse, daily_data

############################
# Visualization
############################

def plot_crime_trend(df):
    daily_data = df.groupby("day", as_index=False)["crimes"].sum()
    fig, ax = plt.subplots()
    ax.plot(daily_data["day"], daily_data["crimes"], marker='o')
    ax.set_title("Crime Trend Over Time")
    ax.set_xlabel("Day")
    ax.set_ylabel("Number of Crimes")
    ax.grid(True)
    st.pyplot(fig)

def plot_hourly_pattern(df):
    hourly = df.groupby("hour")["crimes"].mean().reset_index()
    fig, ax = plt.subplots()
    ax.bar(hourly["hour"], hourly["crimes"])
    ax.set_title("Average Crimes by Hour of Day")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Average Number of Crimes")
    ax.grid(True)
    st.pyplot(fig)

############################
# Streamlit App
############################

def main():
    st.title("Crime Simulation and Prediction Dashboard")

    st.markdown("""
    This application allows you to:
    1. Load and analyze **real historical crime data**.
    2. Generate **synthetic crime data** with adjustable parameters.
    3. Train a simple predictive model to forecast future crime counts.

    **Note:**  
    - For real data, `dispatch_date` (or `dispatch_date_time`) and `hour` columns must be present and properly formatted.
    - Earliest dispatch_date corresponds to day=1.
    """)

    st.sidebar.header("Data Selection")
    data_mode = st.sidebar.selectbox("Select Data Mode", ["real", "synthetic"])

    if data_mode == "synthetic":
        st.sidebar.header("Synthetic Data Parameters")
        days = st.sidebar.slider("Number of Days", 
                                 min_value=2, max_value=365, value=30, step=1)
        
        base_rate = st.sidebar.number_input("Base Crime Rate (crimes/hour)", 
                                            min_value=1,
                                            value=5)
        
        nighttime_factor = st.sidebar.number_input("Nighttime Factor", 
                                                   min_value=1, max_value=5, value=2)
        
        unemployment_rate = st.sidebar.slider("Unemployment Rate", 
                                              0.0, 1.0, 0.05, step=0.01)
        
        weather_factor = st.sidebar.slider("Weather Factor", 
                                           0.5, 2.0, 1.0, step=0.1)
    else:
        days = None
        base_rate = None
        nighttime_factor = None
        unemployment_rate = None
        weather_factor = None

    if st.sidebar.button("Load Data"):
        df = get_data(mode=data_mode,
                      days=days,
                      base_rate=base_rate,
                      nighttime_factor=nighttime_factor,
                      unemployment_rate=unemployment_rate,
                      weather_factor=weather_factor)
        
        if df.empty:
            st.write("No valid data extracted from the CSV. Please check column names and date/hour formatting.")
            return

        st.subheader("Sample of Data")
        st.write(df.head(10))

        # Plot daily trend
        st.subheader("Crime Trend Over Time")
        plot_crime_trend(df)

        # Hourly pattern plot
        st.subheader("Average Crimes by Hour of the Day")
        plot_hourly_pattern(df)

        # Predictive Model
        st.subheader("Predictive Modeling Results")
        model, mse, daily_data = train_predictive_model(df)

        if model is None:
            st.write("Not enough days to train the model (need at least 2 days).")
            return
        
        st.write(f"Mean Squared Error of the Prediction: {mse:.2f}")

        # Future Prediction: 25% of total_days (min 2)
        total_days = daily_data['day'].max()
        future_period = max(2, int(0.25 * total_days))
        last_day = daily_data["day"].max()
        future_days = pd.DataFrame({"day": range(last_day+1, last_day+1+future_period)})
        future_preds = model.predict(future_days[["day"]])

        fig_pred, ax = plt.subplots()
        ax.plot(daily_data["day"], daily_data["crimes"], label="Historical", marker='o')
        ax.plot(future_days["day"], future_preds, label="Predicted", linestyle="--")
        ax.set_title(f"Predicted Crime for the Next {future_period} Days")
        ax.set_xlabel("Day")
        ax.set_ylabel("Crimes")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig_pred)

        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Data as CSV",
            data=csv_data,
            file_name=f'crime_data_{data_mode}.csv',
            mime='text/csv'
        )

if __name__ == "__main__":
    import os
    import sys
    # Check if the script is being executed via `streamlit run`
    if len(sys.argv) > 1 and sys.argv[1] == "streamlit":
        main()
    else:
        # If not running through Streamlit, launch it
        script_path = os.path.abspath(__file__)
        os.system(f"streamlit run {os.path.abspath(__file__)} streamlit")
        sys.exit()