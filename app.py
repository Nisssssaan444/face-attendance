import streamlit as st
import pandas as pd
import time
from datetime import datetime

# Get current date and timestamp
ts = time.time()
date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
timestamp = datetime.fromtimestamp(ts).strftime("%H:%M-%S")

# Function to load and display dataframe
def load_dataframe():
    try:
        df = pd.read_csv(f"Attendance/Attendance_{date}.csv")
        st.table(df)  # Displaying DataFrame using st.table
    except FileNotFoundError:
        st.write(f"No attendance data found for {date}")

# Simulated counter for demonstration
count = st.slider("Select count", 0, 100, 0)  # Use a slider to simulate count change

# Main application logic based on count value
if count == 0:
    st.write("Count is zero")
elif count % 3 == 0 and count % 5 == 0:
    st.write("FizzBuzz")
elif count % 3 == 0:
    st.write("Fizz")
elif count % 5 == 0:
    st.write("Buzz")
else:
    st.write(f"Count: {count}")

# Display the dataframe
load_dataframe()
