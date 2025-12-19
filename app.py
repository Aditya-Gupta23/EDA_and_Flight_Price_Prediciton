import streamlit as st
import pandas as pd
import joblib
from datetime import datetime, time
import random

# ----------------------------
# 🔹 App Configuration
# ----------------------------
st.set_page_config(page_title="Flight Price Predictor", layout="wide")
st.title("✈️ Flight Price Prediction Portal")
st.write("Compare flight options using different ML models — just like IRCTC flight search!")

# ----------------------------
# 🔹 Model Selector
# ----------------------------
model_choice = st.selectbox(
    "Select Prediction Model",
    ["Random Forest (default)", "XGBoost", "Neural Network"],
    index=0,
    help="Compare predictions from different models"
)

# Load corresponding model
model_files = {
    "Random Forest (default)": "pipeline_rf.joblib",
    "XGBoost": "pipeline_xgb.joblib",
    "Neural Network": "pipeline_nn.joblib"
}

try:
    model = joblib.load(model_files[model_choice])
    st.success(f"✅ Loaded {model_choice} successfully.")
except Exception as e:
    st.error(f"❌ Failed to load {model_choice}: {e}")
    st.stop()

# ----------------------------
# 🔹 Sidebar Filters
# ----------------------------
st.sidebar.header("🔍 Filter Your Search")

airlines = ['Air Asia', 'Air India', 'GoAir', 'IndiGo', 'Jet Airways', 'Multiple carriers', 'SpiceJet', 'Vistara']
sources = ['Select Source', 'Banglore', 'Chennai', 'Delhi', 'Kolkata', 'Mumbai']
destinations = ['Select Destination', 'Banglore', 'Cochin', 'Delhi', 'Hyderabad', 'Kolkata']
add_info = ['1 Long layover', 'Change airports', 'In-flight meal not included', 'No info']

source = st.sidebar.selectbox("Source", sources, index=0)
destination = st.sidebar.selectbox("Destination", destinations, index=0)

# Validation
source_error = source == 'Select Source'
destination_error = destination == 'Select Destination'
same_location_error = source == destination and not source_error and not destination_error

total_stops_filter = st.sidebar.multiselect("Total Stops", options=[0, 1, 2, 3, 4], default=[0, 1, 2])
airline_filter = st.sidebar.multiselect("Airlines", options=airlines, default=airlines)
time_slot = st.sidebar.radio("Departure Time", ["00–06", "06–12", "12–18", "18–24"], index=1)
price_range = st.sidebar.slider("Price Range (₹)", 2000, 20000, (4000, 12000))

# ----------------------------
# 🔹 Main Input Section
# ----------------------------
st.subheader("🧳 Enter Your Travel Details")

col1, col2, col3 = st.columns(3)
with col1:
    journey_date = st.date_input("Journey Date", datetime.today())
with col2:
    dep_time = st.time_input("Departure Time", value=time(9, 0))
with col3:
    arr_time = st.time_input("Arrival Time", value=time(11, 30))

dep_hour, dep_min = dep_time.hour, dep_time.minute
arr_hour, arr_min = arr_time.hour, arr_time.minute
duration = ((arr_hour * 60 + arr_min) - (dep_hour * 60 + dep_min)) % (24 * 60)
journey_day, journey_month, journey_year = journey_date.day, journey_date.month, journey_date.year

route_segments = st.number_input("Route Segments", min_value=1, max_value=10, step=1)
add_info_smallcat = st.selectbox("Additional Info", add_info)

# ----------------------------
# 🔹 Generate Mock Flights
# ----------------------------
def generate_flight_rows():
    rows = []
    for a in airlines:
        if a not in airline_filter:
            continue
        total_stops = random.choice(total_stops_filter)
        dep_offset = random.randint(-60, 60)
        dur_var = duration + random.randint(-30, 60)
        dep_h = (dep_hour + dep_offset // 60) % 24
        dep_m = (dep_min + dep_offset % 60) % 60
        arr_h = (dep_h + (dur_var // 60)) % 24
        arr_m = (dep_m + (dur_var % 60)) % 60

        input_data = pd.DataFrame({
            'journey_day': [journey_day],
            'journey_month': [journey_month],
            'journey_year': [journey_year],
            'dep_hour': [dep_h],
            'dep_min': [dep_m],
            'arr_hour': [arr_h],
            'arr_min': [arr_m],
            'duration_mins': [dur_var],
            'total_stops_num': [total_stops],
            'route_segments': [route_segments],
            'Airline': [a],
            'Source': [source],
            'Destination': [destination],
            'add_info_smallcat': [add_info_smallcat]
        })

        price = model.predict(input_data)[0]
        rows.append({
            "Airline": a,
            "Departure": f"{dep_h:02d}:{dep_m:02d}",
            "Arrival": f"{arr_h:02d}:{arr_m:02d}",
            "Duration": f"{dur_var // 60}h {dur_var % 60}m",
            "Stops": total_stops,
            "Price": round(price)
        })

    df = pd.DataFrame(rows)
    df = df[(df["Price"] >= price_range[0]) & (df["Price"] <= price_range[1])]
    return df.sort_values("Price").reset_index(drop=True)

# ----------------------------
# 🔹 Search & Display
# ----------------------------
if st.button("🔎 Search Flights"):
    if source_error or destination_error:
        st.error("Please select both Source and Destination before searching.")
    elif same_location_error:
        st.error("Source and Destination cannot be the same.")
    else:
        flights = generate_flight_rows()
        if flights.empty:
            st.warning("No flights found for the selected filters.")
        else:
            st.success(f"Showing {min(10, len(flights))} of {len(flights)} flights using {model_choice}:")

            # Display cards
            n_show = min(10, len(flights))
            for i, row in flights.head(n_show).iterrows():
                st.markdown(
                    f"""
                    <div style="background-color:#393E46;padding:16px;border-radius:10px;margin-bottom:10px;color:white;">
                        <h4>✈️ {row['Airline']} — ₹{row['Price']:,}</h4>
                        <b>Departure:</b> {row['Departure']} |
                        <b>Arrival:</b> {row['Arrival']} |
                        <b>Duration:</b> {row['Duration']} |
                        <b>Stops:</b> {row['Stops']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            if len(flights) > n_show:
                st.info(f"Showing top {n_show}. Use filters or adjust criteria to narrow results.")
