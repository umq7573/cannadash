# app.py

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import pydeck as pdk
import matplotlib as mpl
import requests
import re
import os
import logging
from sodapy import Socrata
from shapely.geometry import Point
from datetime import datetime
# --------------------------
# Configuration Variables
# --------------------------
dataset_identifier = "jz4z-kudi"
data_directory = './data'
districts_geojson_path = 'districts.geojson'
today_filename = f'Cannabis_Case_Updates_{datetime.today().strftime("%Y-%m-%d")}.csv'

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --------------------------
# Data Processing Functions
# --------------------------
def fetch_data(dataset_identifier, query):
    client = Socrata("data.cityofnewyork.us", 
                     st.secrets["app_token"], 
                     username=st.secrets["username"], 
                     password=st.secrets["password"], 
                     timeout=600)
    results = client.get(dataset_identifier, query=query)
    return pd.DataFrame.from_records(results)

def geocode_address(address):
    base_url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"address": address, "key": st.secrets["api_key"]}
    response = requests.get(base_url, params=params)
    response_json = response.json()
    if response_json['status'] == 'OK':
        location = response_json['results'][0]['geometry']['location']
        return location['lat'], location['lng']
    else:
        return None, None

def apply_geocoding(df, api_key, prior_df=None):
    api_geocode_count = 0
    prior_file_geocode_count = 0

    def get_coordinates(address):
        nonlocal prior_file_geocode_count, api_geocode_count
        if prior_df is not None:
            prior_address = prior_df[prior_df['full_address'] == address]
            if not prior_address.empty:
                prior_file_geocode_count += 1
                return prior_address.iloc[0]['latitude'], prior_address.iloc[0]['longitude']
        api_geocode_count += 1
        return geocode_address(address, api_key)

    df['full_address'] = df.apply(
        lambda row: f"{row['violation_location_house']} {row['violation_location_street_name']}, {row['violation_location_city']}, {row['violation_location_state_name']} {row['violation_location_zip_code']}", 
        axis=1
    )
    df['latitude'], df['longitude'] = zip(*df['full_address'].apply(get_coordinates))
    logging.info(f"Geocodes from prior file: {prior_file_geocode_count}")
    logging.info(f"Geocodes from API: {api_geocode_count}")
    return df

def get_most_recent_file(directory, pattern="Cannabis_Case_Updates_"):
    files = os.listdir(directory)
    date_pattern = re.compile(rf"{pattern}(\d{{4}}-\d{{2}}-\d{{2}}).csv")
    matched_files = [
        (datetime.strptime(date_pattern.match(file).group(1), '%Y-%m-%d'), file)
        for file in files if date_pattern.match(file)
    ]
    if matched_files:
        return max(matched_files)[1]
    return None

def add_districts(df, districts_geojson_path):
    gdf_districts = gpd.read_file(districts_geojson_path)
    geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    gdf = gpd.sjoin(gdf, gdf_districts[['CounDist', 'geometry']], how="left", predicate="within")
    gdf = gdf.rename(columns={'CounDist': 'district_number'})
    gdf = gdf.drop(columns=['geometry', 'index_right'])
    return gdf

def generate_main_summary(df):
    # Convert relevant columns to datetime and numeric
    df['hearing_date'] = pd.to_datetime(df['hearing_date'], errors='coerce')
    df['decision_date'] = pd.to_datetime(df['decision_date'], errors='coerce')
    df['violation_date'] = pd.to_datetime(df['violation_date'], errors='coerce')
    df['penalty_imposed'] = pd.to_numeric(df['penalty_imposed'], errors='coerce').fillna(0)
    df['total_violation_amount'] = pd.to_numeric(df['total_violation_amount'], errors='coerce').fillna(0)
    today = pd.Timestamp('today').floor('D')
    as_of_today = today.strftime('%Y-%m-%d')

    total_issuances = len(df)
    total_issuances_penalty = df['penalty_imposed'].sum()

    completed_cases = df[(df['hearing_status'] == 'HEARING COMPLETED') | (df['hearing_status'] == 'PAID IN FULL')]
    completed_cases_count = len(completed_cases)
    completed_cases_penalty = completed_cases['penalty_imposed'].sum()

    completed_violation_cases = completed_cases[completed_cases['hearing_result'] == 'IN VIOLATION']
    completed_violation_count = len(completed_violation_cases)
    completed_violation_penalty_due = completed_violation_cases[completed_violation_cases['compliance_status'] == 'Penalty Due']
    completed_violation_penalty_due_count = len(completed_violation_penalty_due)
    completed_violation_penalty_due_sum = completed_violation_penalty_due['penalty_imposed'].sum()

    completed_violation_all_terms_met = completed_violation_cases[completed_violation_cases['compliance_status'] == 'All Terms Met']
    completed_violation_all_terms_met_count = len(completed_violation_all_terms_met)
    completed_violation_all_terms_met_sum = completed_violation_all_terms_met['penalty_imposed'].sum()

    completed_dismissed_cases = completed_cases[completed_cases['hearing_result'] == 'DISMISSED']
    completed_dismissed_count = len(completed_dismissed_cases)
    completed_dismissed_sum = completed_dismissed_cases['total_violation_amount'].sum()

    defaulted_cases = df[df['hearing_status'] == 'DEFAULTED']
    defaulted_cases_count = len(defaulted_cases)
    defaulted_cases_penalty = defaulted_cases['penalty_imposed'].sum()

    unique_shops = (df['violation_location_house'] + ", " + df['violation_location_street_name'] + ", " + df['violation_location_city']).nunique()

    status_of_violations = (
        f"**As of {as_of_today}, there have been {total_issuances} total violations issued across {unique_shops} unique shops.**  \n\n"
        f"- {total_issuances - completed_cases_count - defaulted_cases_count} violations are currently pending adjudication.\n"
        f"- Of the {completed_cases_count + defaulted_cases_count} violations adjudicated:\n"
        f"  - {completed_violation_count} were found to be in violation (total penalties: ${completed_cases_penalty}):\n"
        f"    - Penalties remain unpaid for {completed_violation_penalty_due_count} violations (total: ${completed_violation_penalty_due_sum}).\n"
        f"    - Penalties were paid in {completed_violation_all_terms_met_count} cases (total: ${completed_violation_all_terms_met_sum}).\n"
        f"  - {completed_dismissed_count} violations were dismissed (total: ${completed_dismissed_sum}).\n"
        f"  - {defaulted_cases_count} violations resulted in default judgments (total: ${defaulted_cases_penalty}).\n"
    )
    return status_of_violations

def assign_biweek(date, start_date):
    days_since_start = (date - start_date).days
    biweek_num = days_since_start // 14
    biweek_start = start_date + pd.Timedelta(days=biweek_num * 14)
    return pd.Period(biweek_start, freq='2W')

def get_hearing_result(row):
    if pd.isna(row.get('hearing_result')) or row.get('hearing_result') in ['NONE', 'ADJOURNED']:
        return 'PAID IN FULL' if row.get('hearing_status') == 'PAID IN FULL' else 'RESULT PENDING'
    return row.get('hearing_result')

def plot_biweekly_chart(df):
    if df.empty:
        st.write("No data available for biweekly chart.")
        return plt.figure()
    df = df.copy()
    df['violation_date'] = pd.to_datetime(df['violation_date'], errors='coerce')
    df = df.sort_values('violation_date')
    start_date = df['violation_date'].min().to_period('W').start_time
    df['biweek'] = df['violation_date'].apply(lambda date: assign_biweek(date, start_date))
    df['hearing_result'] = df.apply(get_hearing_result, axis=1)
    biweekly_status = df.groupby(['biweek', 'hearing_result']).size().unstack(fill_value=0)
    if not biweekly_status.empty:
        biweekly_status = biweekly_status.iloc[:-1]  # remove incomplete period if any
    biweekly_status_pct = biweekly_status.div(biweekly_status.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(20, 10))
    biweekly_status_pct.plot(kind='bar', stacked=True, ax=ax)
    ax.set_title('Hearing Outcomes by Violation Issue Date (Biweekly)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Biweekly Period')
    ax.set_ylabel('Percentage')
    ax.legend(title='Hearing Result', bbox_to_anchor=(1.05, 1), loc='upper left')
    x_labels = [f"{p.start_time.strftime('%m-%d')} to {p.end_time.strftime('%m-%d')}" for p in biweekly_status_pct.index]
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f%%', label_type='center')
    total_violations = biweekly_status.sum(axis=1)
    for i, total in enumerate(total_violations):
        ax.text(i, 100, f'{total} violations', ha='center', va='bottom')
    ax.set_ylim(0, 110)
    plt.tight_layout()
    return fig

def generate_heatmap(df, districts_geojson_path):
    if df.empty:
        return None
    # Compute violation counts by district
    violation_counts = df.groupby('district_number').size().reset_index(name='violation_count')
    # Load districts geojson
    gdf_districts = gpd.read_file(districts_geojson_path)
    gdf_districts['district_number'] = gdf_districts['CounDist'].astype(str)
    violation_counts['district_number'] = violation_counts['district_number'].astype(str)
    merged = gdf_districts.merge(violation_counts, on='district_number', how='left')
    merged['violation_count'] = merged['violation_count'].fillna(0)
    max_count = merged['violation_count'].max() if merged['violation_count'].max() > 0 else 1

    # Create a Pydeck GeoJsonLayer using a fill color expression based on violation_count
    geojson = merged.to_json()
    layer = pdk.Layer(
        "GeoJsonLayer",
        data=geojson,
        opacity=0.8,
        stroked=True,
        filled=True,
        get_fill_color=f"[255, 255 - (255 * properties.violation_count / {max_count}), 0, 150]",
        get_line_color=[0, 0, 0],
        pickable=True,
    )
    view_state = pdk.ViewState(latitude=40.7128, longitude=-74.0060, zoom=9, pitch=0)
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={"text": "District: {district_number}\nViolations: {violation_count}"}
    )
    return deck

# --------------------------
# Data Loading (with caching)
# --------------------------
@st.cache_data(ttl=3600)
def get_data():
    query = """
    SELECT *
    WHERE issuing_agency = 'NYC SHERIFF'
    AND violation_date >= '2024-05-01'
    LIMIT 20000
    """
    df = fetch_data(dataset_identifier, query)
    most_recent_prior_file = get_most_recent_file(data_directory)
    if most_recent_prior_file:
        prior_df = pd.read_csv(os.path.join(data_directory, most_recent_prior_file))
        df = apply_geocoding(df, prior_df)
    else:
        df = apply_geocoding(df)
    df['violation_date'] = pd.to_datetime(df['violation_date'], errors='coerce')
    gdf = add_districts(df, districts_geojson_path)
    return df, gdf

# --------------------------
# Streamlit Dashboard
# --------------------------
def main():
    st.title("Cannabis Case Violations Dashboard")

    # Load data (cached)
    df, gdf = get_data()

    # District Picker in Sidebar
    unique_districts = sorted(gdf['district_number'].dropna().unique())
    selected_districts = st.sidebar.multiselect("Select District(s)", options=unique_districts, default=unique_districts)

    # Filter data based on selected districts
    if selected_districts:
        filtered_gdf = gdf[gdf['district_number'].isin(selected_districts)]
    else:
        filtered_gdf = gdf

    # Display Summary
    st.subheader("Summary")
    summary = generate_main_summary(filtered_gdf)
    st.markdown(summary)

    # Display Heatmap
    st.subheader("Heatmap of Violations by District")
    heatmap_chart = generate_heatmap(filtered_gdf, districts_geojson_path)
    if heatmap_chart:
        st.pydeck_chart(heatmap_chart)
    else:
        st.write("No heatmap available due to missing data.")

    # Display Biweekly Status Chart
    st.subheader("Biweekly Hearing Outcomes")
    fig = plot_biweekly_chart(filtered_gdf)
    st.pyplot(fig)

if __name__ == "__main__":
    main()
