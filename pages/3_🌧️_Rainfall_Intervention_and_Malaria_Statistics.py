#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 16:14:00 2024

@author: viviensiew
"""

import streamlit as st
import numpy as np
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# load constants
# =============================================================================
# url = "https://raw.githubusercontent.com/M4riaLoureiro/malaria-predictive-modeling-app/eda_vivien/pages/data/precipitation_intervention_cases.csv"
# meta_url = "https://raw.githubusercontent.com/M4riaLoureiro/malaria-predictive-modeling-app/eda_vivien/pages/data/precipitation_intervention_cases_metadata.csv"
# =============================================================================
data_folder = "./data/"
url = data_folder + "precipitation_intervention_cases.csv"
meta_url = data_folder + "precipitation_intervention_cases_metadata.csv"
video1_url = data_folder + "liberia_malaria_cases.mp4"

sns.set_style("whitegrid")
    
# takes in precipitation dataset and downsample to annual level dataset
def format_dataset():
    df_conso = pd.read_csv(url, parse_dates=['date'])
    # Generate annual values for rainfall data: r1h(rainfall 1 month rolling agg), rfh (10 day rainfall), rfq (rainfall anomaly)
    group_county = ["County", "Year"]
    group_agg = [np.mean, np.min, np.max]
    df = df_conso.groupby(by=group_county).agg(r1h_min = pd.NamedAgg(column="r1h", aggfunc=np.min),
                                                 r1h_max = pd.NamedAgg(column="r1h", aggfunc=np.max),
                                                 r1h_mean = pd.NamedAgg(column="r1h", aggfunc=np.mean),
                                                 rfh_min = pd.NamedAgg(column="rfh", aggfunc=np.min),
                                                 rfh_max = pd.NamedAgg(column="rfh", aggfunc=np.max),
                                                 rfh_mean = pd.NamedAgg(column="rfh", aggfunc=np.mean),
                                                 rfq_min = pd.NamedAgg(column="rfq", aggfunc=np.min),
                                                 rfq_max = pd.NamedAgg(column="rfq", aggfunc=np.max),
                                                 rfq_mean = pd.NamedAgg(column="rfq", aggfunc=np.mean))
    
    df.reset_index(inplace=True)
    df_conso_annual = df_conso[["County", "Year", "IRS_coverage_per_100_household", "itn_access_per_100_people", "itn_use_per_100_people",
                           "itn_userate_in_people_with_access", "effective_treatment_per_100_cases", "Cases_Value", "Deaths_Value"]]
    df_conso_annual = df_conso_annual.drop_duplicates(subset=["County", "Year"], keep="first")
    df = pd.merge(df, df_conso_annual, how='left', on=["County", "Year"])
    return df

# format time series dataset for death values
def formatTimeSeriesDf(df):
    df['date'] = pd.to_datetime(df['date'])
    return df.groupby(['date', 'County']).agg({'Deaths_Value': 'sum'}).reset_index()
    
# plot annual trends in precipitation data
def plotTrendsWithPlotly(df, variable, variable_text):
    title = f"{variable_text} trend from year 2010 - 2020 by county"
    fig = px.line(df, x="Year", y=variable, title=title, color="County", markers=True)
    return fig

# plot 10-days trends in precipitation data
def plotTrendsWithSeaborn(data, variable, variable_text):
    fig = plt.figure(figsize=(8, 4))
    sns.histplot(data[variable], kde=True, bins=30)
    plt.title(f'Distribution of {variable_text.lower()}')
    plt.xlabel(variable)
    plt.ylabel('Frequency')
    plt.tight_layout()
    return fig

def plotTimeSeriesDeaths(df, county):    
    fig = plt.figure(figsize=(8, 4))   
    county_data = df[df['County'] == county]
    plt.plot(county_data['date'], county_data['Deaths_Value'], label=county, marker='o', linestyle='-', markersize=2, linewidth=0.5)
    plt.title(f'Time Series of Death Values in {county}')
    plt.xlabel('Date')
    plt.ylabel('Aggregated Death Values')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

    
df = pd.read_csv(url)
df_annual = format_dataset()
county_deaths_time_series = formatTimeSeriesDf(df)
meta_df = pd.read_csv(meta_url)

#Format form 1
variables_to_plot = ['rfh', 'r1h', 'r3h', 'Cases_Value', 'Deaths_Value']
dict_meta = dict(zip(meta_df['Column'].to_list(), meta_df['Column Description'].to_list()))
dict_variables = { v:dict_meta[v] for v in variables_to_plot}
#Format form 2
annual_columns = df_annual.columns.tolist()
annual_columns.remove("County")
annual_columns.remove("Year")
dict_annual = {a:dict_meta[a] for a in annual_columns}
#Format form 3
unique_counties = county_deaths_time_series['County'].unique()

min_year = df_annual['Year'].min()
max_year = df_annual['Year'].max()

# Main section
header_text1 = "Rainfall, Intervention and Malaria statistics in Liberia,"
header_text2 = f" {min_year} to {max_year}"
purpose_text = "We aim to provide users a greater understanding of factors, such as rainfall and interventions from health organizations, which covers IRS (Indoor Residual Spraying), ITN (Insecticide Treated Net) and medical treatments, to show how these factors influence the number of malaria cases and deaths across Liberia."
video_text = "The following time lapse video highlights the number of malaria cases across various counties, from highest to lowest. As such, we hope this video could provide users some useful insights on the counties that are in dire need of intervention from health authorities in their effort to control and combat malaria."
st.header(header_text1 + header_text2 + ":rain_cloud:")
st.write(purpose_text)
st.write(video_text)
with st.expander("Watch video:", expanded=True):
    video_file = open(video1_url, 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)

st.caption("Source: [Malaria Atlas Project](https://malariaatlas.org/), [Humanitarian Data Exchange(HDX) Liberia Rainfall Indicators at Subnational Level](https://data.humdata.org/dataset/a1f60b8a-51ff-4ee7-87ab-897983714595)")

st.divider()

# Section to display sidebar plots

st.write("This section displays charts containing information on rainfall, intervention, malaria cases and deaths.")
st.write(":arrow_left: To view charts, select the information required on the sidebar and press the view button.")

st.sidebar.subheader("Liberia statistics on rainfall, intervention, cases and deaths.")
with st.sidebar.form(key='multiselect_1'):
    st.markdown("Distribution of rainfall measurements and malaria cases/deaths")
    variables = st.multiselect('Select measurement(s):', options=list(dict_variables.keys()), 
                               format_func = lambda x: dict_variables[x])
    submit_1 = st.form_submit_button(label='View distribution chart')

with st.sidebar.form(key='multiselect_2'):
    st.markdown("Annual trends of rainfall, intervention, cases and deaths in various counties")
    annual_features = st.multiselect('Select annual measurement(s):', options=list(dict_annual.keys()), 
                                     format_func = lambda x: dict_annual[x])
    submit_2 = st.form_submit_button(label='View annual trends chart')

with st.sidebar.form(key='multiselect_3'):
    st.markdown("Trend of Malaria Deaths in various counties")
    ts_counties = st.multiselect('Select counties:', unique_counties)
    submit_3 = st.form_submit_button(label='View trends chart')

if submit_1 and variables:
    st.subheader("Distribution of various rainfall measurements and malaria cases/deaths")
    for v in variables:
        plot = plotTrendsWithSeaborn(df, v, dict_variables[v])
        st.pyplot(plot.get_figure())
        
if submit_2 and annual_features:
    for f in annual_features:
        fig = plotTrendsWithPlotly(df_annual, f, dict_annual[f])  
        st.plotly_chart(fig, use_container_width=False, sharing="streamlit", theme="streamlit")

if submit_3 and ts_counties:
    for county in ts_counties:
        plot = plotTimeSeriesDeaths(county_deaths_time_series, county)
        st.pyplot(plot.get_figure())


