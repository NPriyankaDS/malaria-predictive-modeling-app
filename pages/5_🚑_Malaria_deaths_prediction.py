# -*- coding: utf-8 -*-
"""
Created on Sat May 04 2024

@author: N Priyanka 
"""

import os
import joblib
import streamlit as st
import pandas as pd

@st.cache_resource
def model_load(model_path):
    model = joblib.load(model_path)
    return model

@st.cache_resource
def pipeline_load(pipeline_path):
    pipeline = joblib.load(pipeline_path)
    return pipeline

def main():

    # Set page configuration
    st.set_page_config(page_title="Malaria deaths prediction app",
                    layout="wide",
                    page_icon=":ambulance:")
    
    st.title("Malaria Deaths predictor for Liberia :ambulance:")
    st.divider()
    st.markdown("*This application helps to predict the malaria deaths in Liberia at the District level.*")
    st.write("**Select the County and the District in the sidebar. Then fill in the values for below indicators.**")
    st.divider()

    df = pd.read_csv("./data/precipitation_intervention_cases.csv")    

    model = model_load("./saved_models/RFR_MODEL_1.pkl")
    pipeline = pipeline_load("./saved_models/fitted_pipeline_MODEL_1.pkl")

    # Dynamic dropdown for selecting district
    county = st.sidebar.selectbox("**Select County**", sorted(df['County'].unique()))

    # Get county and area based on selected district
    district_in_counties = df[df['County'] == county]['District'].unique()
    district = st.sidebar.selectbox("**Select District**", district_in_counties)
    district_area_SQKM = df[df['District'] == district]['district_area_SQKM'].iloc[0]
    st.write(f"**Information**: The area of the the District **{district}** in **{county}** is **{district_area_SQKM} square kilometer.**")

    col1, col2 , col3  = st.columns(3)

   # Add input fields for each feature
    with col1:
        rfh = st.number_input("**10 day rainfall [mm]**",min_value=0.0)
    with col2:
        rfh_avg = st.number_input("**Rainfall long term average [mm]**",min_value=0.0)
    with col3:
        r1h = st.number_input("**Rainfall 1-month rolling aggregation [mm]**",min_value=0.0)
    with col1:
        r1h_avg = st.number_input("**Rainfall 1-month rolling aggregation long term average [mm]**",min_value=0.0)
    with col2:
        r3h = st.number_input("**Rainfall 3-month rolling aggregation [mm]**",min_value=0.0)
    with col3:
        r3h_avg = st.number_input("**Rainfall 3-month rolling aggregation long term average [mm]**",min_value=0.0)
    with col1:
        rfq = st.number_input("**Rainfall anomaly [%]**",min_value=0.0)
    with col2:
        r1q = st.number_input("**Rainfall 1-month anomaly [%]**",min_value=0.0)
    with col3:
        r3q = st.number_input("**Rainfall 3-month anomaly [%]**",min_value=0.0)
    with col1:
        IRS_coverage_per_100_household = st.number_input("**Indoor residual spraying coverage proportion per 100 households in the county**",min_value=0.0,max_value=100.0, step=0.1)
    with col2:
        itn_access_per_100_people = st.number_input("**Insecticide-treated net access proportion per 100 people in the county**",min_value=0.0,max_value=100.0, step=0.1)
    with col3:
        itn_use_per_100_people = st.number_input("**Insecticide-treated net use proportion per 100 people in the county**",min_value=0.0,max_value=100.0, step=0.1)
    with col1:
        itn_userate_in_people_with_access = st.number_input("**Insecticide-treated net use rate amongst those who have access to an ITN and sleeps under an ITN in the county**",min_value=0.0, max_value=100.0, step=0.1)
    with col2:
        effective_treatment_per_100_cases = st.number_input("**Effective treatment per 100 cases in the county**",min_value=0.0,max_value=100.0, step=0.1)
    with col3:
        Cases_Value = st.number_input("**Estimated malaria cases in the county in defined year**",min_value=0.0)

    st.divider()
    st.write("*Click on Submit button to submit your entered data for prediction.*")        
    submitted = st.button("**Submit**")
                  
    if submitted:
        input_data = pd.DataFrame({
            'rfh': [rfh],
            'rfh_avg': [rfh_avg],
            'r1h': [r1h],
            'r1h_avg': [r1h_avg],
            'r3h': [r3h],
            'r3h_avg' : [r3h_avg],
            'rfq': [rfq],
            'r1q': [r1q],
            'r3q': [r3q],
            'district_area_SQKM': [district_area_SQKM],
            'IRS_coverage_per_100_household': [IRS_coverage_per_100_household],
            'itn_access_per_100_people': [itn_access_per_100_people],
            'itn_use_per_100_people':[itn_use_per_100_people],
            'itn_userate_in_people_with_access': [itn_userate_in_people_with_access],
            'effective_treatment_per_100_cases': [effective_treatment_per_100_cases],
            'Cases_Value': [Cases_Value],
            'District': [district],
            'County': [county]
            })

        st.write("**You have submitted the below data.**")
        st.write(input_data)
        processed_data = pipeline.transform(input_data)
        # Make prediction
        prediction = model.predict(processed_data)

        # Display prediction
        st.write("**The number of malaria cases:** ",Cases_Value)
        st.info(f"**The predicted number of malaria deaths:** {prediction[0]}")
    
if __name__=="__main__":
    main()
