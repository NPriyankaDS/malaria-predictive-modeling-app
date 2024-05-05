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
    
    st.title("Malaria Deaths predictor for Liberia")
    st.markdown("This application helps to predict the malaria deaths provided the inputs in the below form.")

    df = pd.read_csv("./data/precipitation_intervention_cases.csv")    

    model = model_load("./saved_models/RFR_MODEL_1.pkl")
    pipeline = pipeline_load("./saved_models/fitted_pipeline_MODEL_1.pkl")

    
    with st.form("input_form"):
        st.write("Fill in the values for below indicators.")
        # Add input fields for each feature
        rfh = st.number_input("10 day rainfall [mm]")
        rfh_avg = st.number_input("rainfall 1-month rolling aggregation [mm]")
        r1h = st.number_input("rainfall 3-month rolling aggregation [mm]")
        r1h_avg = st.number_input("rainfall long term average [mm]")
        r3h = st.number_input("rainfall 1-month rolling aggregation long term average [mm]")
        r3h_avg = st.number_input("rainfall 3-month rolling aggregation long term average [mm]")
        rfq = st.number_input("rainfall anomaly [%]")
        r1q = st.number_input("rainfall 1-month anomaly [%]")
        r3q = st.number_input("rainfall 3-month anomaly [%]")
        district_area_SQKM = st.number_input("Liberia district area in square kilometre")
        IRS_coverage_per_100_household = st.number_input("Indoor residual spraying coverage proportion per 100 households in the county")
        itn_access_per_100_people = st.number_input("Insecticide-treated net access proportion per 100 people in the county")
        itn_use_per_100_people = st.number_input("Insecticide-treated net use proportion per 100 people in the county")
        itn_userate_in_people_with_access = st.number_input("Insecticide-treated net use rate amongst those who have access to an ITN and sleeps under an ITN in the county")
        effective_treatment_per_100_cases = st.number_input("Effective treatment per 100 cases in the county")
        Cases_Value = st.number_input("Estimated malaria cases in the county in defined year")
		
        # Add input fields for categorical variables
        district = st.selectbox("District", df['District'].unique())
        county = st.selectbox("County", df['County'].unique()) 
        
        submitted = st.form_submit_button("Submit")
                  
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
            st.write(input_data)
            processed_data = pipeline.transform(input_data)
            # Make prediction
            prediction = model.predict(processed_data)

            # Display prediction
            st.write("The predicted number of malaria deaths:", prediction)
        
if __name__=="__main__":
    main()
