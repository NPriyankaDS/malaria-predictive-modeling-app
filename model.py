import streamlit as st
import numpy as np
import pandas as pd
from pycaret.regression import *

def main():
    st.title("Malaria Forecasting for Liberia")
    st.markdowm("This application helps in predicting malaria for Liberia")

    with st.sidebar():
        st.input("IRS_coverage_per_100_househol")
        st.input("itn_access_per_100_people")
        st.input("effective_treatment_per_100_cases")

        button = st.button("Predict")

    if button:
        pass


if __name__=="__main__":
    main()