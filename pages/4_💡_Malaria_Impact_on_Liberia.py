"""
Created on Apr 2024

@author: Maria Loureiro
"""
from page_utils import font_modifier, display_image
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from streamlit_extras.add_vertical_space import add_vertical_space
import plotly.express as px

#font_modifier.make_font_poppins()

st.markdown("# :bulb: Impact of Malaria in Liberia")

st.markdown("""

Malaria remains a significant public health concern in Liberia, impacting communities across its counties.
This page provides insights into the prevalence and toll of malaria over the years and across counties in Liberia, helping to inform prevention strategies and resource allocation.

**Exploring Malaria Trends:**

- The bar charts below showcase the median number of malaria cases and malaria-related deaths per county. Use the dropdown menus to filter by year or county for a more granular view.

**Understanding County Dynamics:**

- The scatter chart illustrates the relationship between malaria cases and related deaths, offering insights into the severity of the disease in different regions.

**Temporal Analysis:**
            
- The line charts depict the trend of malaria cases and deaths over the years, highlighting any fluctuations or patterns that may emerge.

Together, these visualizations aim to empower stakeholders with the information needed to combat malaria effectively, ultimately working towards a healthier Liberia.
""")

#Import and filter dataset
df = pd.read_csv("./data/pivot_data_Liberia_malariapercounty.csv")
df.rename(columns={'Cases_Value': 'Malaria Cases'}, inplace=True)
df.rename(columns={'Deaths_Value': 'Malaria related Deaths'}, inplace=True)
df.rename(columns={'Name': 'County'}, inplace=True)
df.Year=df.Year.astype(str)

add_vertical_space(2)

select_years=st.multiselect('Filter by year', options=['All']+list(df.Year.unique()),default='All')
add_vertical_space(2)
if 'All' in select_years:
    df_to_plot=df.groupby('County')[['Malaria Cases','Malaria related Deaths']].median()
    df_to_plot = df_to_plot.reset_index()
else:
    df_to_plot=df[df.Year.isin(select_years)].groupby('County')[['Malaria Cases','Malaria related Deaths']].median()
    df_to_plot = df_to_plot.reset_index()
st.bar_chart(data=df_to_plot,x="County", y="Malaria Cases", use_container_width=True)
st.bar_chart(data=df_to_plot,x="County", y="Malaria related Deaths", use_container_width=True)

fig = px.scatter(df_to_plot, x="Malaria Cases", y="Malaria related Deaths", title="Scatter Plot")
st.plotly_chart(fig)
#st.scatter_chart(data=df_to_plot,x="Malaria Cases", y="Malaria related Deaths", use_container_width=True)

#Show cases and deaths by county
select_county=st.multiselect('Filter by county', options=['All']+list(df.County.unique()),default='All')

if 'All' in select_county:
    df_grouped=df.groupby(['Year'])[['Malaria Cases','Malaria related Deaths']].sum()
    df_grouped.reset_index(inplace=True)
    df_to_plot=df_grouped.copy()
else:
    df_grouped=df[df.County.isin(select_county)].groupby(['Year'])[['Malaria Cases','Malaria related Deaths']].sum()
    df_grouped.reset_index(inplace=True)
    df_to_plot=df_grouped.copy()

add_vertical_space(4)

st.line_chart(data=df_to_plot, x='Year', y='Malaria Cases', use_container_width=True)

st.line_chart(data=df_to_plot, x='Year', y='Malaria related Deaths', use_container_width=True)

