import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import contextily as ctx
import plotly.express as px
import folium
from folium.plugins import MarkerCluster
import plotly.graph_objects as go


def read_dataset():
    gdf = gpd.read_file("LB_2022_MIS_02262024_207_206552/LBGE81FL/LBGE81FL.shp")
    df = pd.read_csv("LB_2022_MIS_02262024_207_206552/LBGC81FL/LBGC81FL/LBGC81FL.csv")
    columns = list(df.columns)
    gdf_combined = pd.concat([gdf,df],axis=1)
    malaria_columns = ['DHSREGNA', 'Malaria_Incidence_2000', 'Malaria_Incidence_2005', 'Malaria_Incidence_2010', 'Malaria_Incidence_2015', 'Malaria_Incidence_2020']
    malaria_data = gdf_combined[malaria_columns]
    itn_columns = ['DHSREGNA', 'ITN_Coverage_2000','ITN_Coverage_2005','ITN_Coverage_2010','ITN_Coverage_2015','ITN_Coverage_2020']
    itn_data = gdf_combined[itn_columns]
    wet_days_columns = ['DHSREGNA', 'Wet_Days_2000','Wet_Days_2005','Wet_Days_2010','Wet_Days_2015','Wet_Days_2020']
    month_temp_columns = ['DHSREGNA', 'Temperature_January','Temperature_February','Temperature_March','Temperature_April',
                'Temperature_May','Temperature_June','Temperature_July','Temperature_August','Temperature_September','Temperature_October',
                'Temperature_November','Temperature_December']
    rainfall_columns = ['DHSREGNA', 'Rainfall_2000','Rainfall_2005','Rainfall_2010','Rainfall_2015','Rainfall_2020']
    return gdf_combined, malaria_data ,itn_data, columns, wet_days_columns, month_temp_columns, rainfall_columns



# Define your visualizations for each category
def visualization_category_1(data,year):
    st.header("Demography")
    # Add your visualization for category 1 here
    st.markdown("")
    
    if year == '2020':
        fig1 = px.scatter_mapbox(data, lon='LONGNUM', lat='LATNUM',size='UN_Population_Density_2020',color='URBAN_RURA',title='UN Population Density - 2020',
                                labels={'LATNUM': 'Latitude', 'LONGNUM': 'Longitude', 'UN_Population_Density_2020': 'UN Population Density 2020'},
                                center=dict(lat=data['LATNUM'].mean(), lon=data['LONGNUM'].mean()),
                                zoom=10,
                                mapbox_style="carto-positron",color_discrete_map={'U': 'blue', 'R': 'green'})
        
        # Display the Folium map and Plotly figure in Streamlit
        st.plotly_chart(fig1)
    elif year == '2015':
        fig1 = px.scatter_mapbox(data, lon='LONGNUM', lat='LATNUM',size='UN_Population_Density_2015',color='URBAN_RURA',title='UN Population Density - 2015',
                                labels={'LATNUM': 'Latitude', 'LONGNUM': 'Longitude', 'UN_Population_Density_2015': 'UN Population Density 2015'},
                                center=dict(lat=data['LATNUM'].mean(), lon=data['LONGNUM'].mean()),
                                zoom=10,
                                mapbox_style="carto-positron",color_discrete_map={'U': 'blue', 'R': 'green'})
        
        # Display the Folium map and Plotly figure in Streamlit
        st.plotly_chart(fig1)
    else:
        pass
        

    

def visualization_category_2(data,malaria_data,itn_data,columns):
    st.header("Health")
    # Add your visualization for category 2 here
    # Create a Folium map with a basemap
    with st.expander("About",expanded=False):
        st.markdown("Malaria Incidence : Number of clinical cases of Plasmodium falciparum malaria per person.")
        st.markdown("Malaria Prevalence : Parasite rate of plasmodium falciparum (PfPR) in children between the ages of 2 and 10 years old.")

    fig1 = px.scatter_mapbox(data, lon='LONGNUM', lat='LATNUM',size='Malaria_Incidence_2020',color='URBAN_RURA',title='Malaria Incidence in Urban and Rural areas - 2020',
                            labels={'LATNUM': 'Latitude', 'LONGNUM': 'Longitude', 'Malaria_Incidence_2020': 'Malaria Incidence'},
                            center=dict(lat=data['LATNUM'].mean(), lon=data['LONGNUM'].mean()),
                            zoom=10,
                            mapbox_style="carto-positron",color_discrete_map={'U': 'blue', 'R': 'green'})
    
    # Display the Folium map and Plotly figure in Streamlit
    st.plotly_chart(fig1)

    fig2 = px.scatter_mapbox(data, lon='LONGNUM', lat='LATNUM',size='Malaria_Prevalence_2020',color='URBAN_RURA',title='Malaria Prevalence in Urban and Rural areas - 2020',
                            labels={'LATNUM': 'Latitude', 'LONGNUM': 'Longitude', 'Malaria_Prevalence_2020': 'Malaria Prevalence'},
                            center=dict(lat=data['LATNUM'].mean(), lon=data['LONGNUM'].mean()),
                            zoom=10,
                            mapbox_style="carto-positron",color_discrete_map={'U': 'blue', 'R': 'green'})
    
    st.plotly_chart(fig2)
    
    with st.expander("Statistics",expanded=False):
        df = data.groupby('DHSREGNA')[['Malaria_Prevalence_2020','Malaria_Incidence_2020','ITN_Coverage_2020']].mean().sort_values(by='Malaria_Prevalence_2020',ascending=False)
        st.dataframe(df)
    with st.expander("Malaria incidence"):
        df1 = malaria_data.groupby('DHSREGNA').mean()
        st.dataframe(df1)
        malaria_data.groupby('DHSREGNA').mean().plot.line(figsize=(10,6),marker='o')
        st.pyplot(plt.gcf())
    
    with st.expander("ITN Coverage"):
        df2 = itn_data.groupby('DHSREGNA').mean()
        st.dataframe(df2)
        itn_data.groupby('DHSREGNA').mean().plot.line(figsize=(10,6),marker='o')
        st.pyplot(plt.gcf() )


    

def visualization_category_3(data,wet_days_columns, month_temp_columns, rainfall_columns):
    st.header("Climate")
    # Add your visualization for category 3 here
    with st.expander("Mean Wet Days"):
        st.markdown("Wet Days(Mean): The average number of days per month receiving ≥0.1 mm precipitation at the DHS survey cluster location.")
        df1 = data.groupby('DHSREGNA')[wet_days_columns].mean()
        st.dataframe(df1)
    with st.expander("Average Monthly temperature"):
        st.markdown("Average monthly temperature: Average temperature for months January to December in degrees Celsius.")
        df2 = data.groupby('DHSREGNA')[month_temp_columns].mean()
        st.dataframe(df2)
    with st.expander("Average annual rainfall"):
        st.markdown("The average annual rainfall at the DHS survey cluster location.")
        df3 = data.groupby('DHSREGNA')[rainfall_columns].mean()
        st.dataframe(df3)



def visualization_category_4(data):
    st.header("Environment")
    # Add your visualization for category 3 here

def visualization_category_5(data):
    st.header("Agriculture")
    st.subheader("Drought Episodes")
    st.markdown("The average number of drought episodes (categorized between 1 (low) and 10 (high)) at the DHS survey cluster location.")

    fig3 = px.scatter_mapbox(data['Drought_Episodes'], lon='LONGNUM', lat='LATNUM',color='URBAN_RURA',title='Drought episodes',
                            labels={'LATNUM': 'Latitude', 'LONGNUM': 'Longitude', 'Drought_Episodes': 'Drought episodes'},
                            center=dict(lat=data['LATNUM'].mean(), lon=data['LONGNUM'].mean()),
                            zoom=10,
                            mapbox_style="carto-positron",color_discrete_map={'U': 'blue', 'R': 'green'})
    
    st.plotly_chart(fig3)
    


def main():
    st.title("Liberia Statistics")

    data, malaria_data, itn_data, columns, wet_days_columns, month_temp_columns, rainfall_columns = read_dataset()

    # Create a sidebar for selecting categories
    category = st.sidebar.selectbox("Select Category", options = ["Demography", "Health", "Climate","Environment","Agriculture"])
    year = st.sidebar.selectbox("Select the year", options= ['2000', '2010', '2015', '2020'])
    region = st.sidebar.selectbox("Select the region",options = ['Greater Monrovia','South Eastern','South Western'])
    counties = st.sidebar.selectbox("Select the county", options = ['Boma','Grand Bassa'])
    # Display the selected visualization based on the chosen category
    if category == "Demography":
        visualization_category_1(data,year)
    elif category == "Health":
        visualization_category_2(data,malaria_data,itn_data,columns)
    elif category == "Climate":
        visualization_category_3(data,wet_days_columns, month_temp_columns, rainfall_columns)
    elif category == "Environment":
        visualization_category_4(data)
    elif category == "Agriculture":
        visualization_category_5(data)

    


if __name__=="__main__":
    main()



