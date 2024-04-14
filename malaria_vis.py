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
import requests
import re
import time
from shapely.geometry import Point
import altair as alt
from streamlit_folium import folium_static


def read_health_facilities():
    #Loading the data
    points_path = "data/hotosm_lbr_health_facilities_points_geojson.geojson"
    polygons_path = "data/hotosm_lbr_health_facilities_polygons_geojson.geojson"
    points_gdf = gpd.read_file(points_path)
    polygons_gdf = gpd.read_file(polygons_path)
    # Merge the two datasets into a single GeoDataFrame
    combined_gdf = pd.concat([points_gdf, polygons_gdf], ignore_index=True)
    #Converting Polygons to Representative Points
    combined_gdf['point_geometry'] = combined_gdf.apply(lambda row: row['geometry'].representative_point() if row['geometry'].geom_type == 'Polygon' else row['geometry'],axis=1)
    return combined_gdf

def read_dataset():
    #Loading the data for LBGE81FL and LBGC81FL
    gdf = gpd.read_file("data/LB_2022_MIS_02262024_207_206552/LBGE81FL/LBGE81FL.shp")
    df = pd.read_csv("data/LB_2022_MIS_02262024_207_206552/LBGC81FL/LBGC81FL/LBGC81FL.csv")
    

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


def preprocess_health_facilities():
    combined_gdf = read_health_facilities()
    # Fill missing values in the 'name' column with 'Unknown'
    combined_gdf['name'] = combined_gdf['name'].fillna('Unknown')
    # Fill missing values in the 'amenity' column with 'Unknown'
    combined_gdf['amenity'] = combined_gdf['amenity'].fillna('Unknown')
    # Applying the logic to infer missing 'building' values from 'amenity'
    conditions_1 = [
        (combined_gdf['building'].isna()) & (combined_gdf['amenity'] == 'clinic'),
        (combined_gdf['building'].isna()) & (combined_gdf['amenity'] == 'hospital'),
    ]
    # The choices must correspond to the conditions order
    choices = ['clinic', 'hospital']  # Corresponding building types

    # apply these conditions and choices
    combined_gdf['building'] = np.select(conditions_1, choices, default=combined_gdf['building'])
    # Conditions for replacing "yes" in 'building' based on 'amenity' values
    conditions_2 = [
        (combined_gdf['building'] == 'yes') & (combined_gdf['amenity'] == 'clinic'),
        (combined_gdf['building'] == 'yes') & (combined_gdf['amenity'] == 'hospital'),
    ]
    # Apply the conditions and choices to infer 'building' values
    combined_gdf['building'] = np.select(conditions_2, choices, default=combined_gdf['building'])
    # Fill missing values in the 'building' column with 'Unknown'
    combined_gdf['building'] = combined_gdf['building'].fillna('Unknown')
    # Categorizing missing values in 'healthcare' as 'Unknown'
    combined_gdf['healthcare'] = combined_gdf['healthcare'].fillna('Unknown')

    # Categorizing missing values in 'healthcare:speciality' as 'Unknown'
    combined_gdf['healthcare:speciality'] = combined_gdf['healthcare:speciality'].fillna('Unknown')


    # Filter rows where 'addr:city' is missing
    missing_city_gdf = combined_gdf[combined_gdf['addr:city'].isna()]

    # Categorizing missing values in 'healthcare:speciality' as 'Unknown'
    combined_gdf['operator:type'] = combined_gdf['operator:type'].fillna('Unknown')

    # Creating a mapping of variations to standardized names
    source_mapping = {
        "Ministry of Health and National Public Health Institute of Liberia": ["Ministry of Health and National Public Health Institute of Liberia", "Ministry of Health", "MOH & NPHIL", "National Public Health Institute of Liberia (NPHIL)", "Ministry of Health and national :Public Institute", "Ministry of Health and National Public Institite of Liberia", "Ministry of Health & Natinal Public Health Institude", "Ministry of Health and National Public Health Institute", "Ministry of Health and National Public Institute of Liberia", "MoH & NPHIL", "Ministry of Health & National Public Health Institutue of Liberia", "Ministry of Health & National Public Health Institute of Liberia", "MOH&NPHIL"],
        "Red Cross Field Survey": ["Red Cross Field Survey"],
        "Open Cities Monrovia - HOT Field Survey": ["Open Cities Monrovia - HOT Field Survey", "HOTOSM-DAI-LEGIT Field Survey"],
        "WNA Hub - TFiRL project": ["WNA Hub - TFiRL project", "WNA Hub -TFiRL project"],
        "UNMEER": [ "UNMEER ;"],
        "GNS": ["GNS", "GNS;Personal knowledge"],
        "Bing": ["Bing", "bing"],
    }

    # Function to standardize source names
    def standardize_source_name(source):
        for standardized, variations in source_mapping.items():
            if source in variations:
                return standardized
        return source

    # Apply the standardization function to the 'source' column
    combined_gdf['source'] = combined_gdf['source'].apply(standardize_source_name)
    # Categorizing missing values in 'source' as 'Unknown'
    combined_gdf['source'] = combined_gdf['source'].fillna('Unknown')
    def preprocess_source(source):
        # Regular expression to match URLs
        url_pattern = r'https?://\S+|www\.\S+'
        # Replace URLs with an empty string
        cleaned_source = re.sub(url_pattern, '', source)

        # Remove semicolons and strip trailing spaces
        cleaned_source = cleaned_source.replace(';', '').strip()

        return cleaned_source

    # Apply the enhanced preprocessing function to the 'source' column
    combined_gdf['source'] = combined_gdf['source'].apply(preprocess_source)
    # Dropping the 'name:en', 'capacity:persons', and 'addr:full' columns
    combined_gdf = combined_gdf.drop(columns=['name:en', 'capacity:persons', 'addr:full'])
    return combined_gdf


# Define your visualizations for each category
def visualization_category_1(data,year,subcategory):
    if subcategory == "Population": 
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
        

    

def visualization_category_2(data,malaria_data,itn_data,columns,subcategory):
    
    # Add your visualization for category 2 here
    # Create a Folium map with a basemap
    with st.expander("About",expanded=False):
        st.markdown("Malaria Incidence : Number of clinical cases of Plasmodium falciparum malaria per person.")
        st.markdown("Malaria Prevalence : Parasite rate of plasmodium falciparum (PfPR) in children between the ages of 2 and 10 years old.")
    with st.expander("Statistics for the year 2020",expanded=False):
        df = data.groupby('DHSREGNA')[['Malaria_Prevalence_2020','Malaria_Incidence_2020','ITN_Coverage_2020']].mean().sort_values(by='Malaria_Prevalence_2020',ascending=False)
        st.dataframe(df)

    if subcategory =="Malaria Incidence":
        fig1 = px.scatter_mapbox(data, lon='LONGNUM', lat='LATNUM',size='Malaria_Incidence_2020',color='URBAN_RURA',title='Malaria Incidence in Urban and Rural areas - 2020',
                                labels={'LATNUM': 'Latitude', 'LONGNUM': 'Longitude', 'Malaria_Incidence_2020': 'Malaria Incidence'},
                                center=dict(lat=data['LATNUM'].mean(), lon=data['LONGNUM'].mean()),
                                zoom=10,
                                mapbox_style="carto-positron",color_discrete_map={'U': 'blue', 'R': 'green'})
        
        # Display the Folium map and Plotly figure in Streamlit
        st.plotly_chart(fig1)
        with st.expander("Malaria incidence"):
            df1 = malaria_data.groupby('DHSREGNA').mean()
            st.dataframe(df1)
            malaria_data.groupby('DHSREGNA').mean().plot.line(figsize=(10,6),marker='o')
            st.pyplot(plt.gcf())
    
    elif subcategory == "Malaria Prevalence":
        fig2 = px.scatter_mapbox(data, lon='LONGNUM', lat='LATNUM',size='Malaria_Prevalence_2020',color='URBAN_RURA',title='Malaria Prevalence in Urban and Rural areas - 2020',
                                labels={'LATNUM': 'Latitude', 'LONGNUM': 'Longitude', 'Malaria_Prevalence_2020': 'Malaria Prevalence'},
                                center=dict(lat=data['LATNUM'].mean(), lon=data['LONGNUM'].mean()),
                                zoom=10,
                                mapbox_style="carto-positron",color_discrete_map={'U': 'blue', 'R': 'green'})
        
        st.plotly_chart(fig2)

    elif subcategory == "ITN Coverage": 
        with st.expander("ITN Coverage"):
            df2 = itn_data.groupby('DHSREGNA').mean()
            st.dataframe(df2)
            itn_data.groupby('DHSREGNA').mean().plot.line(figsize=(10,6),marker='o')
            st.pyplot(plt.gcf())

    elif subcategory == "Health facilities":
        with st.expander("Health facilities"):
            df3 = preprocess_health_facilities()
            
        st.subheader("Spatial distribution of all amenities")
        # Extract longitude and latitude from the Point geometry
        df3['longitude'] = df3.point_geometry.x
        df3['latitude'] = df3.point_geometry.y

        # Create a filter for healthcare types
        selected_types = st.multiselect('Select healthcare types', df3['healthcare'].unique())

        # Filter the GeoDataFrame based on selected types
        filtered_df = df3[df3['healthcare'].isin(selected_types)]

        # Create a Folium map
        m = folium.Map(location=[df3['latitude'].mean(), df3['longitude'].mean()], zoom_start=10)

        # Define colors for different healthcare types
        colors = {'pharmacy': 'blue', 'clinic': 'green', 'hospital': 'red', 'doctor': 'orange', 'dentist': 'purple', 'alternative': 'cyan', 'midwife': 'magenta', 'laboratory': 'yellow'}

        # Add markers for filtered healthcare facilities
        for idx, row in filtered_df.iterrows():
            folium.Marker([row['latitude'], row['longitude']], popup=row['healthcare'], icon=folium.Icon(color=colors.get(row['healthcare'], 'gray'))).add_to(m)

        # Display the Folium map using streamlit_folium
        folium_static(m)

        # Display statistics based on city
        st.subheader('Statistics by City')

        # Get unique city names
        cities = df3['addr:city'].unique()

        # Create a filter for city
        selected_city = st.selectbox('Select a city', ['All'] + list(cities))

        if selected_city != 'All':
            # Filter the GeoDataFrame based on selected city
            filtered_df = df3[df3['addr:city'] == selected_city]

        # Display statistics for selected city or all cities
        st.write("Total number of selected healthcare facilities:", len(filtered_df))
        st.write("Number of healthcare facilities by type:")
        st.write(filtered_df['healthcare'].value_counts())




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

    # Define main categories and their corresponding subcategories
    categories = {
        'Demography': ['Population'],
        'Health': ['Health facilities', 'ITN Coverage', 'Malaria Incidence','Malaria Prevalence'],
        'Agriculture': ['Schools', 'Universities', 'Libraries'],
        # Add more main categories and subcategories as needed
    }

    # Create a select box for main categories
    main_category = st.sidebar.selectbox('Select a main category', list(categories.keys()))

    # Create a select box for subcategories based on the selected main category
    if main_category:
        subcategories = categories[main_category]
        subcategory = st.selectbox('Select a subcategory', ['All'] + subcategories)
    # Create a sidebar for selecting categories
    #category = st.sidebar.selectbox("Select Category", options = ["Demography", "Health", "Climate","Environment","Agriculture"])
    year = st.sidebar.selectbox("Select the year", options= ['2000', '2010', '2015', '2020'])
    region = st.sidebar.selectbox("Select the region",options = ['Greater Monrovia','South Eastern','South Western'])
    counties = st.sidebar.selectbox("Select the county", options = ['Boma','Grand Bassa'])
    # Display the selected visualization based on the chosen category
    if main_category == "Demography":
        visualization_category_1(data,year,subcategory)
    elif main_category == "Health":
        visualization_category_2(data,malaria_data,itn_data,columns,subcategory)
    elif category == "Climate":
        visualization_category_3(data,wet_days_columns, month_temp_columns, rainfall_columns)
    elif category == "Environment":
        visualization_category_4(data)
    elif category == "Agriculture":
        visualization_category_5(data)

    


if __name__=="__main__":
    main()



