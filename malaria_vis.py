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
import re
from shapely.geometry import Point
import altair as alt
from streamlit_folium import folium_static
from folium.plugins import HeatMap

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
    malaria_prevalence_columns = ['DHSREGNA', 'Malaria_Prevalence_2000','Malaria_Prevalence_2005','Malaria_Prevalence_2010','Malaria_Prevalence_2015','Malaria_Prevalence_2020']
    malaria_prevalence_data = gdf_combined[malaria_prevalence_columns]
    itn_columns = ['DHSREGNA', 'ITN_Coverage_2000','ITN_Coverage_2005','ITN_Coverage_2010','ITN_Coverage_2015','ITN_Coverage_2020']
    itn_data = gdf_combined[itn_columns]
    wet_days_columns = ['DHSREGNA', 'Wet_Days_2000','Wet_Days_2005','Wet_Days_2010','Wet_Days_2015','Wet_Days_2020']
    month_temp_columns = ['DHSREGNA', 'Temperature_January','Temperature_February','Temperature_March','Temperature_April',
                'Temperature_May','Temperature_June','Temperature_July','Temperature_August','Temperature_September','Temperature_October',
                'Temperature_November','Temperature_December']
    rainfall_columns = ['DHSREGNA', 'Rainfall_2000','Rainfall_2005','Rainfall_2010','Rainfall_2015','Rainfall_2020']
    return gdf_combined, malaria_data ,itn_data, columns, wet_days_columns, month_temp_columns, rainfall_columns, malaria_prevalence_data


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
        

    

def visualization_category_2(data,malaria_data,itn_data,columns,subcategory,malaria_prevalence_data):
    # Add your visualization for category 2 here
    # Create a Folium map with a basemap
    if subcategory == "All":
        with st.expander("About",expanded=False):
            st.markdown("Malaria Incidence : Number of clinical cases of Plasmodium falciparum malaria per person.")
            st.markdown("Malaria Prevalence : Parasite rate of plasmodium falciparum (PfPR) in children between the ages of 2 and 10 years old.")
        with st.expander("Statistics for the year 2020",expanded=False):
            df = data.groupby('DHSREGNA')[['Malaria_Prevalence_2020','Malaria_Incidence_2020','ITN_Coverage_2020']].mean().sort_values(by='Malaria_Prevalence_2020',ascending=False)
            st.dataframe(df)

    elif subcategory =="Malaria Incidence":
        df1 = malaria_data.groupby('DHSREGNA').mean()
        year = st.sidebar.selectbox("Select the year", options= ['2000', '2005', '2010', '2015', '2020'])
        st.subheader("Malaria Incidence based on MIS Survey 2022")
        st.markdown(f"The below map shows the Malaria incidence for the year {year}. Here the red circles correspond to Malaria incidence (>50%), yellow for malaria incidence in the range (20-49%) and green for malaria incidence (<20%)")
        m = folium.Map(location=[data['LATNUM'].mean(), data['LONGNUM'].mean()], zoom_start=8)
        # Add circle markers for each location with varying radius based on ITN coverage intensity
        # Define color gradient based on ITN coverage intensity
        color_gradient = {
            'Low (<20%)': 'green',
            'Medium (20-49%)': 'yellow',
            'High (>=50%)': 'red'
        }
        for index, row in data.iterrows():
            # Determine color based on ITN coverage intensity
            malaria_incidence = row[f'Malaria_Incidence_{year}'] * 100  #Convert to percentage
            if malaria_incidence < 20:
                color = color_gradient['Low (<20%)']
            elif malaria_incidence < 50:
                color = color_gradient['Medium (20-49%)']
            else:
                color = color_gradient['High (>=50%)']

            # Create a circle marker with popup displaying ITN coverage value
            folium.CircleMarker(
                location=[row['LATNUM'], row['LONGNUM']],  # Latitude and longitude from geometry column
                radius=8,
                popup=f"DHS Cluster ID: {row['DHSCLUST']}<br>Region:{row['DHSREGNA']}<br>Malaria Incidence: {malaria_incidence:.2f}%",  # Format as percentage
                color=color,
                fill=True,
                fill_color=color
            ).add_to(m)

        # Display the map
        folium_static(m)

        st.dataframe(df1)
        # Define function to plot line chart with interactive dropdown
        def plot_interactive_line_chart(data):
            selected_features = st.multiselect('Select feature:', data.columns, default=['Malaria_Incidence_2020'])
            if selected_features:
                numeric_data = data[selected_features].select_dtypes(include='number')
                if not numeric_data.empty:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    numeric_data.plot.line(ax=ax, marker='o')
                    ax.set_xlabel('DHSREGNA')
                    ax.set_ylabel('Malaria Incidence')
                    ax.set_title('Malaria Incidence')
                    st.pyplot(fig)
                else:
                    st.warning('No numeric data selected. Please choose numeric features.')
            else:
                st.warning('Please select one or more features.')

        # Display the interactive plot
        plot_interactive_line_chart(df1)

        fig1 = px.scatter_mapbox(data, lon='LONGNUM', lat='LATNUM',size=f'Malaria_Incidence_{year}',color='URBAN_RURA',title= f'Malaria Incidence in Urban and Rural areas - {year}',
                        labels={'LATNUM': 'Latitude', 'LONGNUM': 'Longitude', f'Malaria_Incidence_{year}': 'Malaria Incidence'},
                        center=dict(lat=data['LATNUM'].mean(), lon=data['LONGNUM'].mean()),
                        zoom=5,
                        mapbox_style="carto-positron",color_discrete_map={'U': 'blue', 'R': 'pink'})
        
        # Display the Folium map and Plotly figure in Streamlit
        st.plotly_chart(fig1)
        
        
    
    elif subcategory == "Malaria Prevalence":
        df2 = malaria_prevalence_data.groupby('DHSREGNA').mean()
        year = st.sidebar.selectbox("Select the year", options= ['2000', '2005', '2010', '2015', '2020'])
        st.subheader("Malaria Prevalence based on MIS Survey 2022")
        st.markdown(f"The below map shows the Malaria prevalence for the year {year}. Here the red circles correspond to Malaria incidence (>50%), yellow for malaria incidence in the range (20-49%) and green for malaria incidence (<20%)")
        m = folium.Map(location=[data['LATNUM'].mean(), data['LONGNUM'].mean()], zoom_start=8)
        # Add circle markers for each location with varying radius based on ITN coverage intensity
        # Define color gradient based on ITN coverage intensity
        color_gradient = {
            'Low (<20%)': 'green',
            'Medium (20-49%)': 'yellow',
            'High (>=50%)': 'red'
        }
        for index, row in data.iterrows():
            # Determine color based on ITN coverage intensity
            malaria_prevalence = row[f'Malaria_Prevalence_{year}'] * 100  #Convert to percentage
            if malaria_prevalence < 20:
                color = color_gradient['Low (<20%)']
            elif malaria_prevalence < 50:
                color = color_gradient['Medium (20-49%)']
            else:
                color = color_gradient['High (>=50%)']

            # Create a circle marker with popup displaying ITN coverage value
            folium.CircleMarker(
                location=[row['LATNUM'], row['LONGNUM']],  # Latitude and longitude from geometry column
                radius=8,
                popup=f"DHS Cluster ID: {row['DHSCLUST']}<br>Region:{row['DHSREGNA']}<br>Malaria Prevalence: {malaria_prevalence:.2f}%",  # Format as percentage
                color=color,
                fill=True,
                fill_color=color
            ).add_to(m)

        # Display the map
        folium_static(m)

        st.dataframe(df2)
        # Define function to plot line chart with interactive dropdown
        def plot_interactive_line_chart(data):
            selected_features = st.multiselect('Select feature:', data.columns, default=['Malaria_Prevalence_2020'])
            if selected_features:
                numeric_data = data[selected_features].select_dtypes(include='number')
                if not numeric_data.empty:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    numeric_data.plot.line(ax=ax, marker='o')
                    ax.set_xlabel('DHSREGNA')
                    ax.set_ylabel('Malaria Prevalence')
                    ax.set_title('Malaria Prevalence')
                    st.pyplot(fig)
                else:
                    st.warning('No numeric data selected. Please choose numeric features.')
            else:
                st.warning('Please select one or more features.')

        # Display the interactive plot
        plot_interactive_line_chart(df2)
        fig2 = px.scatter_mapbox(data, lon='LONGNUM', lat='LATNUM',size=f'Malaria_Prevalence_{year}',color='URBAN_RURA',title=f'Malaria Prevalence in Urban and Rural areas - {year}',
                                labels={'LATNUM': 'Latitude', 'LONGNUM': 'Longitude', f'Malaria_Prevalence_{year}': 'Malaria Prevalence'},
                                center=dict(lat=data['LATNUM'].mean(), lon=data['LONGNUM'].mean()),
                                zoom=10,
                                mapbox_style="carto-positron",color_discrete_map={'U': 'blue', 'R': 'green'})
        
        st.plotly_chart(fig2)

    elif subcategory == "ITN Coverage": 
        year = st.sidebar.selectbox("Select the year", options= ['2000', '2005', '2010', '2015', '2020'])
        st.subheader("ITN Coverage based on MIS Survey 2022")
        st.markdown(f"The below map shows the ITN distribution for the year {year}. Here the red circles correspond to ITN Coverage (>50%), yellow for medium coverage(20-49%) and green for low coverage (<20%)")
        m = folium.Map(location=[data['LATNUM'].mean(), data['LONGNUM'].mean()], zoom_start=8)
        # Add circle markers for each location with varying radius based on ITN coverage intensity
        # Define color gradient based on ITN coverage intensity
        color_gradient = {
            'Low (<20%)': 'green',
            'Medium (20-49%)': 'yellow',
            'High (>=50%)': 'red'
        }
        for index, row in data.iterrows():
            # Determine color based on ITN coverage intensity
            itn_coverage = row[f'ITN_Coverage_{year}'] * 100  #Convert to percentage
            if itn_coverage < 20:
                color = color_gradient['Low (<20%)']
            elif itn_coverage < 50:
                color = color_gradient['Medium (20-49%)']
            else:
                color = color_gradient['High (>=50%)']

            # Create a circle marker with popup displaying ITN coverage value
            folium.CircleMarker(
                location=[row['LATNUM'], row['LONGNUM']],  # Latitude and longitude from geometry column
                radius=8,
                popup=f"DHS Cluster ID: {row['DHSCLUST']}<br>Region:{row['DHSREGNA']}<br>ITN Coverage: {itn_coverage:.2f}%",  # Format as percentage
                color=color,
                fill=True,
                fill_color=color
            ).add_to(m)

        # Display the map
        folium_static(m)

        df2 = itn_data.groupby('DHSREGNA').mean()
        st.dataframe(df2)
        # Define function to plot line chart with interactive dropdown
        def plot_interactive_line_chart(data):
            selected_features = st.multiselect('Select feature:', data.columns, default=['ITN_Coverage_2020'])
            if selected_features:
                numeric_data = data[selected_features].select_dtypes(include='number')
                if not numeric_data.empty:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    numeric_data.plot.line(ax=ax, marker='o')
                    ax.set_xlabel('DHSREGNA')
                    ax.set_ylabel('ITN Coverage')
                    ax.set_title('ITN Coverage')
                    st.pyplot(fig)
                else:
                    st.warning('No numeric data selected. Please choose numeric features.')
            else:
                st.warning('Please select one or more features.')

        # Display the interactive plot
        plot_interactive_line_chart(df2)

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
        
        # Group the data by city and count the occurrences
        city_counts = df3['addr:city'].value_counts()

        # Display the top 10 cities with the most facilities
        top_cities = city_counts.head(10)
        # Rename the column to 'count' for Plotly compatibility
        top_cities_df = top_cities.reset_index().rename(columns={'Number of Facilities': 'count'})
        st.subheader("Top 10 Cities with Most Healthcare Facilities")
        # Create an interactive bar plot using Plotly
        fig = px.bar(top_cities_df, x='addr:city', y='count', labels={'addr:city': 'City/Area', 'count': 'Number of Facilities'})
        fig.update_layout(xaxis_tickangle=-45)  # Rotate x-axis labels

        # Display the interactive plot
        st.plotly_chart(fig)

        st.subheader("Top 10 Cities with least Healthcare Facilities")
        # Display the top 10 cities with the most facilities
        least_cities = city_counts.nsmallest(10)
        # Rename the column to 'count' for Plotly compatibility
        least_cities_df = least_cities.reset_index().rename(columns={'Number of Facilities': 'count'})
        # Create an interactive bar plot using Plotly
        color = 'pink'
        fig = px.bar(least_cities_df, x='addr:city', y='count', color_discrete_sequence = [color], labels={'addr:city': 'City/Area', 'count': 'Number of Facilities'})
        fig.update_layout(xaxis_tickangle=-45)  # Rotate x-axis labels

        # Display the interactive plot
        st.plotly_chart(fig)


        st.subheader("Heatmap of Top 10 cities with least healthcare facilities")
        # Filter the original GeoDataFrame for only the cities in 'least_cities'
        least_cities_gdf = df3[df3['addr:city'].isin(least_cities.index)]

        # Aggregate point data for these cities
        least_cities_points = least_cities_gdf.groupby('addr:city')['point_geometry'].apply(lambda x: x.iloc[0])

        # Convert this to a simple DataFrame for easier processing
        least_cities_df1 = least_cities_points.to_frame(name='point_geometry').reset_index()
        least_cities_df1['facility_count'] = least_cities_df1['addr:city'].map(least_cities)

        # Ensure that the 'point_geometry' column is used for mean location calculation
        mean_location = [df3['point_geometry'].y.mean(), df3['point_geometry'].x.mean()]

        # Initialize the map with the mean location
        m1 = folium.Map(location=mean_location, zoom_start=6)

        # Assuming 'least_cities_df' contains the top 10 least cities with their 'point_geometry'
        # Create the heat map layer using the 'point_geometry' column
        heat_data = [
            [geom.y, geom.x] for geom in least_cities_df1['point_geometry']
        ]

        HeatMap(heat_data).add_to(m1)

        # Add city names as markers to the map
        for idx, row in least_cities_df1.iterrows():
            folium.Marker(location=[row['point_geometry'].y, row['point_geometry'].x], popup=row['addr:city']).add_to(m1)

        #save to an HTML file to open in a browser
        folium_static(m1)



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

    data, malaria_data, itn_data, columns, wet_days_columns, month_temp_columns, rainfall_columns, malaria_prevalence_data = read_dataset()

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
        subcategory = st.sidebar.selectbox('Select a subcategory', ['All'] + subcategories)
    # Create a sidebar for selecting categories
    #category = st.sidebar.selectbox("Select Category", options = ["Demography", "Health", "Climate","Environment","Agriculture"])
    #year = st.sidebar.selectbox("Select the year", options= ['2000', '2010', '2015', '2020'])
    #region = st.sidebar.selectbox("Select the region",options = ['Greater Monrovia','South Eastern','South Western'])
    #counties = st.sidebar.selectbox("Select the county", options = ['Boma','Grand Bassa'])
    # Display the selected visualization based on the chosen category
    if main_category == "Demography":
        visualization_category_1(data,year,subcategory)
    elif main_category == "Health":
        visualization_category_2(data,malaria_data,itn_data,columns,subcategory,malaria_prevalence_data)
    elif category == "Climate":
        visualization_category_3(data,wet_days_columns, month_temp_columns, rainfall_columns)
    elif category == "Environment":
        visualization_category_4(data)
    elif category == "Agriculture":
        visualization_category_5(data)

    


if __name__=="__main__":
    main()



