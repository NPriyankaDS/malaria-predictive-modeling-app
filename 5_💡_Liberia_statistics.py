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
from branca.colormap import LinearColormap, StepColormap
import plotly.graph_objects as go
import re
from shapely.geometry import Point
import altair as alt
from streamlit_folium import folium_static
from folium.plugins import HeatMap

@st.cache_data
def read_health_facilities():
    #Loading the data
    points_path = "./data/hotosm_lbr_health_facilities_points_geojson.geojson"
    polygons_path = "./data/hotosm_lbr_health_facilities_polygons_geojson.geojson"
    points_gdf = gpd.read_file(points_path)
    polygons_gdf = gpd.read_file(polygons_path)
    # Merge the two datasets into a single GeoDataFrame
    combined_gdf = pd.concat([points_gdf, polygons_gdf], ignore_index=True)
    #Converting Polygons to Representative Points
    combined_gdf['point_geometry'] = combined_gdf.apply(lambda row: row['geometry'].representative_point() if row['geometry'].geom_type == 'Polygon' else row['geometry'],axis=1)
    return combined_gdf

@st.cache_data
def read_dataset():
    #Loading the data for LBGE81FL and LBGC81FL
    gdf = gpd.read_file("./data/LB_2022_MIS_02262024_207_206552/LBGE81FL/LBGE81FL.shp")
    df = pd.read_csv("./data/LB_2022_MIS_02262024_207_206552/LBGC81FL/LBGC81FL/LBGC81FL.csv")
    df_demography = pd.read_csv("./data/Population_LB_year.csv")
    df_agri = pd.read_csv("data/STATcompilerExport2024221_16431.csv")
    df_children_malaria = pd.read_csv("./data/Prevalence_of_malaria_in_children.csv")
    df_children_malaria = df_children_malaria.astype(object).fillna(value='null')
    columns = list(df.columns)
    gdf_combined = pd.concat([gdf,df],axis=1)
    
    malaria_columns = ['DHSREGNA', 'Malaria_Incidence_2000', 'Malaria_Incidence_2005', 'Malaria_Incidence_2010', 'Malaria_Incidence_2015', 'Malaria_Incidence_2020']
    malaria_data = gdf_combined[malaria_columns]
    malaria_prevalence_columns = ['DHSREGNA', 'Malaria_Prevalence_2000','Malaria_Prevalence_2005','Malaria_Prevalence_2010','Malaria_Prevalence_2015','Malaria_Prevalence_2020']
    malaria_prevalence_data = gdf_combined[malaria_prevalence_columns]
    itn_columns = ['DHSREGNA', 'ITN_Coverage_2000','ITN_Coverage_2005','ITN_Coverage_2010','ITN_Coverage_2015','ITN_Coverage_2020']
    itn_data = gdf_combined[itn_columns]
    pop_density_columns = ['DHSREGNA', 'UN_Population_Density_2000','UN_Population_Density_2005','UN_Population_Density_2010','UN_Population_Density_2015','UN_Population_Density_2020']
    pop_density_data = gdf_combined[pop_density_columns]
    wet_days_columns = ['DHSREGNA', 'Wet_Days_2000','Wet_Days_2005','Wet_Days_2010','Wet_Days_2015','Wet_Days_2020']
    month_temp_columns = ['DHSREGNA', 'Temperature_January','Temperature_February','Temperature_March','Temperature_April',
                'Temperature_May','Temperature_June','Temperature_July','Temperature_August','Temperature_September','Temperature_October',
                'Temperature_November','Temperature_December']
    rainfall_columns = ['DHSREGNA', 'Rainfall_2000','Rainfall_2005','Rainfall_2010','Rainfall_2015','Rainfall_2020']
    evi_columns = ['DHSREGNA', 'Enhanced_Vegetation_Index_2000','Enhanced_Vegetation_Index_2005','Enhanced_Vegetation_Index_2010','Enhanced_Vegetation_Index_2015','Enhanced_Vegetation_Index_2020']
    evi_data = gdf_combined[evi_columns]
    pet_columns = ['DHSREGNA', 'PET_2000','PET_2005','PET_2010','PET_2015','PET_2020']
    pet_data = gdf_combined[pet_columns]
    aridity_columns = ['DHSREGNA', 'Aridity_2000', 'Aridity_2005', 'Aridity_2010', 'Aridity_2015','Aridity_2020']
    aridity_data = gdf_combined[aridity_columns]

    return gdf_combined, malaria_data ,itn_data, columns, wet_days_columns, month_temp_columns, rainfall_columns, malaria_prevalence_data, pop_density_data, evi_data, pet_data, df_demography, df_agri, df_children_malaria, aridity_data


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
def visualization_category_1(data,subcategory,pop_density_data, df_demography):
    if subcategory == "UN Population": 
    # Add your visualization for category 1 here
        
        year = st.sidebar.selectbox("Select the year", options= ['2000', '2005', '2010', '2015', '2020'])
        st.subheader(f"UN Population density {year}")
        st.info("The average UN-adjusted population density of the area at the DHS survey cluster location(Number of people per square kilometer).")
        st.markdown("Select the year in the sidebar. The map changes in accordance with the year chosen.")
        df1 = pop_density_data.groupby('DHSREGNA').mean()
        m = folium.Map(location=[data['LATNUM'].mean(), data['LONGNUM'].mean()], zoom_start=8)
        cmap = LinearColormap(['green', 'yellow', 'orange', 'red'], vmin=data[f'UN_Population_Density_{year}'].min(), vmax=data[f'UN_Population_Density_{year}'].max())

        for index, row in data.iterrows():
            # Determine color based on ITN coverage intensity
            pop_density = row[f'UN_Population_Density_{year}']

            # Create a circle marker with popup
            folium.CircleMarker(
                location=[row['LATNUM'], row['LONGNUM']],  # Latitude and longitude from geometry column
                radius=8,
                popup=f"Urban/Rural:{row['URBAN_RURA']}<br>DHS Cluster ID: {row['DHSCLUST']}<br>Region:{row['DHSREGNA']}<br>UN Population Density: {pop_density:.2f}",  
                color=cmap(pop_density),
                fill=True,
                fill_color=cmap(pop_density),
            ).add_to(m)
        
        cmap.add_to(m)
        # Display the map
        folium_static(m)

        st.dataframe(df1)
        # Define function to plot line chart with interactive dropdown
        def plot_interactive_line_chart(data):
            selected_features = st.multiselect('Select feature:', data.columns, default=['UN_Population_Density_2020'])
            button = st.button("Submit")
            if button:
                numeric_data = data[selected_features].select_dtypes(include='number')
                if not numeric_data.empty:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    numeric_data.plot.line(ax=ax, marker='o')
                    ax.set_xlabel('Region')
                    ax.set_ylabel('UN Population density')
                    ax.set_title('UN Population density')
                    st.pyplot(fig)
                else:
                    st.warning('No numeric data selected. Please choose numeric features.')
            else:
                st.warning('Please select one or more features.')

        # Display the interactive plot
        plot_interactive_line_chart(df1)
        st.markdown("Source: https://dhsprogram.com/")

    elif subcategory == "Under-5 Population":
        under5_pop_columns = ['DHSREGNA', 'U5_Population_2000','U5_Population_2005','U5_Population_2010','U5_Population_2015','U5_Population_2020']
        under5_pop_data = data[under5_pop_columns]
        year = st.sidebar.selectbox("Select the year", options= ['2000', '2005', '2010', '2015', '2020'])
        st.subheader(f"Under-5 Population {year}")
        st.info("The number of people under the age of 5 (U5) at the time of measurement (year) living in the 5 x 5 km pixel in which the DHS survey cluster is located. (Number of people).")
        st.markdown("Select the year in the sidebar. The map changes in accordance with the year chosen.")
        df2 = under5_pop_data.groupby('DHSREGNA').mean()
        m = folium.Map(location=[data['LATNUM'].mean(), data['LONGNUM'].mean()], zoom_start=8)
        cmap = LinearColormap(['green', 'yellow', 'orange', 'red'], vmin=data[f'U5_Population_{year}'].min(), vmax=data[f'U5_Population_{year}'].max())

        for index, row in data.iterrows():
            # Determine color based on ITN coverage intensity
            under5_pop = row[f'U5_Population_{year}']
                        # Create a circle marker with popup displaying ITN coverage value
            folium.CircleMarker(
                location=[row['LATNUM'], row['LONGNUM']],  # Latitude and longitude from geometry column
                radius=8,
                popup=f"Region:{row['URBAN_RURA']}<br>DHS Cluster ID: {row['DHSCLUST']}<br>Region:{row['DHSREGNA']}<br>Under-5 Population: {under5_pop:.3f}",  
                color=cmap(under5_pop),
                fill=True,
                fill_color=cmap(under5_pop),
            ).add_to(m)

        cmap.add_to(m)
        # Display the map
        folium_static(m)

        st.dataframe(df2)
        # Define function to plot line chart with interactive dropdown
        def plot_interactive_line_chart(data):
            selected_features = st.multiselect('Select feature:', data.columns, default=['U5_Population_2000'])
            button = st.button("Submit")
            if button:
                numeric_data = data[selected_features].select_dtypes(include='number')
                if not numeric_data.empty:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    numeric_data.plot.line(ax=ax, marker='o')
                    ax.set_xlabel('Region')
                    ax.set_ylabel('Under-5 Population')
                    ax.set_title('Under-5 Population')
                    st.pyplot(fig)
                else:
                    st.warning('No numeric data selected. Please choose numeric features.')
            else:
                st.warning('Please select one or more features.')

        # Display the interactive plot
        plot_interactive_line_chart(df2)
        st.markdown("Source: https://dhsprogram.com/")

    elif subcategory == "Population":
        # Create a multiselect widget for variable selection
        selected_variables = st.multiselect('Select variables', ['Total_population', 'no_of_females', 'no_of_males'], default=['Total_population'])
        st.markdown("There are three variables: Total population, Number of females and Number of males")
        # Define custom colors for the bars
        custom_colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] 
        # Create the bar chart using Plotly based on the selected variables
        fig = px.bar(df_demography, x='Year', y=selected_variables, 
                    title='Demographic Data', 
                    labels={'Year': 'Year', 'value': 'Population', 'variable': 'Variable'}, 
                    color_discrete_sequence=custom_colors)  # Optional: Set a custom color

        # Customize layout (optional)
        fig.update_layout(xaxis=dict(title='Year'), yaxis=dict(title='Population'), xaxis_tickangle=-45)

        # Display the plot in Streamlit
        st.plotly_chart(fig)
        # Define the range of years
        year_range = list(range(2000, 2023))
        year = st.sidebar.selectbox('Select year',year_range)
        df = df_demography[df_demography['Year'] == year]

        select_plot = st.selectbox("Select the plot",['Distribution by Gender','Distribution by Region','Age-pyramid'])
        st.markdown("""There are three plots: Distribution by Gender, Distribution by Region, and Age-pyramid.
        In the sidebar, there is a selectbox for selecting the years. The year ranges from 2000 to 2022.
        The data displayed below changes in accordance with the year chosen. """)
        
        # Get the percentage of females and males for the year selected
        if select_plot == 'Distribution by Gender':
            percentage_females = df['Percentage_of_females'].iloc[0]
            percentage_males = df['Percentage_of_males'].iloc[0]

            # Create a DataFrame for the pie chart
            df1 = {'Gender': ['Females', 'Males'], 'Percentage': [percentage_females, percentage_males]}
            df_pie = pd.DataFrame(df1)

            # Plot the pie chart using Plotly
            fig = px.pie(df_pie, values='Percentage', names='Gender', title=f'Percentage of Females and Males in {year}',
                        color_discrete_sequence=['lightpink', 'lightblue'])

            # Display the pie chart in the Streamlit app
            st.plotly_chart(fig)

        elif select_plot == 'Distribution by Region':
            percentage_rural = df['Percentage_of_Rural_population'].iloc[0]
            percentage_urban = df['Percentage_of_urban_population'].iloc[0]

            # Create a DataFrame for the pie chart
            df2 = {'Region': ['Rural', 'Urban'], 'Percentage': [percentage_rural, percentage_urban]}
            df_pie = pd.DataFrame(df2)

            # Plot the pie chart using Plotly
            fig = px.pie(df_pie, values='Percentage', names='Region', title=f'Percentage of Rural and Urban population in {year}',
                        color_discrete_sequence=['lightgreen', 'lightblue'])

            # Display the pie chart in the Streamlit app
            st.plotly_chart(fig)
        
        else:

            columns_for_females = df[['Population ages 00-04, female', 'Population ages 05-09, female','Population ages 10-14, female','Population ages 15-19, female',
                       'Population ages 20-24, female','Population ages 25-29, female', 'Population ages 30-34, female',
                       'Population ages 35-39, female','Population ages 40-44, female','Population ages 45-49, female','Population ages 50-54, female','Population ages 55-59, female',
                       'Population ages 60-64, female','Population ages 65-69, female','Population ages 70-74, female','Population ages 75-79, female','Population ages 80 and above, female']].iloc[0]
            columns_for_males = df[['Population ages 00-04, male', 'Population ages 05-09, male','Population ages 10-14, male','Population ages 15-19, male',
                       'Population ages 20-24, male','Population ages 25-29, male', 'Population ages 30-34, male',
                       'Population ages 35-39, male','Population ages 40-44, male','Population ages 45-49, male','Population ages 50-54, male','Population ages 55-59, male',
                       'Population ages 60-64, male','Population ages 65-69, male','Population ages 70-74, male','Population ages 75-79, male','Population ages 80 and above, male']].iloc[0]
            age_data = {
                    'Age Group': ['0-04','05-09', '10-14', '15-19','20-24', '25-29', '30-34', '35-39', '40-44', '45-49',
                                '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', 'Above 80'],
                    'Male': columns_for_males.to_list(),
                    'Female': columns_for_females.to_list()
                }
            df_age = pd.DataFrame(age_data)
            # Plot pyramid plot
            fig, ax = plt.subplots(figsize=(12,6))

            # Plot males
            ax.barh(df_age['Age Group'], df_age['Male'], color='blue', label='Male')

            # Plot females with negative values to mirror the bars
            ax.barh(df_age['Age Group'], df_age['Female'], color='red', label='Female')

            # Set labels and title
            ax.set_xlabel('Population')
            ax.set_ylabel('Age Group')
            ax.set_title(f'Population Pyramid {year}')

            # Invert y-axis to display youngest age group at the top
            ax.invert_yaxis()

            # Add legend
            ax.legend()

            # Show plot
            st.pyplot(fig)

            y1 = ['Population ages 0-14, total','Population ages 15-64, total', 'Population ages 65 and above, total']
            fig = px.line(df_demography, x='Year', y=y1, title="Population Distribution Over Time")
            st.plotly_chart(fig)
        st.markdown("Source: https://datacatalog.worldbank.org/public-licenses#cc-by")


    else:
        st.markdown("**About**")
        st.write("--------------------------------------------")
        st.write("""There are three subcategories for Demography: 
        UN Population , Under-5 Population and Population. \n\n
        UN Population:\nUnder this category, we can see the UN Population density of Liberia on the map for the years 2000, 2005, 2010, 2015 and 2020.\n
        Under-5 Population:\nUnder this category, we can see the Under -5 Population of Liberia on the map for the years 2000, 2005, 2010, 2015 and 2020. The data for both the categories has been obtained from DHS Program.\n
        Population:\nUnder this category, the population statistics for Liberia has been displayed.
        The statistics include the Total population, Number of Females, Number of Males, Percentage distribution by Gender, by Region, and Age-pyramid.
        """)
        st.write("--------------------------------------------")

def visualization_category_2(data,malaria_data,itn_data,columns,subcategory,malaria_prevalence_data, df_children_malaria, df_demography):
    # Add your visualization for category 2 here
    # Create a Folium map with a basemap
    if subcategory == "About":
        with st.expander("About",expanded=False):
            st.markdown("Malaria Incidence : Number of clinical cases of Plasmodium falciparum malaria per person.")
            st.markdown("Malaria Prevalence : Parasite rate of plasmodium falciparum (PfPR) in children between the ages of 2 and 10 years old.")
        with st.expander("Statistics for the year 2020",expanded=False):
            df = data.groupby('DHSREGNA')[['Malaria_Prevalence_2020','Malaria_Incidence_2020','ITN_Coverage_2020']].mean().sort_values(by='Malaria_Prevalence_2020',ascending=False)
            st.dataframe(df)

    elif subcategory =="Malaria Incidence":
        df1 = malaria_data.groupby('DHSREGNA').mean()
        year = st.sidebar.selectbox("Select the year", options= ['2000', '2005', '2010', '2015', '2020'])
        st.subheader(f"Malaria Incidence for the year {year}")
        
        m = folium.Map(location=[data['LATNUM'].mean(), data['LONGNUM'].mean()], zoom_start=8)
  
        # Define color map
        cmap = LinearColormap(['green', 'yellow', 'orange', 'red'], vmin=data[f'Malaria_Incidence_{year}'].min(), vmax=data[f'Malaria_Incidence_{year}'].max())

        for index, row in data.iterrows():
            # Determine color based on ITN coverage intensity
            malaria_incidence = row[f'Malaria_Incidence_{year}'] #Convert to percentage

            # Create a circle marker with popup displaying ITN coverage value
            folium.CircleMarker(
                location=[row['LATNUM'], row['LONGNUM']],  # Latitude and longitude from geometry column
                radius=8,
                popup=f"Urban/Rural: {row['URBAN_RURA']}<br>DHS Cluster ID: {row['DHSCLUST']}<br>Region:{row['DHSREGNA']}<br>Malaria Incidence: {malaria_incidence:.4f}",  # Format as percentage
                color=cmap(malaria_incidence),
                fill=True,
                fill_color=cmap(malaria_incidence)
            ).add_to(m)

        cmap.add_to(m)
        # Display the map
        folium_static(m)

        st.dataframe(df1)
        # Define function to plot line chart with interactive dropdown
        def plot_interactive_line_chart(data):
            selected_features = st.multiselect('Select feature:', data.columns, default=['Malaria_Incidence_2020'])
            button = st.button("Submit")
            if button:
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
        st.markdown("Source: https://dhsprogram.com/")
        
    elif subcategory == "Malaria Prevalence":
        df2 = malaria_prevalence_data.groupby('DHSREGNA').mean()
        year = st.sidebar.selectbox("Select the year", options= ['2000', '2005', '2010', '2015', '2020'])
        st.subheader(f"Malaria Prevalence for the year {year}")
        
        m = folium.Map(location=[data['LATNUM'].mean(), data['LONGNUM'].mean()], zoom_start=8)
        # Add circle markers for each location with varying radius based on ITN coverage intensity
        # Define color gradient based on ITN coverage intensity
        
        cmap = LinearColormap(['green', 'yellow', 'orange', 'red'], vmin=data[f'Malaria_Prevalence_{year}'].min(), vmax=data[f'Malaria_Prevalence_{year}'].max())
        for index, row in data.iterrows():
            # Determine color based on ITN coverage intensity
            malaria_prevalence = row[f'Malaria_Prevalence_{year}'] #Convert to percentage
            
            # Create a circle marker with popup displaying ITN coverage value
            folium.CircleMarker(
                location=[row['LATNUM'], row['LONGNUM']],  # Latitude and longitude from geometry column
                radius=8,
                popup=f"Urban/Rural: {row['URBAN_RURA']}<br>DHS Cluster ID: {row['DHSCLUST']}<br>Region:{row['DHSREGNA']}<br>Malaria Prevalence: {malaria_prevalence:.3f}",
                color=cmap(malaria_prevalence),
                fill=True,
                fill_color=cmap(malaria_prevalence)
            ).add_to(m)
        
        cmap.add_to(m)
        # Display the map
        folium_static(m)

        st.dataframe(df2)
        # Define function to plot line chart with interactive dropdown
        def plot_interactive_line_chart(data):
            selected_features = st.multiselect('Select feature:', data.columns, default=['Malaria_Prevalence_2020'])
            button = st.button("Submit")
            if button:
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
        st.markdown("Source: https://dhsprogram.com/")

    elif subcategory == "Malaria prevalence in children(6-59) months":
        # Define a function to extract numeric values from the strings
        def extract_numeric_value(text):
            match = re.search(r'(\d+\.\d+)',text)
            if match:
                return float(match.group(0))
            else:
                return None

        df_children_malaria['Malaria prevalence according to RDT'] = df_children_malaria['Malaria prevalence according to RDT'].apply(extract_numeric_value)
        df_children_malaria['Malaria prevalence according to microscopy'] = df_children_malaria['Malaria prevalence according to microscopy'].apply(extract_numeric_value)
        df_2022_MIS = df_children_malaria.loc[np.where(df_children_malaria['Survey']=='2022 MIS')]
        st.subheader("Malaria prevalence in children(6-59) months based on 2022 Malaria Indicator Survey(MIS)")
        
        with st.expander("About"):
            st.write("Malaria prevalence according to RDT : " "Percentage of children age 6-59 months tested using a rapid diagnostic test (RDT) who are positive for malaria. The confidence intervals (CIs) presented may differ from CIs in DHS and MIS final reports as the CIs in STATcompiler were calculated using the Jackknife Estimation Method rather the Taylor Series Linearization Method used in DHS and MIS final reports.\n\n"
                        "Number of children 6-59 months tested using RDT: ","Number of children age 6-59 months tested for malaria using a rapid diagnostic test (RDT)\n\n",
                        "Malaria prevalence according to microscopy : " "Number of children age 6-59 months tested for malaria using a rapid diagnostic test (RDT) (unweighted)\n\n" 
                        "Number of children 6-59 months tested using microscopy: ","Percentage of children age 6-59 months tested using microscopy who are positive for malaria. The confidence intervals (CIs) presented may differ from CIs in DHS and MIS final reports as the CIs in STATcompiler were calculated using the Jackknife Estimation Method rather the Taylor Series Linearization Method used in DHS and MIS final reports. \n\n"
                        "Source: ICF, 2015. The DHS Program STATcompiler. Funded by USAID. http://www.statcompiler.com. February 22 2024")
        
        st.markdown("Choose the category in the sidebar. There are six categories: Group of Counties, Age group, Gender, Education level, Wealth quintiles and Residence type. The graph and the data changes here as per the chosen category.")
        select_cat = st.sidebar.selectbox("Choose the category", ['Group of Counties', 'Age group', 'Gender', 'Education level', 'Wealth quintiles','Residence type'])
        if select_cat == "Age group":
            df_subset = df_2022_MIS[19:26]  # Ensure to make a copy to avoid modifying the original DataFrame
        elif select_cat == "Education level":
            df_subset = df_2022_MIS[5:9]
        elif select_cat == "Gender":
            df_subset = df_2022_MIS[1:3]
        elif select_cat == "Residence type":
            df_subset = df_2022_MIS[3:5]
        elif select_cat == "Group of Counties":
            df_subset = df_2022_MIS[26:32]
        else:
            df_subset = df_2022_MIS[11:16]  

        # Create a horizontal bar plot using Plotly Express
        fig = px.bar(df_subset, y='Characteristic', x=['Malaria prevalence according to RDT', 'Malaria prevalence according to microscopy'], orientation='h',
                    title=f"Malaria prevalence in children by {select_cat}",
                    labels={'value': 'Percentage', 'Characteristic': f'{select_cat}'})

        # Customize the legend position
        fig.update_layout(legend=dict(x=1.02, y=1.0))

        # Display the Plotly chart in Streamlit
        st.plotly_chart(fig)
        st.write(df_subset.iloc[:,0:5].reset_index(drop=True))

        with st.expander("Analysis"):
            st.write("""***Malaria prevalence in children (6-59 months)***

                * Malaria prevalence according to RDT:

                1) Age groups(in months) : 48-59 > 36-47 > 24-35 > 18-23 > 12-17 > 9-11 > 6-8

                2) Education level: Primary > No education > Secondary > Higher

                3) Gender : Male > Female

                4) Residence: Rural > Urban

                5) Region: South eastern B > South Eastern A > North Central > North Western > South Central > Monrovia

                6) Weath quintile: Lowest > Second >Middle > Fourth > Highest
                
                * Malaria prevalence according to microscopy:

                1) Age groups(in months) : 48-59 > 36-47 > 24-35 > 18-23 > 9-11 > 12-17 > 6-8

                2) Education level : Primary > No education > Secondary > Higher

                3) Gender: Male > Female

                4) Residence: Rural > Urban

                5) Region: South Eastern B > North Central > South eastern A > North Western > South Central > Monrovia

                6) Wealth quintile : Lowest > Second > Middle > Fourth > Highest

                """)
            st.markdown("Source: The DHS Program STATcompiler. Funded by USAID. http://www.statcompiler.com. February 22 2024")
            
    elif subcategory == "ITN Coverage": 
        year = st.sidebar.selectbox("Select the year", options= ['2000', '2005', '2010', '2015', '2020'])
        st.subheader(f"ITN Coverage for the year {year}")
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
            button = st.button("Submit")
            if button:
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
        st.markdown("Source: https://dhsprogram.com/")

    elif subcategory == "Health facilities":
        
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

    elif subcategory == "Malaria cases by Species":
        df = {
            'Year': [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],
            'Indigenous': [922173, 1921159, 1412629, 1244220, 881224, 941711, 1191137, 1093115, None, 915845, None, 912436, 696684],
            'P.falciparum': [212927, 577641, 1407455, 1244220, 864204, None, 809356, None, None, 915845, None, 912436, 696684],
        }
        df = pd.DataFrame(df)
        df['Indigenous'] = df['Indigenous'].fillna(df['Indigenous'].mean())
        df['P.falciparum'] = df['P.falciparum'].fillna(df['P.falciparum'].mean())
        
        # Create the line plot using Plotly Express
        fig = px.line(df, x='Year', y=['Indigenous','P.falciparum'],
                    title='Reported malaria cases by species',
                    labels={'Year': 'Year','value':'Number of cases','variable':'Species'},
                    color_discrete_map={'Indigenous': 'red', 'P.falciparum': 'blue'})

        # Show the plot
        st.plotly_chart(fig)
        
        st.markdown("Note: The missing values have been imputed with the mean values of the respective cases by species.")
        df_demography = df_demography.copy()[0:13]
        df_demography['malaria_deaths'] = df_demography['malaria_deaths'].fillna(df_demography['malaria_deaths'].mean())
        fig = px.line(df_demography, x='Year', y='malaria_deaths', title='Malaria Deaths Over Time')

        # Display the plot using streamlit's plotly_chart
        st.plotly_chart(fig)
        df_demo = df_demography[['Year','malaria_deaths']][0:13]
        df_merged =pd.merge(df,df_demo,on='Year',how='inner')
        st.dataframe(df_merged)
        st.markdown("Note: The missing values for the malaria deaths have been imputed with the mean value.")

def visualization_category_3(data,wet_days_columns, month_temp_columns, rainfall_columns,subcategory):
    
    year = st.sidebar.selectbox("Select the year", options= ['2000', '2005', '2010', '2015', '2020'])

    if subcategory == "Mean wet days":
    
        st.subheader(f"Mean wet days for the year {year}")
        st.write("Definition: \nThe average number of days per month receiving ≥0.1 mm precipitation at the DHS survey cluster location.")
        
        m = folium.Map(location=[data['LATNUM'].mean(), data['LONGNUM'].mean()], zoom_start=8)
        # Add circle markers for each location with varying radius based on ITN coverage intensity
        # Define colormap
        cmap = LinearColormap(['green', 'yellow', 'orange', 'red'], vmin=data[f'Wet_Days_{year}'].min(), vmax=data[f'Wet_Days_{year}'].max())

        for index, row in data.iterrows():
            # Determine color based on Mean number of wet days
            wet_days = row[f'Wet_Days_{year}']  
            # Create circle marker with color based on wet days value
            folium.CircleMarker(
                location=[row['LATNUM'], row['LONGNUM']],
                radius=8,
                popup=f"DHS Cluster ID: {row['DHSCLUST']}<br>Region:{row['DHSREGNA']}<br>Mean wet days: {wet_days:.4f}",
                color=cmap(wet_days),
                fill=True,
                fill_color=cmap(wet_days)
            ).add_to(m)

        # Add colormap legend to map
        cmap.add_to(m)

        # Display the map
        folium_static(m)

        wet_days_data = data[wet_days_columns]
        df2 = wet_days_data.groupby('DHSREGNA').mean()
        st.dataframe(df2)
        # Define function to plot line chart with interactive dropdown
        def plot_interactive_line_chart(data):
            selected_features = st.multiselect('Select feature:', data.columns, default=['Wet_Days_2020'])
            button = st.button("Submit")
            if button:
                numeric_data = data[selected_features].select_dtypes(include='number')
                if not numeric_data.empty:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    numeric_data.plot.line(ax=ax, marker='o')
                    ax.set_xlabel('DHSREGNA')
                    ax.set_ylabel('Mean wet days')
                    ax.set_title('Mean wet days')
                    st.pyplot(fig)
                else:
                    st.warning('No numeric data selected. Please choose numeric features.')
            else:
                st.warning('Please select one or more features.')

        # Display the interactive plot
        plot_interactive_line_chart(df2)

    elif subcategory == "Rainfall":
        st.subheader(f"Average annual rainfall for the year {year}")
        st.write("Definition: \nThe average annual rainfall at the DHS survey cluster location.")
        
        m = folium.Map(location=[data['LATNUM'].mean(), data['LONGNUM'].mean()], zoom_start=8)
        #marker_cluster = MarkerCluster().add_to(m)
        min_rainfall = data[f'Rainfall_{year}'].min()
        max_rainfall = data[f'Rainfall_{year}'].max()

        # Define colormap thresholds (sorted)
        thresholds = [min_rainfall, min_rainfall + (max_rainfall - min_rainfall) / 3, min_rainfall + 2 * (max_rainfall - min_rainfall) / 3, max_rainfall]
        # Define colormap
        cmap = StepColormap(['green', 'yellow', 'orange', 'red'], vmin=min_rainfall, vmax=max_rainfall, index=thresholds)
        
        for index, row in data.iterrows():
            # Determine color based on Mean number of wet days
            rainfall = row[f'Rainfall_{year}']  
            # Create circle marker with color based on wet days value
            folium.CircleMarker(
                location=[row['LATNUM'], row['LONGNUM']],
                radius=8,
                popup=f"Urban/Rural:{row['URBAN_RURA']}<br>DHS Cluster ID: {row['DHSCLUST']}<br>Region:{row['DHSREGNA']}<br>Average annual rainfall: {rainfall:.4f}",
                color=cmap(rainfall),
                fill=True,
                fill_color=cmap(rainfall)
            ).add_to(m)

        # Add colormap legend to map
        m.add_child(cmap)

        # Display the map
        folium_static(m)

        rainfall_data = data[rainfall_columns]
        df2 = rainfall_data.groupby('DHSREGNA').mean()
        st.dataframe(df2)
        # Define function to plot line chart with interactive dropdown
        def plot_interactive_line_chart(data):
            selected_features = st.multiselect('Select feature:', data.columns, default=['Rainfall_2020'])
            button = st.button("Submit")
            if button:
                numeric_data = data[selected_features].select_dtypes(include='number')
                if not numeric_data.empty:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    numeric_data.plot.line(ax=ax, marker='o')
                    ax.set_xlabel('DHSREGNA')
                    ax.set_ylabel('Average annual rainfall')
                    ax.set_title('Average annual rainfall')
                    st.pyplot(fig)
                else:
                    st.warning('No numeric data selected. Please choose numeric features.')
            else:
                st.warning('Please select one or more features.')

        # Display the interactive plot
        plot_interactive_line_chart(df2)
    
    elif subcategory == "Temperature":
        st.subheader(f"Average temperature for the year {year}")
        st.write("Definition: \nThe average temperature at the DHS survey cluster location for a given year.")
        
        m = folium.Map(location=[data['LATNUM'].mean(), data['LONGNUM'].mean()], zoom_start=8)
        #marker_cluster = MarkerCluster().add_to(m)
        min_temp = data[f'Mean_Temperature_{year}'].min()
        max_temp = data[f'Mean_Temperature_{year}'].max()

        # Define colormap thresholds (sorted)
        thresholds = [min_temp, min_temp + (max_temp - min_temp) / 3, min_temp + 2 * (max_temp - min_temp) / 3, max_temp]
        # Define colormap
        cmap = StepColormap(['green', 'yellow', 'orange', 'red'], vmin=min_temp, vmax=max_temp, index=thresholds)
        
        for index, row in data.iterrows():
            # Determine color based on Mean number of wet days
            temp = row[f'Mean_Temperature_{year}']  
            # Create circle marker with color based on wet days value
            folium.CircleMarker(
                location=[row['LATNUM'], row['LONGNUM']],
                radius=8,
                popup=f"Urban/Rural:{row['URBAN_RURA']}<br>DHS Cluster ID: {row['DHSCLUST']}<br>Region:{row['DHSREGNA']}<br>Mean Temperature: {temp:.4f}",
                color=cmap(temp),
                fill=True,
                fill_color=cmap(temp)
            ).add_to(m)

        # Add colormap legend to map
        m.add_child(cmap)

        # Display the map
        folium_static(m)

        temp_columns = ['DHSREGNA', 'Mean_Temperature_2000','Mean_Temperature_2005','Mean_Temperature_2010','Mean_Temperature_2015','Mean_Temperature_2020']
        temp_data = data[temp_columns]
        df2 = temp_data.groupby('DHSREGNA').mean()
        st.dataframe(df2)
        # Define function to plot line chart with interactive dropdown
        def plot_interactive_line_chart(data):
            selected_features = st.multiselect('Select feature:', data.columns, default=['Mean_Temperature_2020'])
            button = st.button("Submit")
            if button:
                numeric_data = data[selected_features].select_dtypes(include='number')
                if not numeric_data.empty:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    numeric_data.plot.line(ax=ax, marker='o')
                    ax.set_xlabel('DHSREGNA')
                    ax.set_ylabel('Mean Temperature')
                    ax.set_title('Mean Temperature')
                    st.pyplot(fig)
                else:
                    st.warning('No numeric data selected. Please choose numeric features.')
            else:
                st.warning('Please select one or more features.')

        # Display the interactive plot
        plot_interactive_line_chart(df2)


def visualization_category_4(data, subcategory, evi_data, pet_data, aridity_data):
    year = st.sidebar.selectbox("Select the year", options= ['2000', '2005', '2010', '2015', '2020'])
    # Add your visualization for category 3 here
    if subcategory == "Enhanced vegetation index":
        st.subheader(f"{subcategory}")
        
        st.info("The average vegetation index value at the DHS survey cluster at the time of measurement (year).\
            Vegetation index value between -1 (least vegetation) and 1 (most vegetation).")
        st.markdown("Select the year in the sidebar. The map changes in accordance with the year chosen.")
        df1 = evi_data.groupby('DHSREGNA').mean()
        m = folium.Map(location=[data['LATNUM'].mean(), data['LONGNUM'].mean()], zoom_start=8)
        cmap = LinearColormap(['green', 'yellow', 'orange', 'red'], vmin=data[f'Enhanced_Vegetation_Index_{year}'].min(), vmax=data[f'Enhanced_Vegetation_Index_{year}'].max())

        for index, row in data.iterrows():
            
            evi = row[f'Enhanced_Vegetation_Index_{year}']  

            # Create a circle marker with popup 
            folium.CircleMarker(
                location=[row['LATNUM'], row['LONGNUM']],  # Latitude and longitude from geometry column
                radius=8,
                popup=f"DHS Cluster ID:{row['DHSCLUST']}<br>Region:{row['DHSREGNA']}<br>Enhanced vegetation index: {evi:.4f}",  
                color=cmap(evi),
                fill=True,
                fill_color=cmap(evi),
            ).add_to(m)

        cmap.add_to(m)
        # Display the map
        folium_static(m)

        st.dataframe(df1)
        # Define function to plot line chart with interactive dropdown
        def plot_interactive_line_chart(data):
            selected_features = st.multiselect('Select feature:', data.columns, default=['Enhanced_Vegetation_Index_2020'])
            if selected_features:
                numeric_data = data[selected_features].select_dtypes(include='number')
                if not numeric_data.empty:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    numeric_data.plot.line(ax=ax, marker='o')
                    ax.set_xlabel('Regions')
                    ax.set_ylabel('Enhanced vegetation index')
                    ax.set_title('Enhanced vegetation index(EVI)')
                    st.pyplot(fig)
                else:
                    st.warning('No numeric data selected. Please choose numeric features.')
            else:
                st.warning('Please select one or more features.')

        # Display the interactive plot
        plot_interactive_line_chart(df1)

    elif subcategory == "Potential Evapotranspiration":
        st.subheader(f"{subcategory}")
        st.info("The average potential evapotranspiration (PET) at the DHS survey cluster location. This dataset was produced by taking the average of the twelve monthly datasets, which represent millimeters\
                    per day, for a given year.")
        st.info("PET: The number shows the number millimeters of water that would be evaporated into the air over the course of a year if there was\
                    unlimited water at the location.")
        st.markdown("Select the year in the sidebar. The map changes in accordance with the year chosen.")
        df2 = pet_data.groupby("DHSREGNA").mean()
        m = folium.Map(location=[data['LATNUM'].mean(), data['LONGNUM'].mean()], zoom_start=8)
        cmap = LinearColormap(['green', 'yellow', 'orange', 'red'], vmin=data[f'PET_{year}'].min(), vmax=data[f'PET_{year}'].max())

        for index, row in data.iterrows():
     
            pet = row[f'PET_{year}']  
            # Create a circle marker with popup

            folium.CircleMarker(
                location=[row['LATNUM'], row['LONGNUM']],  # Latitude and longitude from geometry column
                radius=8,
                popup=f"DHS Cluster ID:{row['DHSCLUST']}<br>Region:{row['DHSREGNA']}<br>Urban/Rural:{row['URBAN_RURA']}<br>PET: {pet:.4f}",  
                color=cmap(pet),
                fill=True,
                fill_color=cmap(pet),
            ).add_to(m)

        cmap.add_to(m)
        # Display the map
        folium_static(m)

        st.dataframe(df2)
        # Define function to plot line chart with interactive dropdown
        def plot_interactive_line_chart(data):
            selected_features = st.multiselect('Select feature:', data.columns, default=['PET_2020'])
            if selected_features:
                numeric_data = data[selected_features].select_dtypes(include='number')
                if not numeric_data.empty:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    numeric_data.plot.line(ax=ax, marker='o')
                    ax.set_xlabel('Regions')
                    ax.set_ylabel('PET(in mm)')
                    ax.set_title('Average Potential Evapotranspiration(PET)')
                    # Add annotations to data points
                    for column in numeric_data.columns:
                        for i, value in enumerate(numeric_data[column]):
                            ax.annotate(f'{value:.4f}', (i, value), textcoords="offset points", xytext=(0,12), ha='center')
                    st.pyplot(fig)
                else:
                    st.warning('No numeric data selected. Please choose numeric features.')
            else:
                st.warning('Please select one or more features.')
        plot_interactive_line_chart(df2)

    elif subcategory == "Aridity Index":
        st.subheader(f"{subcategory}")
        st.markdown("""The dataset represents the average monthly precipitation divided by average monthly potential
            evapotranspiration, an aridity index defined by the United Nations Environmental Programme
            (UNEP).""")
        st.info("Aridity Index (AI), which is defined as the ratio of annual precipitation to annual potential\
            evapotranspiration, is a key parameter in drought characterization. Index between 0 (most arid) and 300 (most wet)")
        st.markdown("Select the year in the sidebar. The map changes in accordance with the year chosen.")
        df3 = aridity_data.groupby("DHSREGNA").mean()
        m = folium.Map(location=[data['LATNUM'].mean(), data['LONGNUM'].mean()], zoom_start=8)
        cmap = LinearColormap(['green', 'yellow', 'orange', 'red'], vmin=data[f'Aridity_{year}'].min(), vmax=data[f'Aridity_{year}'].max())

        for index, row in data.iterrows():
            # Determine color based on ITN coverage intensity
            aridity = row[f'Aridity_{year}']  #Convert to percentage

            # Create a circle marker with popup displaying ITN coverage value
            folium.CircleMarker(
                location=[row['LATNUM'], row['LONGNUM']],  # Latitude and longitude from geometry column
                radius=8,
                popup=f"Urban/Rural:{row['URBAN_RURA']}<br>DHS Cluster ID:{row['DHSCLUST']}<br>Region:{row['DHSREGNA']}<br>Urban/Rural:{row['URBAN_RURA']}<br>Aridity: {aridity:.4f}",  
                color=cmap(aridity),
                fill=True,
                fill_color=cmap(aridity),
            ).add_to(m)

        cmap.add_to(m)
        # Display the map
        folium_static(m)

        st.dataframe(df3)
        # Define function to plot line chart with interactive dropdown
        def plot_interactive_line_chart(data):
            selected_features = st.multiselect('Select feature:', data.columns, default=['Aridity_2020'])
            if selected_features:
                numeric_data = data[selected_features].select_dtypes(include='number')
                if not numeric_data.empty:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    numeric_data.plot.line(ax=ax, marker='o')
                    ax.set_xlabel('Regions')
                    ax.set_ylabel('Aridity index')
                    ax.set_title('Aridity index')
                    # Add annotations to data points
                    for column in numeric_data.columns:
                        for i, value in enumerate(numeric_data[column]):
                            ax.annotate(f'{value:.4f}', (i, value), textcoords="offset points", xytext=(0,12), ha='center')
                    st.pyplot(fig)
                else:
                    st.warning('No numeric data selected. Please choose numeric features.')
            else:
                st.warning('Please select one or more features.')
        plot_interactive_line_chart(df3)
    
    else:
        st.markdown("About")
        st.markdown("-----------------------------------------")
        st.write("The page on Agriculture has been divided into three subcategories: \
            1)Enhanced vegetation index,\
            2)Potential Evapotranspiration and \
            3)Aridity Index.\
                The map of Liberia is displayed for the years 2000, 2005, 2010, 2015 and 2020.This data has been obtained from The DHS Program.\
            ")
        st.markdown("-----------------------------------------")


def visualization_category_5(data, subcategory, df_agri):
        
    if subcategory == "Households with farm animals and agricultural land based on 2019-20 DHS Survey":
        df_agri = df_agri.loc[np.where(df_agri['Survey'] == "2019-20 DHS")] #Selecting the MIS 2022 Survey data
        df_agri = df_agri.reset_index(drop=True)
        groups_counties = [val for val in df_agri['Characteristic'].values if val.startswith('Groups of Counties')]
        counties = [val for val in df_agri['Characteristic'].values if val.startswith('Counties')]
        st.subheader("2019-20 DHS Survey data")
        st.markdown("Choose the category in the sidebar. There are four categories: 'Group of Counties','Counties','Wealth quintiles' and 'Residence type'. The graph and the data changes here as per the chosen category.")
        select_cat = st.sidebar.selectbox("Choose the category", ['Group of Counties', 'Counties','Wealth quintiles','Residence type'])
        if select_cat == "Residence type":
            df_subset = df_agri[1:3].copy()  # Ensure to make a copy to avoid modifying the original DataFrame
        elif select_cat == "Wealth quintiles":
            df_subset = df_agri[3:8].copy()
        elif select_cat == "Group of Counties":
            mask = df_agri.isin(groups_counties).any(axis=1)
            groups_of_counties = df_agri[mask]
            df_subset = groups_of_counties
        else:
            mask2 = df_agri.isin(counties).any(axis=1)
            counties_df = df_agri[mask2]
            df_subset = counties_df
            
        # Create a horizontal bar plot using Plotly Express
        fig = px.bar(df_subset, y='Characteristic', x=['Households owning agricultural land', 'Households owning farm animals'], orientation='h',
                    title=f"Households owning agriculture lands and farm animals by {select_cat}",
                    labels={'value': 'Percentage', 'Characteristic': f'{select_cat}'})

        # Customize the legend position
        fig.update_layout(legend=dict(x=1.02, y=1.0))

        # Display the Plotly chart in Streamlit
        st.plotly_chart(fig)
        st.write(df_subset.reset_index(drop=True))
        with st.expander("Findings"):
            st.write("""
            1) ***Households owning agricultural land :***
            * Counties : Nimba > Lofa > Bomi > River Gee > River Cess
            * Regions: North Central > North Western > South Eastern A > South Eastern B > South Central

            2) ***Households owning farm animals:***
            * Counties: Grand Kru > River Gee > Nimba > River Cess > Gharpola
            * Regions: South Eastern A > South Eastern B > North Central > North Western > South Central
            """)

    elif subcategory == "Households with farm animals and agricultural land based on 2022 MIS":
        df_agri = df_agri.loc[np.where(df_agri['Survey'] == "2022 MIS")] #Selecting the MIS 2022 Survey data
        st.subheader("2022 MIS Survey data")
        st.markdown("Choose the category in the sidebar. There are three categories: 'Group of Counties','Wealth quintiles' and 'Residence type'. The graph and the data changes here as per the chosen category.")
        select_cat = st.sidebar.selectbox("Choose the category", ['Group of Counties','Wealth quintiles','Residence type'])
        if select_cat == "Group of Counties":
            df_subset = df_agri[8:13].copy()  # Ensure to make a copy to avoid modifying the original DataFrame
        elif select_cat == "Wealth quintiles":
            df_subset = df_agri[3:8].copy()
        else:
            df_subset = df_agri[1:3].copy()
        # Create a horizontal bar plot using Plotly Express
        fig = px.bar(df_subset, y='Characteristic', x=['Households owning agricultural land', 'Households owning farm animals'], orientation='h',
                    title=f"Households owning agriculture lands and farm animals by {select_cat}",
                    labels={'value': 'Percentage', 'Characteristic': f'{select_cat}'})

        # Customize the legend position
        fig.update_layout(legend=dict(x=1.02, y=1.0))

        # Display the Plotly chart in Streamlit
        st.plotly_chart(fig)
        st.write(df_subset.reset_index(drop=True))
        with st.expander("Findings"):
            st.write("""
            1) ***Households with agricultural land holdings:***
            * Region: South eastern A >South eastern B > North Western > South Central > Monrovia
            * Residence: Rural > Urban
            * Wealth : Second > Lowest > Middle > Fourth > Highest


            2) ***Households owning farm animals:***
            * Region: South eastern A> South eastern B >North Western > South Central > Monrovia
            * Wealth quintiles : Second >Lowest > middle >Fourth > Highest
            * Residence : Rural > Urban
            """)

def main():
    st.title("Liberia Statistics")

    data, malaria_data, itn_data, columns, wet_days_columns, month_temp_columns, rainfall_columns, malaria_prevalence_data, pop_density_data, evi_data, pet_data, df_demography, df_agri, df_children_malaria, aridity_data = read_dataset()

    # Define main categories and their corresponding subcategories
    categories = {
        'Demography': ['UN Population','Under-5 Population','Population'],
        'Health': ['Health facilities', 'ITN Coverage', 'Malaria Incidence','Malaria Prevalence','Malaria prevalence in children(6-59) months', 'Malaria cases by Species'],
        'Environment': ['Enhanced vegetation index','Potential Evapotranspiration','Aridity Index'],
        'Climate':['Mean wet days','Rainfall','Temperature'],
        'Agriculture':['Households with farm animals and agricultural land based on 2022 MIS','Households with farm animals and agricultural land based on 2019-20 DHS Survey']
        # Add more main categories and subcategories as needed
    }

    # Create a select box for main categories
    main_category = st.sidebar.selectbox('Select a main category', list(categories.keys()))

    # Create a select box for subcategories based on the selected main category
    if main_category:
        subcategories = categories[main_category]
        subcategory = st.sidebar.selectbox('Select a subcategory', ['About'] + subcategories)

    if main_category == "Demography":
        visualization_category_1(data,subcategory,pop_density_data, df_demography)
    elif main_category == "Health":
        visualization_category_2(data,malaria_data,itn_data,columns,subcategory,malaria_prevalence_data, df_children_malaria, df_demography)
    elif main_category == "Climate":
        visualization_category_3(data,wet_days_columns, month_temp_columns, rainfall_columns,subcategory)
    elif main_category == "Environment":
        visualization_category_4(data,subcategory,evi_data, pet_data, aridity_data)
    elif main_category == "Agriculture":
        visualization_category_5(data, subcategory, df_agri)
    else:
        pass

    


if __name__=="__main__":
    main()



