import streamlit as st
from streamlit_extras.let_it_rain import rain
from streamlit_extras.colored_header import colored_header
from page_utils import font_modifier, display_image


################### HEADER SECTION #######################
display_image.display_image('https://cdn-images-1.medium.com/max/800/0*vBDO0wwrvAIS5e1D.png')

st.markdown("<h1 style='text-align: center; '>Develop an AI-powered App for Predictive Modeling and Forecasting of Malaria Prevention in Liberia</h1>",
            unsafe_allow_html=True)

display_image.display_image('./static/home_page_cover.jpg')

################### INFORMATION SECTION #######################
st.header('🌊 Challenge Background')
st.markdown(
            """
            <style>
            .tab {
                text-indent: 0px;  /* adjust as needed */
                text-align: justify;  /* Add this line */
            }
            </style>
            <div class="tab" style="text-align=justify;">Malaria is a significant public health burden in Liberia, causing widespread illness and death, particularly among children and pregnant women. Traditional methods of malaria prevention and control, such as distribution of antimalarial drugs, and Mosquito nets have been effective to some extent, but the challenge in preventing malaria remains a serious concern, according to the WHO's Country Disease Outlook for Liberia (August 2023), the estimated mortality rate for malaria in 2021 was 3,548 deaths. This translates to a 1.9 deaths per 1,000 population (incidence rate of 358.5 cases). One major challenge in tackling malaria prevention in Liberia has been weak surveillance systems that is Inadequate to monitor malaria cases.</div>
            <p></p>
            
            """
            ,unsafe_allow_html=True)

st.header('🧩 The Problem')
st.markdown(
            """
            <style>
            .tab {
                text-indent: 0px;  /* adjust as needed */
                text-align: justify;  /* Add this line */
            }
            </style>
            <div class="tab" style="text-align=justify;">In the West African nation of Liberia, malaria remains a significant public health burden. Despite concerted efforts with traditional intervention strategies like vector control and chemoprevention, malaria transmission persists, disproportionately impacting vulnerable populations like children and pregnant women. This mosquito-borne parasite claims thousands of lives every year. The seamless integration of this AI-powered application is anticipated to yield substantial positive outcomes for malaria prevention in Liberia, including: 1. Decreased rates of malaria transmission. 2. Optimized allocation of resources for precisely targeted interventions. 3. Strengthened preparedness and rapid response capabilities during malaria outbreaks. 4. Enhanced health outcomes for vulnerable populations. 5. Valuable contribution towards the overarching objective of malaria elimination in Liberia.</div>
            <p></p>
            
            """
            ,unsafe_allow_html=True)

st.header('🎯 Goal of the Project')
st.markdown(
            """
            <style>
            .tab {
                text-indent: 0px;  /* adjust as needed */
                text-align: justify;  /* Add this line */
            }
            </style>
            <div class="tab" style="text-align=justify;">The goal of this project is to develop an AI-powered app that utilizes predictive modeling and forecasting techniques to enhance malaria prevention efforts in Liberia. The app should incorporate three key functionalities: 1)Predicting malaria transmission risk: Identify areas and populations at high risk of malaria outbreaks based on historical data, climate patterns, and human behavior. 2)Forecasting malaria outbreaks: Predict the timing and severity of future malaria outbreaks using real-time data on weather patterns, mosquito populations, and human mobility. 3)Identifying environmental and social determinants of malaria: Analyze large datasets to identify factors that contribute to malaria transmission and vulnerability, informing broader interventions. The development of an AI-powered app for predictive modeling and forecasting of malaria prevention in Liberia will require a range of AI tools and technologies. Here are some of the key tools that will be needed: - Machine learning algorithms: These algorithms will be used to analyze historical data on malaria transmission, climate patterns, and human behavior to identify areas at high risk of malaria outbreaks, forecast the timing and severity of future outbreaks.                    - Data visualization tools: These tools will be used to display the results of the AI analyses in a user-friendly way, such as using maps, charts, or graphs. This will make it easier for public health officials and other users to understand the information and act.</div>
            <p></p>
            
            """
            ,unsafe_allow_html=True)

font_modifier.make_font_poppins()


# st.markdown("<h4 style='text-align: center; color: #FFFFD0; font-family: Segoe UI;'>A web-based Machine Learning Model for Classifying Malaria based on certain parameters.</h3>", unsafe_allow_html=True)

################### FILE UPLOAD SECTION #######################
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

font_modifier.make_font_poppins()