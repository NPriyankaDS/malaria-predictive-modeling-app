import streamlit as st
from streamlit_extras.let_it_rain import rain
from streamlit_extras.colored_header import colored_header
from page_utils import font_modifier, display_image


################### HEADER SECTION #######################
display_image.display_image('https://cdn-images-1.medium.com/max/800/0*vBDO0wwrvAIS5e1D.png')

st.markdown("<h1 style='text-align: center; color: #F5EFE6;'>Developing an AI-powered App for Predictive Modeling and Forecasting of Malaria Prevention in Liberia</h1>",
            unsafe_allow_html=True)

display_image.display_image('https://miro.medium.com/v2/resize:fit:10368/1*Aa35cz76rGh6PiDb2UEs8w.jpeg')

st.markdown("<h4 style='text-align: center; color: #FFFFD0; font-family: Segoe UI;'>A web-based Machine Learning Model for Classifying Malaria based on certain parameters.</h3>", unsafe_allow_html=True)

################### FILE UPLOAD SECTION #######################
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])





font_modifier.make_font_poppins()