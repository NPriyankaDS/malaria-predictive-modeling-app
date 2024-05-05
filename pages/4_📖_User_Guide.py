from page_utils import font_modifier, display_image
import streamlit as st
import requests
import base64

font_modifier.make_font_poppins()

st.header('User Guides')
# st.markdown(
#             """
#             <style>
#             .tab {
#                 text-indent: 0px;  /* adjust as needed */
#                 text-align: justify;  /* Add this line */
#             }
#             </style>
#             <div class="tab" style="text-align=justify;">TBD</div>
#             <p></p>
            
#             """
#             ,unsafe_allow_html=True)

st.subheader("The Geospatial Covariate Datasets Manual from DHS Program")

# Function to fetch PDF data from URL
def fetch_pdf_data(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.content
    else:
        st.error(f"Failed to fetch PDF data from {url}. Status code: {response.status_code}")
        return None

# URL from which to fetch PDF data
pdf_url = "https://spatialdata.dhsprogram.com/references/DHS_Covariates_Extract_Data_Description_3.pdf"

# Fetch PDF data
pdf_data = fetch_pdf_data(pdf_url)

# Check if PDF data is fetched successfully
if pdf_data is not None:
    pdf_base64 = base64.b64encode(pdf_data).decode('utf-8')
    # Display PDF
    
    st.markdown(f'<iframe src="data:application/pdf;base64,{pdf_base64}" width="700" height="1000" style="border: none;"></iframe>', unsafe_allow_html=True)