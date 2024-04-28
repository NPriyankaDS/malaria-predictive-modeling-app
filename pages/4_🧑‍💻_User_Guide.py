from page_utils import font_modifier, display_image
import streamlit as st
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

# Function to read and display PDF content
def displayPDF(file):
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # Embedding PDF in HTML
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)

displayPDF("./data/DHS Covariates Extract Data Description_3.pdf")
