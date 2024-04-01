from page_utils import font_modifier, display_image
import streamlit as st

font_modifier.make_font_poppins()

st.header('Usage Guidelines')
st.markdown(
            """
            <style>
            .tab {
                text-indent: 0px;  /* adjust as needed */
                text-align: justify;  /* Add this line */
            }
            </style>
            <div class="tab" style="text-align=justify;">TBD</div>
            <p></p>
            
            """
            ,unsafe_allow_html=True)
