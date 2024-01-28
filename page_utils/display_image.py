import streamlit as st


def display_image(url, caption=None):
    st.markdown("<p></p>",unsafe_allow_html=True)
    image_url = url
    st.image(image_url, caption=caption, use_column_width=True)
    st.markdown("<p></p>",unsafe_allow_html=True)