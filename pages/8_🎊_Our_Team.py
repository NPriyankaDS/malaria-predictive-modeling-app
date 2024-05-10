import streamlit as st
from page_utils import font_modifier, display_image

#display_image.display_image('https://cdn-images-1.medium.com/max/800/0*vBDO0wwrvAIS5e1D.png')

st.header(":orange[Meet the Team] :tada:")
st.divider()
#st.markdown("<h1 style='text-align: center; '>Meet the Team :tada:</h1>",unsafe_allow_html=True)

team = {
    "SENIOR RESEARCH FELLOW": ["Oscar Daniel Murgueytio Panana"],
	"RESEARCHERS": ["Paulo Nascimento", "Ahmed Daker", "Fauzia Khan Mohamed Abdulla", "Jannath Fatima Mohammed", "Shanawaz Anwar"],
	"DATA COLLECTORS": ["Arnav Kumar", "Bartequa Blackmon", "Elias Dzobo", "Fathima Shanavas", "Friday Emmanuel James", "Hareesh Haridas", "Jayashree Subramanian",\
        "Mansi Upadhyay","Peter Mahuthu","Poornachander Pothana","Rishab Bandodkar","Sahil Bhandari","Tejansh Sachdeva",\
        "Tinotenda Mangarai","Trishika Boyila"],
	"LEAD DATA ANALYST": ["Ekeke Chidiebere Chiwuikem"],
	"DATA ANALYSTS": ["Dorothea Paulssen","Linda Oraegbunam","Raul Catacora","Rohit Bhalode","Satvik Rajesh"],
	"DATA SCIENTISTS": ["Ndong Henry Ndang","Sindhusha Nannapaneni","Vivien Siew"],
    "LEAD ML ENGINEER": ["Thomas James"],
	"ML ENGINEERS": ["Aditya Sharma","Dinesh Kumar M","Muhtasim Ibteda Shochcho","N Priyanka", "Sahar Nikoo", "Ayaluri Sri Kaushik"],
    "DATA VISUALIZATION DEVELOPER": ["Maria Loureiro"],
    "MODEL DEPLOYMENT ENGINEERS": ["Lanz Vincent T. Vencer","Varshitha M"],
    }



st.markdown("<h1 style= 'color: blue; text-align: center; '>PROJECT MANAGER</h1>",unsafe_allow_html=True)
st.markdown(f"<h2 style= 'text-align: center; '>Daikukai Bindah</h2>", unsafe_allow_html=True)
st.markdown("-------------------------------------")

# Define the number of columns for displaying roles
num_columns = 2

# Calculate the number of roles per column
roles_per_column = len(team) // num_columns
remainder = len(team) % num_columns

for i in range(roles_per_column + (1 if remainder > 0 else 0)):
    columns = st.columns(num_columns)
    for j in range(num_columns):
        role_index = i + j * roles_per_column
        if j < remainder:
            role_index += j
        else:
            role_index += remainder
        if role_index < len(team):
            role, members = list(team.items())[role_index]
            with columns[j]:
                st.header(f":blue[{role}]")
                for member in members:
                    st.write(member)