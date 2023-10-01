import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot  as plt
import altair as alt
import pickle
import folium
from streamlit_folium import st_folium

#---MAP---
embarked_ports = {
    "Southampton": {"lat": 50.9097, "lon": -1.4044},
    "Cherbourg": {"lat": 49.6301, "lon": -1.619},
    "Queenstown": {"lat": 51.8496, "lon": -8.2976},
}

# Create a Folium map centered on the first embarked port
default_location = list(embarked_ports.values())[0]
m = folium.Map(location=[default_location["lat"], default_location["lon"]], zoom_start=5)

# Add markers for embarked ports
for port, coordinates in embarked_ports.items():
    folium.Marker([coordinates["lat"], coordinates["lon"]], tooltip=port).add_to(m)

# Function to render Folium map
def render_folium_map():
    st_folium(m, width=700, height=300)

#load pickle file
model = pickle.load(open(r"titanic_v0.pkl", 'rb'))

#---MAIN---
def main():
    st.set_page_config(
        page_title="Titanic Survival Predicition App",
        page_icon="🚢",
        layout="centered",
        initial_sidebar_state="expanded"
    )
    #main title/img/caption
    st.title("Titanic Survival Prediction with ML")
    st.image(r"imgs\titanicSinking.jpg", caption="The Titanic | 14th April 1912", use_column_width=True)
    st.write("""
    Ahoy there! Welcome to my Titanic Survival Prediction App! Ever wondered if you'd have made it on the Titanic's infamous voyage in 1912? Well, now you can find out. Just input your details, click "Predict," and uncover your fate on that fateful night. Let's set sail and explore your odds of survival together!
    """)
    url = "https://github.com/AnnaSmith3110/Data-Analysis/blob/main/titanic_v0.ipynb"
    st.markdown("<br>***Interest in the machine learning model behind this app?*** Check out this [link](%s)"% url, unsafe_allow_html=True)
    st.divider()
    st.header("""Would you have survived the Titanic Disaster?""")

    #sidebar info
    #get user input
    st.sidebar.title("Your Results:")
    col1, col2 = st.columns(2)
    
    with col1:
        #containers for ui
        age = st.number_input('What is your Age?', step=1)
        sex = st.selectbox("Select yout Gender: ", ["male", "female"])
        if sex == "male":
            Sex = 0
        else:
            Sex = 1

        SibSp = st.selectbox("How many siblings/spouses do you have on board?", [0,1,2,3,4,5,6,7,8])
        Parch = st.selectbox("How many parents/children do you have on borard?", [0,1,2,3,4,5,6,7,8])
    with col2:
        fare = st.slider("Enter your Ticket Fare: ", 15, 500, 40)
        Pclass = st.selectbox("Select Passenger Class: ", [1,2,3])
        boarded_location = st.selectbox("Boarded From: ", ["Cherbourg", "Queenstown", "Southampton"])
        Embarked_C, Embarked_Q, Embarked_S = 0,0,0
        if boarded_location == "Cherbourg":
            Embarked_C = 1
        elif boarded_location == "Queenstown":
            Embarked_Q = 1
        else:
            Embarked_S = 1
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Titanic Embarking Ports**")
    render_folium_map()

    #framing user data into pandas df
    #pclass, sex, age, sibsp, parch, fare, embarked_c, embarked_q, embarked_s
    data = {
        "Pclass": Pclass,
        "Sex": Sex,
        "Age": age,
        "SibSp": SibSp,
        "Parch": Parch,
        "Fare": fare,
        "Embarked_C": Embarked_C,
        "Embarked_Q": Embarked_Q,
        "Embarked_S": Embarked_S
    }
    df = pd.DataFrame(data, index=[0])
    return df
    
#calling main
data = main()   

# Predict function
def predict_survival(data):
    result = model.predict(data)
    proba = model.predict_proba(data)
    return result[0], proba[0,0]*100, proba[0,1]*100


#prediction
col1, col2, col3 = st.columns(3)
with col1:
    st.write(' ')
with col2:
    if st.button("Predict Results", key="predict_button", help="Click to see your survival chances", type="primary"):
        result, no_prob, yes_prob = predict_survival(data)

        if result == 1:
            st.sidebar.success("Congrats you made it!")
            st.sidebar.write("Your chances of survival are high")
            st.sidebar.markdown("<br>", unsafe_allow_html=True)
        else:
            st.sidebar.error("Oops..RIP")
            st.sidebar.write("Your chances of survival are low")
            st.sidebar.markdown("<br>", unsafe_allow_html=True)
        
        st.sidebar.subheader("Survival chances:")
        col1, col2 = st.columns(2)

        # Define custom colors (green and red shades)
        colors = ['#D9534F', '#5CB85C']  # Green and Red

        with col1:
            # labels = ['Not Survived', 'Survived']
            # sizes = [no_prob, yes_prob]
            # fig, ax = plt.subplots()
            # ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
            # ax.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
            # st.sidebar.pyplot(fig)
            
            # Define your data
            labels = ['Not Survived', 'Survived']
            sizes = [no_prob, yes_prob]
            data = pd.DataFrame({'labels': labels, 'sizes': sizes})

            # Create a pie chart using Altair
            pie_chart = alt.Chart(data).mark_arc().encode(
                theta='sizes:Q',
                color=alt.Color('labels:N', scale=alt.Scale(range=['#D9534F', '#5CB85C'])),
                tooltip=['labels', 'sizes']
            ).properties(
                width=250,
                height=250
            )

            # Display the pie chart in the sidebar
            st.sidebar.altair_chart(pie_chart, use_container_width=True)

        with col2:
            # Define your table data
            table_data = {
                "Survival Chances": ["No", "Yes"],
                "Percentage": [f"{no_prob:.2f}%", f"{yes_prob:.2f}%"]
            }
            df = pd.DataFrame(table_data)
            st.sidebar.markdown(df.style.hide(axis="index").to_html(), unsafe_allow_html=True)
            st.sidebar.markdown("<br>", unsafe_allow_html=True)

    else:
        st.sidebar.write("Fill the form and click the predict button!")
with col3:
    st.write(' ')
