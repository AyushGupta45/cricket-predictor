import streamlit as st
import pickle
import pandas as pd
from pathlib import Path

# Get the path of the current directory
current_directory = Path(__file__).resolve().parent

# Load CSS for styling
css_file_path = current_directory / "style.css"
with open(css_file_path) as f:
    css = f.read()

st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

teams = [
    'Sunrisers Hyderabad',
    'Mumbai Indians',
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders',
    'Punjab Kings',
    'Chennai Super Kings',
    'Rajasthan Royals',
    'Delhi Capitals',
    'Gujarat Titans',
    'Lucknow Super Giants'
]

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi', 'Jaipur', 'Chennai', 'Ahmedabad', 'Dharamsala', 'Mohali', 'Lucknow']


model_file_path = current_directory.parent / "Model" / "logistic_regression.pkl"
pipe = pickle.load(open(model_file_path, 'rb')) 

def limit_overs_input():
    current_overs = st.session_state.overs
    decimal_part = current_overs % 1
    if decimal_part >= 0.5:
        new_overs = int(current_overs) + 1
    else:
        new_overs = current_overs
    st.session_state.overs = new_overs

if 'overs' not in st.session_state:
    st.session_state['overs'] = 15.0

st.title('Live Match Prediction Tool')

col1, col2, col3 = st.columns(3)

with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams), index=0)
with col2:
    bowling_team = st.selectbox('Select the bowling team', sorted(teams), index=5)
with col3:
    selected_city = st.selectbox('Select host city', sorted(cities), index=11)

col1, col2 = st.columns(2)
with col1:
    target = st.number_input('Target', value=240)
with col2:
    score = st.number_input('Score', value=150)

col1, col2 = st.columns(2)

with col1:
    overs = st.number_input('Overs completed', min_value=0.0, max_value=20.0, value=float(st.session_state['overs']), format="%.1f", step=0.1, key='overs', on_change=limit_overs_input)
with col2:
    wickets = st.number_input('Wickets out', min_value=0, max_value=10, value=5)


if st.button('Predict Probability'):
    runs_left = target - score
    balls_left = 120 - (overs*6)
    wickets_left = 10 - wickets
    crr = score/overs
    rrr = (runs_left*6)/balls_left

    input_df = pd.DataFrame({'Batting_Team': [batting_team], 'Bowling_Team': [bowling_team], 'City': [selected_city],
                             'runs_left': [runs_left], 'balls_left': [balls_left], 'wickets_left': [wickets_left],
                             'Total_Runs_x': [target], 'crr': [crr], 'rrr': [rrr]})

    

    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]
    st.header(batting_team + "- " + str(round(win*100)) + "%")
    st.header(bowling_team + "- " + str(round(loss*100)) + "%")
