import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Load CSS for styling
with open('./chart.css') as f:
    css = f.read()

st.set_page_config(layout="wide")
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

# Load the pre-trained model and match data
pipe = pickle.load(open('../Model/logistic_regression.pkl', 'rb')) 
final_match_data = pd.read_csv('../Dataset/final_match_data_with_id.csv')
final_match_data['Match_Date'] = pd.to_datetime(final_match_data['Match_ID'].str.split('_').str[0], format='%Y-%m-%d')
final_match_data = final_match_data.sort_values(by='Match_Date', ascending=False)

final_match_data['Match_Format'] = final_match_data['Match_Date'].dt.strftime('%d-%m-%Y') + ': ' + final_match_data['Batting_Team'] + ' vs ' + final_match_data['Bowling_Team']

st.title('Cricket Match Prediction App')

years = final_match_data['Match_Date'].dt.year.unique()

col1, col2 = st.columns([0.4, 1])

with col1:
    selected_year = st.selectbox('Select Year', years)
    matches_of_selected_year = final_match_data[final_match_data['Match_Date'].dt.year == selected_year]

with col2:
    match_formats = matches_of_selected_year['Match_Format'].unique()
    selected_match = st.selectbox('Select Match', match_formats)
    selected_match_data = matches_of_selected_year[matches_of_selected_year['Match_Format'] == selected_match]


# Data preparation
batting_team = selected_match_data['Batting_Team']
bowling_team = selected_match_data['Bowling_Team']
selected_city = selected_match_data['City']
target = selected_match_data['Total_Runs_x']
runs_left = selected_match_data['runs_left']
balls_left = selected_match_data['balls_left']
wickets_left = selected_match_data['wickets_left']
crr = selected_match_data['crr']
rrr = selected_match_data['rrr']

input_df = pd.DataFrame({'Batting_Team': batting_team, 'Bowling_Team': bowling_team, 'City': selected_city, 'Total_Runs_x': target,
                         'runs_left': runs_left, 'balls_left': balls_left, 'wickets_left': wickets_left,
                         'crr': crr, 'rrr': rrr})
input_df = input_df.sort_values(by='balls_left', ascending=False)

predictions_win = []
predictions_loss = []

for index, row in input_df.iterrows():
    input_data = row.to_frame().T
    prediction = pipe.predict_proba(input_data)
    win_probability = round(prediction[0][0] * 100)  
    loss_probability = round(prediction[0][1] * 100) 

    predictions_win.append(win_probability)
    predictions_loss.append(loss_probability)

input_df = input_df[['runs_left', 'balls_left', 'wickets_left']]

win_column_name = f"{selected_match_data['Bowling_Team'].iloc[0]} Win %"
loss_column_name = f"{selected_match_data['Batting_Team'].iloc[0]} Win %"

input_df[win_column_name] = predictions_win
input_df[loss_column_name] = predictions_loss

input_df['Overs'] = (20 - input_df['balls_left'] // 6)

# Plotting with Matplotlib
fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(input_df['Overs'], input_df[win_column_name], label=f"{selected_match_data['Bowling_Team'].iloc[0]} Win %", linewidth=2)
ax.plot(input_df['Overs'], input_df[loss_column_name], label=f"{selected_match_data['Batting_Team'].iloc[0]} Win %", linewidth=2)
ax.set_xlabel('Overs')
ax.set_ylabel('Runs Left/Win %')
ax.set_title('Runs Left vs Overs')
ax.set_xticks(range(0, 21))
ax.set_yticks(range(0, 101, 10))
ax.legend()
st.pyplot(fig)
