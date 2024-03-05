import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go

with open('./style.css') as f:
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

short_forms = {
    'Sunrisers Hyderabad': 'SRH',
    'Mumbai Indians': 'MI',
    'Royal Challengers Bangalore': 'RCB',
    'Kolkata Knight Riders': 'KKR',
    'Punjab Kings': 'PBKS',
    'Chennai Super Kings': 'CSK',
    'Rajasthan Royals': 'RR',
    'Delhi Capitals': 'DC',
    'Gujarat Titans': 'GT',
    'Lucknow Super Giants': 'LSG'
}

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
          'Chandigarh', 'Jaipur', 'Chennai',
          'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
          'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
          'Sharjah', 'Mohali', 'Bengaluru']


team_colors = {
    'Sunrisers Hyderabad': '#FE6B18',
    'Mumbai Indians': '#004AA0',
    'Royal Challengers Bangalore': '#BC1520',
    'Kolkata Knight Riders': '#2E0854',
    'Punjab Kings': '#D50032',
    'Chennai Super Kings': '#FFF05B',
    'Rajasthan Royals': '#FFC0CB',
    'Delhi Capitals': '#004D80',
    'Gujarat Titans': '#1E1E28',
    'Lucknow Super Giants': '#660066'
}

ball = [0,1,2,3,4,5]
filtered_ball = [1,2,3,4,5]
def_ball = 0


pipe = pickle.load(open('logistic_regression.pkl', 'rb'))


st.title('IPL Win Predictor')

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams), index=0)
with col2:
    bowling_team = st.selectbox('Select the bowling team', sorted(teams), index=5)

selected_city = st.selectbox('Select host city', sorted(cities), index=14)

col1, col2 = st.columns(2)

with col1 :
    target = st.number_input('Target', value=250, min_value=1)

with col2:
    score = st.number_input('Current Score', value=200, min_value=0,max_value = target-1)

col4, col5, col6 = st.columns(3)

with col4:
    overs = st.number_input('Overs completed', value=16,max_value=19, step=1,min_value=0)


if(overs==20):
    with col5:
        balls = st.selectbox('Maximum overs are bowled',sorted(ball), disabled=True)

elif(overs==0):
    with col5:
        balls = st.selectbox('Balls bowled in current over',sorted(filtered_ball), index=ball.index(def_ball))
else:
    with col5:
        balls = st.selectbox('Balls bowled in current over',sorted(ball))
with col6:
    wickets = st.number_input('Wickets out', max_value=10, value=5,min_value=0)


    

button = st.button('Predict Probability')



    
def ipl_win_predictor():    
    runs_left = target - score
    balls_left = 120 - ((overs*6) + (balls))
    wickets_left = 10 - wickets
    crr = score/(overs + balls)
    rrr = (runs_left*6)/(balls_left)
    input_df = pd.DataFrame({'Batting_Team': [batting_team], 'Bowling_Team': [bowling_team], 'City': [selected_city],
                            'runs_left': [runs_left], 'balls_left': [balls_left], 'wickets_left': [wickets_left],
                            'Total_Runs_x': [target], 'crr': [crr], 'rrr': [rrr]})

    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]
    
    if(wickets >= 10 or balls_left==0 or rrr>36):
        win_chart=0
    else:
        win_chart= round(win*100)
    loss_chart= round(loss*100)


    labels = [batting_team, bowling_team]
    values = [win_chart, loss_chart]

    colors = [team_colors.get(batting_team, 'gray'), team_colors.get(bowling_team, 'gray')]
    
    explode = [0.1, 0]

    fig = go.Figure()

    fig.add_trace(go.Pie(labels=labels, values=values, hole=0.0, marker=dict(colors=colors), pull=explode))

    st.write("\n")
    st.write("<h3>Probability of Result</h3>", unsafe_allow_html=True)
    fig.update_layout( showlegend=True)
    fig.update_layout(width=400, height=400)
    fig.update_layout(legend=dict(x=1, y=0, orientation='h'))

    run_req = short_forms[batting_team] + " need " + str(runs_left) + " runs in " + str(balls_left ) + " balls"
    crr_text = "Current Run Rate : " + str(crr)
    rrr_text = "Required Run Rate : " + str(rrr)
    scorecard = str(score)+ "/" + str(wickets) + " in " + str(overs) + "." + str(balls) + " Overs" 

    col1, col2 = st.columns(2)
    with col2:
        st.write("\n")
        st.write("<h3>Scorecard</h3>" + "<h3>" + short_forms[batting_team] + " vs " + short_forms[bowling_team] + "</h3>", unsafe_allow_html=True)
        st.write(scorecard)
        st.write(run_req)
        st.write(crr_text)
        st.write(rrr_text)
        
    with col1:
        st.plotly_chart(fig)

if button:
    ipl_win_predictor()
