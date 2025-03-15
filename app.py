import streamlit as st
import pickle as pkl
import pandas as pd
import os

st.set_page_config(layout='wide')

st.title('IPL Win Predictor')

# Check if model.pkl exists in the current directory
if not os.path.exists('model.pkl'):
    st.error("Error: The model.pkl file is missing.")
    st.stop()

# Try loading the pickle files and handling errors
try:
    # Importing data and model from pickle files
    teams = pkl.load(open('team.pkl', 'rb'))
    cities = pkl.load(open('city.pkl', 'rb'))
    model = pkl.load(open('model.pkl', 'rb'))
except Exception as e:
    st.error(f"Error loading pickle files: {e}")
    st.stop()

col1, col2, col3 = st.columns(3)
with col1:
    batting_team = st.selectbox("Select the batting team", sorted(teams))
with col2:
    bowling_team = st.selectbox("Select the bowling_team", sorted(teams))
with col1:
    selected_city = st.selectbox("Select the Host city", sorted(cities))

target = st.number_input('Target Score', min_value=0, max_value=720, step=1)

col4, col5, col6 = st.columns(3)
with col4:
    score = st.number_input('Score', min_value=0, max_value=720, step=1)
with col5:
    overs = st.number_input('Overs_completed', min_value=0, max_value=20, step=1)
with col6:
    wickets = st.number_input('Wickets Fell', min_value=0, max_value=20, step=1)

if st.button("Predict Probabilities"):
    if overs == 0:
        st.warning("Overs cannot be zero, please enter a valid number of overs.")
    else:
        runs_left = target - score
        balls_left = 120 - (overs * 6)
        wickets = 10 - wickets
        crr = score / overs if overs > 0 else 0



        rrr = (runs_left * 6) / balls_left
        input_df = pd.DataFrame({'batting_team': [batting_team],
                                 'bowling_team': [bowling_team],
                                 'city': [selected_city],
                                 'Score': [score],
                                 'Wickets': [wickets],
                                 'Remaining_balls': [balls_left],
                                 'target_left': [runs_left],
                                 'crr': [crr],
                                 'rrr': [rrr]
                                 })
        try:
            result = model.predict_proba(input_df)
            loss = result[0][0]
            win = result[0][1]
            st.header(f"{batting_team} - {round(win * 100)}%")
            st.header(f"{bowling_team} - {round(loss * 100)}%")
        except Exception as e:
            st.error(f"Error during prediction: {e}")

