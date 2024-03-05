# IPL MATCH WIN PREDICTOR

## Description

The IPL Match Win Predictor is an application designed to calculate the probabilities of winning or losing for cricket teams based on the current state of the match. It considers various factors such as the 'Current Score', 'Wickets Left', 'Venue', and more. The predictions are made using historical data from IPL matches between 2008 to 2023. The application utilizes the Random Forest algorithm for prediction, which is selected by default.

## Installation and Running

1. **Clone the Repository:**
   - Clone the repository to your local machine using the following command:
     ```
     git clone https://github.com/AyushGupta45/cricket-predictor.git
     ```

2. **Install Streamlit:**
   - If Streamlit is not installed on your system, you can install it using:
     ```
     pip install streamlit
     ```

3. **Run the Application:**
   - Navigate to the cloned directory.
   - Run the following command in your terminal:
     ```
     streamlit run app.py
     ```
   
4. **Open in Browser:**
   - The project will open in your default web browser.

## How to Use

1. **Select Batting Team (Chasing Team):**
   - Select the team currently batting (chasing the target).

2. **Select Bowling Team (Defending Team):**
   - Select the team currently bowling (defending the target).

3. **Select Host City:**
   - Choose the city where the match is taking place.

4. **Enter Target:**
   - Input the total target set by the batting team.

5. **Enter Current Score:**
   - Enter the current score of the chasing team.

6. **Enter Overs Completed:**
   - Input the number of overs completed by the chasing team.

7. **Enter Wickets Fallen:**
   - Enter the number of wickets fallen for the chasing team.

8. **Click on 'Predict Probability' Button:**
   - After entering all the required information, click on the 'Predict Probability' button to get the predicted probabilities of winning and losing for the teams.

## Credits

This IPL Match Win Predictor was developed by:

- Ayush Gupta
- Bhavesh Choudhari
- Om Chaudhari

Feel free to use and explore the application for your IPL match predictions!

### Note:
- The predictions are based on historical IPL match data from 2008 to 2023.
- Results may vary, and the application is meant for informational purposes only.
