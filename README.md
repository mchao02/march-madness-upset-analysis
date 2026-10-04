# March Madness Upset Analysis
A python project exploring NCCAA men's basketball tournament results, seeding, and regular-season team statistics dating from 2008 to 2025

## Objective
Investiage whether differences in regular season performance can help identify tournament upsets which is when a team with a higher seed defeats a team with a lower seed.

## Approach
- Transform regular season results into team level season averages
- Calculate advanced basketball metrics such as net rating, effective field goal percentage, turnover rate, offensive rebounding rate, free-throw rate, tempo, and defensive effiency
- Combine team statistics with tournament results and seeds
- Train a logistic regression model using eight matchup features
- Evaluate perfoormance using accuracy, precision, recall, F1 sore, and a confusion matrix

## Technologies
Python, pandas, NumPy, scikit-learn, and Matplotlib

## Running the Project
1. Download the required datasets, instructions listed in data/README.md
2. Place the three CSV files in the data/folder
3. Install dependencies from the project's folder

python -m pip install -r requirements.txt

4. Run the analysis:

python main.py

The program prints evaluation metrics and displays charts of model coefficients, class balance, and the test-set confusion matrix

# Results and Insights
Indentified rebounding, Shooting effectiveness, turnovers, and free throw shooting as the most important factors in determining an upset in a matchuph