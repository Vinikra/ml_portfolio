import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load and investigate the data here:

df = pd.read_csv(r'/home/vinicius/Downloads/tennis_ace_starting/tennis_ace_starting/tennis_stats.csv')

print(df.columns)
print(df.head())

# perform exploratory analysis here:
plt.scatter(df['BreakPointsOpportunities'], df['Winnings'])
plt.title('BreakPointsOpportunities vs Winnings')
plt.xlabel('BreakPointsOpportunities')
plt.ylabel('Winnings')
plt.show()

## perform single feature linear regressions here:
features = df[['FirstServeReturnPointsWon']]
outcome = df[['Winnings']]
features_train, features_test, outcome_train, outcome_test = train_test_split(features, outcome, train_size = 0.8)

model = LinearRegression()
model.fit(features_train, outcome_train)
print(model.score(features_test, outcome_test))
prediction = model.predict(features_test)
plt.scatter(outcome_test, prediction, alpha=0.4)
plt.show()

## perform two feature linear regressions here:

features_2 = df[['BreakPointsOpportunities', 'FirstServeReturnPointsWon']]
outcome_2 = df[['Winnings']]
features_train_2, features_test_2, outcome_train_2, outcome_test_2 = train_test_split(features_2, outcome_2, train_size = 0.8)
model.fit(features_train_2, outcome_train_2)
print(model.score(features_test_2, outcome_test_2))
prediction_2 = model.predict(features_test_2)
plt.scatter(outcome_test_2, prediction_2, alpha=0.4)
plt.show()

## perform multiple feature linear regressions here:
features_3 = df[['FirstServe', 'FirstServePointsWon', 'FirstServeReturnPointsWon', 'SecondServePointsWon', 'SecondServeReturnPointsWon', 'Aces', 'BreakPointsConverted', 'BreakPointsFaced', 'BreakPointsOpportunities', 'BreakPointsSaved', 'DoubleFaults', 'ReturnGamesPlayed', 'ReturnGamesWon', 'ReturnPointsWon', 'ServiceGamesPlayed', 'ServiceGamesWon', 'TotalPointsWon', 'TotalServicePointsWon']]
outcome_3 = df[['Winnings']]
features_train_3, features_test_3, outcome_train_3, outcome_test_3 = train_test_split(features_3, outcome_3, train_size = 0.8)
model.fit(features_train_3, outcome_train_3)
print(model.score(features_test_3, outcome_test_3))
prediction_3 = model.predict(features_test_3)
plt.scatter(outcome_test_3, prediction_3, alpha=0.4)
plt.show()