### DotA 2 Win Prediction Model

This is a work in progress and my first attempt at top 100 on a major Kaggle leaderboard.

#### Model 1:

Knowing the radiant win/loss outcomes and having a binary classification problem at hand, we use a supervised leraning model like random forest as our rudimentary model. 

$$MSE = \frac{1}{N}\sum^{N}_{i=1}(fi-yi)^2 \text{ where N is the # of data points, fi is the value returned by the model, and yi is the actual value for data point i.$$

![image](https://github.com/user-attachments/assets/32188fb4-d938-4e55-89d4-2de73c086027)

Random Forest
