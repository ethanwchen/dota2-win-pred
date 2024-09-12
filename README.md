### DotA 2 Win Prediction Model

This is a work in progress and my first attempt at top 100 on a major Kaggle leaderboard.

#### Model 1:

Knowing the radiant win/loss outcomes and having a binary classification problem at hand, we use a supervised leraning model like Random Forest as our rudimentary model. This is good for tabular data with numerical features or categorical features with fewer than hundreds of categories. This is where we are able to capture non-linear interaction between the features and targets. Tree based models are also very good with higher feature counts.

![image](https://github.com/user-attachments/assets/32188fb4-d938-4e55-89d4-2de73c086027)

##### Mathematically Understanding RF

$$MSE = \frac{1}{N}\sum^{N}_{i=1}(fi-yi)^2$$

- where N is the number of data points, fi is the value returned by the model, and yi is the actual value for data point i.

Random Forest uses the mean squared error to calculate the distance of each node from the predicted actual value. This helps decide which branch is the better decision for our forest. We are using the Gini index to decide how our nodes on the decision tree branch. This can be written out as 

$$ Gini = 1-\sum^{C}_{i=1}(p_{i})^2$$

Using class and probability, we can determine the Gini of each branch on a node. This index calculates the purity of groups of data created by a split point. 0 would be perfect purity where class values are perfectly separated into two groups. To find the best split point value, we must evaluate the cost of each value in the training dataset for each input variable. $p_{i}$ here represents the relative frequency of the class we are observing in the dataset and $c$ represents the number of classes.

#### Tuning

RF has many tuning parameters.

- num_trees contrals the number of trees in the final model. The mode trees, the higher the accuracy but also the higher the prediction time
- max_depth restricts the depth of trees to prevent overfitting
- step_size is shrinkage and works like gradient descent learning rate
- min_child_rate is the minimum observations required at a leaf node. Larger values have simpler trees
- min_loss_reduction is a pruning criteria for constructing the tree. This restricts reduction of loss function meaning that a larger value produes a simpler tree



Random Forest
