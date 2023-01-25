# Neural Network Charity Analysis
## Overview of the analysis: Explain the purpose of this analysis.


- help the foundation predict where to make investments.

- With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to help  create a binary classifier that is capable of predicting whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, Beks received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as the following:

EIN and NAME—Identification columns
APPLICATION_TYPE—Alphabet Soup application type
AFFILIATION—Affiliated sector of industry
CLASSIFICATION—Government organization classification
USE_CASE—Use case for funding
ORGANIZATION—Organization type
STATUS—Active status
INCOME_AMT—Income classification
SPECIAL_CONSIDERATIONS—Special consideration for application
ASK_AMT—Funding amount requested
IS_SUCCESSFUL—Was the money used effectively



## Preprocessing Data for a Neural Network Model
### Data Preprocessing
- What variable(s) are considered the target(s) for your model?
- What variable(s) are considered to be the features for your model?
- What variable(s) are neither targets nor features, and should be removed from the input data?

Although one-hot encoding is a very robust solution, it can be very memory-intensive. Therefore, categorical variables with a large number of unique values (or very large variables with only a few unique values) might become difficult to navigate or filter once encoded. To address this issue, we must reduce the number of unique values in the categorical variables. The process of reducing the number of unique categorical values in a dataset is known as bucketing or binning. Bucketing data typically follows one of two approaches:

Collapse all of the infrequent and rare categorical values into a single "other" category.
Create generalized categorical values and reassign all data points to the new corresponding values.
The first bucketing approach takes advantage of the fact that uncommon categories and "edge cases" are rarely statistically significant. Therefore, regression and classification models are unlikely to be able to use rare categorical values to produce robust models, and instead will ignore the rare events altogether and focus on more informative values.

The second bucketing approach collapses the number of unique categorical values and maintains relative order and magnitude so that the machine learning model can train on the categorical variable with minimal impact to performance. This approach is particularly useful when dealing with a categorical variable whose distribution of unique values is relatively even. Once we have bucketed our categorical variables, we can proceed to transform the categorical variable using one-hot encoding.

## Compile, Train, and Evaluate the Model
### Compiling, Training, and Evaluating the Model
- How many neurons, layers, and activation functions did you select for your neural network model, and why?  READ 19.2.5

Looking at the output of our SVM model, the model was able to correctly predict the customers who subscribed roughly 87% of the time, which is a respectable first-pass model. Next, we need to compile and evaluate our deep learning model. Again, we'll use our typical binary classifier parameters:

Our first hidden layer will have an input_dim equal to the length of the scaled feature data X , 10 neuron units, and will use the relu activation function.
Our second hidden layer will have 5 neuron units and also will use the relu activation function.
The loss function should be binary_crossentropy, using the adam optimizer.

- Were you able to achieve the target model performance?
- What steps did you take to try and increase model performance?

## Neural Network Model Summary: 
Summarize the overall results of the deep learning model. 
Include a recommendation for how a different model could solve this classification problem, and explain your recommendation.