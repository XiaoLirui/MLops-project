# MLops-project : Mobile Price Prediction
##  Problem Statement
The dataset I am working on is from a mobile company. It conducted a market survey to collect the prices of many mobiles, which helps to price its newly produced mobile phones. This dataset comprises 2001 entries, each detailing various features of the mobile phones, such as RAM, memory, battery power, and Front Camera mega pixels. The aim of my project is to employ machine learning to identify the relationship between these features and the phone’s price range (categorized from 0 to 3, where 0 represents the lowest price range and 3 represents the highest) and then make predictions
for new data.
One interesting aspect of this dataset that inspires me is its practical purpose. By gaining a deeper understanding of how mobile phone features influence pricing, I can make more informed decisions when purchasing a phone in the future. It can help me avoid being misled by sales personnel and ensure that I am paying a fair price for the features that matter most to me.

## Dataset
This is a dataset from Kaggle, and the link is https://www.kaggle.com/code/vikramb/mobile-price-prediction. It has two data files -- train.csv and test.csv. I have stored them in the dataset folder.

## Model
To address this multiple class classification problem, I utilize three kinds of different algorithms like the above dataset. But some algorithms are different:
Support Vector Machines (SVM): For this problem, I experiment with two different kernel functions: the modified version –SVC with the linear kernel and the radial basis function (RBF) kernel, which is very suitable to multiple class classification.
k-Nearest Neighbors (k-NN): I used the same algorithms as in the previous dataset.
Decision Tree Boosting (Extra credit): I use a boosting algorithm to enhance the performance of decision trees. Specifically, it creates a base decision tree classifier and uses the AdaBoost algorithm to boost it.

## Setting up the environment

