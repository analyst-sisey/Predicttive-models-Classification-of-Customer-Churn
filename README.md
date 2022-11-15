Predicting Customer Churn – Comparison of 4 Machine Learning Models

Customer attrition is one of the biggest expenditures of any organization. Customer churn otherwise known as customer attrition or customer turnover is the percentage of customers that stopped using an organization’s product or service within a specified timeframe.
Customer attrition also known as customer churn is the rate at which customers stop using an organization’s product or service.
Organizations spends a lot of resources-time and money- in acquiring customers. Knowing customers possibility of churning helps organizations put in place effective strategies for retention.
The goal of this project is to build a classification model that predicts the likelihood of customer churn. This article will explore the processes of accomplishing this goal using the CRISP-DM methodology. 

•	Perform EDA
•	Answer some questions
•	Test some hypothesis
•	Create 4 different models and compare their results and confusion matrixes
•	Perform cross validation and hyperparameter tuning


A notebook containing the detailed codes, explanations and visualization can be found here on my GitHub page.
The data for this project is the dataset of a telecommunications company. It contains customer’s demographic, account, and subscription information. 
We will be answering the following questions and hypothesis.
1. What is the gender distribution of customers
2. What percentage of customers are senior citizens
3. What percentage of customers have partners
4. What percentage of customers have dependant
5. What percentage of customers have more than 1 service subscription
6. What is the churn rate by the above distributions
7. What is the distribution of customers by Contract type
8. What is the distribution of customers by Payment Method
9. Is there a correlation between charges and churn


Data Exploration 

Exploration of the dataset revealed the dataset is made up of both categorical and numerical columns. The target column "Churn" is numeric with binary classes 0 and 1. The dataset can be summarized as containing the following data:
Demographic data: Gender, SeniorCitizen, Partner, and Dependents
Subscribed services: PhoneService, MultipleLine, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, and  StreamingMovies
Account information: CustomerID, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges, and Tenure.

The data has no missing values. However, the TotalCharges column contains empty spaces. We assume those rows contain information for new customers who have not been billed yet since their tenure is 0. We therefore change the empty spaces with 0. We also change the datatype of the TotalCharges column to float.
Analysis and Visualization
Our dataset contains 7043 customers. The minimum monthly charge is 18.25 whiles the maximum monthly charge is 118.75. The gender distribution is 49.5% female and 50.5% male. 
 

There are 1142 senior citizens out of the 7043 customers representing 16.2% of total customers.
 
There are 3402 customers with partners representing 48% of total customers.
 
There are 2110 customers with dependents representing 30% of total customers. 4933 customers have no dependents.
 
3875 customers are on a Month-to-month, 1473 on one year contract and 1695 on two year contract. 
 
There are 4 payment methods. 2365 customers pay through the electronic check payment method, 1522 through automatic credit card , 1544 through automatic bank transfer, and 1612 pay through mailed check.
 


Churn Rates

From the analysis, 26.5% of customers churned whiles 73.5% stayed. 
 
customers on the Month-to-month contract churned more, followed by those on one year and then two year contracts. There is therefore a direct relationship between contract longevity and customers not churning.
 
Of those who churned, 1,446 did not receive technical support whiles 310 received technical support.  

1,543 of those who churned had no dependents whiles 326 had dependents. 
 

Feature Engineering

Encoding Categorical Columns

Models perform better with numeric values. Hence, we encoded the categorical columns using sklearn's OneHotEncoder function as we dont have ordinal columns. We also changed "Yes" and "No" in the Churn column to 0 and 1 respectively.
Since we are predicting customer churn, we used the churn column as our target variables and use the rest of the columns as our predictor variables.  Again, our predictive variables (Churn) contains imbalance data. There are 5174 entries for Yes and 1869 entries for No. This has automatically created minority and majority classes which can create problems with the accuracy of our model due to this bias. We upsampled the minority class using sklearn resample().
In order to adjust our features to a common scale, we performed feature scaling on our predictive variables. We  used the MinMaxScaler to perform feature scaling.

Modeling and Prediction

We built the following models and compared their accuracy, precision, recall and F1 & F2 scores:
1. LogisticRegression Model
2. DecisionTree Model
3. RandomForest Model
4. XGB Classifier Model

Model	Accuracy	Precision	Recall	F1	F2
0	LogisticRegression	0.821859	0.861086	0.903475	0.881771	0.869242
1	DecisionTreeClassifier	0.710433	0.804854	0.800193	0.802517	0.803918
2	RandomForestClassifier	0.798439	0.831570	0.910232	0.869124	0.846195
3	GradientBoostingClassifier	0.770759	0.829178	0.866795	0.847570	0.836438
From the table above, we see that generally, all our models performed fairly well with F1 scores of 80% or above. However, the LogisticRegression Model performs better than the rest of the models. The results of the model's confusion matrix shows that the model predicted 222 True positives (TP), 936 true negatives (TN), 151 false positives (FP), and 100 false negatives(FN). 
 
In other words, the model rightly predicted those who churn 222 times and those who did not churn 936 times. It wrongly predicted those who churned 151 times and those who did not churn 100 times.000
We therefore selected it as our best model and optimized it to see if it can perform even better.
We used the GridSearchCV technique to find the optimal parameters for 3 of the best performing models ie the LogisticRegression, RandomForest, and the DecisionTree models. We passed all the possible values for the various parameters for the function to choose the best parameters for us.
We get the best parameters through the best_params_ method. From the output of the GridSearchCV , the best parameters and values are :
Logistic Regression
'C': 225.0, 'penalty': 'l2', 'solver': 'sag'
‘C’ determines the Inverse of regularization strength and solver is the algorithm to use in the optimization problem
Decission Tree
'entropy' for 'criterion', a 'max_depth' of 6,  'max_features' of 0.6 and 'random' for 'splitter':
Random Forest
'class_weight' of 'balanced_subsample','gini' for 'criterion', a 'max_depth' of None,  'max_features' of 0.4.**

These parameters where fitted in our model to see if they will perform better than they did earlier. After Fitting the model with tuned parameters, The logistic regression model did not perform better with the tuned parameters compared with our initial model.
In conclusion, the minimum monthly charge is 18.25 whiles the maximum monthly charge is 118.75. 26.5% of customers churned whiles 73.5% stayed. Out of the 1869 customers who churned, 939 where female whiles 930 where male. 16.2% are senior citizens. From the analysis, customers on the Month-to-month contract churned more, followed by those on one year and then two year contracts. There is therefore a direct relationship between contract longevity and customers not churning.  1,446 of those who churned did not receive technical support whiles 310 received technical support.
All the models we tested performed better on the imbalanced data than the balanced data. generally, all our models performed fairly well with F1 scores of 80% or above. However, the Logistic Regression Model performs better than the rest of the models.
A notebook containing the detailed codes, explanations and visualization can be found here on my GitHub page
