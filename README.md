# Home-Credit-Default
The main goal of this project is to build a machine learning model to predict the likelihood of customers defaulting on a loan. By developing a robust credit scoring system, financial institutions can make more informed lending decisions, reducing the risk of non-performing loans.
1. Business Objective:
1.	Many people struggle to get loans due to insufficient or non-existent credit histories. And, unfortunately, this population is often taken advantage of by untrustworthy lenders. Home Credit strives to broaden financial inclusion for the unbanked population by providing a positive and safe borrowing experience. In order to make sure this underserved population has a positive loan experience, Home Credit makes use of a variety of alternative data--including telco and transactional information--to predict their clients' repayment abilities.
2.	While Home Credit is currently using various statistical and machine learning methods to make these predictions, they're challenging Kagglers to help them unlock the full potential of their data. Doing so will ensure that clients capable of repayment are not rejected and that loans are given with a principal, maturity, and repayment calendar that will empower their clients to be successful.
3.	The main goal of this project is to build a machine learning model to predict the likelihood of customers defaulting on a loan. By developing a robust credit scoring system, financial institutions can make more informed lending decisions, reducing the risk of non-performing loans.


2.Approach:
The following steps were taken to achieve the objective:
1.	Data Understanding: Explore the dataset, check for missing values, and understand the correlation between different features and the target variable (TARGET).
2.	Data Preprocessing: Handle missing data, scale the numerical features, and perform feature selection.
3.	Modeling: Train machine learning models like Logistic Regression, Random Forest, AdaBoost, and Decision Trees on the preprocessed data.
4.	Model Evaluation: Assess model performance using accuracy, precision, recall, and ROC AUC scores.
3. Detailed Explanation of Algorithms:
Each step in the process is explained with the relevant code and line-by-line explanation:
Data Preprocessing:
 

pandas and numpy are libraries for data manipulation and numerical operations.train_test_split from sklearn.model_selection is used to split the dataset into training and testing sets.pd.set_option('display.max_columns', 100) configures pandas to display up to 100 columns in the DataFrame output.sklearn.linear_model, sklearn.metrics, sklearn.ensemble, sklearn.tree, sklearn.svm, sklearn.preprocessing are modules from scikit-learn for different machine learning models and utilities.RandomizedSearchCV from sklearn.model_selection is used for hyperparameter tuning.Reads a CSV file from the specified path into a pandas DataFrame named data.Displays the first few rows of the DataFrame to give a preview of the data.
data.columns.values returns an array of the column names from the DataFrame data. 
quick way to see all the column names in your dataset.
 
data.describe() generates a summary of statistics for numerical columns in  DataFrame data. It provides a quick overview of the central tendency, dispersion, and shape of the dataset’s distribution.
Here's what data.describe() typically returns:
1.	Count: The number of non-null values in each column.
2.	Mean: The average value of each numerical column.
3.	Standard Deviation: The measure of the amount of variation or dispersion of values.
4.	Min: The minimum value in each column.
5.	25%: The 25th percentile value (first quartile).
6.	50%: The median value (50th percentile or second quartile).
7.	75%: The 75th percentile value (third quartile).
8.	Max: The maximum value in each colum. 
a.Fraud: Filters the DataFrame data to include only rows where the 'TARGET' column has a      value of 1, indicating fraudulent transactions.
b.	Valid: Filters the DataFrame data to include only rows where the 'TARGET' column has a value of 0, indicating valid transactions.
len(Fraud): Number of fraudulent transactions.
c.	len(Valid): Number of valid transactions.
d.	outlier_fraction: Proportion of fraudulent transactions relative to the total number of transactions. This gives a measure of how common fraud is in the dataset.
total_cases: Total number of transactions in the dataset (sum of fraud and valid transactions).
e.	fraud_percentage: Percentage of fraudulent transactions in the dataset.
f.	valid_percentage: Percentage of valid transactions in the dataset.
outlier_fraction: Prints the proportion of fraud cases.
g.	len(Fraud): Prints the number of fraudulent transactions.
h.	len(Valid): Prints the number of valid transactions.
i.	fraud_percentage: Prints the percentage of fraudulent transactions.
valid_percentage: Prints the percentage of valid transactions :
 

Returns the dimensions of the DataFrame data :
 

Shows the count of each data type present in the DataFrame. It  gives a Series with the data types as the index and the number of columns of each type as the values.

 

Finds the number of unique values in each column of type object . data.select_dtypes('object'): Selects only the columns with data type object (usually strings or categorical data).
.apply(pd.Series.nunique, axis=0): Applies the nunique function to each column to count the number of unique values. 

It gives a Series with column names as the index and the number of unique values in each column as the values
 

Counts the frequency of each unique value in the CNT_CHILDREN column.
 
plot_heatmap that visualizes the correlation matrix of a dataset using a heatmap. It starts by importing the necessary libraries (pandas, seaborn, matplotlib.pyplot, and numpy). The function takes a DataFrame dataframe, a figure size tuple figsize, and an annotation font size annot_size as parameters. It creates a figure with the specified size and sets a white background style using seaborn. It then computes the correlation matrix of the DataFrame, generates a mask to hide the upper triangle of the matrix for clarity, and creates a custom diverging colormap. Using seaborn's heatmap function, it draws the heatmap with the mask applied, adjusts visual properties like color mapping and annotation sizes, and displays the plot with a title and customized tick labels. The code includes commented-out example usage for applying the function to a DataFrame named application_train.

 

to plot the heat map of all the variables in the data set:

 
Extract the EXT_SOURCE variables and show correlations :
 

 Filters the DataFrame data to select columns that are not numerical.
 exclude=[np.number]: Excludes columns with numerical data types (int, float, etc.) from the selection.
Creates a new DataFrame non_numerical_data from the selected non-numerical columns. This line is somewhat redundant because non_numerical_columns already is a DataFrame with the required columns. This step essentially duplicates non_numerical_columns into non_numerical_data.
Displays the first few rows of the non_numerical_data DataFrame to give a preview of the non-numerical columns in the dataset.
Saves the non_numerical_data DataFrame to a CSV file named non_numerical_data.csv without including the row index.
Reads a CSV file named numerical_data.csv into a DataFrame nnd. This assumes that the numerical_data.csv file exists and contains the numerical columns that were not included in the non_numerical_data DataFrame.
the code filters out the non-numerical columns, creates a new DataFrame with those columns, prints a preview of this DataFrame, saves it to a CSV file, and finally reads another CSV file presumably containing numerical data
 
takes numerical columns from the original DataFrame, creates a new DataFrame with these columns, displays a preview, saves the numerical data to a CSV file, and then reads that file back into a new DataFrame. This process effectively separates numerical data from non-numerical data in the dataset and demonstrates data handling and storage techniques.
 
calculates and analyzes the absolute correlation values between the 'TARGET' column and all other numerical columns in the DataFrame nd. First, it computes the absolute correlations, sorts them in descending order, and then prints the top 30 most positively correlated features and the bottom 15 most negatively correlated features. It also determines a correlation threshold based on the value of the 11th highest correlation, which is used as a cutoff for identifying significant correlations. This threshold value is printed to provide a benchmark for evaluating which features are most strongly related to 'TARGET'.
 
 
 
It first defines a function, missing_values_table, which takes a DataFrame df as input and calculates the total number and percentage of missing values for each column. The results are stored in a DataFrame, where the percentage of missing values is calculated by dividing the number of missing values by the total number of rows in the DataFrame, then multiplying by 100.
Next, a threshold is defined, which in this case is 50%. Columns with missing values greater than 50% are dropped from the DataFrame. The remaining columns, which have missing values below or equal to the threshold, are filled with their respective column medians (for numeric columns). If there are any remaining missing values in the dataset after this process, they are filled with the mean of the column. Finally, the code checks if any missing values still remain in the DataFrame and prints the total count of missing values after the cleaning process.
This approach ensures that columns with excessive missing data are removed, while the remaining columns are cleaned by imputing appropriate values (medians or means).
 
print("Remaining column names:"): This line prints a simple message indicating that the following output will be the names of the remaining columns in the DataFrame X.
print(list(X.columns)):
•	X.columns retrieves the names of all columns in the DataFrame X.
•	list(X.columns) converts the column names, which are returned as a pandas Index object, into a Python list.
•	Finally, the print() function displays this list of column names.

pd.set_option('display.max_rows', None): This line modifies the pandas display settings to ensure that all rows of output are shown when printed. By default, pandas limits the number of rows displayed, but this setting removes that limit so that the entire correlation output can be seen.
•	correlations = X.corr()['TARGET'].abs().sort_values(ascending=False):
•	X.corr() calculates the correlation matrix for all numerical columns in the DataFrame X. This matrix contains correlation coefficients that indicate the strength and direction of the linear relationship between each pair of columns.
•	X.corr()['TARGET'] selects the column from the correlation matrix that corresponds to the correlation of each column in X with the 'TARGET' column.
•	.abs() takes the absolute value of the correlations, ignoring whether they are positive or negative, and focuses on the strength of the correlation.
•	.sort_values(ascending=False) sorts these absolute correlations in descending order, with the most strongly correlated columns (positively or negatively) appearing first.
•	print("\nColumns in ascending to descending order of their correlation with 'TARGET':"): This line prints a message indicating that the following output will be the columns sorted by their correlation with 'TARGET'.
•	print(correlations): This prints the sorted list of absolute correlation values between each feature in X and the 'TARGET' column.
•	The goal of this code is to identify which features in X have the strongest relationships with the target variable, helping to inform feature selection or importance.
 
5.Conclusions :
1. Logistic Regression:
•	Training Accuracy: 0.92
•	Testing Accuracy: 0.92
•	Precision (Train/Test): 0.43 / 0.46
•	Recall (Train/Test): 0.0 / 0.0
•	ROC AUC Score (Train/Test): 0.72 / 0.71
Conclusion:
•	Accuracy: The model achieves a good overall accuracy (92%) on both training and testing data, suggesting it's performing well in terms of correctly classifying the majority of instances.
•	Precision: The precision (46% for test data) indicates that while the model makes some correct positive predictions, many of the positives are incorrect.
•	Recall: The recall is 0, meaning the model fails to identify any true positives (i.e., it’s not identifying instances of the positive class). This is a serious issue if recall is important in the context of the problem.
•	ROC AUC: The ROC AUC score of 0.71 indicates the model has a moderate ability to distinguish between classes, though there’s room for improvement.
2. Random Forest:
•	Training Accuracy: 1.0
•	Testing Accuracy: 0.92
•	Precision (Train/Test): 1.0 / 0.5
•	Recall (Train/Test): 1.0 / 0.01
•	ROC AUC Score (Train/Test): 1.0 / 0.69
Conclusion:
•	Overfitting: The Random Forest model achieves perfect accuracy (100%) on the training set, which is a strong indication of overfitting—it learned the training data too well but doesn't generalize as effectively to unseen data.
•	Testing Metrics: On the test set, the model still performs decently in terms of accuracy (92%) but has low recall (0.01). This means that while it accurately classifies the majority of instances, it struggles to correctly identify true positives.
•	Precision and ROC AUC: The precision of 50% on test data is decent, but the low recall and AUC (0.69) suggest the model might not be reliable in high-stakes decision-making where catching all positive instances is important.
3. AdaBoost:
•	Training Accuracy: 0.92
•	Testing Accuracy: 0.92
•	Precision (Train/Test): 0.49 / 0.48
•	Recall (Train/Test): 0.01 / 0.01
•	ROC AUC Score (Train/Test): 0.73 / 0.72
Conclusion:
•	Balanced Performance: AdaBoost performs similarly to Logistic Regression in terms of accuracy but suffers from the same issue of low recall, meaning it's not identifying many positive instances.
•	Precision and ROC AUC: Precision is moderate (around 48-49%), and the ROC AUC score (around 0.72) shows that the model is okay at distinguishing between classes but is not particularly strong in this regard.
4. Decision Tree:
•	Training Accuracy: 0.93
•	Testing Accuracy: 0.91
•	Precision (Train/Test): 0.91 / 0.2
•	Recall (Train/Test): 0.2 / 0.05
•	ROC AUC Score (Train/Test): 0.82 / 0.64
Conclusion:
•	Overfitting: While the decision tree doesn't overfit as much as Random Forest, there’s still a clear performance gap between the training and test precision (91% vs. 20%) and recall (20% vs. 5%).
•	Testing Metrics: Despite a good accuracy, the model struggles with precision and recall on the test data, indicating poor performance in correctly classifying positive instances.
•	ROC AUC: A ROC AUC score of 0.64 on test data suggests this model isn’t great at distinguishing between classes and may need better tuning (e.g., adjusting max_depth, pruning) to improve.
________________________________________
Overall Conclusion:
1.	Accuracy: All models perform similarly in terms of accuracy (around 91-92%) on both training and test sets, indicating they are somewhat consistent in correctly classifying the majority of instances.
2.	Recall: All models exhibit poor recall (close to 0) on the test set. This indicates that none of them are effectively identifying positive instances, which is critical depending on the application.
3.	Overfitting: The Random Forest and Decision Tree models show signs of overfitting, as they perform significantly better on training data compared to test data, particularly in terms of precision and recall.
4.	Precision: While Logistic Regression and AdaBoost maintain moderate precision (around 0.45-0.5), they struggle with recall, resulting in an incomplete performance.
5.	ROC AUC: The ROC AUC scores for all models indicate only a moderate ability to distinguish between classes, with values ranging from 0.64 to 0.73, suggesting none of the models are particularly strong classifiers for this problem.

6.References :
https://www.kaggle.com/competitions/home-credit-default-risk
https://github.com/rakshithvasudev/Home-Credit-Default-Risk
https://github.com/anshikaahuja/Home-Credit-Default-Risk
https://github.com/rishabhrao1997/Home-Credit-Default-Risk
https://github.com/kozodoi/Kaggle_Home_Credit
https://github.com/harshitlikhar/Home-Credit-Default-Risk
https://github.com/yakupkaplan/Home-Credit-Default-Risk
https://github.com/hungchun-lin/Home-credit-default-risk
https://github.com/jayborkar/Home-Credit
https://github.com/richardzefan/Home-Credit-Default-Risk





