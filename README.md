# Hands-On-Machine-Learning
<h2> 1. Classification </h2> 
<br>  It is the process of identifying to which set of categories a new observation belongs.Here On classification we used the library Sckit learn.
Datasets loaded by Scikit-Learn generally have a similar dictionary structure including: 

<li>  A DESCR key describing the dataset </li>
<li>  A data key containing an array with one row per instance and one column per feature </li>
<li>  A target key containing an array with the labels </li>
<br>
Looking at the arrays we got to know there are 70,000 images, and each image has 784 features. This is because each image
is 28×28 pixels, and each feature simply represents one pixel’s intensity, from 0
(white) to 255 (black). Let’s take a peek at one digit from the dataset. All you need to
do is grab an instance’s feature vector, reshape it to a 28×28 array, and display it using
Matplotlib’s imshow() function.
<h4>Image of the Result</h4>
<img src=https://res.cloudinary.com/adeshpokhrel/image/upload/v1621076062/1_v7nx5p.png> </img>
This looks like a 5, and indeed that’s what the label tells us.


# 2.Linear Regression Model
Linear Regression Model makes prediction by simply computing a weighted sum of the input features plus a constant called the bias term. Here I have done the hourse price predication using the dataset for finding out price of house on different parameters. I did exploratory Data Analysis, split the training and testing data, Model Evaluation and Predictions. 

### Problem Statement
Lets predict the house price for regions in the USA. The dataset is brought from kaggle, and used Linear Regressioon Model. I created a model which help me to estimate price of house to sell for.

Dataset contains 7 columns and 5000 rows with CSV extension. 
The data contains the following columns :
- 'Avg. Area Income': Avg. Income of householder of the city house is located in.
- 'Avg. Area House Age': Avg. Age of Houses in same city.
- 'Avg. Area Number of Rooms': Avg. Number of Rooms for Houses in same city.
- 'Avg. Area Number of Bedrooms': Avg. Number of Bedrooms for Houses in same city.
- 'Area Population': Population of city.
- 'Price': Price that the house sold at.
- 'Address': Address of the houses.

<h3>Lets see the Regression model graph: </h3>
 <br>
 <img src=https://res.cloudinary.com/adeshpokhrel/image/upload/v1621148749/LinearRefression_byq72d.png> </img> </br>
 
 # 3.Logistic Regression Model
Logistic regression is one of the most popular Machine Learning algorithms, which comes under the Supervised Learning technique. It is used for predicting the categorical dependent variable using a given set of independent variables. Logistic regression predicts the output of a categorical dependent variable.

### Library we used: 
- numpy 
- pandas 
- matplotlib
- matplotlib
- seaborn

### Problem Statement 
Lets see here How the Loan Status is heavily dependent on the Credit History . The Logistic Regression algorithm gives us how much accuracy results.
Data set contains the 13 columns and 614 rows.
The data contains the following columns :
- 'Loan_ID': For their loan id
- 'Gender' : Male or FeMale
- 'Married'	: Weither married or not
- Dependents' : Depedent or Independent  
- 'Education' : Education Level
- 'Self_Employed': Employed or Not
-	'ApplicantIncome': Thier Income
-	'CoapplicantIncome': To know applicant income
-	'LoanAmount': How much loan they have taken
-	'Loan_Amount_Term': Their loan amount terms
-	'Credit_History':	Their hostory of credit
-	'Property_Area': Their area of property
-	'Loan_Status' : What their Loan status is
