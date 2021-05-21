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

# 4.SVM 
SVM works by mapping data to a high-dimensional feature space so that data points can be categorized,even when the data are not otherwise linearly separable (This gets done by kernel function of SVM classifier). A separator between the categories is found, then the data is transformed in such a way that the separator could be drawn as a hyperplane.

Here, we used SVM to build and train a model using human cell records, and classify cells to whether the samples are benign (mild state) or malignant (evil state).

Here the Dataset contains 12 columns where ID number,Clump,	UnifSize,	UnifShape,	MargAdh,	SingEpiSize,	BareNuc,	BlandChrom,	NormNucl,	Mit	Class helps to find the weather the person is suffering from cancer or not. We collected 700 persons data for this work.

Here we train our data using SVM library using sklearn model. 
Finally for out results we can check weather the person is suffering from Cancer or not.
<h3>Lets see the SVM model graph: </h3>
<br>
 <img src=https://res.cloudinary.com/adeshpokhrel/image/upload/v1621405022/svm_qvyxaq.png> </img> </br>

# 5.Decisions Tree
A decision tree is a decision support tool that uses a tree-like model of decisions and their possible consequences, including chance event outcomes, resource costs, and utility. It is one way to display an algorithm that only contains conditional control statements. Decision trees are commonly used in operations research, specifically in decision analysis, to help identify a strategy most likely to reach a goal, but are also a popular tool in machine learning.

### Problem Statement
Lets predict the patients suffering from heart diseases or not. The dataset is brought from kaggle, and used Decision Tree Model. I created a model which help me to estimate for the patients suffering or not.

Dataset contains 14 columns and 303 rows with CSV extension. 
The data contains the following columns :
- 'age' :			age
- 'sex' :			1: male, 0: female
- 'cp' :			chest pain type, 1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic
- 'trestbps' :			resting blood pressure
- 'chol' :		 serum cholestoral in mg/dl
- 'fbs' :			fasting blood sugar > 120 mg/dl
- 'restecg' :			resting electrocardiographic results (values 0,1,2)
- 'thalach' :			 maximum heart rate achieved
- 'exang' :			exercise induced angina
- 'oldpeak' :			oldpeak = ST depression induced by exercise relative to rest
- 'slope' :			the slope of the peak exercise ST segment
- 'ca' :			number of major vessels (0-3) colored by flourosopy
- 'thal' :			thal: 3 = normal; 6 = fixed defect; 7 = reversable defect

<h3>Lets see the Decision Tree graph: </h3>
 <br>
 <img src=https://res.cloudinary.com/adeshpokhrel/image/upload/v1621405016/tumblr_inline_o24t595vJi1tcmcm4_1280_qkii15.png height=300 width=300> </img> </br>
 
 # 6.KNN(K-nearest Neighbors)
 K-Nearest Neighbors (KNN) is a conceptually simple yet very powerful algorithm, and for those reasons, it’s one of the most popular machine learning algorithms. Let’s take a deep dive into the KNN algorithm and see exactly how it works. Having a good understanding of how KNN operates will let you appreciated the best and worst use cases for KNN.
 
 #### Problem Statement
Lets predict the office workers weather they will leave company next year or not. The dataset is brought from kaggle, and used KNN Model. Based on the workers factors we can conclude how many workers will leave the company using KNN model we can solved it.
The data contains the following columns:
- 'employee_id'	
- 'number_project'	
- 'average_montly_hours'
- 'time_spend_company'
- 'Work_accident'	
- 'left	promotion_last_5years'
- 'department	salary'
- 'satisfaction_level'
- 'last_evaluation'
 
Here based on salary and satisfication level we can say weather the workers will leave the company or not.
<h3>Lets see the KNN model: </h3>
 <br>
 <img src=https://res.cloudinary.com/adeshpokhrel/image/upload/v1621482214/knn_bose1s.jpg height=300 width=300> </img> </br>
 
# 7.Random Forest
Random forests or random decision forests are an ensemble learning method for classification, regression and other tasks that operates by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean/average prediction (regression) of the individual trees.

#### Problem Statement
Lets predict the quality of the WINE. The dataset is brought from kaggle, and used Random Forest Model. Based on the several factors of the wine we can conclude how is the conditions of the wine and weather it is drinkable or not.

The data sets contains 1599 rown & 12 columns.
Here the columns contains the data like:
- 'Fixed Acidity'
- 'Volatile Acidity'	
- 'Citric Acid'
- 'Residual Sugar'	
- 'Chlorides'
- 'Free Sulfur Dioxide'
- 'Total Sulfur Dioxide'
- 'Density'
- 'pH'
- 'Sulphates'	
- 'Alcohol'
- 'Quality'
Here based on factors of the coulumns we can say weather the wink is good or not.
<h3>Lets see the SNS heatmap of the dataset: </h3>
 <br>
 <img src=https://res.cloudinary.com/adeshpokhrel/image/upload/v1621567400/RF_dhvhfv.png height=300 width=300> </img> </br>
















