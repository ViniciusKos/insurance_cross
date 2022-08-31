# Health Insurance Cross Sell Prediction 🏠 🏥


![image](https://user-images.githubusercontent.com/73034020/181102567-2d26df7a-77c7-4ddf-b080-d6789ac4f87b.png)


**Disclaimer**: this project was inspired by the "Health Insurance Cross Sell Prediction" challenge published on kaggle (https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction?sort=votes). Although it is a fictitious project it will follow all steps of a real project.

## 1. Business Issue
Our client is an Insurance company that has provided Health Insurance to its customers now they need your help in building a model to predict whether the policyholders (customers) from past year will also be interested in Vehicle Insurance provided by the company.

## 2. Solution methodology
The resolution of the challenge was carried out following the CRISP-DM (CRoss-Industry Standard Process for data mining) methodology, which is a cyclical approach that streamlines the delivery of value to the business through fast and valuable MVPs.

Benefits of Using CRISP-DM
This method is cost-effective as it includes a number of processes to take out simple data mining tasks.
CRISP-DM encourages best practices and allows projects to replicate.
This methodology provides a uniform framework for planning and managing a project.
Being cross-industry standard, CRISP-DM can be implemented in any Data Science project irrespective of its domain.

For more details about CRISP-DM metodology: https://analyticsindiamag.com/why-is-crisp-dm-gaining-grounds/


![image](https://user-images.githubusercontent.com/73034020/180753015-7945d745-3420-4fd0-9681-6487fb066c80.png)


My strategy to solve this challenge was:
Step 01. Data Description:
Step 02. Feature Engineering:
Step 03. Data Filtering:
Step 04. Exploratory Data Analysis:
Step 05. Data Preparation:
Step 06. Feature Selection:
Step 07. Machine Learning Modelling:
Step 08. Hyperparameter Fine Tunning:
Step 09. Convert Model Performance to Business Values:
Step 10. Deploy Model to Production:

## 3. Top 3 data insights

### 1) Customers with driving license buys MORE insurance.
There's a propensity increase of 135% (12% vs 5%)!

![image](https://user-images.githubusercontent.com/73034020/187721686-03a03de8-b548-44e0-bb1e-b06ec36ee619.png)

### 2) Underaged vehicles are MORE LIKELY to be insured by customers.

![image](https://user-images.githubusercontent.com/73034020/187724043-05f603f5-fec5-4d75-b30f-fe6629be9966.png)

### 3) Previously damaged vehicles are much MORE LIKELY to be insured by customers.

![image](https://user-images.githubusercontent.com/73034020/187725801-349dac8b-1d7d-48cc-89b0-6bf33b01e0f7.png)

## 4). Machine Learning Model Applied and Performance
In order to predict whether or not a client wil buy vehicle insurance 4 models were tested using Stratified K-Fold Cross Validation.
This technique is especially helpful in problems with unbalanced data, like this. The Stratified K-Fold cross-validation guarantee that each fold has the same proportion of positive classes

Below it's shown how it works
![image](https://user-images.githubusercontent.com/73034020/187759783-03fe8033-b828-43c8-8f4a-56be10fa12a8.png).

The models Cross Validation performance's are shown below:

![image](https://user-images.githubusercontent.com/73034020/187763496-60f71302-2b41-4523-a81b-4a3aa2751138.png)


LGBMClassifier was chosen because it has both precision and recall above the baseline.
Why not the XGBClassifier which has scored the higher precision?
It has a good precision but a poor recall, remember that recall measures the "capture" power of the models.

In our problem, our goal is to deliver an ordained propensity list to the business team. So we need to find a balance between the number of clients in this list and their propensity, e.g a list of 10 extremely high propensity clients may not be useful at all nor a list with 70000 clients with low propensity.
In summary, for the first CRISP-DM circle it has chosen LGBMClassifier because it has an F1-Score above the baseline given that this metric combine both precision and recall.

![image](https://user-images.githubusercontent.com/73034020/187764056-d9492e21-88d1-4442-816a-1178b1cd0037.png)





