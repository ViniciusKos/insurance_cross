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

## 4). Machine Learning Models Applied
In order to predict whether or not a client wil buy vehicle insurance 4 models were tested using Stratified K-Fold Cross Validation.
This technique is especially helpful in problems with unbalanced data, like this. The Stratified K-Fold cross-validation guarantee that each fold has the same proportion of positive classes

Below it's shown how it works
![image](https://user-images.githubusercontent.com/73034020/187759783-03fe8033-b828-43c8-8f4a-56be10fa12a8.png).

The models Cross Validation performance's are shown below:

![image](https://user-images.githubusercontent.com/73034020/187763496-60f71302-2b41-4523-a81b-4a3aa2751138.png)


LGBMClassifier was chosen because it has both precision and recall above the baseline (F1-score above the baseline).
Why not the XGBClassifier which has scored the higher precision?
It has a good precision but a poor recall, remember that recall measures the "capture" power of the models.
In our problem, our goal is to deliver an ordained propensity list to the business team. So we need to find a balance between the number of clients in this list and their propensity, e.g a list of 10 extremely high propensity clients may not be useful at all nor a list with 70000 clients with low propensity.
In summary, for the first CRISP-DM circle it has chosen LGBMClassifier because it has an F1-Score above the baseline given that this metric combine both precision and recall.

Below it's shown the intuition of these metrics and the F1-score formula.

![image](https://user-images.githubusercontent.com/73034020/187764056-d9492e21-88d1-4442-816a-1178b1cd0037.png)

![image](https://user-images.githubusercontent.com/73034020/187766374-2e3c71c3-8c4d-4270-884b-0779842be694.png)


## 5). Final Machine Learning Models Performance

### Machine Learning Metrics

After the Hyperparameter tuning our final model has the metrics shown below (in houldout set).

![image](https://user-images.githubusercontent.com/73034020/187790867-b897f565-16df-40ef-8197-3641fb3e7dab.png)

### Cumulative Gain Curve

Below, it's shown 

## 6). Business Impact

Disclaimer: For evaluation purposes, it was considered that the commercial/business team can reach 40% of the entire customer base. But according to the business context, this number can be lesser or greater.
Also, it was considered that each lead has a cost of $250.

How much the is model better than a random guess of whether or not a client will buy an insurance?

The chart below shows us that if the business team prospects,let's say, the first 40% customers, it will me 2.3 times more precise than a random guess.

![image](https://user-images.githubusercontent.com/73034020/187791963-a5770f0b-a82b-4897-af0a-c3ec6cff36aa.png)

This can be translated into cost reduction. Let's say the company has a human sales channel, surely it has a cost for each lead. And each lead without a sale convertion is a loss. Thus, most of leads must be successful in order to maximize our profit.


The chart below shows us that if the business team prospect the same first 40% customers it will reach 90% of all the propense clients thanks to the model propensity score.
On the other hand, if the business team hadn't the model, the same aproach of prospecting randomly 40% customers, it will reach only 4,8% of all propense clients (40% times the real proportion of 12%)

![image](https://user-images.githubusercontent.com/73034020/187791862-163d457f-9069-4721-af46-99c06c47d6b3.png)

###  How much does the model increase the company's profit?

The amount of leads needed to reach 40% of the customers it's 60978.
Each lead has a cost of $250.

If the business team would prospect the first 40% of the entire clients base WITHOUT the model, it would have a profit of:  $ 4601077.41

If the business team would prospect the first 40% of the entire clients base WITH the model, it would have a profit of:  $ 30787938.75

Therefore, the absolute monthly profit increase would be of $26186861, and the relative monthly profit increase would be of **569%**)

![image](https://user-images.githubusercontent.com/73034020/187796310-eedbd2f6-fc87-4e2b-856d-e559b2f738c5.png)

At the first moment, this increase may be astonishing and maybe doubtful, but in our scenario where there's a considerable lead cost and very few propense customers, it's feasible that a propensity ranked list increases the sales funnel efficiency and consequently increases considerably the profit.







