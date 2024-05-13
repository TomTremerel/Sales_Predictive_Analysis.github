
Hi there!

This project aims to predict the sales of the different Walmart supermarkets throughout the USA. The prediction is based on various variables such as fuel price, CPI (Consumer Price Index), temperature, holidays, and unemployment.

The dataset has already been cleaned; I made no changes except for converting the temperature into Celsius.

Next, I conducted an EDA to understand the relationship between all variables. 

![Figure 2024-05-13 141312](https://github.com/TomTremerel/Sales_Predictive_Analysis.github/assets/156415815/743abcc6-e462-49f0-a4bd-3d13b9403eeb)

This graph illustrates the evolution of the CPI, which represents inflation. Apart from tracking the CPI, the "Date" variable proved unnecessary for my analysis of this dataset, so I decided to drop it.

Next, I plotted the correlation matrix:

![corr matrix](https://github.com/TomTremerel/Sales_Predictive_Analysis.github/assets/156415815/d0f12e54-7eb9-4204-b061-d5b13cdf51e8)

As observed, the two most relevant variables are the "Holidays flag" and the "Fuel price."

Next, I attempted to visualize the density of the fuel prices:

![density fuel](https://github.com/TomTremerel/Sales_Predictive_Analysis.github/assets/156415815/58d50e8a-0be5-4477-b8b2-321d362ff6ad)

After gaining an understanding of the various variables, I began creating the model. For this phase, I opted to test three different regression models and compare their error measures. So, I trained a RandomForest model, a Linear Regression model, and a Gradient Boosting Regressor.

Upon training them, here are the results:

![realvspred](https://github.com/TomTremerel/Sales_Predictive_Analysis.github/assets/156415815/b296a504-0521-43c1-ad0c-4ff1122e70f9)

Here, the blue areas are the real value and the other values are the prediction. As we can see  the most efficient model without calculating the RMSE or the MAE, it is the Random Forest. 

After I computed the differents MAE, MSE and RMSE to compare the different errors value of the models : 

![image](https://github.com/TomTremerel/Sales_Predictive_Analysis.github/assets/156415815/a17ef9ef-0b78-4534-938b-2882085d73ec)

The most efficient model is the Random Forest Regressor with 6% of error. 

