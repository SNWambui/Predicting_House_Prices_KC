# Overview
Perform correlation and regression analysis of the KC House Sales data. Using a linear model and Pearson's R, estimated the price range of houses based on square footage.<br>
Computed predictive analysis on multiple features of the houses. Built and Fine tuned Linear Regression, Ridge Regression, Decision tree Regression and Random Forest Regressor models. Used cross-validation and grid-search to obtain the best estimators for predicting the prices of houses. 


## Correlation and Regression
This is correlation and regression analysis of the King County House Sales data. The dataset is obtained from Kaggle at: https://www.kaggle.com/harlfoxem/housesalesprediction. the dataset includes homes sold between May 2014 and May 2015

The first file answers a specific question I developed for a CS51 class assignment:
- Does the square footage of the home have a significant impact on the price of the house?


The second file is an iteration and further improvement of the same dataset that includes more features that are evaluated to predict the prices of the house.
Sample data is used to predict the prices of houses in the entire population

### The dataset features
- <b>id</b> : A notation for a house
- <b> date</b>: Date house was sold
- <b>price</b>: Price is prediction target
- <b>bedrooms</b>: Number of bedrooms
- <b>bathrooms</b>: Number of bathrooms
- <b>sqft_living</b>: Square footage of the home
- <b>sqft_lot</b>: Square footage of the lot
- <b>floors</b> :Total floors (levels) in house
- <b>waterfront</b> :House which has a view to a waterfront
- <b>view</b>: Has been viewed
- <b>condition</b> :How good the condition is overall
- <b>grade</b>: overall grade given to the housing unit, based on King County grading system
- <b>sqft_above</b> : Square footage of house apart from basement
- <b>sqft_basement</b>: Square footage of the basement
- <b>yr_built</b> : Built Year
- <b>yr_renovated</b> : Year when house was renovated
- <b>zipcode</b>: Zip code
- <b>lat</b>: Latitude coordinate
- <b>long</b>: Longitude coordinate
- <b>sqft_living15</b> : Living room area in 2015(implies-- some renovations) This might or might not have affected the lotsize area
- <b>sqft_lot15</b> : LotSize area in 2015(implies-- some renovations)

## Predicting Price with Square footage of home
The distribution of the price and square footage are skewed to the right but this does not affect the analysis.
To perform a correlation analysis, use the least squares line and check for these assumptions:
1. Linear relationship between the variables
2. The residual model is approximately normal: observed from residual histogram and normal probability plot
3. Variability of residuals is constant
4. The observations are independent: less than 10% of population and simple random observations\
The same assumptions are considered when making inference from the regression model.\
The price vs square footage correlation plot shows that some values may have been underpredicted by the model. The normal probability plot show a deviation from the line which shows that there are outliers or the distribution of data is skewed. The residuals vs fitted values plot shows that the homoscedastic criteria is not met which may be because we are only using one predictor. These are important confounding factors to keep in mind as they may affect the interpretation

The correlation coefficient, coefficient of determination (R^2), slope of least squares line, confidence interval of least squares line and regression equation are computed from the mathematical definitions of each of the terms.

### Analysis of Results
The correlation coefficient (Pearson's r) is 0.702 which means that there is a strong positive linear relationship between the price and square footage of the home.The outliers seen in the correlation plot could have reduced the value of r since they increase the standard deviation of the variables used in correlation coefficient formula. The outliers may be caused by some houses being much larger than the median houses such as penthouses. 

The value of r^2 is 0.493. This means that 49.3% in the variation of price can be explained by the variation in square footage of living. This means that 49.3% of the variation of the data points from the fitted line can be explained by the model. This implies that the model does an average job in fitting the data to the regression line which means that while square footage of the home is a good predictor, it is not enought to predict the prices of the houses overall.

For the regression equation, the point estimate of the slope of the line is 280.636 which means that the price of the house increases by $281 for every square foot increase in the home. The y-intercept means that the price of a house is $-43580 when there is no square footage of the home. We can use the equation to predict the price of a house based on the given value of the square footage.

The confidence interval here means that if we were to take many samples of the slope estimate of the least squares line, there is a 99% chance that the slope estimate will be between 276 and 285. This is under the frequentist interpretation. Furthermore, since the confidence interval does not inclue 0, there is a statistically significant linear relationship between the variables.

### Conclusion
Overall, using induction, conclude that in the total population, there is a correlation between the square footage of the home and the price of the house. Second, by the regression model, the price of the house can be predicted using the square footage of the home but one predictor is not enough.

## Regression with Multiple features
### Wrangling and Exploratory Analysis
As this uses multiple features, remove the columns that are not necessary using `df.drop`.\\
Check for null values and if any, replace with the mean value. This will ensure that the model created accounts for these outliers in predicting the price of the houses.
Check for outliers using `sns.boxplot` in various features. While outliers will affect the standard deviation of the features, I shall not remove them because it is expected that the real-world data has outliers in cases such as penthouses or mansions that may have extra features that are pricier than the median price.

A correlation dataframe of price against other features helps to see which is the least correlated variable and which is the most correlated variable with price. The square footage of price analyzed above in 'Predicting Price with Square footage of home' is the most correlated. The correlation table or the correlation heatmap checks for correlation of the features with other features. This will help identify cases of multicollinearity. Multicollinearity affects the coefficients and p-values but not the prediction from the model and therefore there is no need to worry about it.

### Model Development
Feature scaling of numerical variables is important to ensure that the scale of the variables is similar for better model performance. I use standardization which removes the mean and divides by the standard deviation to ensure the distribution has unit variance. I use pipelines to simultaneously perform the standardization and to replace any missing values with the median value. The data is then split into training and testing sets to use in evaluating the performance of the model. The testing size is only 15% of the dataset.

The linear regression model trains a model using the data provided and can be used to make predictions on test data. To make a prediction, one can replace the values of the desired feature in the equation of the form `Yhat = b0 + b1x1 + b2x2 + b3x3...` where x are the independent variables and b(i) is the coefficeint of the independent variable. The distribution plot of the actual values and predicted values of price of houses based on the features shows that the predicted values are reasonaly close. Better models can improve the prediction while reducing the test error rate. Here, I discuss:
- Ridge regression which is a regularized form of the linear model that checks for overfitting and could reduce the mean-squared error while increasing the r^squared, 
- Decision tree regression which fits a sine curve and helps check for complex nonlinear relationships
- Random forest regression which decorrelates the variables and uses ensemble learning and bagging to predict results with a lower variance and reduce overfitting
I use the root mean-squared error (which gives the prediction error on the test data and predicted value) and r^squared (which as described shows how much of the variation in price is described by the model) as the metric for evaluation of how well the model will perform.


### Model Evaluation and Refinement
I perform cross-validation on all four regression models. The dataset is split into training and validation data sets using cross-validation and train and evaluate the three regression models using k = 5. The advantage of cross-validation is that it reduces testing error since it divides the dataset into equal sections, k, and trains k-1 sections and tests on 1 section. This repeats until all sections have been used as validation data. It helps identify which is the best model that best describes the real world scenario. From this, compare the results of the rmse from each of the models and select the one that minimizes the rmse while having a high r^squared. 

I then fine-tune the best model from the cross-validation result using grid-search and get which are the best estimators to be used in predicting price from an unseen dataset. The best model in this case is the random forest. Finally, using this model and the test set, to get the rmse and compute the confidence interval for this model to see how well it performs.


