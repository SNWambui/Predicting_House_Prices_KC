# Correlation-and-Regression
This is correlation and regression analysis of the King County House Sales data. 
The first file answers a specific question I developed for a CS51 class assignment:
- Does the square footage of the home have a significant impact on the price of the house?
The second file is an iteration of the same dataset that includes more features that are evaluated to predict the prices of the house.
Sample data is used to predict the prices of houses in the entire population

## The dataset features
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
4. The observations are independent: less than 10% of population and simple random observations
The same assumptions are considered when making inference from the regression model.
The price vs square footage correlation plot shows that some values may have been underpredicted by the model. The normal probability plot show a deviation from the line which shows that there are outliers or the distribution of data is skewed.. The residuals vs fitted values plot shows that the homoscedastic criteria is not met which may be because we are only using one predictor. These are important confounding factors to keep in mind as they may affect the interpretation

The correlation coefficient, coefficient of determination (R^2), slope of least squares line, confidence interval of least squares line and regression equation are computed from the mathematical definitions of each of the terms.

### Analysis of Results
The correlation coefficient (Pearson's r) is 0.702 which means that there is a strong positive linear relationship between the price and square footage of the home.The outliers seen in the correlation plot could have reduced the value of r since they increase the standard deviation of the variables used in correlation coefficient formula. As stated above, the outliers may be caused by using only one predictor.\

The value of r^2 is 0.493. This means that 49.3% in the variation of price can be explained by the variation in square footage of living. This means that 49.3% of the variation of the data points from the fitted line can be explained by the model. This implies that the model does an average job in fitting the data to the regression line which means that while square footage of the home is a good predictor, it is not enought to predict the prices of the houses overall.\

For the regression equation, the point estimate of the slope of the line is 280.636 which means that the price of the house increases by $281 for every square foot increase in the home. The y-intercept means that the price of a house is $-43580 when there is no square footage of the home. We can use the equation to predict the price of a house based on the given value of the square footage.\

The confidence interval here means that if we were to take many samples of the slope estimate of the least squares line, there is a 99% chance that the slope estimate will be between 276 and 285. This is under the frequentist interpretation. Furthermore, since the confidence interval does not inclue 0, there is a statistically significant linear relationship between the variables.\

### Conclusion
Overall, using induction, conclude that in the total population, there is a correlation between the square footage of the home and the price of the house. Second, by the regression model, the price of the house can be predicted using the square footage of the home but one predictor is not enough.
