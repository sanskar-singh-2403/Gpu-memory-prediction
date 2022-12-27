import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import seaborn as sns
import matplotlib.pyplot as plt
from lmfit.models import StepModel, LinearModel
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

def memory_prediction(para_year):
    #importing dataset

    dataset = pd.read_csv('All_GPUs.csv')

    #defining key coplumns

    key_columns = ['Best_Resolution', 'Core_Speed', 'Manufacturer', 'Memory', 'Memory_Bandwidth', 'Name', 'Release_Date']
    dataset = dataset[key_columns]

    #combining year and month to create a release column 

    dataset['Release_Date']=dataset['Release_Date'].str[1:-1]
    dataset=dataset[dataset['Release_Date'].str.len()==11]
    dataset['Release_Date']=pd.to_datetime(dataset['Release_Date'], format='%d-%b-%Y')
    dataset['Release_Year']=dataset['Release_Date'].dt.year
    dataset['Release_Month']=dataset['Release_Date'].dt.month
    dataset['Release']=dataset['Release_Year'] + dataset['Release_Month']/12
    dataset['Memory'] = dataset['Memory'].str[:-3].fillna(0).astype(int)
    # Numpy array that holds unique release year values
    year_arr = dataset.sort_values("Release_Year")['Release_Year'].unique()
    # Numpy array that holds mean values of GPUs memory for each year
    memory_arr_mean = dataset.groupby('Release_Year')['Memory'].mean().values
    # Numpy array that holds median values of GPUs memory for each year
    memory_arr_median = dataset.groupby('Release_Year')['Memory'].median().values

    # Minimal value of release year from dataset
    year_min = year_arr[0]
    # Median size of memory in year_min
    memory_min = memory_arr_median[0]
    def calculateMooresValue(x, y_trans):
        return memory_arr_median[0] * 2**((x-y_trans)/2)

    # GPU Memory Size calculation based on Moore's Law
    y_pred_moore_law_teoretic = calculateMooresValue(year_arr, int(year_min))

    def exponentialCurve(x, a, b, c):
        return a*2**((x-c)*b)

    # popt,pcov= curve_fit(exponentialCurve,  year_arr, memory_arr_mean,  p0=(2, 0.5, 1998))
    # y_pred_moore_law_fitted = exponentialCurve(year_arr, *popt)

    # Fitting Polynomial Regression to the dataset
    poly_reg_2 = PolynomialFeatures(degree = 2, include_bias=False)
    poly_reg_3 = PolynomialFeatures(degree = 3, include_bias=False)

    X_poly_2 = poly_reg_2.fit_transform(year_arr.reshape(-1, 1))
    X_poly_3 = poly_reg_3.fit_transform(year_arr.reshape(-1, 1))

    lin_reg_2 = LinearRegression()
    lin_reg_3 = LinearRegression()

    lin_reg_2.fit(X_poly_2, memory_arr_mean)
    lin_reg_3.fit(X_poly_3, memory_arr_mean)
    
    y_pred_lin_reg_2 = lin_reg_2.predict(poly_reg_2.fit_transform(year_arr.reshape(-1, 1)))
    y_pred_lin_reg_3 = lin_reg_3.predict(poly_reg_3.fit_transform(year_arr.reshape(-1, 1)))

    # ear=int(input("Enter the year you want the memory size: "))
    memory_2025 = exponentialCurve(int(para_year),2,0.5,1998)
    return (round(int(memory_2025) / 1024, 2))

# para_year=int(input("Enter the year you want the memory size: "))
# memory_prediction(para_year)
