import pandas as pd
import numpy as np
import scipy.optimize as opt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def process_training_data(data):
    # Create new variables
    data['price_diff'] = data['output_own_price'] - data['output_comp_price']
    data['total_sales'] = data['output_own_sales'] / data['output_own_share']

    # Time variables
    for i in data['mkt_id'].unique():
        data.loc[data['mkt_id'] == i, 'index_day_of_year'] = range(1, len(data.loc[data['mkt_id'] == i])+1)
    data['bool_week_day'] = data['index_day_of_year'].apply(lambda x: 1 if (((x + 1) % 7 < 6) and ((x + 1) % 7 != 0)) else 0)

    return data

def train_model(processed_data):
    # Share of sales regression
    params_init = np.array([1, 1, 1, 1, 1])
    results_mle = opt.minimize(share_log_likelihood, params_init, args=(processed_data))

    # Total sales regression - weekdays
    data_to_reg = processed_data.loc[processed_data['bool_week_day'] == 1]

    degree = 6
    poly = PolynomialFeatures(degree)
    output_X_poly = np.array(poly.fit_transform(data_to_reg['output_X'].values.reshape(-1, 1)))

    X = np.concatenate((output_X_poly[:, 1:], data_to_reg['output_own_price'].values.reshape(-1, 1), data_to_reg['output_comp_price'].values.reshape(-1, 1)), axis=1)
    y = data_to_reg['total_sales']

    reg_weekdays = LinearRegression().fit(X, y)

    # Total sales regression - weekends
    data_to_reg = processed_data.loc[processed_data['bool_week_day'] == 0]

    degree = 5
    poly = PolynomialFeatures(degree)
    output_X_poly = np.array(poly.fit_transform(data_to_reg['output_X'].values.reshape(-1, 1)))

    X = np.concatenate((output_X_poly[:, 1:], data_to_reg['output_own_price'].values.reshape(-1, 1), data_to_reg['output_comp_price'].values.reshape(-1, 1)), axis=1)
    y = data_to_reg['total_sales']

    reg_weekends = LinearRegression().fit(X, y)

    # P2 regression
    X = pd.concat((processed_data['bool_week_day'],processed_data['output_X'], processed_data['output_X']*processed_data['bool_week_day']), axis=1)
    y = processed_data['output_comp_price']

    X.columns = X.columns.astype(str)
    reg_comp = LinearRegression().fit(X, y)

    return (results_mle, reg_weekdays, reg_weekends, reg_comp)

def process_testing_data(data):
    # Time variables
    for i in data['mkt_id'].unique():
        data.loc[data['mkt_id'] == i, 'index_day_of_year'] = range(1, len(data.loc[data['mkt_id'] == i])+1)
    data['bool_week_day'] = data['index_day_of_year'].apply(lambda x: 1 if (((x + 1) % 7 < 6) and ((x + 1) % 7 != 0)) else 0)

    return data

def predict_price(test_data, model):
    predictions = np.array([])
    exp_profits = np.array([])
    if isinstance(test_data, pd.DataFrame):
        for _, row in test_data.iterrows():
            cost = row['output_own_cost']
            weekday = row['bool_week_day']
            mkt_cond = row['output_X']
            
            params_init = np.array([cost+0.001])
            cons = ({'type': 'ineq', 'fun': lambda x: x[0] - cost})
            price_optimisation = opt.minimize(profit_function, params_init, args=(cost, weekday, mkt_cond, *model), constraints=cons)

            predictions = np.append(predictions, price_optimisation.x[0])
            exp_profits = np.append(exp_profits, -price_optimisation.fun)
    else:
        cost = test_data['output_own_cost']
        weekday = test_data['bool_week_day']
        mkt_cond = test_data['output_X']
        
        params_init = np.array([cost+0.001])
        cons = ({'type': 'ineq', 'fun': lambda x: x[0] - cost})
        price_optimisation = opt.minimize(profit_function, params_init, args=(cost, weekday, mkt_cond, *model), constraints=cons)

        predictions = np.append(predictions, price_optimisation.x[0])
        exp_profits = np.append(exp_profits, -price_optimisation.fun)

    return predictions, exp_profits

# Sigmoid mu Normal MLE
def share_log_likelihood(params, data):
    # params = [beta_0, beta_1, beta_2, beta_3, sigma]
    mu = 1/ (1 + np.exp(-( params[0] + params[1] * data['bool_week_day'] + params[2] * data['price_diff'] + params[3] * data['price_diff'] * data['bool_week_day'])))
    sigma = params[4]
    n = len(data)
    ll = n/2 * np.log(sigma**2 * 2 * np.pi) + 1/ (2 * sigma**2) * sum((data['output_own_share'] - mu)**2)
    return ll

# Profit function to be maximised
def profit_function(p1, c, weekday, mkt_cond, results_mle, reg_weekdays, reg_weekends, reg_comp):
    
    p2 = reg_comp.intercept_ + reg_comp.coef_[0] * weekday + reg_comp.coef_[1] * mkt_cond + reg_comp.coef_[2] * mkt_cond * weekday

    if weekday:
        degree = 6
        ts = reg_weekdays.intercept_ + np.sum([reg_weekdays.coef_[i]*mkt_cond**(i+1) for i in range(degree)],axis=0) + reg_weekdays.coef_[degree]*p1 + reg_weekdays.coef_[degree+1]*p2
        mu = 1/(1+np.exp(-(results_mle.x[0] + results_mle.x[1] + (results_mle.x[2] + results_mle.x[3]) * (p1 - p2))))

    else:
        degree = 5
        ts = reg_weekends.intercept_ + np.sum([reg_weekends.coef_[i]*mkt_cond**(i+1) for i in range(degree)],axis=0) + reg_weekends.coef_[degree]*p1 + reg_weekends.coef_[degree+1]*p2
        mu = 1/(1+np.exp(-(results_mle.x[0] + results_mle.x[2] * (p1 - p2))))
    
    return - (p1 - c) * mu * ts