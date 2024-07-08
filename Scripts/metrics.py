
"""
Useful methods to compute all the different metrics such as MSE, MRE, R^2, confidence intervals, ...
"""
import sklearn.metrics as metrics
import torch.nn as nn
import pandas as pd
import numpy as np


def compute_CI(preds, targets):
    """
    Compute MSE confidence intervals for 1D array predictions

    Parameters
    ----------
    preds: predictions array
    targets: targets array

    Returns
    --------
    Lower bound, Higher bound and confidence interval
    """
    mse_ls  = ((targets-preds)**2).tolist()
    mean = np.mean(mse_ls)
    std = np.std(mse_ls)
    z = 1.96 # 95% CI

    return mean - (z*(std/np.sqrt(len(mse_ls)))), mean + (z*(std/np.sqrt(len(mse_ls)))), z*(std/np.sqrt(len(mse_ls)))

def all_metrics(preds, targets):
    """
    Compute the different metrics for predictions and returns the mean.
    Both must be of same shape.

    Parameters
    ----------
    preds: predictions array
    targets: targets array

    Returns
    --------
    MSE, RMSE, MAPE (MRE) and R^2

    """
    mape = metrics.mean_absolute_percentage_error(targets, preds)
    mse = metrics.mean_squared_error(targets, preds)
    rmse = metrics.mean_squared_error(targets, preds, squared=False)
    r2 = metrics.r2_score(targets, preds)

    bound_inf, bound_sup, diff = compute_CI(preds, targets)

    print(f"MRE: {mape}")
    print(f"MAE: {metrics.mean_absolute_error(targets, preds)}")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"R2: {r2}")
    print(f"CI: {bound_inf, bound_sup, diff}")

    return mse, rmse, mape, r2


def output_wise_metrics(preds, targets):
    """
    Compute the different metrics for predictions for each target varaible separately.
    Both must be of same shape.

    Parameters
    ----------
    preds: predictions array
    targets: targets array

    Returns
    --------
    Dataframe with the number of rows equal to number of targets

    """
    mse_fn = nn.MSELoss()
    df = pd.DataFrame(columns=['MSE', 'RMSE', 'R2'])

    for i in range(preds.shape[1]):
        current_pred = preds[:, i]
        current_target = targets[:, i]
        mse = mse_fn(current_pred, current_target)
        rmse = np.sqrt(mse)
        mape = metrics.mean_absolute_percentage_error(current_target, current_pred)
        r2 = metrics.r2_score(current_target, current_pred)

        bound_inf, bound_sup, diff = compute_CI(current_pred, current_target)
        
        df = df.append({'MSE': float(mse), 'RMSE': float(rmse), 'MAPE': mape, 'R2': r2, 'CI-':bound_inf, 'CI+':bound_sup, 'diff':diff}, ignore_index=True)
    
    print(f"MSE uniform weight: {metrics.mean_squared_error(targets, preds)}")
    print(f"MSE raw: {metrics.mean_squared_error(targets, preds, multioutput='raw_values')}")
    print(f"RMSE uniform weight: {metrics.mean_squared_error(targets, preds, squared=False)}")
    print(f"MAPE uniform weight: {metrics.mean_absolute_percentage_error(targets, preds)}")
    print(f"R2 uniform weight: {metrics.r2_score(targets, preds)}")
    return df