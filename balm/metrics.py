import torch
from scipy import stats
from sklearn.metrics import average_precision_score

def get_aupr(Y, P, threshold=7.0):
    """
    Calculate the Area Under the Precision-Recall Curve (AUPR) for binary classification.

    Args:
        Y (Tensor): Ground truth labels.
        P (Tensor): Predicted labels or probabilities.
        threshold (float, optional): The threshold to binarize the labels. Default is 7.0.

    Returns:
        float: The AUPR score.
    """
    # Binarize the labels based on the threshold
    Y = (Y >= threshold).float()
    P = (P >= threshold).float()
    # Compute the AUPR score
    aupr = average_precision_score(Y.cpu().numpy(), P.cpu().numpy())
    return aupr

def get_cindex(Y, P):
    """
    Calculate the Concordance Index (C-index) for evaluating the predictions.

    Args:
        Y (Tensor): Ground truth labels.
        P (Tensor): Predicted labels or probabilities.

    Returns:
        float: The C-index score.
    """
    # Sort the labels and predictions
    indices = torch.argsort(Y)
    Y = Y[indices]
    P = P[indices]
    
    # Calculate concordant pairs
    summ = torch.sum(
        (Y[:-1].unsqueeze(0) < Y[1:].unsqueeze(1)).float()
        * (P[:-1].unsqueeze(0) < P[1:].unsqueeze(1)).float()
    )
    
    # Calculate total comparable pairs
    total = torch.sum(Y[:-1].unsqueeze(0) < Y[1:].unsqueeze(1)).float()
    return summ / total

def r_squared_error(y_obs, y_pred):
    """
    Calculate the R-squared (coefficient of determination) between observed and predicted values.

    Args:
        y_obs (Tensor): Observed/ground truth values.
        y_pred (Tensor): Predicted values.

    Returns:
        float: The R-squared value.
    """
    y_obs_mean = y_obs.mean()
    y_pred_mean = y_pred.mean()
    
    # Compute the numerator (covariance squared)
    mult = torch.sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean)) ** 2
    
    # Compute the denominator (product of variances)
    y_obs_sq = torch.sum((y_obs - y_obs_mean) ** 2)
    y_pred_sq = torch.sum((y_pred - y_pred_mean) ** 2)
    
    return mult / (y_obs_sq * y_pred_sq)

def get_k(y_obs, y_pred):
    """
    Calculate the slope (k) of the regression line between observed and predicted values.

    Args:
        y_obs (Tensor): Observed/ground truth values.
        y_pred (Tensor): Predicted values.

    Returns:
        float: The slope k of the regression line.
    """
    return torch.sum(y_obs * y_pred) / torch.sum(y_pred * y_pred)

def squared_error_zero(y_obs, y_pred):
    """
    Calculate the squared error for the regression line passing through the origin.

    Args:
        y_obs (Tensor): Observed/ground truth values.
        y_pred (Tensor): Predicted values.

    Returns:
        float: The squared error for the zero-intercept regression line.
    """
    k = get_k(y_obs, y_pred)
    y_obs_mean = y_obs.mean()
    
    # Compute the upper part (sum of squared residuals)
    upp = torch.sum((y_obs - (k * y_pred)) ** 2)
    
    # Compute the lower part (total sum of squares)
    down = torch.sum((y_obs - y_obs_mean) ** 2)
    
    return 1 - (upp / down)

def get_rm2(ys_orig, ys_line):
    """
    Calculate the RM2 metric, a variant of R-squared adjusted for model robustness.

    Args:
        ys_orig (Tensor): Observed/ground truth values.
        ys_line (Tensor): Predicted values.

    Returns:
        float: The RM2 value.
    """
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)
    
    return r2 * (1 - torch.sqrt(torch.abs((r2 * r2) - (r02 * r02))))

def get_rmse(y, f):
    """
    Calculate the Root Mean Squared Error (RMSE) between observed and predicted values.

    Args:
        y (Tensor): Observed/ground truth values.
        f (Tensor): Predicted values.

    Returns:
        float: The RMSE value.
    """
    return torch.sqrt(torch.mean((y - f) ** 2))

def get_mse(y, f):
    """
    Calculate the Mean Squared Error (MSE) between observed and predicted values.

    Args:
        y (Tensor): Observed/ground truth values.
        f (Tensor): Predicted values.

    Returns:
        float: The MSE value.
    """
    return torch.mean((y - f) ** 2)

def get_pearson(y, f):
    """
    Calculate the Pearson correlation coefficient between observed and predicted values.

    Args:
        y (Tensor): Observed/ground truth values.
        f (Tensor): Predicted values.

    Returns:
        float: The Pearson correlation coefficient.
    """
    return torch.nn.functional.cosine_similarity(y - y.mean(), f - f.mean(), dim=0)

def get_spearman(y, f):
    """
    Calculate the Spearman rank correlation coefficient between observed and predicted values.

    Args:
        y (Tensor): Observed/ground truth values.
        f (Tensor): Predicted values.

    Returns:
        float: The Spearman rank correlation coefficient.
    """
    y = y.cpu().numpy()
    f = f.cpu().numpy()
    
    return stats.spearmanr(y, f)[0]

def get_ci(y, f):
    """
    Calculate the Concordance Index (CI) for evaluating the predictions.

    Args:
        y (Tensor): Ground truth labels.
        f (Tensor): Predicted labels or probabilities.

    Returns:
        float: The CI score.
    """
    ind = torch.argsort(y)
    y = y[ind]
    f = f[ind]
    
    # Calculate the total number of comparable pairs
    z = torch.sum(y[:-1].unsqueeze(0) < y[1:].unsqueeze(1)).float()
    
    # Calculate the number of concordant pairs
    S = torch.sum(
        (y[:-1].unsqueeze(0) < y[1:].unsqueeze(1)).float()
        * (f[:-1].unsqueeze(0) < f[1:].unsqueeze(1)).float()
    )
    
    return S / z
