import torch
from scipy import stats
from sklearn.metrics import average_precision_score


def get_aupr(Y, P, threshold=7.0):
    Y = (Y >= threshold).float()
    P = (P >= threshold).float()
    aupr = average_precision_score(Y.cpu().numpy(), P.cpu().numpy())
    return aupr


def get_cindex(Y, P):
    indices = torch.argsort(Y)
    Y = Y[indices]
    P = P[indices]
    summ = torch.sum(
        (Y[:-1].unsqueeze(0) < Y[1:].unsqueeze(1)).float()
        * (P[:-1].unsqueeze(0) < P[1:].unsqueeze(1)).float()
    )
    total = torch.sum(Y[:-1].unsqueeze(0) < Y[1:].unsqueeze(1)).float()
    return summ / total


def r_squared_error(y_obs, y_pred):
    y_obs_mean = y_obs.mean()
    y_pred_mean = y_pred.mean()
    mult = torch.sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean)) ** 2
    y_obs_sq = torch.sum((y_obs - y_obs_mean) ** 2)
    y_pred_sq = torch.sum((y_pred - y_pred_mean) ** 2)
    return mult / (y_obs_sq * y_pred_sq)


def get_k(y_obs, y_pred):
    return torch.sum(y_obs * y_pred) / torch.sum(y_pred * y_pred)


def squared_error_zero(y_obs, y_pred):
    k = get_k(y_obs, y_pred)
    y_obs_mean = y_obs.mean()
    upp = torch.sum((y_obs - (k * y_pred)) ** 2)
    down = torch.sum((y_obs - y_obs_mean) ** 2)
    return 1 - (upp / down)


def get_rm2(ys_orig, ys_line):
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)
    return r2 * (1 - torch.sqrt(torch.abs((r2 * r2) - (r02 * r02))))


def get_rmse(y, f):
    return torch.sqrt(torch.mean((y - f) ** 2))


def get_mse(y, f):
    return torch.mean((y - f) ** 2)


def get_pearson(y, f):
    return torch.nn.functional.cosine_similarity(y - y.mean(), f - f.mean(), dim=0)


def get_spearman(y, f):
    y = y.cpu().numpy()
    f = f.cpu().numpy()
    return stats.spearmanr(y, f)[0]


def get_ci(y, f):
    ind = torch.argsort(y)
    y = y[ind]
    f = f[ind]
    z = torch.sum(y[:-1].unsqueeze(0) < y[1:].unsqueeze(1)).float()
    S = torch.sum(
        (y[:-1].unsqueeze(0) < y[1:].unsqueeze(1)).float()
        * (f[:-1].unsqueeze(0) < f[1:].unsqueeze(1)).float()
    )
    return S / z
