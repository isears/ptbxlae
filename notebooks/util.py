from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


# Util function to do a quick LR on a specified target based on latent variables
def eval_predictive_power_binary_outcome(x, y):
    lr = LogisticRegression()

    ret = {"Total usable": len(y), "% positive": sum(y) / len(y)}

    if sum(y) < 10:  # Won't be able to do CV
        ret["Avg CV score"] = float("nan")
        return ret

    try:
        scores = cross_val_score(lr, x, y, cv=5, scoring="roc_auc")
        ret["Avg CV score"] = sum(scores) / len(scores)
        return ret
    except ValueError:
        ret["Avg CV score"] = float("nan")
        return ret
