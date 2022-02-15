import pandas as pd
from baselines.baseline1 import Baseline
from new_method import NewMethod
from sklearn.metrics import accuracy_score
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def run_methdos(X_train, X_test, y_train, y_test, params):
    # TODO : update run_methods
    """Run all params['methods'] methods.
    This function will run the methods and generate a pd.DataFrame with the results.
    Args:
        X_train: Features of training set.
        X_test: Features of testing set.
        y_train: Target of training set.
        y_test: Target of testing set.
        params: Dictionary with parameters.
    Returns:
        pd.Dataframe with the results.
    """
    output = pd.DataFrame(columns=['method', 'config',
                                   'metric_train', 'metric_test'])

    for method in params.get('methods', ['new_method']):
        if method == 'new_method':
            model = NewMethod()
        elif method == 'baseline':
            model = Baseline()
        else:
            raise ValueError(f"Method {method} not implemented.")

        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        out = {
            'method': method,
            'config': params.get('config', 'None'),
            'metric_train': metric(y_train, y_train_pred),
            'metric_test': metric(y_test, y_test_pred),
        }
        output = output.append(out, ignore_index=True)

    return output
