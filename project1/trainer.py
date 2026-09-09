import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    mean_squared_error, r2_score, mean_absolute_error
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

import io
import base64
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.base import clone

class ModelTrainer:
    def __init__(self, df: pd.DataFrame, target_col: str, seed: int):
        self.df = df.dropna(subset=[target_col]) # Safety clean target
        self.target_col = target_col
        self.seed = seed
        self.problem_type = self._detect_problem_type()
        
    def _detect_problem_type(self) -> str:
        """Determines whether the target is classification or regression."""
        target_series = self.df[self.target_col]
        if pd.api.types.is_numeric_dtype(target_series):
            # Unique ratio rule: few unique values relative to size suggests classification
            unique_count = target_series.nunique()
            if unique_count <= 15 or (unique_count / len(target_series)) < 0.05:
                return 'classification'
            return 'regression'
        return 'classification'

    def prepare_data(self, holdout_pct: float, test_pct: float):
        """
        Performs a 2-stage split:
        1. Holds out the Validation/Holdout set from full data.
        2. Splits remaining data into Train and Test sets.
        """
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]

        # Stage 1: Separate Holdout Set
        X_temp, X_holdout, y_temp, y_holdout = train_test_split(
            X, y, test_size=holdout_pct, random_state=self.seed
        )

        # Stage 2: Train / Test Split on Remaining Data
        X_train, X_test, y_train, y_test = train_test_split(
            X_temp, y_temp, test_size=test_pct, random_state=self.seed
        )

        return (X_train, y_train), (X_test, y_test), (X_holdout, y_holdout)

    def get_algorithm_instance(self, algo_name: str, params: dict):
        """Instantiates the correct Scikit-Learn estimator based on form parameters."""
        is_class = self.problem_type == 'classification'

        if algo_name == 'dt':
            max_depth = int(params.get('max_depth', 5)) if params.get('max_depth') else None
            min_samples_split = int(params.get('min_samples_split', 2))
            criterion = params.get('criterion', 'gini' if is_class else 'squared_error')
            
            # Map frontend criterion choices to valid scikit-learn names
            if criterion == 'entropy' and not is_class:
                criterion = 'absolute_error'

            if is_class:
                return DecisionTreeClassifier(
                    max_depth=max_depth, 
                    min_samples_split=min_samples_split, 
                    criterion=criterion, 
                    random_state=self.seed
                )
            else:
                return DecisionTreeRegressor(
                    max_depth=max_depth, 
                    min_samples_split=min_samples_split, 
                    criterion=criterion, 
                    random_state=self.seed
                )

        elif algo_name == 'knn':
            n_neighbors = int(params.get('n_neighbors', 5))
            weights = params.get('weights', 'uniform')
            metric = params.get('metric', 'minkowski')

            if is_class:
                return KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric)
            else:
                return KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, metric=metric)

        elif algo_name in ('linear', 'logistic'):
            max_iter = int(params.get('max_iter', 1000))

            if is_class:
                c_param = float(params.get('c_param', 1.0))
                return LogisticRegression(C=c_param, max_iter=max_iter, random_state=self.seed)
            else:
                return LinearRegression()

        raise ValueError(f"Unsupported algorithm key: '{algo_name}' for problem type '{self.problem_type}'")

    def evaluate(self, model, X, y):
        """Calculates regression or classification performance metrics."""
        preds = model.predict(X)
        if self.problem_type == 'classification':
            acc = accuracy_score(y, preds)
            p, r, f1, _ = precision_recall_fscore_support(y, preds, average='weighted', zero_division=0)
            return {
                'accuracy': round(acc, 4),
                'precision': round(p, 4),
                'recall': round(r, 4),
                'f1_score': round(f1, 4)
            }
        else:
            mse = mean_squared_error(y, preds)
            return {
                'rmse': round(np.sqrt(mse), 4),
                'mae': round(mean_absolute_error(y, preds), 4),
                'r2': round(r2_score(y, preds), 4)
            }

    def train_and_eval(self, algo_name, params, holdout_pct=0.15, test_pct=0.20):
        # 1. Perform splits
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]

        # Split into Holdout set and Training Pool
        X_pool, X_holdout, y_pool, y_holdout = train_test_split(
            X, y, test_size=holdout_pct, random_state=self.seed
        )

        # Split Training Pool into Train and Test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_pool, y_pool, test_size=test_pct, random_state=self.seed
        )

        # 2. Instantiate and fit model
        estimator = self.get_algorithm_instance(algo_name, params)
        estimator.fit(X_train, y_train)

        # 3. Calculate metrics
        train_metrics = self.evaluate(estimator, X_train, y_train)
        test_metrics = self.evaluate(estimator, X_test, y_test)
        holdout_metrics = self.evaluate(estimator, X_holdout, y_holdout)

        metrics = {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'holdout_metrics': holdout_metrics
        }

        # 4. Return everything needed for persistence and plotting
        return {
            'metrics': metrics,
            'estimator': estimator,
            'X_train': X_train,
            'y_train': y_train,
            'X_holdout': X_holdout,
            'y_holdout': y_holdout
        }
    

def generate_decision_boundary_plot(model, X_train, y_train, X_holdout, y_holdout, x_col, y_col, target_col, is_classification=True):
    fig, ax = plt.subplots(figsize=(7, 5), dpi=100)

    # 1. Extract 2D feature slices
    X_train_2d = X_train[[x_col, y_col]].copy()
    X_holdout_2d = X_holdout[[x_col, y_col]].copy()

    # One-hot encode features if x_col or y_col happen to be categorical
    X_train_2d = pd.get_dummies(X_train_2d, drop_first=True)
    X_holdout_2d = pd.get_dummies(X_holdout_2d, drop_first=True)

    # Convert to numeric numpy arrays
    X_tr_vals = X_train_2d.values.astype(float)
    X_ho_vals = X_holdout_2d.values.astype(float)

    # 2. Convert string/categorical target values to numeric IDs
    if is_classification or isinstance(y_train.iloc[0], str):
        unique_labels, y_train_num = np.unique(y_train, return_inverse=True)
        label_map = {lbl: idx for idx, lbl in enumerate(unique_labels)}
        y_holdout_num = np.array([label_map[v] for v in y_holdout])
        num_classes = len(unique_labels)
    else:
        y_train_num = y_train.values.astype(float)
        y_holdout_num = y_holdout.values.astype(float)

    # 3. Fit 2D surrogate model
    surrogate = clone(model)
    surrogate.fit(X_tr_vals, y_train_num)

    # 4. Create dense grid
    margin_x = (X_tr_vals[:, 0].max() - X_tr_vals[:, 0].min()) * 0.1 or 0.5
    margin_y = (X_tr_vals[:, 1].max() - X_tr_vals[:, 1].min()) * 0.1 or 0.5

    x_min, x_max = X_tr_vals[:, 0].min() - margin_x, X_tr_vals[:, 0].max() + margin_x
    y_min, y_max = X_tr_vals[:, 1].min() - margin_y, X_tr_vals[:, 1].max() + margin_y

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )

    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = surrogate.predict(grid_points).reshape(xx.shape)

    # 5. Plotting with strict color alignment
    if is_classification:
        # Use discrete Tab10 colormap and explicit normalization
        cmap = plt.get_cmap('Set3', max(num_classes, 3))
        norm = Normalize(vmin=0, vmax=max(num_classes - 1, 1))

        # Filled contours for decision regions
        ax.contourf(
            xx, yy, Z, 
            levels=np.arange(-0.5, num_classes, 1), 
            cmap=cmap, 
            norm=norm, 
            alpha=0.45
        )

        # Holdout dots overlay
        scatter = ax.scatter(
            X_ho_vals[:, 0], 
            X_ho_vals[:, 1], 
            c=y_holdout_num, 
            cmap=cmap, 
            norm=norm, 
            edgecolors='black', 
            linewidths=1.0, 
            s=50, 
            label='Holdout Validation Data'
        )

    else:
        # Continuous regression surface
        vmin = min(Z.min(), y_holdout_num.min())
        vmax = max(Z.max(), y_holdout_num.max())
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.cm.viridis

        contour = ax.pcolormesh(
            xx, yy, Z, 
            cmap=cmap, 
            norm=norm, 
            alpha=0.5, 
            shading='auto'
        )
        fig.colorbar(contour, ax=ax, label=f'Predicted {target_col}')

        ax.scatter(
            X_ho_vals[:, 0], 
            X_ho_vals[:, 1], 
            c=y_holdout_num, 
            cmap=cmap, 
            norm=norm, 
            edgecolors='black', 
            linewidths=1.0, 
            s=50
        )

    ax.set_xlabel(x_col, fontweight='bold')
    ax.set_ylabel(y_col, fontweight='bold')
    ax.set_title(f'2D Decision Surface: {target_col}', fontsize=11, fontweight='bold')

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)

    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"