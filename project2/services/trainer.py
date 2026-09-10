import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from django.conf import settings
from django.core.cache import cache

from palmerpenguins import load_penguins
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


class ModelTrainer:
    """
    Trains classifiers on the Palmer Penguins dataset using Scikit-Learn Pipelines.
    Selects the best model based on the penalized score: acc_test - (lambda * complexity).
    Saves performance plots and model visualizers to Django's MEDIA_ROOT.
    """

    def __init__(self, model_type="dt", lambda_val=0.05):
        self.model_type = model_type
        self.lambda_val = float(lambda_val)
        self.best_pipeline = None
        
        # Load and preprocess raw data
        self._prepare_data()

    def _prepare_data(self):
        penguins = load_penguins().dropna()
        X = penguins.drop("species", axis=1)
        y = penguins["species"]

        self.num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    def _get_preprocessor(self):
        """Creates a ColumnTransformer to handle raw categorical and numeric inputs."""
        if self.model_type == "lr":
            numeric_transformer = StandardScaler()
        else:
            numeric_transformer = 'passthrough'

        categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)

        return ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.num_cols),
                ('cat', categorical_transformer, self.cat_cols)
            ]
        )

    def train_and_evaluate(self):
        if self.model_type == "lr":
            return self._train_logistic_regression()
        else:
            return self._train_decision_tree()

    def _train_decision_tree(self):
        max_depth_range = range(1, 11)
        candidates = []
        regularized_scores = []

        for depth in max_depth_range:
            preprocessor = self._get_preprocessor()
            clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
            pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', clf)])
            
            pipeline.fit(self.X_train, self.y_train)

            acc_test = pipeline.score(self.X_test, self.y_test)
            complexity = clf.get_n_leaves()
            score = acc_test - (self.lambda_val * complexity)

            candidates.append({
                'pipeline': pipeline,
                'max_depth': depth,
                'acc_test': acc_test,
                'complexity': complexity,
                'score': score
            })
            regularized_scores.append(score)

        best_candidate = max(candidates, key=lambda item: item['score'])
        self.best_pipeline = best_candidate['pipeline']

        y_pred = self.best_pipeline.predict(self.X_test)
        report = classification_report(self.y_test, y_pred, output_dict=True)

        # Plot optimization curve
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.plot(list(max_depth_range), regularized_scores, marker='o', color='#275CB2', linewidth=2)
        ax.set_xlabel("Max Depth")
        ax.set_ylabel("Penalized Score: Acc - λ(Leaves)")
        ax.axvline(x=best_candidate['max_depth'], color='#e11d48', linestyle='--', label=f"Optimal Depth: {best_candidate['max_depth']}")
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend()

        _, relative_url = _save_plot("dt_optimization.png")
        plt.close(fig)

        # Generate Decision Tree Diagram
        tree_viz_url = self._generate_tree_plot()

        return {
            "model_name": "Decision Tree",
            "best_param_name": "max_depth",
            "best_param_val": best_candidate['max_depth'],
            "test_accuracy": round(best_candidate['acc_test'] * 100, 2),
            "complexity": best_candidate['complexity'],
            "penalized_score": round(best_candidate['score'], 4),
            "report": report,
            "plot_url": relative_url,
            "model_viz_url": tree_viz_url,
            "complexity_penalty": "Leaves"
        }

    def _train_logistic_regression(self):
        c_values = np.logspace(-3, 2, 20)
        candidates = []
        regularized_scores = []

        for c in c_values:
            preprocessor = self._get_preprocessor()
            clf = LogisticRegression(l1_ratio=1.0, C=c, solver='saga', max_iter=5000, random_state=42)
            pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', clf)])

            pipeline.fit(self.X_train, self.y_train)

            acc_test = pipeline.score(self.X_test, self.y_test)
            non_zero_weights = np.sum(clf.coef_ != 0)
            complexity = non_zero_weights
            score = acc_test - (self.lambda_val * complexity)

            candidates.append({
                'pipeline': pipeline,
                'param_value': c,
                'acc_test': acc_test,
                'complexity': complexity,
                'score': score
            })
            regularized_scores.append(score)

        best_candidate = max(candidates, key=lambda item: item['score'])
        self.best_pipeline = best_candidate['pipeline']

        y_pred = self.best_pipeline.predict(self.X_test)
        report = classification_report(self.y_test, y_pred, output_dict=True)

        # Optimization Curve Plot
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.plot(c_values, regularized_scores, marker='o', color='#275CB2')
        ax.set_xscale('log')
        ax.set_xlabel("C Value (Inverse Regularization)")
        ax.set_ylabel("Penalized Score: Acc - λ(L1 Norm)")
        ax.axvline(x=best_candidate['param_value'], color='red', linestyle='--', label=f"Best C: {best_candidate['param_value']:.4f}")
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend()

        _, relative_url = _save_plot("lr_optimization.png")
        plt.close(fig)

        # Generate Weight/Zero Coefficient Matrix Heatmap Plot
        lr_viz_url = self._generate_lr_weights_plot()

        return {
            "model_name": "Logistic Regression",
            "best_param_name": "C",
            "best_param_val": round(best_candidate['param_value'], 4),
            "test_accuracy": round(best_candidate['acc_test'] * 100, 2),
            "complexity": int(best_candidate['complexity']),
            "penalized_score": round(best_candidate['score'], 4),
            "report": report,
            "plot_url": relative_url,
            "model_viz_url": lr_viz_url,
            "complexity_penalty": "Non-zero coefficients"
        }

    def _get_feature_names(self):
        """Extract transformed feature names directly from the fitted ColumnTransformer."""
        preprocessor = self.best_pipeline.named_steps['preprocessor']
        return list(preprocessor.get_feature_names_out())

    def _generate_tree_plot(self):
        """Generates visual representation of the fitted DecisionTree."""
        classifier = self.best_pipeline.named_steps['classifier']
        feature_names = self._get_feature_names()

        fig, ax = plt.subplots(figsize=(14, 8))
        plot_tree(
            classifier,
            feature_names=feature_names,
            class_names=classifier.classes_,
            filled=True,
            rounded=True,
            fontsize=8,
            ax=ax
        )
        _, relative_url = _save_plot("dt_structure.png")
        plt.close(fig)
        return relative_url

    def _generate_lr_weights_plot(self):
        """Generates heatmap matrix highlighting non-zero weights vs eliminated zero-weights."""
        classifier = self.best_pipeline.named_steps['classifier']
        feature_names = self._get_feature_names()
        weights = classifier.coef_

        fig, ax = plt.subplots(figsize=(11, 4))
        im = ax.imshow(weights, cmap='RdBu', aspect='auto', vmin=-np.max(np.abs(weights)), vmax=np.max(np.abs(weights)))
        
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Coefficient Weight Strength", rotation=270, labelpad=15)

        ax.set_xticks(np.arange(len(feature_names)))
        ax.set_yticks(np.arange(len(classifier.classes_)))
        ax.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(classifier.classes_, fontsize=9)

        for i in range(len(classifier.classes_)):
            for j in range(len(feature_names)):
                val = weights[i, j]
                is_zero = np.isclose(val, 0, atol=1e-5)
                
                text_color = "black" if abs(val) < 0.5 else "white"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=7.5, fontweight='bold')
                
                if is_zero:
                    rect = plt.Rectangle((j - 0.45, i - 0.45), 0.9, 0.9, fill=False, edgecolor='red', linewidth=2)
                    ax.add_patch(rect)

        _, relative_url = _save_plot("lr_weights.png")
        plt.close(fig)
        return relative_url

def _save_plot(filename):
    media_dir = os.path.join(settings.MEDIA_ROOT, "plots")
    os.makedirs(media_dir, exist_ok=True)
    unique_filename = f"{int(time.time())}_{filename}"
    filepath = os.path.join(media_dir, unique_filename)
    
    plt.savefig(filepath, format="png", bbox_inches="tight", dpi=150)
    relative_url = os.path.join(settings.MEDIA_URL, "plots", unique_filename)
    return filepath, relative_url


def generate_counterfactuals_model(session_key, sample_id, num_samples, target_class):
    cached_data = cache.get(f"model_{session_key}")
    if not cached_data:
        return {"error": "No trained model found. Please train a model first."}
    
    # Extract trained Pipeline object
    pipeline = cached_data["model"]

    # 1. Load raw dataset
    df_raw = load_penguins().dropna().reset_index(drop=True)
    sample_id = int(sample_id)
    if not (0 <= sample_id < len(df_raw)):
        return {"error": "Invalid sample ID"}

    original_sample = df_raw.iloc[sample_id].copy()
    feature_cols = [c for c in df_raw.columns if c != "species"]

    num_cols_raw = df_raw.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df_raw.select_dtypes(include=['object', 'category']).columns if c != 'species']

    # MAD calculation for raw numeric features
    feature_mads = (df_raw[num_cols_raw] - df_raw[num_cols_raw].median()).abs().median()
    feature_mads[feature_mads == 0] = 1.0

    best_cf_raw = None
    best_dist = float('inf')

    # 2. Search loop
    for _ in range(int(num_samples)):
        cf_raw = original_sample.copy()

        # Mutate numerical values
        for col in num_cols_raw:
            std = df_raw[col].std()
            cf_raw[col] += np.random.normal(0, 2 * std)

        # Mutate categorical values
        for col in cat_cols:
            if np.random.rand() > 0.5:
                cf_raw[col] = np.random.choice(df_raw[col].unique())

        # 3. Create raw un-encoded DataFrame for single prediction
        cf_df_raw = pd.DataFrame([cf_raw[feature_cols]])

        # 4. Predict target directly via pipeline without manual dummy encoding
        predicted_class = str(pipeline.predict(cf_df_raw)[0])

        if predicted_class == target_class:
            num_dist = np.sum(np.abs(cf_raw[num_cols_raw] - original_sample[num_cols_raw]) / feature_mads)
            cat_dist = sum(1.0 for col in cat_cols if cf_raw[col] != original_sample[col])
            total_mad = (num_dist + cat_dist) / (len(num_cols_raw) + len(cat_cols))

            if total_mad < best_dist:
                best_dist = total_mad
                best_cf_raw = cf_raw.to_dict()
                best_cf_raw["predicted_species"] = predicted_class
                best_cf_raw["mad_distance"] = float(round(total_mad, 4))

    if best_cf_raw is None:
        return {"error": f"Could not find a valid counterfactual predicting '{target_class}' within {num_samples} iterations."}

    return best_cf_raw


def generate_pdp_plot(session_key, col):
    """
    Generate Partial Dependence Plot (PDP) for all numerical features and save plots to disk.
    Returns a dict mapping feature names to plot relative URLs.
    """
    cached_data = cache.get(f"model_{session_key}")
    if not cached_data:
        return {"error": "No trained model found. Please train a model first."}

    pipeline = cached_data["model"]
    
    df_raw = load_penguins().dropna().reset_index(drop=True)
    X_raw = df_raw.drop("species", axis=1)
    
    # Extract numerical feature names
    num_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()

    classes = pipeline.named_steps['classifier'].classes_

    fig, ax = plt.subplots(figsize=(8, 4))

    value_range = np.linspace(X_raw[col].min(), X_raw[col].max(), 20)
    pdp_values = []
    
    for val in value_range:
        df_temp = X_raw.copy()
        df_temp[col] = val  # Force feature value across all rows
        
        probs = pipeline.predict_proba(df_temp)
        mean_probs = probs.mean(axis=0)
        pdp_values.append(mean_probs)

    pdp_values = np.array(pdp_values)

    for i, class_name in enumerate(classes):
        ax.plot(value_range, pdp_values[:, i], label=class_name, linewidth=2)

    ax.set_title(f"Partial Dependence Plot for {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Predicted Probability")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()

    _, relative_url = _save_plot(f"pdp_{col}.png")

    plt.close(fig)

    return relative_url
