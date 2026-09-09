import io
import base64
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Django
import matplotlib.pyplot as plt

def generate_decision_boundary_plot(model, X_train, y_train, X_holdout, y_holdout, x_col, y_col, target_col, is_classification=True):
    """
    Generates a 2D decision boundary visualization using contourf/pcolormesh,
    overlaying holdout validation data points.
    """
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

    # 1. Select the two features chosen for axes
    X_train_2d = X_train[[x_col, y_col]].values
    X_holdout_2d = X_holdout[[x_col, y_col]].values

    # 2. Fit a visual surrogate model on just the 2 selected 2D features
    # (Since the full model may expect N features, a 2D surrogate faithfully shows 
    # decision boundaries for the selected slice)
    from sklearn.base import clone
    surrogate_model = clone(model)
    surrogate_model.fit(X_train_2d, y_train)

    # 3. Create a dense meshgrid over the plot bounds
    x_min, x_max = X_train_2d[:, 0].min() - 0.5, X_train_2d[:, 0].max() + 0.5
    y_min, y_max = X_train_2d[:, 1].min() - 0.5, X_train_2d[:, 1].max() + 0.5
    
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )

    # 4. Predict over the meshgrid grid points
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    if is_classification:
        Z = surrogate_model.predict(grid_points)
        
        # Convert text labels to numeric IDs for contour plotting if necessary
        if isinstance(Z[0], str):
            unique_labels = np.unique(np.concatenate([y_train, y_holdout]))
            label_map = {label: idx for idx, label in enumerate(unique_labels)}
            Z = np.array([label_map[v] for v in Z])
            y_holdout_num = np.array([label_map[v] for v in y_holdout])
        else:
            y_holdout_num = y_holdout

        Z = Z.reshape(xx.shape)

        # Plot shaded decision boundary regions
        cmap = plt.cm.Set3
        ax.contourf(xx, yy, Z, alpha=0.4, cmap=cmap)

        # Scatter plot holdout validation points on top
        scatter = ax.scatter(
            X_holdout_2d[:, 0], 
            X_holdout_2d[:, 1], 
            c=y_holdout_num, 
            cmap=cmap, 
            edgecolors='k', 
            linewidths=1.2, 
            s=50, 
            alpha=0.9, 
            label='Holdout Validation Data'
        )

    else:
        # Regression background continuous prediction gradient
        Z = surrogate_model.predict(grid_points).reshape(xx.shape)
        
        contour = ax.pcolormesh(xx, yy, Z, cmap='viridis', alpha=0.5, shading='auto')
        fig.colorbar(contour, ax=ax, label=f'Predicted {target_col}')

        scatter = ax.scatter(
            X_holdout_2d[:, 0], 
            X_holdout_2d[:, 1], 
            c=y_holdout, 
            cmap='viridis', 
            edgecolors='white', 
            linewidths=1.2, 
            s=50, 
            label='Holdout Actual'
        )

    ax.set_xlabel(x_col, fontweight='bold')
    ax.set_ylabel(y_col, fontweight='bold')
    ax.set_title(f'Decision Surface for target: "{target_col}"', fontsize=11, fontweight='bold')
    
    plt.tight_layout()

    # Save to Base64 URI string for HTMX rendering
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    encoded_img = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{encoded_img}"