import numpy as np
import matplotlib.pyplot as plt

def plot_decision_boundary(X, y, model, test_x=None, test_y=None):
    # Step 3: Generate the mesh grid
    x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
    y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
    x, y = np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100)
    XX, YY = np.meshgrid(x, y)
    
    # Step 4: Predict over the grid
    Z = model.predict(np.c_[XX.ravel(), YY.ravel()])
    Z = Z.reshape(XX.shape)
    
    # Step 5: Plot the results
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.contourf(XX, YY, Z, cmap=plt.cm.RdYlBu, alpha=0.5)
    
    # Plot test data if provided
    if test_x is not None and test_y is not None:
        ax.scatter(test_x[:,0], test_x[:,1], c=test_y, cmap=plt.cm.RdYlBu, edgecolor='k')
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    plt.show()

# Example usage
# plot_decision_boundary(X, y, model, test_x, test_y)