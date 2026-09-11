import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_decision_boundary(model, features, target, step_size=0.02, show_probabilities=False):
    """
    Plots the decision boundary (decision regions) for a TensorFlow/Keras model.
    
    Parameters:
    - model: Trained TensorFlow/Keras Sequential model
    - features: Input dataset features (Pandas DataFrame or NumPy array, shape: [n_samples, 2])
    - target: True target labels/classes (Pandas Series or NumPy array, shape: [n_samples])
    - step_size: Resolution of the background mesh grid
    - show_probabilities: Toggles continuous confidence gradients vs sharp class boundaries
    """
    # 1. Extract raw values for calculations, preserving column names if using Pandas
    if isinstance(features, pd.DataFrame):
        feature_names = features.columns.tolist()
        features_arr = features.values
    else:
        feature_names = ["Feature 1", "Feature 2"]
        features_arr = features

    # Ensure target is a standard NumPy array format
    target_arr = target.values if isinstance(target, pd.Series) else target

    # 2. Define grid limits based on the feature spaces
    x_min, x_max = features_arr[:, 0].min() - 0.5, features_arr[:, 0].max() + 0.5
    y_min, y_max = features_arr[:, 1].min() - 0.5, features_arr[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step_size), 
                         np.arange(y_min, y_max, step_size))

    # print(xx,yy)
    
    # 3. Create the flat grid coordinates for prediction
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Reconstruct a DataFrame if the model expects named inputs
    # if isinstance(features, pd.DataFrame):
    #     grid_points = pd.DataFrame(grid_points, columns=feature_names)

    # 4. Predict over the mesh grid points
    predictions = model.predict(grid_points, verbose=0)
    
    # 5. Reshape predictions back into the 2D grid matrix
    if show_probabilities:
        Z = predictions.reshape(xx.shape)
    else:
        Z = (predictions > 0.5).astype(int).reshape(xx.shape)
        
    # 6. Generate the Plot background
    # plt.figure(figsize=(8, 6))
    
    if show_probabilities:
        contour = plt.contourf(xx, yy, Z, levels=20, cmap=plt.cm.plasma, alpha=0.4)
        plt.colorbar(contour, label="Model Confidence / Probability")
    else:
        plt.pcolormesh(xx, yy, Z, cmap=plt.cm.plasma, alpha=0.3, shading="auto")
    
    # 7. Scatter the true historical data points on top
    plt.scatter(features_arr[:, 0], features_arr[:, 1], c=target_arr, cmap=plt.cm.RdYlBu, edgecolor="k", s=40)
    
    # 8. Apply dynamic styling
    plt.title("Model Decision Boundary")
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    # plt.show()
    


def plot_random_image_classification(model, images, labels, num_images=10):
    i = random.randint(0, len(images) - 1)

    target_image = images[i]
    target_label = labels[i]

    pred = model.predict(target_image.reshape(1, 28, 28))
    pred_label = np.argmax(pred)

    plt.imshow(target_image, cmap='gray')

    if pred_label == target_label:
        plt.title('Correct Prediction', color='green')
    else:
        plt.title('Wrong Prediction', color='red')

    plt.xlabel(f"Predicted Label: {pred_label} - {(pred[0][pred_label])*100:.2f}")