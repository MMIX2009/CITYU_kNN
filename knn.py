import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import plotly.graph_objs as go
from palmerpenguins import load_penguins
from PIL import Image

# Load datasets
@st.cache_data
def load_iris_data():
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    return X, y, feature_names, target_names

@st.cache_data
def load_penguin_data():
    penguins = load_penguins()
    penguins = penguins.dropna()
    
    features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
    X = penguins[features].values
    
    le = LabelEncoder()
    y = le.fit_transform(penguins['species'])
    
    feature_names = features
    target_names = le.classes_
    
    return X, y, feature_names, target_names

@st.cache_data
def load_wine_data():
    wine = load_wine()
    X = wine.data
    y = wine.target
    feature_names = wine.feature_names
    target_names = wine.target_names
    return X, y, feature_names, target_names

# Load all datasets at the start
iris_data = load_iris_data()
penguin_data = load_penguin_data()
wine_data = load_wine_data()

# Plotting function
def plot_3d_scatter(X, y, feature_names, target_names, test_point=None, predicted_class=None, colors=None):
    traces = []
    
    for i, target in enumerate(target_names):
        indices = y == i
        traces.append(go.Scatter3d(
            x=X[indices, 0],
            y=X[indices, 1],
            z=X[indices, 2],
            mode='markers',
            name=target,
            marker=dict(size=5, color=colors[i])
        ))
    
    if test_point is not None:
        traces.append(go.Scatter3d(
            x=[test_point[0]],
            y=[test_point[1]],
            z=[test_point[2]],
            mode='markers',
            name='Test Point',
            marker=dict(size=8, color='black', symbol='diamond')
        ))
    
    layout = go.Layout(
        scene=dict(
            xaxis_title=feature_names[0],
            yaxis_title=feature_names[1],
            zaxis_title=feature_names[2]
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        legend=dict(x=0.7, y=0.9)
    )
    
    fig = go.Figure(data=traces, layout=layout)
    return fig

def main():
    st.title("k-NN Classification Visualization")

    # Dataset selection
    dataset = st.sidebar.selectbox("Select Dataset", ["Iris", "Penguin", "Wine"])

    # Load and display the appropriate image or information
    if dataset == "Iris":
        image = Image.open("iris_species.png")
        st.image(image, caption="Iris Species", use_column_width=True)
        X, y, feature_names, target_names = iris_data
    elif dataset == "Penguin":
        col1, col2 = st.columns(2)
        with col1:
            image1 = Image.open("Penguin_Bill.png")
            st.image(image1, caption="Penguin Bill Measurements", width=300)
        with col2:
            image2 = Image.open("penguins_species.png")
            st.image(image2, caption="Penguin Species", width=300)
        X, y, feature_names, target_names = penguin_data
    else:  # Wine
        image = Image.open("wine.jpg")
        st.image(image, caption="Wine Varieties", width=400)
        st.write("The Wine dataset contains 13 features of 178 wine samples from three different cultivars in Italy.")
        X, y, feature_names, target_names = wine_data

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create a k-NN classifier
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_scaled, y)

    # Initialize colors in session state if they don't exist
    if 'colors' not in st.session_state or len(st.session_state.colors) != len(target_names):
        st.session_state.colors = [f"#{np.random.randint(0, 0xFFFFFF):06x}" for _ in target_names]

    st.sidebar.header("Feature Selection for Visualization")
    selected_features = st.sidebar.multiselect("Select 3 features for 3D plot", feature_names, default=feature_names[:3])
    
    if len(selected_features) != 3:
        st.warning("Please select exactly 3 features for the 3D plot.")
        return

    feature_indices = [feature_names.index(feature) for feature in selected_features]

    st.sidebar.header("Test Sample Selection")
    test_sample = []
    for feature in feature_names:
        min_val, max_val = X[:, feature_names.index(feature)].min(), X[:, feature_names.index(feature)].max()
        value = st.sidebar.slider(f"{feature}", float(min_val), float(max_val), float(X[:, feature_names.index(feature)].mean()))
        test_sample.append(value)

    st.sidebar.header("k-NN Settings")
    k_neighbors = st.sidebar.slider("Number of Neighbors (k)", 1, 20, 3)
    knn.set_params(n_neighbors=k_neighbors)

    st.sidebar.header("Color Settings")
    for i, target in enumerate(target_names):
        st.session_state.colors[i] = st.sidebar.color_picker(f"Color for {target}", st.session_state.colors[i])

    # Scale the test sample
    test_sample_scaled = scaler.transform(np.array(test_sample).reshape(1, -1))

    # Predict the class of the test sample
    predicted_class = knn.predict(test_sample_scaled)[0]
    predicted_proba = knn.predict_proba(test_sample_scaled)[0]

    # Create the 3D scatter plot
    fig = plot_3d_scatter(X_scaled[:, feature_indices], y, selected_features, target_names, 
                          test_sample_scaled[0, feature_indices], predicted_class, st.session_state.colors)
    st.plotly_chart(fig, use_container_width=True)

    # Display prediction results
    st.subheader("Prediction Results")
    st.write(f"Predicted Class: {target_names[predicted_class]}")
    st.write("Class Probabilities:")
    for target, proba in zip(target_names, predicted_proba):
        st.write(f"{target}: {proba:.4f}")

    # Display test sample values
    st.subheader("Test Sample Features")
    for feature, value in zip(feature_names, test_sample):
        st.write(f"{feature}: {value:.4f}")

if __name__ == "__main__":
    main()