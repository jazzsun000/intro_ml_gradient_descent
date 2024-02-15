import streamlit as st
import numpy as np
import plotly.graph_objs as go
import plotly.express as px

st.title("Gradient Descent Method Demonstrator")

st.write("""
    ### Explore How Different Gradient Descent Methods Optimize Loss Functions
    This interactive tool demonstrates the behavior of various gradient descent methods 
    (Vanilla, Momentum, and Adagrad) on different loss functions. Adjust the parameters 
    to see how these methods approach optimization differently.
""")

# Define different loss functions and their gradients
def quadratic_loss(x):
    return x ** 2, 2 * x

def complex_loss(x):
    return np.sin(5 * x) + x ** 2, 5 * np.cos(5 * x) + 2 * x

# Implement gradient descent methods with tracking of each iteration
def gradient_descent(loss_func, start, learn_rate, n_iter=100, method='vanilla', alpha=0.9, epsilon=1e-8):
    x = start
    v = 0
    grad_accumulate = 0
    path = [start]  # Track the path of x values
    for _ in range(n_iter):
        _, grad = loss_func(x)
        if method == 'vanilla':
            x = x - learn_rate * grad
        elif method == 'momentum':
            v = alpha * v + learn_rate * grad
            x = x - v
        elif method == 'adagrad':
            grad_accumulate += grad ** 2
            adjusted_lr = learn_rate / (np.sqrt(grad_accumulate) + epsilon)
            x = x - adjusted_lr * grad
        path.append(x)
    return np.array(path)

# Sidebar settings for the app
st.sidebar.header("Gradient Descent Settings")
start = st.sidebar.slider("Starting Point", -2.0, 2.0, 0.0, help="Initial starting point for the gradient descent.")
learn_rate = st.sidebar.slider("Learning Rate", 0.001, 0.1, 0.01, help="Step size for each iteration.")
n_iter = st.sidebar.slider("Number of Iterations", 10, 100, help="Total number of iterations for the descent.")
method = st.sidebar.selectbox("Method", ['vanilla', 'momentum', 'adagrad'])
loss_function = st.sidebar.selectbox("Choose the loss function", ["Quadratic", "Complex"])

# Get the corresponding loss function and gradient
loss_func = quadratic_loss if loss_function == "Quadratic" else complex_loss

# Run gradient descent method and track the path
path = gradient_descent(loss_func, start, learn_rate, n_iter, method)

# Generate data for the loss function plot
x_values = np.linspace(-2.5, 2.5, 400)
y_values = np.array([loss_func(x)[0] for x in x_values])

# Create Plotly figure
fig = px.line(x=x_values, y=y_values, labels={'x': 'x', 'y': 'Loss'}, title='Gradient Descent Visualization')
fig.add_trace(go.Scatter(x=path, y=[loss_func(x)[0] for x in path], mode='markers+lines', name='GD Path'))

# Animate the path
fig.update_layout(transition={'duration': 30}, title_text='Gradient Descent Animation')
fig.update_traces(marker=dict(size=10),
                  selector=dict(mode='markers+lines'))

# Display the figure in the Streamlit app
st.plotly_chart(fig)

st.write(f"### Method: {method}")
st.write(f"### Starting Point: {start}")
st.write(f"### Learning Rate: {learn_rate}")
st.write(f"### Iterations: {n_iter}")
