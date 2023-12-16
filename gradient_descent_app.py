# Required installations:
# pip install streamlit numpy matplotlib

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Gradient Descent Method Demonstrator")

st.write("""
    ### Explore How Different Gradient Descent Methods Optimize Loss Functions
    This interactive tool demonstrates the behavior of various gradient descent methods 
    (Vanilla, Momentum, and Adagrad) on different loss functions. Adjust the parameters 
    to see how these methods approach optimization differently.
""")

# Define different loss functions and their gradients
def quadratic_loss(x):
    return x**2, 2*x

def complex_loss(x):
    return np.sin(5*x) + x**2, 5*np.cos(5*x) + 2*x

# Gradient Descent Methods
def vanilla_gradient_descent(loss_func, start, learn_rate, n_iter=100):
    x = start
    for _ in range(n_iter):
        _, grad = loss_func(x)  # Unpack to get the gradient
        x = x - learn_rate * grad
    return x

def momentum_gradient_descent(loss_func, start, learn_rate, n_iter=100, alpha=0.9):
    x = start
    v = 0
    for _ in range(n_iter):
        _, grad = loss_func(x)  # Unpack to get the gradient
        v = alpha * v + learn_rate * grad
        x = x - v
    return x

def adagrad(loss_func, start, learn_rate, n_iter=100, epsilon=1e-8):
    x = start
    grad_accumulate = 0
    for _ in range(n_iter):
        _, grad = loss_func(x)  # Unpack to get the gradient
        grad_accumulate += grad**2
        adjusted_lr = learn_rate / (np.sqrt(grad_accumulate) + epsilon)
        x = x - adjusted_lr * grad
    return x

# Sidebar settings for the app
st.sidebar.header("Gradient Descent Settings")
start = st.sidebar.slider("Starting Point", -2.0, 2.0, 0.0, help="Initial starting point for the gradient descent.")
learn_rate = st.sidebar.slider("Learning Rate", 0.001, 0.1, 0.01, help="Step size for each iteration.")
n_iter = st.sidebar.slider("Number of Iterations", 10, 1000, 100, help="Total number of iterations for the descent.")

# Choose the loss function
loss_function = st.sidebar.selectbox(
    "Choose the loss function",
    ("Quadratic", "Complex"),
    help="Select a loss function to optimize."
)

# Get the corresponding loss function and gradient
if loss_function == "Quadratic":
    loss_func = quadratic_loss
else:
    loss_func = complex_loss

# Run gradient descent methods
vanilla_result = vanilla_gradient_descent(loss_func, start, learn_rate, n_iter)
momentum_result = momentum_gradient_descent(loss_func, start, learn_rate, n_iter)
adagrad_result = adagrad(loss_func, start, learn_rate, n_iter)

# Plotting
x_values = np.linspace(-2, 2, 400)
y_values = [loss_func(x)[0] for x in x_values]  # Get only the loss values

fig, ax = plt.subplots()
ax.plot(x_values, y_values, label='Loss function')
ax.scatter([vanilla_result], [loss_func(vanilla_result)[0]], color='red', zorder=5, label='Vanilla GD Result')
ax.scatter([momentum_result], [loss_func(momentum_result)[0]], color='blue', zorder=5, label='Momentum GD Result')
ax.scatter([adagrad_result], [loss_func(adagrad_result)[0]], color='green', zorder=5, label='Adagrad Result')
ax.set_xlabel('x')
ax.set_ylabel('Loss')
ax.legend()
st.pyplot(fig)

st.write("### Explanation and Results")
st.write(f"""
    - **Starting Point:** {start}  
    - **Learning Rate:** {learn_rate}  
    - **Iterations:** {n_iter}  
    - **Selected Loss Function:** {loss_function}  
""")

st.write(f"Vanilla GD final position: {vanilla_result:.2f}")
st.write(f"Momentum GD final position: {momentum_result:.2f}")
st.write(f"Adagrad final position: {adagrad_result:.2f}")
st.write("""
    The plot above shows the selected loss function and the paths taken by different 
    gradient descent methods. The red, blue, and green dots represent the final positions 
    reached by Vanilla, Momentum, and Adagrad methods, respectively.
""")
