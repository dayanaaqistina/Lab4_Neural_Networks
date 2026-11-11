import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="ReLU Activation", page_icon="📈")
st.title("Rectified Linear Unit (ReLU)")
st.write("ReLU returns 0 for negative inputs and returns the input itself for positive inputs.")

x = np.linspace(-10, 10, 400)
y = np.maximum(0, x)

fig, ax = plt.subplots()
ax.plot(x, y, label="ReLU: f(x) = max(0, x)")
ax.axhline(0, linewidth=0.8)
ax.axvline(0, linewidth=0.8)
ax.set_xlabel("Input (x)")
ax.set_ylabel("Output f(x)")
ax.set_title("ReLU Activation Function")
ax.grid(True)
ax.legend()
st.pyplot(fig)

value = st.slider("Choose an input value", -10.0, 10.0, 2.0, 0.5)
result = max(0.0, value)
st.metric("ReLU output", f"{result:.2f}")
st.caption("ReLU is commonly used in hidden layers because it is simple and helps neural networks learn non-linear patterns.")
