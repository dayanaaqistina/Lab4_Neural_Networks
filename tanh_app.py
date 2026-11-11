import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Tanh Activation", page_icon="📈")
st.title("Hyperbolic Tangent (Tanh) Activation")
st.write("Tanh maps input values to an output range between -1 and 1.")

x = np.linspace(-10, 10, 400)
y = np.tanh(x)

fig, ax = plt.subplots()
ax.plot(x, y, label="Tanh: f(x) = tanh(x)")
ax.axhline(0, linewidth=0.8)
ax.axvline(0, linewidth=0.8)
ax.set_xlabel("Input (x)")
ax.set_ylabel("Output f(x)")
ax.set_title("Tanh Activation Function")
ax.set_ylim(-1.1, 1.1)
ax.grid(True)
ax.legend()
st.pyplot(fig)

value = st.slider("Choose an input value", -10.0, 10.0, 0.0, 0.5)
result = np.tanh(value)
st.metric("Tanh output", f"{result:.4f}")
st.caption("Tanh is zero-centred and is often used in hidden layers in some neural network architectures.")
