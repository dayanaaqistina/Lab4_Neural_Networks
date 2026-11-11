import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sigmoid Activation", page_icon="📈")
st.title("Sigmoid Activation Function")
st.write("Sigmoid converts any real-valued input into an output between 0 and 1.")

x = np.linspace(-10, 10, 400)
y = 1 / (1 + np.exp(-x))

fig, ax = plt.subplots()
ax.plot(x, y, label="Sigmoid: f(x) = 1 / (1 + e^(-x))")
ax.axhline(0, linewidth=0.8)
ax.axvline(0, linewidth=0.8)
ax.set_xlabel("Input (x)")
ax.set_ylabel("Output f(x)")
ax.set_title("Sigmoid Activation Function")
ax.set_ylim(-0.05, 1.05)
ax.grid(True)
ax.legend()
st.pyplot(fig)

value = st.slider("Choose an input value", -10.0, 10.0, 0.0, 0.5)
result = 1 / (1 + np.exp(-value))
st.metric("Sigmoid output", f"{result:.4f}")
st.caption("Sigmoid is useful when an output needs to represent a probability, such as binary classification.")
