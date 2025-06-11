import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
})

x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

plt.plot(x, y1, label=r'\textbf{Thinking}', linestyle='-', marker='^')
plt.plot(x, y2, label=r'\textbf{No Thinking}', linestyle='--', marker='o')
plt.xlabel(r'\textit{Inference Compute Budget (Tokens)}')
plt.ylabel(r'\textit{pass@1k}')
plt.legend()
plt.grid(True)
plt.show()
