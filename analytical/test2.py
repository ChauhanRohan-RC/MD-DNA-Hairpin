import matplotlib.pyplot as plt
import numpy as np

import sp_impl

x = np.linspace(-0.5, 0, 50)
y = sp_impl.critical_bias(x)

plt.plot(x, y)
plt.title("Critical Bias as a function of Depth")
plt.xlabel("Depth (A)")
plt.ylabel("Critical Bias")
plt.show()
