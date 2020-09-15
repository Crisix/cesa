import matplotlib.pyplot as plt
import numpy as np

from search_utils.Sentence import scale_relatedness

X = np.linspace(0, 20, 1000)
Y = [scale_relatedness(x) for x in X]

plt.figure(figsize=(7, 5))
plt.plot(X, Y, label="Abbildung von $w_{i,j}$ zu Entfernung")
plt.axvspan(-100, 5, alpha=0.3, color='red', label="Ignorierte Wörter")
plt.axvspan(5, 15, alpha=0.3, color='yellow', label="Meistens geeignete Wörter")
plt.axvspan(15, 200, alpha=0.3, color='green', label="Gut geeignete Wörter")
plt.xlim(0, 20)
plt.xlabel(r"Wert $w_{i,j}$ : Je größer desto besser passt das Wort")
plt.ylabel(r"Entfernung zum Original")
plt.legend()
# plt.show()
plt.savefig("relatedness_scaling.pdf")
plt.savefig("relatedness_scaling.png")
