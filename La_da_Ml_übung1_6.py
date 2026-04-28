import numpy as np 
from matplotlib import pyplot as plt

def design_matrix_linear(x):
    return np.column_stack((x, np.ones_like(x)))

def design_matrix_quadratic(x):
    return np.column_stack((x**2, x, np.ones_like(x)))


x = np.array([1, 2, 3, 4])
y = np.array([1.5, 2.5, 3, 4.5])

#(a) 
# A = design_matrix_linear(x)
A = design_matrix_quadratic(x)
print(f"Dim(A) = {A.shape}\nA = \n{A}")

#(b)
ATA = A.T @ A # A transponiert mal A 
ATy = A.T @ y # A transponiert mal y (vektor y) 
print(f"Dim(ATA) = {ATA.shape}\nATA = \n{ATA}")
print(f"Dim(ATy) = {ATy.shape}\nATy = \n{ATy}")

#(c) 
sol = np.linalg.inv(ATA) @ ATy
print(f"[m n] = {sol}")

#(e)
sol_poly = np.polyfit(x, y, deg=2)
sol_lest_squares = np.linalg.lstsq(A, y)
print(f"[m n] = {sol_poly}")
print(f"[m n] = {sol_lest_squares[0]}") # [0] da np.linalg.lstsq() mehrere Werte zurückgibt, wir interessieren uns nur für die Koeffizienten

#(d)
ATr = A.T @ (A@sol - y) #keine lösung aber fehler soll senkrecht zu A liegen Lösung gibt es nicht, da nicht alle Punkte auf der geraden liegen 
print(f"ATr = {ATr}")

plt.plot(x, y, '+')
plt.show()

