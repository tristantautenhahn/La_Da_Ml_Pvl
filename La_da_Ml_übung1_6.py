import numpy as np 
from matplotlib import pyplot as plt

p = 2 # polynomgrad

def design_matrix_linear(x):
    return np.column_stack((x, np.ones_like(x)))

def design_matrix_quadratic(x):
    return np.column_stack((x**2, x, np.ones_like(x)))

def design_beliebige_polynom_matrix(x, p): # p = polynomgrad 
    return np.column_stack((x**p, np.ones_like(x))) # rechnet alles hoch p und fügt eine spalte mit 1 hinzu --> design Matrix 


def regression_normalgleichung(x, y, p): # x muss nicht übergeben werden, da die funktion design_poly... sich darum kümmert 
    b = y.T # transponiert y 
    lsg = np.array()
    lsg = np.linalg.inv(ATA) @ A.T @ b 
    return lsg


x = np.array([1, 2, 3, 4])
y = np.array([1.5, 2.5, 3, 4.5])

lsg = regression_normalgleichung(x, y, p)
print(f"Regressionsparameter des Polynoms {p}.grades: {lsg.shape}")

#(a) 
# A = design_matrix_linear(x)
A = design_beliebige_polynom_matrix(x) # Aufpassen, wenn man reg testen will muss hier design_belibige_polynom(x) stehen !!
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

