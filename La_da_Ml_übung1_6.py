import numpy as np 
from matplotlib import pyplot as plt

p = 4 # polynomgrad

def design_matrix_linear(x):
    return np.column_stack((x, np.ones_like(x)))

def design_matrix_quadratic(x):
    return np.column_stack((x**2, x, np.ones_like(x)))

def design_beliebige_polynom_matrix(x, p): # p = polynomgrad 
    return np.column_stack([x**k for k in range(p, -1, -1)]) # -1 wird nicht erreicht --> geht bis 1  

x = np.array([1, 2, 3, 4])
y = np.array([1.5, 2.5, 3, 4.5])

lsg = np.array([])

#(a) 
# A = design_matrix_linear(x)
A = design_beliebige_polynom_matrix(x, p) # Aufpassen, wenn man reg testen will muss hier design_belibige_polynom(x) stehen !!
print(f"Dim(A) = {A.shape}\nA = \n{A}")

#(b)

ATA = A.T @ A # A transponiert mal A 
ATy = A.T @ y # A transponiert mal y (vektor y) 
'''
print(f"Dim(ATA) = {ATA.shape}\nATA = \n{ATA}")
print(f"Dim(ATy) = {ATy.shape}\nATy = \n{ATy}") '''

# spagetticode unbedingt später beheben!!! 
def regression_normalgleichung(x, y, p): # x muss nicht übergeben werden, da die funktion design_poly... sich darum kümmert 
    b = y.T # transponiert y 
    lsg = np.array([])
    A = design_beliebige_polynom_matrix(x, p)
    ATA = A.T @ A
    print(f"Dim(ATA) = {ATA.shape}\nATA = \n{ATA}")
    ATy = A.T @ y 
    print(f"Dim(ATy) = {ATy.shape}\nATy = \n{ATy}")
    lsg = np.linalg.inv(ATA) @ A.T @ b
    return lsg


lsg = regression_normalgleichung(x, y, p)
print(f"Regressionsparameter des Polynoms {p}.grades: \n{lsg}")
print("Test")
#(c) 
sol = np.linalg.inv(ATA) @ ATy
print(f"[m n] = {sol}")

#(e)
sol_poly = np.polyfit(x, y, deg=4)
sol_lest_squares = np.linalg.lstsq(A, y)
print(f"Sol_Polyfit [m n] = {sol_poly}") # zum testen ob p = 2 und polyfit die gleichen Werte ausgeben 
print(f"[m n] = {sol_lest_squares[0]}") # [0] da np.linalg.lstsq() mehrere Werte zurückgibt, wir interessieren uns nur für die Koeffizienten

#(d)
ATr = A.T @ (A@sol - y) #keine lösung aber fehler soll senkrecht zu A liegen Lösung gibt es nicht, da nicht alle Punkte auf der geraden liegen 
print(f"ATr = {ATr}")

plt.plot(x, y, '+')
plt.show()

