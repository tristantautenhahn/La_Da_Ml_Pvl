import numpy as np 
from matplotlib import pyplot as plt

p = 8 # polynomgrad

def design_matrix_linear(x):
    return np.column_stack((x, np.ones_like(x)))

def design_matrix_quadratic(x):
    return np.column_stack((x**2, x, np.ones_like(x)))

def design_beliebige_polynom_matrix(x, p): # p = polynomgrad 
    return np.column_stack([x**k for k in range(p, -1, -1)]) # -1 wird nicht erreicht --> geht bis 1  

x = np.array([-4, -3, -2, -1, 0, 2 , 4, 3])
y = np.array([18.7, 9.9, 4.3, 1.8, 1.5, 2.0, 4.9, 10.4])

lsg = np.array([])

#(a) 
# A = design_matrix_linear(x)
A = design_beliebige_polynom_matrix(x, p) 
print(f"Dim(A) = {A.shape}\nA = \n{A}")

#(b)

ATA = A.T @ A # A transponiert mal A 
ATy = A.T @ y # A transponiert mal y (vektor y) 
'''
print(f"Dim(ATA) = {ATA.shape}\nATA = \n{ATA}")
print(f"Dim(ATy) = {ATy.shape}\nATy = \n{ATy}") '''


def regression_normalgleichung(x, y, p): 
    lsg = np.array([])
    A = design_beliebige_polynom_matrix(x, p)
    ATA = A.T @ A
    print(f"ATA = \n{ATA}")
    ATy = A.T @ y 
    print(f"ATy = \n{ATy}")
    lsg = np.linalg.inv(ATA) @ A.T @ y
    # Überprüfung der Ortogonalitätsbedingung (mit 10^-6 als Toleranz)
    fehler = A.T @ (A @ lsg - y)
    if np.any(fehler > 1e-6): # weil das ein Vektor ist und alle einträge kleiner sein müssen 
        print(f"Warnung: Die Ortogonalitätsbedingung ist nicht erfüllt. \n Fehler: {fehler}")
    else :
        print(f"Die Ortogonalitätsbedingung ist erfüllt. \n Fehler: {fehler}")
    return lsg


lsg = regression_normalgleichung(x, y, p)
print(f"Regressionsparameter des Polynoms {p}.grades: \n{lsg}")
#(c) 
sol = np.linalg.inv(ATA) @ ATy
print(f"[m n] = {sol}")

#(e)
sol_poly = np.polyfit(x, y, deg=p)
sol_lest_squares = np.linalg.lstsq(A, y)
print(f"Sol_Polyfit [m n] = {sol_poly}") # zum testen ob p = 2 und polyfit die gleichen Werte ausgeben 
print(f"[m n] = {sol_lest_squares[0]}") # [0] da np.linalg.lstsq() mehrere Werte zurückgibt, wir interessieren uns nur für die Koeffizienten

#(d)
ATr = A.T @ (A@sol - y) #keine lösung aber fehler soll senkrecht zu A liegen Lösung gibt es nicht, da nicht alle Punkte auf der geraden liegen 
print(f"ATr = {ATr}")

# plt.plot(x, y, '+')
# plt.show()


## Auswertung: 
# bei den in der Übung gegebenen Datensätzen x = np.array([1, 2, 3, 4]) und y = np.array([1.5, 2.5, 3, 4.5]) erhält man 
# bei einem Polynomgrad von 2 die Koeffizienten [0.125 0.325 1.125] und bei Polyfit exakt die gleichen der Fehler liegt bei [9.14823772e-13 2.60236277e-13 7.77156117e-14].
# bei einem Polynomgrad von 3 die Koeffizienten [ 0.25 -1.75  4.5  -1.5 ] und bei Polyfit exakt die gleichen der Fehler liegt bei [5.73540326e-10 1.52373225e-10 4.15285584e-11 1.17923449e-11].
# bei einem Polynomgrad von 4 die Koeffizienten [ 0.0831604  -0.31079102 -0.46679688  3.45117188 -1.01025391] und bei Polyfit [ 0.03805017 -0.13050168 -0.41824411  2.59749158 -0.58679596] das liegt daran, dass der polynomgrad jetzt höher ist als die Anzahl an Datenpunkten, was dazu führt, dass das Modell instabiel wird. Der fehler liegt bei: [614.95245361 158.68493652  41.68951416  11.34143066   3.32647705] was sehr sehr hoch ist. 

# jetzt die Test für die Datenpunkte x = np.array([-4, -3, -2, -1, 0, 2 , 4, 3]) y = np.array([18.7, 9.9, 4.3, 1.8, 1.5, 2.0, 4.9, 10.4]) erhält man für 
# den Polynomgrad von 3 die Koeffizienten [-0.14960727  0.69334194  0.81394624  1.65714559] und bei Polyfit exakt das gleiche der Fehler liegt bei [-1.96109795e-12  1.52766688e-13 -1.61648472e-13  7.77156117e-15] 
# den Polynomgrad von 7 die Koeffizienten [-0.00339782 -0.01292163  0.06603671  0.27770337 -0.27490079 -0.49156746 -0.31452381  1.5] und bei Polyfit exakt das gleiche der Fehler liegt bei [ 5.90026623e-07 -1.29092848e-09  3.68375555e-08 -6.87807589e-11 2.29633224e-09 -3.37174733e-12  1.41815670e-10 -5.61772850e-14]
# bei 8 sind die Polynome Instabiel was zu Unterschieden Zwischen den Koeffizienten und dem Polyfit führt auch wird der Fehler sehr groß: [-1.79617442e+06  8.55402118e+04 -1.13706589e+05  5.56142174e+03 -7.27259075e+03  3.71984161e+02 -4.74309025e+02  2.60931032e+01 -3.21864061e+01]