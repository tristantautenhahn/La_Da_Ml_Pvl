import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

 # Bild einlesen
img = np.array(Image.open('artemis_fenster.png')).astype(float)


 # Singulaerwertzerlegung berechnen
 # TODO: Berechnen Sie U, S und VT mit np.linalg.svd
U, S, VT = np.linalg.svd(img)

 # Anzahl der verwendeten Singulaerwerte
k = 90 #128 

 # Auswahl der ersten k Singulaerwerte und Singulaervektoren

 # TODO: Definieren Sie U_k, S_k und VT_k
U_k = U[:, :k] # Alle Zeilen, aber nur die ersten k Spalten von U
S_k = S[:k] # Die ersten k Singulaerwerte
VT_k = VT[:k, :] # Die ersten k Zeilen von VT

 # Rekonstruktion
compressed_img = (U_k * S_k) @ VT_k
compressed_img = np.clip(compressed_img, 0, 255)

 # Kompressionsrate berechnen
m, n = img.shape
original_values = m * n
compressed_values = k * (m + n + 1)

storage_percent = 100 * compressed_values / original_values
compression_factor = original_values / compressed_values

 # Relativen Fehler berechnen
rel_error = np.linalg.norm(img - compressed_img, 'fro') \
            / np.linalg.norm(img, 'fro')

 # Ausgabe
print(f"k = {k}")
print(f"Speicherbedarf: {storage_percent:.1f}% des Originals")
print(f"Kompressionsfaktor: {compression_factor:.2f}x")
print(f"Relativer Fehler: {rel_error:.4f}")

 # Plot erzeugen
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].imshow(img, cmap='gray', vmin=0, vmax=255)
ax[0].set_title(f"Originalbild\n{m} x {n} Pixel")
ax[0].axis('off')

ax[1].imshow(compressed_img, cmap='gray', vmin=0, vmax=255)
ax[1].set_title(
f"Rang-{k}-Approximation\n"
f"{storage_percent:.1f}% Speicher, Fehler {rel_error:.3f}"
)
ax[1].axis('off')

plt.savefig('svd_bildkompression.png', dpi=150, bbox_inches='tight')
plt.show()

# a) Je größer k, desto besser ist die Bildqualität 
# b) ungefähr ab k = 100 
# c) bei 90 18.8% bei 50 10.4% und bei 128 26.7% des speicherbedarfs 
# d) wenn k = diag(img.shape) da alle singularitätswerte benutzt werden 
# e) die großen hellen bzw dunklen Bereiche 
# f) die details wie die Kante des Fensters, oder die Krater des Mondes.  