# creer_templates.py
#   python creer_templates.py                       # 10 templates depuis images/
#   python creer_templates.py 10 images_camera     # depuis les photos de la caméra
# Choisit N images par classe dans images/<classe>/ et enregistre leur chiffre
# normalisé dans templates/<classe>/. La liste des images utilisées est notée
# dans templates/utilises.txt pour que test.py ne les teste pas.
import glob
import os
import random
import shutil
import sys

import cv2

from lire_chiffre import extraire_chiffre

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10          # templates par classe
SOURCE = sys.argv[2] if len(sys.argv) > 2 else "images"    # dossier des photos
random.seed(0)

if os.path.isdir("templates"):
    shutil.rmtree("templates")  # repartir de zéro

utilises = []
for nom in ("1", "2"):
    os.makedirs(f"templates/{nom}", exist_ok=True)
    fichiers = sorted(glob.glob(f"{SOURCE}/{nom}/*.png"))
    random.shuffle(fichiers)

    crees = 0
    for chemin in fichiers:
        if crees == N:
            break
        chiffre, _ = extraire_chiffre(cv2.imread(chemin))
        if chiffre is None:
            continue  # image inutilisable, on passe à la suivante
        cv2.imwrite(f"templates/{nom}/{crees + 1:02d}.png", chiffre)
        utilises.append(os.path.basename(chemin))
        crees += 1
    print(f"classe {nom} : {crees} templates créés")

with open("templates/utilises.txt", "w") as f:
    f.write("\n".join(utilises))
