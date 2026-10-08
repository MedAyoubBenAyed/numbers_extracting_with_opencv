# test.py
#   python test.py                    # teste les images de images/
#   python test.py images_camera      # teste les photos de la caméra
import glob
import os
import sys

import cv2

from lire_chiffre import charger_templates, lire_chiffre

templates = charger_templates()
if not all(templates.values()):
    sys.exit("Aucun template : lancez d'abord  python creer_templates.py")

# images déjà utilisées comme templates : on ne les teste pas
utilises = set()
if os.path.exists("templates/utilises.txt"):
    utilises = set(open("templates/utilises.txt").read().split())

SOURCE = sys.argv[1] if len(sys.argv) > 1 else "images"
classes = ("1", "2")
confusion = {a: {b: 0 for b in classes + ("rien",)} for a in classes}
erreurs = []

for attendu in classes:
    for chemin in sorted(glob.glob(f"{SOURCE}/{attendu}/*.png")):
        nom = os.path.basename(chemin)
        if nom in utilises:
            continue
        img = cv2.imread(chemin)
        pred, scores = lire_chiffre(img, templates)
        confusion[attendu][pred if pred else "rien"] += 1
        if pred != attendu:
            sc = {k: round(v, 2) for k, v in scores.items()}
            erreurs.append(f"{nom:15s} attendu={attendu} prédit={pred} scores={sc}")

total = sum(sum(l.values()) for l in confusion.values())
bons = sum(confusion[c][c] for c in classes)

if erreurs:
    print("Erreurs :")
    print("\n".join(erreurs))

print(f"\nTemplates par classe : { {k: len(v) for k, v in templates.items()} }")
print(f"Images testées : {total} (templates exclus)")
print("\nMatrice de confusion (lignes = vrai, colonnes = prédit)")
print(f"{'':8s}" + "".join(f"{b:>8s}" for b in classes + ("rien",)))
for a in classes:
    print(f"{'vrai ' + a:8s}" + "".join(f"{confusion[a][b]:>8d}" for b in classes + ("rien",)))
print(f"\nPrécision : {bons}/{total} = {100 * bons / total:.1f} %")
