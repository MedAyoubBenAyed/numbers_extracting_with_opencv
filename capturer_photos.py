# capturer_photos.py : prendre des photos d'un chiffre avec la caméra
#
#   python capturer_photos.py 1 20      # 20 photos du chiffre 1  -> images_camera/1/
#   python capturer_photos.py 2 20      # 20 photos du chiffre 2  -> images_camera/2/
#   python capturer_photos.py 1 20 --delai 2   # une photo toutes les 2 s (sans appuyer sur Entrée)
#
# Entre deux photos, changez un peu la distance, l'angle ou l'éclairage :
# c'est ce qui rend la reconnaissance robuste. Si ROI est défini dans config.py,
# seule cette zone est enregistrée.
import argparse
import os
import time

import cv2

import config
from camera import Camera

ap = argparse.ArgumentParser()
ap.add_argument("chiffre", choices=["1", "2"])
ap.add_argument("nombre", type=int, nargs="?", default=20)
ap.add_argument("--delai", type=float, default=0, help="secondes entre photos (0 = Entrée à chaque photo)")
ap.add_argument("--webcam", action="store_true")
ap.add_argument("--dossier", default="images_camera")
args = ap.parse_args()

dossier = os.path.join(args.dossier, args.chiffre)
os.makedirs(dossier, exist_ok=True)
debut = len([f for f in os.listdir(dossier) if f.endswith(".png")])  # continue la numérotation

with Camera(source="webcam" if args.webcam else "auto") as cam:
    for i in range(args.nombre):
        if args.delai > 0:
            print(f"Photo {i + 1}/{args.nombre} dans {args.delai:.0f} s ... (bougez un peu)")
            time.sleep(args.delai)
        else:
            input(f"Photo {i + 1}/{args.nombre} : placez le chiffre {args.chiffre}, puis Entrée ")
        img = cam.prendre_image()
        if config.ROI:
            x, y, w, h = config.ROI
            img = img[y:y + h, x:x + w]
        nom = os.path.join(dossier, f"{args.chiffre}_{debut + i + 1:03d}.png")
        cv2.imwrite(nom, img)
        print("  enregistrée :", nom)

print(f"\nTerminé : {args.nombre} photos dans {dossier}/")
