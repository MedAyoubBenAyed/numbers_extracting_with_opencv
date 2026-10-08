# test_camera.py : lire le chiffre 1 ou 2 avec la caméra
#
#   python test_camera.py --apercu          # photo avec grille pour choisir la ROI
#   python test_camera.py                   # Entrée = une lecture, q = quitter
#   python test_camera.py --auto            # une lecture par seconde (Ctrl+C pour finir)
#   python test_camera.py --attendu 2       # compte les bonnes réponses (chiffre 2 devant la caméra)
#   python test_camera.py --roi 500,200,300,300
#   python test_camera.py --image photo.png # teste sans caméra
#   python test_camera.py --webcam          # force la webcam USB (test sur PC)
#   python test_camera.py --afficher        # ouvre des fenêtres (écran ou VNC requis)
import argparse
import os
import time

import cv2
import numpy as np

import config
from lire_chiffre import charger_templates, extraire_chiffre, lire_chiffre_vote

os.makedirs("resultats", exist_ok=True)


def parse_roi(texte):
    x, y, w, h = (int(v) for v in texte.split(","))
    return (x, y, w, h)


def grille(img, pas=100):
    """Dessine une grille numérotée pour lire les coordonnées de la ROI."""
    out = img.copy()
    for x in range(0, out.shape[1], pas):
        cv2.line(out, (x, 0), (x, out.shape[0]), (0, 255, 255), 1)
        cv2.putText(out, str(x), (x + 3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    for y in range(0, out.shape[0], pas):
        cv2.line(out, (0, y), (out.shape[1], y), (0, 255, 255), 1)
        cv2.putText(out, str(y), (3, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return out


def annoter(img, roi, pred, details, ms):
    """Image de résultat : ROI en vert, prédiction, et (à droite) le chiffre
    tel que l'algorithme l'a extrait."""
    out = img.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    echelle = max(out.shape[1] / 640, 0.5)
    if roi:
        x, y, w, h = roi
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(out, f"lu = {pred} ({ms:.0f} ms)", (8, int(30 * echelle)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8 * echelle, (0, 0, 255), 2)

    # panneau de droite : ce que l'algorithme a réellement extrait
    H = out.shape[0]
    panneau = np.zeros((H, 120, 3), np.uint8)
    chiffre, _ = extraire_chiffre(img, roi)
    if chiffre is not None:
        hv = min(120, max(H - 20, 30))
        wv = int(hv * 2 / 3)
        v = cv2.cvtColor(cv2.resize(chiffre, (wv, hv)), cv2.COLOR_GRAY2BGR)
        panneau[10:10 + hv, 10:10 + wv] = v
    return np.hstack([out, panneau])


def lire_une_fois(cam, templates, roi, args, compteur):
    t0 = time.time()
    images = [cam.prendre_image() for _ in range(config.NB_VOTES)] if cam else [args.img]
    pred, details = lire_chiffre_vote(images, templates, roi, config.SEUIL_SCORE)
    ms = (time.time() - t0) * 1000

    votes = [p for p, _ in details]
    moy = {k: round(float(np.mean([s[k] for _, s in details])), 2) for k in templates} if details else {}
    print(f"  -> lu = {pred}   votes = {votes}   scores moyens = {moy}   ({ms:.0f} ms)")

    if args.attendu:
        compteur["total"] += 1
        compteur["ok"] += (pred == args.attendu)
        print(f"     {'OK' if pred == args.attendu else 'ERREUR'}   "
              f"précision : {compteur['ok']}/{compteur['total']}")

    nom = time.strftime("%H%M%S")
    res = annoter(images[-1], roi, pred, details, ms)
    cv2.imwrite(f"resultats/{nom}_lu{pred}.png", res)
    if args.afficher:
        cv2.imshow("resultat", res)
        cv2.waitKey(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apercu", action="store_true", help="photo avec grille, puis quitter")
    ap.add_argument("--auto", action="store_true", help="une lecture par seconde")
    ap.add_argument("--attendu", choices=["1", "2"], help="chiffre réellement présenté")
    ap.add_argument("--roi", type=parse_roi, help="x,y,w,h")
    ap.add_argument("--image", help="tester sur un fichier au lieu de la caméra")
    ap.add_argument("--webcam", action="store_true", help="utiliser la webcam USB")
    ap.add_argument("--afficher", action="store_true", help="ouvrir des fenêtres")
    args = ap.parse_args()

    roi = args.roi or config.ROI

    cam = None
    if args.image:
        args.img = cv2.imread(args.image)
        if args.img is None:
            raise SystemExit(f"Image introuvable : {args.image}")
    else:
        from camera import Camera
        cam = Camera(source="webcam" if args.webcam else "auto")

    try:
        # ---- mode aperçu ----
        if args.apercu:
            img = cam.prendre_image() if cam else args.img
            cv2.imwrite("apercu.png", img)
            cv2.imwrite("apercu_grille.png", grille(img))
            print(f"Image {img.shape[1]}x{img.shape[0]} enregistrée : apercu.png et apercu_grille.png")
            print("Ouvrez apercu_grille.png, repérez le chiffre, puis notez la ROI x,y,largeur,hauteur")
            print("dans config.py (ROI = (x, y, w, h)) ou avec --roi x,y,w,h")
            return

        templates = charger_templates()
        if not all(templates.values()):
            raise SystemExit("Aucun template. Faites d'abord :\n"
                             "  python capturer_photos.py 1 20\n"
                             "  python capturer_photos.py 2 20\n"
                             "  python creer_templates.py 10 images_camera")
        print(f"Templates : { {k: len(v) for k, v in templates.items()} }   ROI = {roi}")

        compteur = {"ok": 0, "total": 0}

        # ---- test sur un fichier ----
        if args.image:
            lire_une_fois(None, templates, roi, args, compteur)
            return

        # ---- boucle caméra ----
        if args.auto:
            print("Mode automatique, Ctrl+C pour arrêter.")
            while True:
                lire_une_fois(cam, templates, roi, args, compteur)
                time.sleep(1.0)
        else:
            while True:
                rep = input("Entrée = lire, q = quitter : ").strip().lower()
                if rep == "q":
                    break
                lire_une_fois(cam, templates, roi, args, compteur)
    except KeyboardInterrupt:
        pass
    finally:
        if cam:
            cam.fermer()
        if args.attendu and compteur["total"]:
            print(f"\nBilan : {compteur['ok']}/{compteur['total']} "
                  f"= {100 * compteur['ok'] / compteur['total']:.0f} %")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
