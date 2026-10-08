# lire_chiffre.py
import glob
import os
from collections import Counter

import cv2
import numpy as np

TAILLE = (40, 60)  # (largeur, hauteur) du chiffre normalisé


def _plus_grand_contour(bw, aire_min=30):
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    return c if cv2.contourArea(c) >= aire_min else None


def extraire_chiffre(img, roi=None):
    """Retourne (chiffre 40x60 blanc sur noir, image binaire).
    Le chiffre vaut None si rien n'est trouvé."""
    if roi is not None:
        x, y, w, h = roi
        img = img[y:y + h, x:x + w]

    # accepte les images couleur (BGR) ou en niveaux de gris
    gris = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gris = cv2.GaussianBlur(gris, (5, 5), 0)
    _, bw = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Le chiffre est toujours la plus petite région : si le blanc occupe
    # plus de la moitié de l'image, la polarité est inversée
    if (bw > 0).mean() > 0.5:
        bw = 255 - bw

    # nettoyer les petits points ; si le chiffre est très fin, l'ouverture
    # l'effacerait : on retombe alors sur l'image non nettoyée
    nettoye = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    c = _plus_grand_contour(nettoye)
    if c is None:
        c = _plus_grand_contour(bw)
        nettoye = bw
    bw = nettoye
    if c is None:
        return None, bw
    x, y, w, h = cv2.boundingRect(c)
    chiffre = cv2.resize(bw[y:y + h, x:x + w], TAILLE)
    return chiffre, bw


def charger_templates(dossier="templates"):
    """Charge templates/1/*.png et templates/2/*.png.
    Retourne {"1": [img, img, ...], "2": [...]}."""
    templates = {}
    for nom in ("1", "2"):
        fichiers = sorted(glob.glob(os.path.join(dossier, nom, "*.png")))
        templates[nom] = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in fichiers]
    return templates


def lire_chiffre(img, templates, roi=None):
    """templates = {"1": [t1, t2, ...], "2": [...]}.
    Retourne (prediction, scores). prediction = None si lecture impossible."""
    chiffre, _ = extraire_chiffre(img, roi)
    if chiffre is None:
        return None, {}
    scores = {}
    for nom, liste in templates.items():
        # meilleur score parmi tous les templates de la classe
        scores[nom] = max(
            float(cv2.matchTemplate(chiffre, t, cv2.TM_CCOEFF_NORMED).max())
            for t in liste
        )
    return max(scores, key=scores.get), scores


def lire_chiffre_vote(images, templates, roi=None, seuil=0.0):
    """Lit plusieurs images et fait un vote majoritaire.
    Retourne (prediction, details) ; details = liste de (prediction, scores)
    pour chaque image acceptée. prediction = None si aucune lecture valide."""
    details = []
    for img in images:
        pred, scores = lire_chiffre(img, templates, roi)
        if pred is not None and scores[pred] >= seuil:
            details.append((pred, scores))
    if not details:
        return None, details

    nb = Counter(p for p, _ in details)
    somme = {k: sum(s[k] for _, s in details) for k in templates}
    # plus de votes d'abord ; en cas d'égalité, meilleur score cumulé
    pred = max(nb, key=lambda k: (nb[k], somme[k]))
    return pred, details
