# config.py : tous les réglages au même endroit

RESOLUTION = (1280, 720)      # (largeur, hauteur) de la caméra

# Zone de l'image où se trouve le chiffre : (x, y, largeur, hauteur) en pixels.
# None = toute l'image. À régler avec :  python test_camera.py --apercu
ROI = None

VERROUILLER_EXPOSITION = True  # fige exposition / gain / balance des blancs
INVERSER_RB = False            # mettre True si le rouge apparaît bleu

NB_VOTES = 3                   # nombre de photos par lecture (vote majoritaire)
SEUIL_SCORE = 0.30             # score minimal pour accepter une lecture (à calibrer)
