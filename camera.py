# camera.py : accès à la caméra (Pi Camera via picamera2, sinon webcam USB)
import time

import cv2

import config


class Camera:
    """Utilisation :
        with Camera() as cam:
            img = cam.prendre_image()      # image BGR (format OpenCV)
    source : "auto" (Pi Camera, sinon webcam), "picamera" ou "webcam".
    """

    def __init__(self, source="auto", resolution=None, verrouiller=None):
        self.resolution = resolution or config.RESOLUTION
        self.verrouiller = config.VERROUILLER_EXPOSITION if verrouiller is None else verrouiller
        self.mode = None
        self.cam = None
        self.cap = None

        if source in ("auto", "picamera"):
            try:
                self._ouvrir_picamera()
            except Exception as e:
                if source == "picamera":
                    raise
                print(f"[camera] Pi Camera indisponible ({e}) -> essai webcam")
        if self.mode is None:
            self._ouvrir_webcam()
        print(f"[camera] mode = {self.mode}, résolution = {self.resolution}")

    # -- Pi Camera ---------------------------------------------------------
    def _ouvrir_picamera(self):
        from picamera2 import Picamera2

        self.cam = Picamera2()
        cfg = self.cam.create_preview_configuration(
            main={"size": self.resolution, "format": "RGB888"})
        self.cam.configure(cfg)
        self.cam.start()
        time.sleep(1.5)  # laisse l'auto-exposition se stabiliser

        if self.verrouiller:
            # On fige les valeurs que la caméra a choisies elle-même :
            # plus de variation de luminosité ou de couleur entre deux photos.
            m = self.cam.capture_metadata()
            self.cam.set_controls({
                "AeEnable": False,
                "ExposureTime": m["ExposureTime"],
                "AnalogueGain": m["AnalogueGain"],
                "AwbEnable": False,
                "ColourGains": m["ColourGains"],
            })
            time.sleep(0.3)
        self.mode = "picamera2"

    # -- Webcam USB (pour tester sur PC) -----------------------------------
    def _ouvrir_webcam(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Aucune caméra trouvée (Pi Camera ni webcam).")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        time.sleep(0.5)
        self.mode = "webcam"

    # -- Capture ------------------------------------------------------------
    def prendre_image(self):
        if self.mode == "picamera2":
            # Le format "RGB888" de picamera2 est déjà dans l'ordre BGR en mémoire,
            # donc directement utilisable par OpenCV.
            img = self.cam.capture_array()
        else:
            for _ in range(2):  # vide le tampon pour avoir une image récente
                self.cap.grab()
            ok, img = self.cap.read()
            if not ok:
                raise RuntimeError("Lecture webcam impossible.")
        if img.ndim == 3 and img.shape[2] == 4:  # enlève un éventuel canal alpha
            img = img[:, :, :3]
        if config.INVERSER_RB:
            img = img[:, :, ::-1].copy()
        return img

    def fermer(self):
        if self.cam is not None:
            self.cam.stop()
            self.cam.close()
        if self.cap is not None:
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.fermer()
