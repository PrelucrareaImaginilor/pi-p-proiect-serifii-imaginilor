import cv2
import numpy as np
import pytesseract
from PIL import Image
import warnings
from pytesseract import Output
import winsdk.windows.media.speechsynthesis as speech
import winsdk.windows.storage.streams as streams
import simpleaudio as sa

from functii import (
    preprocess_dark_image,
    preprocess_blurred_image,
    preprocess_normal_image,
    preprocess_fundal_image
)

warnings.filterwarnings('ignore')

# =========================
# configurare
# =========================
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
IMAGE_PATH = r"Images\InsulaMisterioasa.png"


# =========================
# clasa pentru sintetizatorul vocal
# =========================
class TextToSpeech:
    def __init__(self):
        # obtine lista tuturor vocilor instalate în Windows
        self.voices = speech.SpeechSynthesizer.all_voices
        self.synth = speech.SpeechSynthesizer()
        self.current_voice = None

    def set_voice_by_lang(self, lang_code: str):
        chosen = False
        for v in self.voices:
            # verifica limba
            if lang_code == "ron" and v.language.lower().startswith("ro"):
                self.synth.voice = v
                self.current_voice = v
                chosen = True
                break
            elif lang_code == "eng" and v.language.lower().startswith("en"):
                self.synth.voice = v
                self.current_voice = v
                chosen = True
                break

        if not chosen:
            print(f"[Avertisment] Nu am gasit voce pentru {lang_code}, folosesc vocea implicita.")
            self.current_voice = self.synth.voice

    async def speak(self, text: str):
        if text.strip():
            print(f"\n Citire vocala ({self.current_voice.display_name}): {text}")
            stream = await self.synth.synthesize_text_to_stream_async(text)

            # converteste streamul WinRT intr-un buffer de bytes
            reader = streams.DataReader(stream)
            buffer = bytearray(stream.size)
            reader.load_async(stream.size)
            reader.read_bytes(buffer)

            # reda cu simpleaudio
            play_obj = sa.play_buffer(buffer, 1, 2, 16000)
            play_obj.wait_done()
        else:
            print("Nu există text de citit")

# =========================
# clasa pentru clasificarea imaginii
# =========================
class ImageTextAnalyzer:
    def __init__(self):
        pass

    def extract_text(self, image):
        try:
            return pytesseract.image_to_string(image, lang='ron').strip()
        except:
            return ""

    def detect_text_presence(self, image_path):
        try:
            text_original = self.extract_text(Image.open(image_path))
            return len(text_original) > 30
        except:
            return False

    def detect_blur(self, image_path, threshold=100):
        image = cv2.imread(image_path)
        if image is None:
            return False
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        height, width = gray.shape
        adjusted_threshold = threshold * (1000000 / (height * width))
        return laplacian_var < adjusted_threshold

    def detect_dark_image(self, image_path, brightness_threshold=50):
        image = cv2.imread(image_path)
        if image is None:
            return False
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        brightness = np.mean(hsv[:, :, 2])
        return brightness < brightness_threshold

    def detect_background_with_text(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return False

        h_img, w_img = img.shape[:2]
        data = pytesseract.image_to_data(
            img, config="--oem 3 --psm 6", output_type=Output.DICT
        )

        total_area = 0
        n_valid_boxes = 0
        for i in range(len(data["text"])):
            txt = (data["text"][i] or "").strip()
            try:
                conf = float(data["conf"][i])
            except:
                continue
            if conf > 40 and len(txt) > 1:
                w, h = int(data["width"][i]), int(data["height"][i])
                if w > 0 and h > 0:
                    total_area += w * h
                    n_valid_boxes += 1

        if n_valid_boxes == 0:
            return False

        coverage_ratio = total_area / float(w_img * h_img)

        print(f"[DEBUG] Boxuri valide: {n_valid_boxes}, Coverage ratio: {coverage_ratio:.2f}")

        if n_valid_boxes <= 4:  # foarte putine zone de text -> text mare, gen poster
            return True
        elif coverage_ratio > 0.15 and n_valid_boxes < 15:
            return True
        else:
            return False

    def analyze_image(self, image_path):
        results = {
            'text_present': self.detect_text_presence(image_path),
            'blurred': self.detect_blur(image_path),
            'dark': self.detect_dark_image(image_path),
            'background': self.detect_background_with_text(image_path),
            'classification': 'imagine normala'
        }

        if results['blurred']:
            results['classification'] = 'imagine blurata'
        elif results['dark']:
            results['classification'] = 'imagine intunecata'
        elif results['background']:
            results['classification'] = 'imagine de fundal'
        else:
            results['classification'] = 'imagine normala'

        return results

# =========================
# clasa procesare
# =========================
class ImageTextProcessor:
    def __init__(self, lang_code="ron"):    # 'eng' sau 'ron'
        self.analyzer = ImageTextAnalyzer()
        self.lang_code = lang_code
        self.tts = TextToSpeech()
        self.tts.set_voice_by_lang(self.lang_code)

    def adaptive_preprocessing(self, image_path, classification):
        if "imagine intunecata" in classification:
            processed = preprocess_dark_image(image_path)
        elif "imagine blurata" in classification:
            processed = preprocess_blurred_image(image_path)
        elif "imagine de fundal" in classification:
            processed = preprocess_fundal_image(image_path)
        else:
            processed = preprocess_normal_image(image_path)
        return processed

    def process(self, image_path):
        print("=" * 50)
        print("=== ANALIZA IMAGINE ===")
        print("=" * 50)

        # Pas 1: clasificare
        results = self.analyzer.analyze_image(image_path)
        classification = results['classification']
        print(f"Clasificare: {classification}")

        # Pas 2: procesare imagine
        final_img = self.adaptive_preprocessing(image_path, classification)

        # Pas 3: extragere text
        print("=" * 50)
        print("=== EXTRAGERE TEXT ===")
        print("=" * 50)

        # setam limba OCR după limba dorita
        text = pytesseract.image_to_string(final_img, lang=self.lang_code, config='--oem 3 --psm 6')
        print(text)

        # Pas 4: citire vocala (Winsdk)
        print("=" * 50)
        print("=== CITIRE VOCALA ===")
        print("=" * 50)

        if text.strip():  # daca s-a recunoscut text
            import asyncio
            asyncio.run(self.tts.speak(text))
        else:
            print("Nu s-a putut recunoaste niciun text")
        return text


# =========================
# main
# =========================
def main():
    processor = ImageTextProcessor()
    processor.process(IMAGE_PATH)

if __name__ == "__main__":
    main()
