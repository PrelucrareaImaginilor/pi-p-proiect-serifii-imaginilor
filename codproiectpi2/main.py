import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pytesseract
import time
import pyttsx3
import os
os.environ['TESSDATA_PREFIX'] = r"D:\Cod proicet PI\Tesseract\tessdata"


# Setează calea către Tesseract
pytesseract.pytesseract.tesseract_cmd = r'D:\\Cod proicet PI\\Tesseract\\tesseract.exe'

#Configurare asistent vocal
engine = pyttsx3.init()

# Funcție pentru redare vocală a textului
def vocalize(text):
    engine.say(text)
    engine.runAndWait()

# Suprascriem funcția print pentru a include redarea vocală
def print_and_speak(*args, **kwargs):
    text = " ".join(map(str, args))
    print(text, **kwargs)  # Afișează în consolă
    vocalize(text)  # Redă vocal textul

# Funcție pentru a încărca imaginea și a o transforma în gri
def load_and_preprocess_image(image_path):
    image = Image.open(image_path).convert('L')  # Convertim imaginea la gri
    image = np.array(image)
    return image


# Funcție pentru a aplica un filtru de detecție a marginilor (precum Sobel)
def detect_edges(image):
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    grad_x = convolve(image, kernel_x)
    grad_y = convolve(image, kernel_y)

    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
    return magnitude


# Funcție de convoluție pentru aplicarea filtrului
def convolve(image, kernel):
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)

    output = np.zeros_like(image)
    for i in range(image_height):
        for j in range(image_width):
            region = padded_image[i:i + kernel_height, j:j + kernel_width]
            output[i, j] = np.sum(region * kernel)

    return output


# Funcție pentru a identifica și extrage contururi fără recursivitate
def find_contours(edge_image):
    contours = []
    visited = np.zeros_like(edge_image, dtype=bool)

    def dfs_iterative(x, y):
        stack = [(x, y)]
        contour = []
        while stack:
            cx, cy = stack.pop()
            if cx < 0 or cy < 0 or cx >= edge_image.shape[0] or cy >= edge_image.shape[1]:
                continue
            if visited[cx, cy] or edge_image[cx, cy] == 0:
                continue
            visited[cx, cy] = True
            contour.append((cx, cy))
            stack.extend([(cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)])
        return contour

    for i in range(edge_image.shape[0]):
        for j in range(edge_image.shape[1]):
            if edge_image[i, j] > 0 and not visited[i, j]:
                contour = dfs_iterative(i, j)
                contours.append(contour)
    return contours


# Funcție de a desena contururile pe imagine
def draw_contours(image, contours):
    image_copy = image.copy()
    for contour in contours:
        for point in contour:
            image_copy[point[0], point[1]] = 255  # Sau o culoare pentru a marca contururile
    return image_copy


# Funcție pentru extragerea și afișarea textului folosind OCR
def extract_text_with_ocr(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image, lang = 'ron')
    print("Text extras din imagine:")
    print_and_speak(text)


# Cod principal de execuție
if __name__ == "__main__":
    start_time = time.time()
    image_path7 = "D://cod proicet PI//Ana.png"
    #image_path7 = "D://cod proicet PI//Diagrama.PNG"

    # 1. Încarcă și preprocesează imaginea
    image = load_and_preprocess_image(image_path7)
    plt.imshow(image, cmap='gray')
    plt.title("Imagine preprocesată (Gri)")
    plt.show()

    # 2. Detectează marginile
    edge_image = detect_edges(image)
    plt.imshow(edge_image, cmap='gray')
    plt.title("Margini detectate (Sobel)")
    plt.show()

    # 3. Găsește contururile
    contours = find_contours(edge_image)
    contour_image = np.zeros_like(image)
    for contour in contours:
        for point in contour:
            contour_image[point[0], point[1]] = 255
    plt.imshow(contour_image, cmap='gray')
    plt.title("Contururi detectate")
    plt.show()

    # 4. Desenează contururile pe imagine
    final_image = draw_contours(image, contours)
    plt.imshow(final_image, cmap='gray')
    plt.title("Imagine finală cu contururi desenate")
    plt.show()

    # 5. Extragere și afișare text OCR
    print("Se extrage textul folosind OCR.")
    extract_text_with_ocr(image_path7)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Timpul de execuție: {execution_time:.4f} secunde")