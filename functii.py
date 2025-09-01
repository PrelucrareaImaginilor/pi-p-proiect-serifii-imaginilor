import cv2
import numpy as np
from matplotlib import pyplot as plt

def remove_noise_and_borders(bin_img, min_ratio=0.0005, max_ratio=0.8):
    nb, output, stats, _ = cv2.connectedComponentsWithStats(bin_img, connectivity=8)
    h, w = bin_img.shape
    total = h * w
    mask = np.zeros_like(bin_img)

    for i in range(1, nb):  # 0 = background
        area = stats[i, cv2.CC_STAT_AREA]
        if min_ratio * total < area < max_ratio * total:
            mask[output == i] = 255

    mask = cv2.bitwise_not(mask)
    return mask

def thin_font(image):
    image = cv2.bitwise_not(image)
    kernel = np.ones((2,2),np.uint8)
    image = cv2.erode(image,kernel,iterations=1)
    return cv2.bitwise_not(image)

def contrast_stretch(image):
    min_val = np.percentile(image, 2)
    max_val = np.percentile(image, 98)
    stretched = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    return stretched

#realizeaza binarizarea adaptiva, alege un prag pt fiecare imagine, nr pixelilor vecini pt calcularea pragului
def adaptive_binarize(image):
    return cv2.adaptiveThreshold(image, 255,
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 21, 10)  #mai multi pixeli devin albi

def light_denoise(image):
    return cv2.GaussianBlur(image, (3, 3), 0)

def thin_text(image):
    image = cv2.bitwise_not(image)
    kernel = np.ones((1, 1), np.uint8)
    thinned = cv2.erode(image, kernel, iterations=1)
    return cv2.bitwise_not(thinned)


def display(img, dpi=80, label=None, position=(10, 30), font_scale=1, color=(255, 0, 0), thickness=2):

    if img is None:
        raise ValueError("Imaginea este None!")

    display_img = img.copy()

    # daca e grayscale, o convertim în BGR pentru a putea pune text colorat
    if len(display_img.shape) == 2:
        display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)

    # adauga textul
    if label:
        cv2.putText(display_img, label, position, cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, thickness, cv2.LINE_AA)

    display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)

    height, width = display_img.shape[:2]
    figsize = width / float(dpi), height / float(dpi)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.imshow(display_img)
    plt.show()
    plt.close(fig)

def preprocess_dark_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Imaginea nu a putut fi încarcata: {image_path}")
    display(img, label="Original")

    # Normalizeaza polaritatea textului
    bin_img = cv2.bitwise_not(img)     #doar inverseaza valorile pixelilor, nu face binarizare
    display(bin_img, label="Inversare polaritate")

    if len(bin_img.shape) == 2:
        gray = bin_img
    else:
        gray = cv2.cvtColor(bin_img, cv2.COLOR_BGR2GRAY)
    display(gray, label="Grayscale")

    if len(np.unique(gray)) > 2:
        stretched = contrast_stretch(gray)
        display(stretched, label="Contrast stretching")
    else:
        stretched = gray
        display(stretched, label="Fara contrast stretching")

    denoised = light_denoise(stretched)
    display(denoised, label="Denoising")

    if len(np.unique(denoised)) > 2:
        adaptive = adaptive_binarize(denoised)
        display(adaptive, label="Adaptive Threshold")
    else:
        adaptive = denoised
        display(adaptive, label="Fara Adaptive Threshold")

    thinned = thin_text(adaptive)
    display(thinned, label="Subtiere text")

    return thinned


def preprocess_blurred_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Imaginea nu a putut fi încarcata: {image_path}")
    display(img, label="Original")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    display(gray, label="Grayscale")

    denoised = cv2.fastNlMeansDenoising(gray, None, h=15, templateWindowSize=7, searchWindowSize=21)
    display(denoised, label="Denoising")

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(denoised)
    gamma = 0.8
    gamma_corrected = np.array(255 * (contrast / 255) ** gamma, dtype='uint8')
    display(gamma_corrected, label="Gamma Correction")

    blurred = cv2.GaussianBlur(gamma_corrected, (9, 9), 10)
    highpass = cv2.addWeighted(gamma_corrected, 1.5, blurred, -0.5, 0)
    display(highpass, label="High-pass Sharpening")

    otsu = cv2.threshold(highpass, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    adaptive = cv2.adaptiveThreshold(highpass, 255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 35, 10)
    contours_otsu, _ = cv2.findContours(otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_adapt, _ = cv2.findContours(adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    binary = otsu if len(contours_otsu) > len(contours_adapt) else adaptive
    display(binary, label="Binarizare finala")

    mask = remove_noise_and_borders(binary, min_ratio=0.0005, max_ratio=0.8)
    display(mask, label="Filtrare componente text")

    mask = remove_noise_and_borders(mask, min_ratio=0.0005, max_ratio=0.8)
    display(mask, label="Filtrare componente text")

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thickened = cv2.dilate(mask, kernel, iterations=1)
    display(thickened, label="Ingrosare text")

    return thickened


def preprocess_normal_image(image_path):
    img = cv2.imread(image_path)
    display(img, label="Original")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    display(gray, label="Grayscale")

    stretched = contrast_stretch(gray)
    display(stretched, label="Contrast stretching")

    denoised = light_denoise(stretched)
    display(denoised, label="Denoising")

    adaptive = adaptive_binarize(denoised)
    display(adaptive, label="Adaptive Threshold")

    thinned = thin_text(adaptive)
    display(thinned, label="Subtiere text")

    return thinned


def preprocess_fundal_image(image_path, min_text_ratio=0.0001, max_text_ratio=0.8, enhance=True):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Imaginea nu a putut fi încarcata: {image_path}")
    display(img, label="Original")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    display(gray, label="Grayscale")

    stretched = contrast_stretch(gray)
    display(stretched, label="Contrast stretching")

    binary = adaptive_binarize(stretched)
    display(binary, label="Adaptive Threshold")


    mask = remove_noise_and_borders(binary, min_ratio=min_text_ratio, max_ratio=max_text_ratio)
    display(mask, label="Filtrare componente text")

    if enhance:
        thinned = thin_font(mask)
        display(thinned, label="Subtiere text")
        processed = thinned
    else:
        processed = mask

    return processed
