"""
Метрики качества водяных знаков.

Метрики незаметности (host vs watermarked):
    PSNR  — Peak Signal-to-Noise Ratio, dB. Больше = лучше. >40 dB — отлично.
    SSIM  — Structural Similarity Index, [0,1]. Ближе к 1 = лучше.

Метрики извлечения (watermark vs extracted):
    NCC   — Normalized Cross-Correlation, [-1,1]. Ближе к 1 = точнее извлечение.
    BER   — Bit Error Rate, [0,1]. Ближе к 0 = меньше ошибок.
"""

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as _ssim
from skimage.metrics import peak_signal_noise_ratio as _psnr


def _to_gray(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("L"), dtype=np.float64)


# ------------------------------------------------------------------ #
#  Незаметность                                                        #
# ------------------------------------------------------------------ #

def psnr(original: Image.Image, watermarked: Image.Image) -> float:
    """Peak Signal-to-Noise Ratio в дБ."""
    a = _to_gray(original)
    b = _to_gray(watermarked)
    # Если изображения идентичны — возвращаем inf
    if np.allclose(a, b):
        return float("inf")
    return float(_psnr(a, b, data_range=255.0))


def ssim(original: Image.Image, watermarked: Image.Image) -> float:
    """Structural Similarity Index."""
    a = _to_gray(original).astype(np.uint8)
    b = _to_gray(watermarked).astype(np.uint8)
    return float(_ssim(a, b, data_range=255))


# ------------------------------------------------------------------ #
#  Качество извлечения                                                 #
# ------------------------------------------------------------------ #

def ncc(watermark: Image.Image, extracted: Image.Image) -> float:
    """
    Normalized Cross-Correlation между оригинальным и извлечённым знаком.
    NCC = 1 → идеальное совпадение.
    """
    a = _to_gray(watermark).flatten()
    b = _to_gray(extracted).flatten()

    # Выровнять размеры на случай небольших расхождений
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]

    a -= a.mean()
    b -= b.mean()
    denom = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2)) + 1e-10
    return float(np.sum(a * b) / denom)


def ber(watermark: Image.Image, extracted: Image.Image, threshold: int = 128) -> float:
    """
    Bit Error Rate — доля пикселей с неверным значением после бинаризации.
    BER = 0 → ошибок нет.
    """
    a = (_to_gray(watermark).flatten() > threshold)
    b = (_to_gray(extracted).flatten() > threshold)
    n = min(len(a), len(b))
    return float(np.sum(a[:n] != b[:n]) / n)


# ------------------------------------------------------------------ #
#  Сводная таблица                                                     #
# ------------------------------------------------------------------ #

def all_metrics(
    host: Image.Image,
    watermarked: Image.Image,
    watermark: Image.Image | None,
    extracted: Image.Image | None,
) -> dict[str, str]:
    """
    Возвращает словарь {метрика: значение_строкой} для отображения в GUI.
    watermark и extracted могут быть None (тогда метрики извлечения пропускаются).
    """
    result: dict[str, str] = {}

    try:
        result["PSNR (dB)"] = f"{psnr(host, watermarked):.2f}"
    except Exception:
        result["PSNR (dB)"] = "—"

    try:
        result["SSIM"] = f"{ssim(host, watermarked):.4f}"
    except Exception:
        result["SSIM"] = "—"

    if watermark is not None and extracted is not None:
        try:
            result["NCC"] = f"{ncc(watermark, extracted):.4f}"
        except Exception:
            result["NCC"] = "—"

        try:
            result["BER"] = f"{ber(watermark, extracted):.4f}"
        except Exception:
            result["BER"] = "—"

    return result
