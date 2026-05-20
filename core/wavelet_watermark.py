"""
Wavelet Watermark — встраивание и извлечение водяного знака
через дискретное вейвлет-преобразование (DWT).

Принцип:
  1. Применяем многоуровневый DWT к изображению.
  2. Встраиваем водяной знак в коэффициенты выбранной субполосы
     (LL — приближение, LH/HL/HH — детали).
  3. Масштабируем поправку относительно std субполосы для инвариантности.
  4. Обратный DWT → помеченное изображение.
  5. Извлечение: разность коэффициентов помеченного и оригинала.
"""

import numpy as np
import pywt
from PIL import Image


class WaveletWatermark:
    """Водяной знак на основе DWT (PyWavelets)."""

    SUBBANDS = ("LL", "LH", "HL", "HH")
    WAVELETS = ("haar", "db1", "db2", "db4", "sym2", "bior1.3", "coif1")

    def __init__(
        self,
        alpha: float = 0.10,
        wavelet: str = "haar",
        level: int = 2,
        subband: str = "LL",
    ):
        """
        Args:
            alpha:   Сила встраивания (доля от std субполосы).
            wavelet: Семейство вейвлета.
            level:   Глубина разложения (1–5).
            subband: Целевая субполоса: 'LL' | 'LH' | 'HL' | 'HH'.
                     LL  — наиболее устойчиво (низкие частоты).
                     HH  — минимально заметно (диагональные детали).
        """
        if subband not in self.SUBBANDS:
            raise ValueError(f"subband должен быть одним из {self.SUBBANDS}")
        self.alpha = alpha
        self.wavelet = wavelet
        self.level = level
        self.subband = subband

    # ------------------------------------------------------------------ #
    #  Вспомогательные методы                                             #
    # ------------------------------------------------------------------ #

    def _get_luma(self, img: Image.Image) -> tuple[np.ndarray, str, np.ndarray | None]:
        mode = img.mode
        if mode in ("RGB", "RGBA"):
            ycbcr = np.array(img.convert("YCbCr"), dtype=np.float64)
            return ycbcr[:, :, 0], mode, ycbcr
        return np.array(img.convert("L"), dtype=np.float64), mode, None

    def _resize_wm(self, wm: Image.Image, shape: tuple[int, int]) -> np.ndarray:
        h, w = shape
        arr = np.array(wm.convert("L").resize((w, h), Image.LANCZOS), dtype=np.float64)
        arr = arr / 255.0 - 0.5   # нормализация в [-0.5, 0.5]
        return arr

    def _get_subband(self, coeffs: list, target: str) -> np.ndarray:
        """Извлекает массив нужной субполосы из структуры pywt.wavedec2."""
        if target == "LL":
            return coeffs[0]
        idx = {"LH": 0, "HL": 1, "HH": 2}[target]
        return coeffs[1][idx]

    def _set_subband(self, coeffs: list, target: str, data: np.ndarray) -> list:
        """Подставляет массив обратно в нужную субполосу."""
        result = list(coeffs)
        if target == "LL":
            result[0] = data
        else:
            idx = {"LH": 0, "HL": 1, "HH": 2}[target]
            detail = list(result[1])
            detail[idx] = data
            result[1] = tuple(detail)
        return result

    # ------------------------------------------------------------------ #
    #  Встраивание                                                         #
    # ------------------------------------------------------------------ #

    def embed(self, host: Image.Image, watermark: Image.Image) -> Image.Image:
        """
        Встраивает watermark в host и возвращает помеченное изображение.

        Алгоритм:
            coeffs = wavedec2(channel, wavelet, level)
            sub    = coeffs[subband]
            wm     = resize(watermark, sub.shape)
            sub_new = sub + alpha * std(sub) * wm
            result  = waverec2(coeffs_new, wavelet)
        """
        channel, mode, color_arr = self._get_luma(host)

        # DWT
        coeffs = pywt.wavedec2(channel, self.wavelet, level=self.level)

        # Целевая субполоса
        sub = self._get_subband(coeffs, self.subband)
        wm_arr = self._resize_wm(watermark, sub.shape)

        # Масштабирование по стандартному отклонению субполосы
        sigma = np.std(sub) + 1e-10
        sub_new = sub + self.alpha * sigma * wm_arr

        # Обновляем коэффициенты
        coeffs_new = self._set_subband(coeffs, self.subband, sub_new)

        # Обратный DWT
        result_ch = pywt.waverec2(coeffs_new, self.wavelet)
        result_ch = result_ch[: channel.shape[0], : channel.shape[1]]
        result_ch = np.clip(result_ch, 0, 255)

        return self._reconstruct(result_ch, mode, color_arr, channel.shape)

    # ------------------------------------------------------------------ #
    #  Извлечение                                                          #
    # ------------------------------------------------------------------ #

    def extract(self, watermarked: Image.Image, original: Image.Image) -> Image.Image:
        """
        Извлекает водяной знак из помеченного изображения.
        Требует оригинал (non-blind extraction).

        Алгоритм:
            sub_wm   = wavedec2(watermarked)[subband]
            sub_orig = wavedec2(original)[subband]
            extracted = (sub_wm - sub_orig) / (alpha * sigma)
        """
        wm_ch, _, _ = self._get_luma(watermarked)
        orig_ch, _, _ = self._get_luma(original)

        coeffs_wm = pywt.wavedec2(wm_ch, self.wavelet, level=self.level)
        coeffs_orig = pywt.wavedec2(orig_ch, self.wavelet, level=self.level)

        sub_wm = self._get_subband(coeffs_wm, self.subband)
        sub_orig = self._get_subband(coeffs_orig, self.subband)

        sigma = np.std(sub_orig) + 1e-10
        extracted = (sub_wm - sub_orig) / (self.alpha * sigma)

        # Нормализация в [0, 255]
        extracted = extracted + 0.5   # сдвиг из [-0.5, 0.5] → [0, 1]
        extracted = np.clip(extracted, 0, 1)
        return Image.fromarray((extracted * 255).astype(np.uint8), "L")

    # ------------------------------------------------------------------ #
    #  Утилита                                                             #
    # ------------------------------------------------------------------ #

    def _reconstruct(
        self,
        channel: np.ndarray,
        mode: str,
        color_arr: np.ndarray | None,
        orig_shape: tuple,
    ) -> Image.Image:
        h, w = orig_shape
        channel = channel[:h, :w]
        if mode in ("RGB", "RGBA"):
            color_arr[:, :, 0] = channel
            return Image.fromarray(color_arr.astype(np.uint8), "YCbCr").convert(mode)
        return Image.fromarray(channel.astype(np.uint8), "L")
