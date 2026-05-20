"""
FFT Watermark — встраивание и извлечение водяного знака
через 2D преобразование Фурье.

Принцип:
  1. Вычисляем 2D FFT изображения → получаем амплитудный и фазовый спектры.
  2. Встраиваем водяной знак, модифицируя амплитуды в кольцевой области
     средних частот (не затрагиваем DC и высокие частоты).
  3. Восстанавливаем изображение обратным FFT.
  4. Извлечение: вычитаем амплитуды оригинала из амплитуд помеченного → IFFT.
"""

import numpy as np
from PIL import Image


class FFTWatermark:
    """Водяной знак на основе 2D FFT."""

    def __init__(self, alpha: float = 25.0, r_low: float = 0.05, r_high: float = 0.35):
        """
        Args:
            alpha:  Сила встраивания. Диапазон 10–60.
                    Больше — знак устойчивее, но заметнее.
            r_low:  Нижняя граница частотного кольца (доля от min(H,W)/2).
            r_high: Верхняя граница частотного кольца.
        """
        self.alpha = alpha
        self.r_low = r_low
        self.r_high = r_high

    # ------------------------------------------------------------------ #
    #  Вспомогательные методы                                             #
    # ------------------------------------------------------------------ #

    def _get_luma(self, img: Image.Image) -> tuple[np.ndarray, str, np.ndarray | None]:
        """
        Возвращает (канал_Y float64, исходный_mode, YCbCr_массив | None).
        Для цветных изображений работаем с Y-каналом YCbCr.
        """
        mode = img.mode
        if mode in ("RGB", "RGBA"):
            ycbcr = np.array(img.convert("YCbCr"), dtype=np.float64)
            return ycbcr[:, :, 0], mode, ycbcr
        else:
            return np.array(img.convert("L"), dtype=np.float64), mode, None

    def _resize_wm(self, wm: Image.Image, shape: tuple[int, int]) -> np.ndarray:
        """Масштабирует водяной знак к нужному размеру, нормализует в [-1, 1]."""
        h, w = shape
        arr = np.array(wm.convert("L").resize((w, h), Image.LANCZOS), dtype=np.float64)
        vmin, vmax = arr.min(), arr.max()
        arr = (arr - vmin) / (vmax - vmin + 1e-10)  # [0, 1]
        return arr * 2.0 - 1.0                       # [-1, 1]

    def _freq_mask(self, shape: tuple[int, int]) -> np.ndarray:
        """Кольцевая маска средних частот (булев массив)."""
        h, w = shape
        cy, cx = h // 2, w // 2
        r_max_px = min(h, w) / 2
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
        return (dist >= self.r_low * r_max_px) & (dist <= self.r_high * r_max_px)

    # ------------------------------------------------------------------ #
    #  Встраивание                                                         #
    # ------------------------------------------------------------------ #

    def embed(self, host: Image.Image, watermark: Image.Image) -> Image.Image:
        """
        Встраивает watermark в host и возвращает помеченное изображение.

        Алгоритм:
            F_host  = fftshift(fft2(Y_channel))
            F_wm    = fftshift(fft2(wm_resized))
            |F_host_new|[mask] += alpha * |F_wm|[mask]
            result  = ifft2(ifftshift(|F_host_new| * exp(i * phase_host)))
        """
        channel, mode, color_arr = self._get_luma(host)
        wm_arr = self._resize_wm(watermark, channel.shape)
        mask = self._freq_mask(channel.shape)

        # FFT хоста
        F_host = np.fft.fftshift(np.fft.fft2(channel))
        mag_host = np.abs(F_host)
        phase_host = np.angle(F_host)

        # FFT водяного знака (только амплитуда)
        F_wm = np.fft.fftshift(np.fft.fft2(wm_arr))
        mag_wm = np.abs(F_wm)

        # Встраивание в амплитуду
        mag_new = mag_host.copy()
        mag_new[mask] += self.alpha * mag_wm[mask]

        # Обратное преобразование
        F_new = mag_new * np.exp(1j * phase_host)
        result_ch = np.fft.ifft2(np.fft.ifftshift(F_new)).real
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
            diff_mag = (|FFT(watermarked)| - |FFT(original)|) / alpha
            extracted = |ifft2(ifftshift(diff_mag))|   (фаза = 0)
        """
        wm_ch, _, _ = self._get_luma(watermarked)
        orig_ch, _, _ = self._get_luma(original)

        F_wm = np.fft.fftshift(np.fft.fft2(wm_ch))
        F_orig = np.fft.fftshift(np.fft.fft2(orig_ch))

        mask = self._freq_mask(wm_ch.shape)

        diff = np.zeros_like(np.abs(F_wm))
        diff[mask] = (np.abs(F_wm)[mask] - np.abs(F_orig)[mask]) / (self.alpha + 1e-10)

        # Реконструкция без фазы → реальная часть
        extracted = np.abs(np.fft.ifft2(np.fft.ifftshift(diff)))

        # Нормализация в [0, 255]
        extracted = (extracted - extracted.min()) / (extracted.max() - extracted.min() + 1e-10)
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
