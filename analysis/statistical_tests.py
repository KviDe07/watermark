"""
Идея 1: Статистический анализ незаметности водяного знака.

Нулевая гипотеза H₀:
    «Встраивание водяного знака не изменяет статистические
     свойства распределения пикселей изображения.»

Используемые критерии:
    • t-тест Стьюдента   — сравнение средних
    • χ²-критерий        — сравнение гистограмм (распределений)
    • Критерий КС        — сравнение эмпирических функций распределения
"""

import numpy as np
from PIL import Image
from scipy import stats
from dataclasses import dataclass


# ═══════════════════════════════════════════════════════════════════════
#  Структуры результатов
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TestResult:
    name: str
    statistic: float
    p_value: float
    alpha_level: float = 0.05

    @property
    def reject_h0(self) -> bool:
        return self.p_value < self.alpha_level

    @property
    def conclusion(self) -> str:
        if self.reject_h0:
            return f"H₀ ОТВЕРГАЕТСЯ (p={self.p_value:.4f} < {self.alpha_level}) — знак статистически заметен"
        return f"H₀ НЕ ОТВЕРГАЕТСЯ (p={self.p_value:.4f} ≥ {self.alpha_level}) — знак статистически незаметен"


@dataclass
class FullTestReport:
    alpha_value: float          # сила встраивания
    method: str                 # FFT / Wavelet
    t_test: TestResult
    chi2_test: TestResult
    ks_test: TestResult

    # описательная статистика
    mean_original: float
    mean_watermarked: float
    std_original: float
    std_watermarked: float

    @property
    def invisible(self) -> bool:
        """True если все три теста не отвергают H₀."""
        return not (self.t_test.reject_h0 or
                    self.chi2_test.reject_h0 or
                    self.ks_test.reject_h0)


# ═══════════════════════════════════════════════════════════════════════
#  Вспомогательные функции
# ═══════════════════════════════════════════════════════════════════════

def _pixels(img: Image.Image) -> np.ndarray:
    """Возвращает пиксели Y-канала как одномерный float64 массив."""
    return np.array(img.convert("L"), dtype=np.float64).flatten()


def _histogram(pixels: np.ndarray, bins: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Нормализованная гистограмма яркостей."""
    counts, edges = np.histogram(pixels, bins=bins, range=(0, 255))
    return counts, edges


# ═══════════════════════════════════════════════════════════════════════
#  Три статистических теста
# ═══════════════════════════════════════════════════════════════════════

def t_test(original: Image.Image, watermarked: Image.Image,
           alpha_level: float = 0.05) -> TestResult:
    """
    Двухвыборочный t-тест Стьюдента.

    H₀: μ_original = μ_watermarked  (средние яркости равны)
    H₁: μ_original ≠ μ_watermarked

    При малом alpha встраивания средние должны совпадать,
    и p-value будет большим → H₀ не отвергается.
    """
    orig = _pixels(original)
    wm   = _pixels(watermarked)

    # Используем подвыборку для скорости (пиксели коррелированы)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(orig), size=min(5000, len(orig)), replace=False)

    stat, p = stats.ttest_ind(orig[idx], wm[idx], equal_var=False)
    return TestResult("t-тест Стьюдента", float(stat), float(p), alpha_level)


def chi2_test(original: Image.Image, watermarked: Image.Image,
              bins: int = 64, alpha_level: float = 0.05) -> TestResult:
    """
    Критерий χ² Пирсона для сравнения гистограмм яркости.

    H₀: гистограммы оригинала и помеченного изображения совпадают
    H₁: гистограммы различаются

    Большой χ² означает что распределение пикселей изменилось.
    """
    orig_counts, _ = _histogram(_pixels(original), bins)
    wm_counts, _   = _histogram(_pixels(watermarked), bins)

    # Избегаем нулей (условие χ²: ожидаемые частоты > 0)
    orig_counts = orig_counts + 1
    wm_counts   = wm_counts + 1

    stat, p = stats.chisquare(wm_counts, f_exp=orig_counts)
    return TestResult("χ²-критерий Пирсона", float(stat), float(p), alpha_level)


def ks_test(original: Image.Image, watermarked: Image.Image,
            alpha_level: float = 0.05) -> TestResult:
    """
    Двухвыборочный критерий Колмогорова-Смирнова.

    H₀: F_original(x) = F_watermarked(x)  (ЭФР совпадают)
    H₁: sup|F_original - F_watermarked| > порог

    Статистика КС = максимальное отклонение между ЭФР.
    Более чувствителен к форме распределения чем t-тест.
    """
    orig = _pixels(original)
    wm   = _pixels(watermarked)

    rng = np.random.default_rng(42)
    n = min(3000, len(orig))
    idx = rng.choice(len(orig), size=n, replace=False)

    stat, p = stats.ks_2samp(orig[idx], wm[idx])
    return TestResult("Критерий Колмогорова-Смирнова", float(stat), float(p), alpha_level)


# ═══════════════════════════════════════════════════════════════════════
#  Полный отчёт для одного alpha
# ═══════════════════════════════════════════════════════════════════════

def full_report(original: Image.Image, watermarked: Image.Image,
                alpha_value: float, method: str = "FFT") -> FullTestReport:
    orig_px = _pixels(original)
    wm_px   = _pixels(watermarked)

    return FullTestReport(
        alpha_value    = alpha_value,
        method         = method,
        t_test         = t_test(original, watermarked),
        chi2_test      = chi2_test(original, watermarked),
        ks_test        = ks_test(original, watermarked),
        mean_original  = float(orig_px.mean()),
        mean_watermarked = float(wm_px.mean()),
        std_original   = float(orig_px.std()),
        std_watermarked= float(wm_px.std()),
    )


# ═══════════════════════════════════════════════════════════════════════
#  Sweep по alpha: отчёт для нескольких значений
# ═══════════════════════════════════════════════════════════════════════

def alpha_sweep(original: Image.Image, watermark: Image.Image,
                alphas: list[float], embedder_factory,
                method: str = "FFT") -> list[FullTestReport]:
    """
    Прогоняет full_report для каждого alpha из списка.

    embedder_factory(alpha) → объект с методом .embed(host, wm)
    """
    reports = []
    for a in alphas:
        embedder = embedder_factory(a)
        wm_img = embedder.embed(original, watermark)
        report = full_report(original, wm_img, a, method)
        reports.append(report)
        print(f"  alpha={a:6.1f} | t p={report.t_test.p_value:.3f} "
              f"χ² p={report.chi2_test.p_value:.3f} "
              f"KS p={report.ks_test.p_value:.3f} "
              f"| {'✓ незаметен' if report.invisible else '✗ заметен'}")
    return reports
