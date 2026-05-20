"""
Идея 2: Шум встраивания и Центральная Предельная Теорема.

Теория:
    Разность d = watermarked - original — это «шум встраивания».
    По ЦПТ: если этот шум складывается из большого числа независимых
    малых поправок (коэффициентов в спектре), то при alpha→0
    распределение d должно сходиться к нормальному N(μ, σ²).

Проверяем:
    1. Строим гистограмму d и сравниваем с N(μ̂, σ̂²)
    2. Тест Шапиро-Уилка на нормальность
    3. QQ-plot (квантиль-квантиль)
    4. Зависимость σ шума от alpha — должна быть линейной
"""

import numpy as np
from PIL import Image
from scipy import stats
from dataclasses import dataclass, field


# ═══════════════════════════════════════════════════════════════════════
#  Структуры результатов
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class NoiseStats:
    alpha_value: float
    method: str

    # Описательная статистика шума
    mean: float               # μ̂
    std: float                # σ̂
    skewness: float           # асимметрия (у нормального = 0)
    kurtosis: float           # эксцесс (у нормального = 0)

    # Тест Шапиро-Уилка
    shapiro_stat: float
    shapiro_p: float

    # Тест Д'Агостино-Пирсона (для больших выборок)
    dagostino_stat: float
    dagostino_p: float

    @property
    def is_normal_shapiro(self) -> bool:
        return self.shapiro_p >= 0.05

    @property
    def is_normal_dagostino(self) -> bool:
        return self.dagostino_p >= 0.05

    @property
    def noise_array(self) -> np.ndarray:
        return self._noise

    def _attach_noise(self, noise: np.ndarray):
        self._noise = noise


# ═══════════════════════════════════════════════════════════════════════
#  Основные функции
# ═══════════════════════════════════════════════════════════════════════

def compute_noise(original: Image.Image,
                  watermarked: Image.Image) -> np.ndarray:
    """
    Вычисляет карту шума: d[i,j] = watermarked[i,j] - original[i,j]
    Возвращает одномерный массив float64.
    """
    orig = np.array(original.convert("L"), dtype=np.float64)
    wm   = np.array(watermarked.convert("L"), dtype=np.float64)
    return (wm - orig).flatten()


def noise_statistics(original: Image.Image,
                     watermarked: Image.Image,
                     alpha_value: float,
                     method: str = "FFT",
                     sample_size: int = 5000) -> NoiseStats:
    """
    Полная статистика шума встраивания.

    sample_size: размер подвыборки для теста Шапиро (макс 5000).
    """
    noise = compute_noise(original, watermarked)

    # Подвыборка для Шапиро (требует n ≤ 5000)
    rng = np.random.default_rng(42)
    n = min(sample_size, len(noise))
    sample = rng.choice(noise, size=n, replace=False)

    # Шапиро-Уилк
    sw_stat, sw_p = stats.shapiro(sample)

    # Д'Агостино-Пирсон (работает на больших выборках)
    da_stat, da_p = stats.normaltest(noise)

    result = NoiseStats(
        alpha_value    = alpha_value,
        method         = method,
        mean           = float(noise.mean()),
        std            = float(noise.std()),
        skewness       = float(stats.skew(noise)),
        kurtosis       = float(stats.kurtosis(noise)),  # excess kurtosis
        shapiro_stat   = float(sw_stat),
        shapiro_p      = float(sw_p),
        dagostino_stat = float(da_stat),
        dagostino_p    = float(da_p),
    )
    result._attach_noise(noise)
    return result


def sigma_vs_alpha(original: Image.Image,
                   watermark: Image.Image,
                   alphas: list[float],
                   embedder_factory,
                   method: str = "FFT") -> tuple[list[float], list[float]]:
    """
    Вычисляет σ шума для каждого alpha.
    Возвращает (alphas, sigmas).

    По теории: σ ∝ alpha → связь должна быть линейной.
    Проверяем коэффициент корреляции Пирсона.
    """
    sigmas = []
    for a in alphas:
        embedder = embedder_factory(a)
        wm_img = embedder.embed(original, watermark)
        noise = compute_noise(original, wm_img)
        sigmas.append(float(noise.std()))
        print(f"  alpha={a:6.1f} | σ_шума = {sigmas[-1]:.4f}")

    r, p = stats.pearsonr(alphas, sigmas)
    print(f"\n  Корреляция Пирсона r={r:.4f}, p={p:.6f}")
    print(f"  {'✓ Линейная зависимость σ(alpha) подтверждена' if r > 0.99 else '⚠ Зависимость нелинейна'}")

    return alphas, sigmas


def fit_normal(noise: np.ndarray) -> tuple[float, float]:
    """МНК-оценки параметров нормального распределения N(μ, σ²)."""
    return float(noise.mean()), float(noise.std())


def theoretical_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Плотность нормального распределения N(mu, sigma²)."""
    return stats.norm.pdf(x, loc=mu, scale=sigma)


def qqplot_data(noise: np.ndarray,
                sample_size: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """
    Данные для QQ-plot: (теоретические квантили, наблюдаемые квантили).
    Если точки лежат на прямой y=x → распределение нормальное.
    """
    rng = np.random.default_rng(42)
    sample = rng.choice(noise, size=min(sample_size, len(noise)), replace=False)
    sample_sorted = np.sort(sample)

    n = len(sample_sorted)
    # Теоретические квантили стандартного нормального
    probs = (np.arange(1, n + 1) - 0.5) / n
    theoretical = stats.norm.ppf(probs)

    # Стандартизируем наблюдаемые
    observed = (sample_sorted - sample_sorted.mean()) / (sample_sorted.std() + 1e-10)

    return theoretical, observed
