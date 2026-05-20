"""
run_analysis.py — статистические тесты незаметности водяного знака.

Запуск из папки files/:
    python run_analysis.py --host фото.png --wm знак.png
    python run_analysis.py   (без аргументов — синтетические данные)

Результат: папка analysis_results/ с графиками и отчётом.
"""

import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageDraw
from scipy import stats

from core.fft_watermark import FFTWatermark
from core.wavelet_watermark import WaveletWatermark

# ── стиль графиков ──────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#161b22",
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  "#c9d1d9",
    "text.color":       "#c9d1d9",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "grid.color":       "#21262d",
    "lines.linewidth":  2.0,
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
})

OUTPUT_DIR = Path("analysis_results")


# ═══════════════════════════════════════════════════════════════════════
#  Тестовые изображения (если не переданы свои)
# ═══════════════════════════════════════════════════════════════════════

def make_test_images():
    rng = np.random.default_rng(0)
    h, w = 256, 256
    X, Y = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    base = 80 + 100 * (0.5 + 0.3 * np.sin(4 * np.pi * X) * np.cos(3 * np.pi * Y))
    ch = np.clip(base + rng.normal(0, 8, (h, w)), 0, 255).astype(np.uint8)
    orig = Image.fromarray(
        np.stack([ch,
                  np.clip(ch + rng.integers(-15, 15, (h, w)), 0, 255).astype(np.uint8),
                  np.clip(ch + rng.integers(-10, 10, (h, w)), 0, 255).astype(np.uint8)],
                 axis=-1), "RGB"
    )
    wm = Image.new("L", (256, 256), 0)
    d = ImageDraw.Draw(wm)
    d.text((30, 100), "WATERMARK", fill=255)
    d.rectangle([5, 5, 250, 250], outline=180, width=3)
    return orig, wm


# ═══════════════════════════════════════════════════════════════════════
#  Три статистических теста для одной пары изображений
# ═══════════════════════════════════════════════════════════════════════

def run_tests(original, watermarked):
    """
    Возвращает словарь с результатами трёх тестов.
    H0: встраивание не изменило распределение пикселей.
    """
    orig_px = np.array(original.convert("L"), dtype=np.float64).flatten()
    wm_px   = np.array(watermarked.convert("L"), dtype=np.float64).flatten()

    # Подвыборка (тесты работают медленно на 65000 пикселях)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(orig_px), size=5000, replace=False)
    a, b = orig_px[idx], wm_px[idx]

    # t-тест: сравниваем средние
    t_stat, t_p = stats.ttest_ind(a, b, equal_var=False)

    # χ²: сравниваем гистограммы
    orig_hist, _ = np.histogram(orig_px, bins=64, range=(0, 255))
    wm_hist,   _ = np.histogram(wm_px,   bins=64, range=(0, 255))
    chi2_stat, chi2_p = stats.chisquare(wm_hist + 1, f_exp=orig_hist + 1)

    # КС: сравниваем функции распределения
    ks_stat, ks_p = stats.ks_2samp(a, b)

    return {
        "t-тест":  {"stat": t_stat,    "p": t_p},
        "χ²":      {"stat": chi2_stat, "p": chi2_p},
        "КС":      {"stat": ks_stat,   "p": ks_p},
    }


# ═══════════════════════════════════════════════════════════════════════
#  Главная функция анализа
# ═══════════════════════════════════════════════════════════════════════

def run_analysis(original, watermark):
    OUTPUT_DIR.mkdir(exist_ok=True)

    alphas_fft = [5, 10, 20, 30, 50, 80, 120]
    alphas_wav = [0.02, 0.05, 0.10, 0.20, 0.40, 0.80, 1.50]

    # Собираем p-value для каждого alpha
    results = {"FFT": {}, "Wavelet": {}}

    print("\n[FFT]")
    print(f"  {'alpha':>6} | {'t p':>6} | {'χ² p':>6} | {'КС p':>6} | статус")
    print("  " + "-" * 50)
    for a in alphas_fft:
        marked = FFTWatermark(alpha=a).embed(original, watermark)
        r = run_tests(original, marked)
        results["FFT"][a] = r
        invisible = all(v["p"] >= 0.05 for v in r.values())
        print(f"  {a:>6} | {r['t-тест']['p']:>6.3f} | {r['χ²']['p']:>6.3f} | "
              f"{r['КС']['p']:>6.3f} | {'✓ незаметен' if invisible else '✗ заметен'}")

    print("\n[Wavelet]")
    print(f"  {'alpha':>6} | {'t p':>6} | {'χ² p':>6} | {'КС p':>6} | статус")
    print("  " + "-" * 50)
    for a in alphas_wav:
        marked = WaveletWatermark(alpha=a).embed(original, watermark)
        r = run_tests(original, marked)
        results["Wavelet"][a] = r
        invisible = all(v["p"] >= 0.05 for v in r.values())
        print(f"  {a:>6.2f} | {r['t-тест']['p']:>6.3f} | {r['χ²']['p']:>6.3f} | "
              f"{r['КС']['p']:>6.3f} | {'✓ незаметен' if invisible else '✗ заметен'}")

    # ── Графики ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Статистические тесты незаметности водяного знака\n"
                 "H₀: встраивание не изменяет распределение пикселей", fontsize=13)

    test_names  = ["t-тест", "χ²", "КС"]
    test_labels = ["t-тест Стьюдента", "χ²-критерий Пирсона", "Критерий КС"]

    for ax, tname, tlabel in zip(axes, test_names, test_labels):
        p_fft = [results["FFT"][a][tname]["p"]     for a in alphas_fft]
        p_wav = [results["Wavelet"][a][tname]["p"] for a in alphas_wav]

        x = range(len(alphas_fft))
        ax.plot(x, p_fft, "o-", color="#818cf8", label="FFT",     markersize=8)
        ax.plot(x, p_wav, "s--", color="#34d399", label="Wavelet", markersize=8)
        ax.axhline(0.05, color="#f87171", lw=1.5, linestyle=":", label="α = 0.05")
        ax.fill_between(x, 0.05, 1.0, alpha=0.06, color="#818cf8")

        ax.set_xticks(x)
        ax.set_xticklabels([str(a) for a in alphas_fft], rotation=45)
        ax.set_xlabel("alpha (FFT)")
        ax.set_ylabel("p-value")
        ax.set_title(tlabel)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)
        ax.grid(True)

        ax.text(0.5, 0.93, "H₀ не отвергается (знак незаметен)",
                transform=ax.transAxes, ha="center", color="#818cf8", fontsize=8)
        ax.text(0.5, 0.01, "H₀ отвергается (знак обнаружен)",
                transform=ax.transAxes, ha="center", color="#f87171", fontsize=8)

    plt.tight_layout()
    plot_path = OUTPUT_DIR / "statistical_tests.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nГрафик сохранён: {plot_path}")

    # ── Гистограммы ──────────────────────────────────────────────────
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
    fig2.suptitle("Сравнение гистограмм яркости: оригинал vs с водяным знаком")

    for ax, (alpha_val, engine, label, color) in zip(axes2, [
        (30,  FFTWatermark(alpha=30),        "FFT  α=30",      "#818cf8"),
        (0.2, WaveletWatermark(alpha=0.2),   "Wavelet  α=0.2", "#34d399"),
    ]):
        marked = engine.embed(original, watermark)
        bins = np.linspace(0, 255, 65)
        ax.hist(np.array(original.convert("L")).flatten(),
                bins=bins, alpha=0.5, color="#8b949e", label="Оригинал", density=True)
        ax.hist(np.array(marked.convert("L")).flatten(),
                bins=bins, alpha=0.6, color=color,
                label=f"С водяным знаком ({label})", density=True)
        ax.set_xlabel("Яркость пикселя")
        ax.set_ylabel("Плотность")
        ax.set_title(f"Гистограмма — {label}")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    hist_path = OUTPUT_DIR / "histograms.png"
    plt.savefig(hist_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Гистограммы сохранены: {hist_path}")

    # ── Текстовый отчёт ──────────────────────────────────────────────
    report_lines = [
        "=" * 60,
        "  ОТЧЁТ: Статистические тесты незаметности",
        "=" * 60,
        "",
        "  H₀: встраивание не изменяет распределение пикселей.",
        "  Уровень значимости α = 0.05",
        "  p > 0.05 → H₀ не отвергается → знак статистически незаметен",
        "",
        "  FFT:",
        f"  {'alpha':>6} | {'t p':>6} | {'χ² p':>6} | {'КС p':>6} | статус",
        "  " + "-" * 46,
    ]
    for a in alphas_fft:
        r = results["FFT"][a]
        invisible = all(v["p"] >= 0.05 for v in r.values())
        report_lines.append(
            f"  {a:>6} | {r['t-тест']['p']:>6.3f} | {r['χ²']['p']:>6.3f} | "
            f"{r['КС']['p']:>6.3f} | {'✓ незаметен' if invisible else '✗ заметен'}"
        )

    report_lines += ["", "  Wavelet:",
                     f"  {'alpha':>6} | {'t p':>6} | {'χ² p':>6} | {'КС p':>6} | статус",
                     "  " + "-" * 46]
    for a in alphas_wav:
        r = results["Wavelet"][a]
        invisible = all(v["p"] >= 0.05 for v in r.values())
        report_lines.append(
            f"  {a:>6.2f} | {r['t-тест']['p']:>6.3f} | {r['χ²']['p']:>6.3f} | "
            f"{r['КС']['p']:>6.3f} | {'✓ незаметен' if invisible else '✗ заметен'}"
        )

    report_lines += ["", "=" * 60]
    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    report_path = OUTPUT_DIR / "report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    print(f"\nОтчёт сохранён: {report_path}")
    print("\n✅ Готово! Результаты в папке analysis_results/")


# ═══════════════════════════════════════════════════════════════════════
#  Точка входа
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None, help="Путь к оригиналу")
    parser.add_argument("--wm",   type=str, default=None, help="Путь к водяному знаку")
    args = parser.parse_args()

    if args.host and args.wm:
        original  = Image.open(args.host)
        watermark = Image.open(args.wm)
        print(f"Загружены: {args.host}, {args.wm}")
    else:
        print("Изображения не переданы — используются синтетические данные.")
        original, watermark = make_test_images()

    run_analysis(original, watermark)
