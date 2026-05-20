"""
Главное окно Watermark Studio.

Компоновка:
  ┌─────────────┬──────────────────────────────────────┐
  │  Левая      │  Правая панель (вкладки)              │
  │  панель     │  ┌──────────────────────────────────┐│
  │  • Загрузка ││  Оригинал | ЗВ | Результат | Извл. ││
  │  • Метод    │└──────────────────────────────────────┘│
  │  • Параметры│  Метрики                              │
  │  • Кнопки   │                                       │
  └─────────────┴──────────────────────────────────────┘
"""

from __future__ import annotations

import sys
from pathlib import Path
from PIL import Image

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QGridLayout,
)

from core.fft_watermark import FFTWatermark
from core.wavelet_watermark import WaveletWatermark
from core import all_metrics


# ═══════════════════════════════════════════════════════════════════════
#  ImageLabel — виджет предпросмотра изображения
# ═══════════════════════════════════════════════════════════════════════

class ImageLabel(QLabel):
    """QLabel с автоматическим масштабированием PIL-изображения."""

    def __init__(self, placeholder: str = "—"):
        super().__init__()
        self._pil: Image.Image | None = None
        self.placeholder = placeholder
        self.setText(placeholder)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(180, 180)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet(
            "border: 2px dashed #444; border-radius: 8px;"
            "background: #12151e; color: #555; font-size: 12px;"
        )

    # ── публичный интерфейс ──────────────────────────────────────────

    def set_image(self, img: Image.Image) -> None:
        self._pil = img
        self._refresh()

    def get_image(self) -> Image.Image | None:
        return self._pil

    # ── внутренние методы ────────────────────────────────────────────

    def _refresh(self) -> None:
        if self._pil is None:
            self.setText(self.placeholder)
            return
        rgb = self._pil.convert("RGB")
        w, h = rgb.size
        data = rgb.tobytes("raw", "RGB")
        qimg = QImage(data, w, h, w * 3, QImage.Format.Format_RGB888)
        px = QPixmap.fromImage(qimg).scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(px)

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        self._refresh()


# ═══════════════════════════════════════════════════════════════════════
#  WorkerThread — фоновый поток (чтобы GUI не замерзал)
# ═══════════════════════════════════════════════════════════════════════

class WorkerThread(QThread):
    finished = pyqtSignal(object, object)   # (PIL Image, dict)
    error = pyqtSignal(str)

    def __init__(self, func, *args):
        super().__init__()
        self._func = func
        self._args = args

    def run(self):
        try:
            img, metrics = self._func(*self._args)
            self.finished.emit(img, metrics)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))


# ═══════════════════════════════════════════════════════════════════════
#  MainWindow
# ═══════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):

    # ── инициализация ────────────────────────────────────────────────

    def __init__(self):
        super().__init__()
        self.host_img: Image.Image | None = None
        self.wm_img: Image.Image | None = None
        self.watermarked_img: Image.Image | None = None
        self.extracted_img: Image.Image | None = None

        self.setWindowTitle("Watermark Studio — FFT & Wavelet")
        self.setMinimumSize(1280, 820)
        self.setStyleSheet(self._stylesheet())
        self._build_ui()

    # ── построение UI ────────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        lay = QHBoxLayout(root)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(12)

        left = self._left_panel()
        left.setFixedWidth(290)
        lay.addWidget(left)
        lay.addWidget(self._right_panel(), 1)

    # ---- левая панель -----------------------------------------------

    def _left_panel(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(10)

        # Заголовок
        title = QLabel("Watermark Studio")
        title.setFont(QFont("Segoe UI", 15, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: #818cf8; padding: 6px 0 2px;")
        sub = QLabel("FFT · Wavelet · Метрики")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sub.setStyleSheet("color: #555; font-size: 11px; margin-bottom: 4px;")
        lay.addWidget(title)
        lay.addWidget(sub)

        # Загрузка
        lay.addWidget(self._load_group())

        # Метод + параметры
        lay.addWidget(self._method_group())

        # Кнопки
        lay.addWidget(self._action_group())

        lay.addStretch()

        # Метрики
        lay.addWidget(self._metrics_group())

        return w

    def _load_group(self) -> QGroupBox:
        g = QGroupBox("Изображения")
        lay = QVBoxLayout(g)

        self.btn_host = QPushButton("Загрузить оригинал")
        self.lbl_host = QLabel("не загружено")
        self.lbl_host.setStyleSheet("color:#555;font-size:11px;")

        self.btn_wm = QPushButton("Загрузить водяной знак")
        self.lbl_wm = QLabel("не загружено")
        self.lbl_wm.setStyleSheet("color:#555;font-size:11px;")

        self.btn_host.clicked.connect(self._load_host)
        self.btn_wm.clicked.connect(self._load_wm)

        lay.addWidget(self.btn_host)
        lay.addWidget(self.lbl_host)
        lay.addWidget(self.btn_wm)
        lay.addWidget(self.lbl_wm)
        return g

    def _method_group(self) -> QGroupBox:
        g = QGroupBox("Метод и параметры")
        lay = QVBoxLayout(g)

        self.combo_method = QComboBox()
        self.combo_method.addItems(["FFT (преобразование Фурье)", "Wavelet (вейвлет)"])
        self.combo_method.currentIndexChanged.connect(self._toggle_params)
        lay.addWidget(self.combo_method)

        # ---- FFT params ----
        self.fft_widget = QWidget()
        fl = QVBoxLayout(self.fft_widget)
        fl.setContentsMargins(0, 0, 0, 0)
        fl.addWidget(QLabel("Сила встраивания (alpha):"))
        self.fft_alpha = QDoubleSpinBox()
        self.fft_alpha.setRange(1.0, 120.0)
        self.fft_alpha.setValue(25.0)
        self.fft_alpha.setSingleStep(5.0)
        fl.addWidget(self.fft_alpha)

        fl.addWidget(QLabel("Нижн. граница частот (r_low):"))
        self.fft_rlow = QDoubleSpinBox()
        self.fft_rlow.setRange(0.01, 0.49)
        self.fft_rlow.setValue(0.05)
        self.fft_rlow.setSingleStep(0.01)
        fl.addWidget(self.fft_rlow)

        fl.addWidget(QLabel("Верхн. граница частот (r_high):"))
        self.fft_rhigh = QDoubleSpinBox()
        self.fft_rhigh.setRange(0.10, 0.90)
        self.fft_rhigh.setValue(0.35)
        self.fft_rhigh.setSingleStep(0.05)
        fl.addWidget(self.fft_rhigh)

        # ---- Wavelet params ----
        self.wav_widget = QWidget()
        wl = QVBoxLayout(self.wav_widget)
        wl.setContentsMargins(0, 0, 0, 0)
        wl.addWidget(QLabel("Сила встраивания (alpha):"))
        self.wav_alpha = QDoubleSpinBox()
        self.wav_alpha.setRange(0.01, 10.0)
        self.wav_alpha.setValue(0.10)
        self.wav_alpha.setSingleStep(0.05)
        wl.addWidget(self.wav_alpha)

        wl.addWidget(QLabel("Вейвлет:"))
        self.wav_type = QComboBox()
        self.wav_type.addItems(WaveletWatermark.WAVELETS)
        wl.addWidget(self.wav_type)

        wl.addWidget(QLabel("Уровень разложения:"))
        self.wav_level = QSpinBox()
        self.wav_level.setRange(1, 5)
        self.wav_level.setValue(2)
        wl.addWidget(self.wav_level)

        wl.addWidget(QLabel("Субполоса:"))
        self.wav_sub = QComboBox()
        self.wav_sub.addItems(WaveletWatermark.SUBBANDS)
        wl.addWidget(self.wav_sub)

        self.wav_widget.hide()
        lay.addWidget(self.fft_widget)
        lay.addWidget(self.wav_widget)
        return g

    def _action_group(self) -> QGroupBox:
        g = QGroupBox("Действия")
        lay = QVBoxLayout(g)

        self.btn_embed = QPushButton("▶  Встроить водяной знак")
        self.btn_embed.setStyleSheet(
            "background:#4f46e5;color:white;font-weight:bold;padding:8px;border-radius:6px;"
        )
        self.btn_embed.clicked.connect(self._do_embed)

        self.btn_extract = QPushButton("🔍  Извлечь водяной знак")
        self.btn_extract.clicked.connect(self._do_extract)

        self.btn_save = QPushButton("💾  Сохранить результат")
        self.btn_save.clicked.connect(self._save)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()
        self.status_lbl = QLabel("")
        self.status_lbl.setStyleSheet("color:#818cf8;font-size:11px;")

        lay.addWidget(self.btn_embed)
        lay.addWidget(self.btn_extract)
        lay.addWidget(self.btn_save)
        lay.addWidget(self.progress)
        lay.addWidget(self.status_lbl)
        return g

    def _metrics_group(self) -> QGroupBox:
        g = QGroupBox("Метрики качества")
        lay = QVBoxLayout(g)
        self.metrics_view = QTextEdit()
        self.metrics_view.setReadOnly(True)
        self.metrics_view.setMaximumHeight(130)
        self.metrics_view.setStyleSheet(
            "background:#0d1117;color:#58a6ff;font-family:monospace;font-size:12px;border:none;"
        )
        self.metrics_view.setPlaceholderText("Метрики появятся после обработки…")
        lay.addWidget(self.metrics_view)
        return g

    # ---- правая панель ---------------------------------------------

    def _right_panel(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)

        tabs = QTabWidget()
        tabs.setStyleSheet(
            "QTabBar::tab{padding:7px 18px;background:#1e2130;color:#666;}"
            "QTabBar::tab:selected{background:#4f46e5;color:white;}"
            "QTabWidget::pane{border:1px solid #2a2d3e;}"
        )

        # Вкладка сравнения
        compare = QWidget()
        gl = QGridLayout(compare)
        gl.setSpacing(8)
        captions = [
            ("Оригинал", "pv_original"),
            ("Водяной знак", "pv_watermark"),
            ("С водяным знаком", "pv_result"),
            ("Извлечённый знак", "pv_extracted"),
        ]
        for col, (cap, attr) in enumerate(captions):
            lbl = QLabel(cap)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("color:#aaa;font-size:11px;font-weight:bold;")
            img_lbl = ImageLabel()
            setattr(self, attr, img_lbl)
            gl.addWidget(lbl, 0, col)
            gl.addWidget(img_lbl, 1, col)
        tabs.addTab(compare, "Сравнение")

        # Вкладка результата
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.pv_result_big = ImageLabel("Встройте водяной знак…")
        scroll.setWidget(self.pv_result_big)
        tabs.addTab(scroll, "Результат (полный)")

        lay.addWidget(tabs)
        return w

    # ── слоты ────────────────────────────────────────────────────────

    def _toggle_params(self, idx: int):
        self.fft_widget.setVisible(idx == 0)
        self.wav_widget.setVisible(idx == 1)

    # ---- загрузка ------------------------------------------------

    def _load_host(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите оригинальное изображение", "",
            "Изображения (*.png *.jpg *.jpeg *.bmp *.tiff *.webp)"
        )
        if path:
            self.host_img = Image.open(path)
            self.pv_original.set_image(self.host_img)
            self.lbl_host.setText(Path(path).name)

    def _load_wm(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите водяной знак", "",
            "Изображения (*.png *.jpg *.jpeg *.bmp *.tiff *.webp)"
        )
        if path:
            self.wm_img = Image.open(path)
            self.pv_watermark.set_image(self.wm_img)
            self.lbl_wm.setText(Path(path).name)

    # ---- встраивание --------------------------------------------

    def _do_embed(self):
        if self.host_img is None or self.wm_img is None:
            QMessageBox.warning(self, "Нет изображений", "Загрузите оба изображения.")
            return
        self._set_busy(True, "Встраивание…")
        engine = self._make_engine()

        def task(host, wm):
            result = engine.embed(host, wm)
            m = all_metrics(host, result, wm, None)
            return result, m

        self._run(task, self.host_img.copy(), self.wm_img.copy(),
                  on_done=self._embed_done)

    def _embed_done(self, result: Image.Image, metrics: dict):
        self.watermarked_img = result
        self.pv_result.set_image(result)
        self.pv_result_big.set_image(result)
        self._show_metrics(metrics)
        self._set_busy(False, "Встраивание завершено ✓")

    # ---- извлечение --------------------------------------------

    def _do_extract(self):
        if self.watermarked_img is None:
            QMessageBox.warning(self, "Нет данных", "Сначала встройте водяной знак.")
            return
        if self.host_img is None:
            QMessageBox.warning(self, "Нет оригинала", "Загрузите оригинальное изображение.")
            return
        self._set_busy(True, "Извлечение…")
        engine = self._make_engine()

        def task(wm_img, orig_img, orig_wm):
            extracted = engine.extract(wm_img, orig_img)
            m = all_metrics(orig_img, wm_img, orig_wm, extracted)
            return extracted, m

        self._run(
            task,
            self.watermarked_img.copy(),
            self.host_img.copy(),
            self.wm_img.copy() if self.wm_img else None,
            on_done=self._extract_done,
        )

    def _extract_done(self, extracted: Image.Image, metrics: dict):
        self.extracted_img = extracted
        self.pv_extracted.set_image(extracted)
        self._show_metrics(metrics)
        self._set_busy(False, "Извлечение завершено ✓")

    # ---- сохранение --------------------------------------------

    def _save(self):
        if self.watermarked_img is None:
            QMessageBox.warning(self, "Нет результата", "Сначала встройте водяной знак.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить изображение", "watermarked.png",
            "PNG (*.png);;JPEG (*.jpg);;BMP (*.bmp)"
        )
        if path:
            self.watermarked_img.save(path)
            self.status_lbl.setText(f"Сохранено: {Path(path).name}")

    # ── вспомогательные методы ───────────────────────────────────────

    def _make_engine(self) -> FFTWatermark | WaveletWatermark:
        if self.combo_method.currentIndex() == 0:
            return FFTWatermark(
                alpha=self.fft_alpha.value(),
                r_low=self.fft_rlow.value(),
                r_high=self.fft_rhigh.value(),
            )
        return WaveletWatermark(
            alpha=self.wav_alpha.value(),
            wavelet=self.wav_type.currentText(),
            level=self.wav_level.value(),
            subband=self.wav_sub.currentText(),
        )

    def _run(self, func, *args, on_done):
        self._worker = WorkerThread(func, *args)
        self._worker.finished.connect(on_done)
        self._worker.finished.connect(lambda *_: self._set_busy(False))
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _set_busy(self, busy: bool, msg: str = ""):
        self.btn_embed.setEnabled(not busy)
        self.btn_extract.setEnabled(not busy)
        self.progress.setVisible(busy)
        if msg:
            self.status_lbl.setText(msg)

    def _show_metrics(self, metrics: dict):
        lines = "\n".join(f"  {k:<22} {v}" for k, v in metrics.items())
        self.metrics_view.setText(lines)

    def _on_error(self, msg: str):
        self._set_busy(False, "Ошибка!")
        QMessageBox.critical(self, "Ошибка обработки", msg)

    # ── стили ────────────────────────────────────────────────────────

    @staticmethod
    def _stylesheet() -> str:
        return """
        QMainWindow, QWidget {
            background: #0d1117;
            color: #c9d1d9;
            font-family: "Segoe UI", "Arial";
            font-size: 13px;
        }
        QGroupBox {
            border: 1px solid #2a2d3e;
            border-radius: 7px;
            margin-top: 10px;
            padding-top: 10px;
            color: #8b949e;
            font-weight: bold;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
        }
        QPushButton {
            background: #1c2033;
            color: #c9d1d9;
            border: 1px solid #2a2d3e;
            border-radius: 6px;
            padding: 6px 14px;
        }
        QPushButton:hover  { background: #252940; border-color: #818cf8; }
        QPushButton:pressed { background: #161b2e; }
        QPushButton:disabled { color: #3a3f55; }
        QComboBox, QDoubleSpinBox, QSpinBox {
            background: #161b2e;
            color: #c9d1d9;
            border: 1px solid #2a2d3e;
            border-radius: 5px;
            padding: 4px 8px;
        }
        QComboBox:focus, QDoubleSpinBox:focus, QSpinBox:focus {
            border-color: #818cf8;
        }
        QProgressBar {
            background: #1c2033;
            border: 1px solid #2a2d3e;
            border-radius: 4px;
            height: 8px;
        }
        QProgressBar::chunk { background: #818cf8; border-radius: 4px; }
        QScrollArea { border: none; }
        QLabel { color: #c9d1d9; }
        """
