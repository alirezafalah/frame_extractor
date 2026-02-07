import argparse
import glob
import os
import sys
from dataclasses import dataclass

import cv2
import numpy as np

from PyQt6.QtCore import Qt, QObject, QThread, pyqtSignal
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
	QApplication, QComboBox, QFileDialog, QGroupBox, QHBoxLayout, QLabel,
	QMainWindow, QPushButton, QProgressBar, QSpinBox, QDoubleSpinBox,
	QStatusBar, QVBoxLayout, QWidget, QSizePolicy, QScrollArea,
)
from PyQt6.QtOpenGLWidgets import QOpenGLWidget

from OpenGL.GL import (
	glBindTexture, glClear, glClearColor, glDeleteTextures, glDisable,
	glEnable, glGenTextures, glTexImage2D, glTexParameteri, glBegin, glEnd,
	glTexCoord2f, glVertex2f, glViewport, glMatrixMode, glLoadIdentity,
	glOrtho, GL_TEXTURE_2D, GL_RGBA, GL_UNSIGNED_BYTE, GL_QUADS,
	GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER, GL_LINEAR, GL_COLOR_BUFFER_BIT,
	GL_PROJECTION, GL_MODELVIEW,
)


@dataclass
class ExtractOptions:
	every_n: int
	fmt: str
	jpg_quality: int
	start_frame: int
	end_frame: int


class GLFrameCanvas(QOpenGLWidget):
	def __init__(self, parent=None):
		super().__init__(parent)
		self._texture_id = None
		self._frame_rgb: np.ndarray | None = None
		self._img_w = 0
		self._img_h = 0
		self._texture_dirty = False

	def set_frame(self, frame_bgr: np.ndarray | None):
		if frame_bgr is None:
			self._frame_rgb = None
			self._img_h = self._img_w = 0
			self._texture_dirty = True
			self.update()
			return
		rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
		self._frame_rgb = rgb
		self._img_h, self._img_w = rgb.shape[:2]
		self._texture_dirty = True
		self.update()

	def initializeGL(self):
		glClearColor(0.05, 0.05, 0.05, 1.0)
		self._texture_id = glGenTextures(1)

	def resizeGL(self, w: int, h: int):
		glViewport(0, 0, w, h)
		glMatrixMode(GL_PROJECTION)
		glLoadIdentity()
		glOrtho(0, w, h, 0, -1, 1)
		glMatrixMode(GL_MODELVIEW)
		glLoadIdentity()

	def paintGL(self):
		glClear(GL_COLOR_BUFFER_BIT)
		if self._frame_rgb is None or self._img_w == 0 or self._img_h == 0:
			return

		if self._texture_dirty:
			glBindTexture(GL_TEXTURE_2D, self._texture_id)
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
			glTexImage2D(
				GL_TEXTURE_2D,
				0,
				GL_RGBA,
				self._img_w,
				self._img_h,
				0,
				GL_RGBA,
				GL_UNSIGNED_BYTE,
				self._rgba_with_alpha(self._frame_rgb),
			)
			glBindTexture(GL_TEXTURE_2D, 0)
			self._texture_dirty = False

		w = self.width()
		h = self.height()
		scale = min(w / self._img_w, h / self._img_h)
		dw = self._img_w * scale
		dh = self._img_h * scale
		ox = (w - dw) * 0.5
		oy = (h - dh) * 0.5

		glEnable(GL_TEXTURE_2D)
		glBindTexture(GL_TEXTURE_2D, self._texture_id)
		glBegin(GL_QUADS)
		glTexCoord2f(0, 0); glVertex2f(ox, oy)
		glTexCoord2f(1, 0); glVertex2f(ox + dw, oy)
		glTexCoord2f(1, 1); glVertex2f(ox + dw, oy + dh)
		glTexCoord2f(0, 1); glVertex2f(ox, oy + dh)
		glEnd()
		glBindTexture(GL_TEXTURE_2D, 0)
		glDisable(GL_TEXTURE_2D)

	def _rgba_with_alpha(self, rgb: np.ndarray) -> np.ndarray:
		if rgb.shape[2] == 4:
			return rgb
		alpha = np.full((rgb.shape[0], rgb.shape[1], 1), 255, dtype=np.uint8)
		return np.concatenate([rgb, alpha], axis=2)

	def cleanup(self):
		if self._texture_id:
			glDeleteTextures([self._texture_id])
			self._texture_id = None


class ExtractWorker(QObject):
	progress = pyqtSignal(int, int)
	finished = pyqtSignal(str)
	error = pyqtSignal(str)

	def __init__(self, video_path: str, output_dir: str, options: ExtractOptions):
		super().__init__()
		self.video_path = video_path
		self.output_dir = output_dir
		self.options = options
		self._stop = False

	def stop(self):
		self._stop = True

	def run(self):
		cap = cv2.VideoCapture(self.video_path)
		if not cap.isOpened():
			self.error.emit(f"Failed to open video: {self.video_path}")
			return

		total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
		start = max(0, self.options.start_frame)
		end = self.options.end_frame
		if total > 0:
			last = total - 1
			end = last if end <= 0 or end > last else end
		if end < start:
			start, end = end, start

		idx = start
		cap.set(cv2.CAP_PROP_POS_FRAMES, start)
		saved = 0
		while idx <= end:
			if self._stop:
				break
			ok, frame = cap.read()
			if not ok:
				break
			if (idx - start) % self.options.every_n == 0:
				if self._save_frame(frame, idx):
					saved += 1
			idx += 1
			if total:
				self.progress.emit(idx, total)
		cap.release()
		if self._stop:
			self.finished.emit(f"Extraction cancelled. Saved {saved} frames to {self.output_dir}")
		else:
			self.finished.emit(f"Saved {saved} frames to {self.output_dir}")

	def _save_frame(self, frame: np.ndarray, idx: int) -> bool:
		os.makedirs(self.output_dir, exist_ok=True)
		ext = ".png" if self.options.fmt == "png" else ".jpg"
		out_path = os.path.join(self.output_dir, f"frame_{idx:06d}{ext}")
		if self.options.fmt == "jpg":
			return cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, self.options.jpg_quality])
		return cv2.imwrite(out_path, frame)


class FrameExtractor(QMainWindow):
	def __init__(self, input_folder: str | None = None, output_folder: str | None = None):
		super().__init__()
		self.setWindowTitle("PixelGrid — Frame Extractor (OpenGL)")
		self.resize(1400, 900)

		self.input_folder = input_folder or os.path.join(os.path.dirname(__file__), "test_data", "videos")
		self.output_folder = output_folder or os.path.join(os.path.dirname(__file__), "test_data", "frames")

		self.video_paths: list[str] = []
		self.current_video_index = -1
		self.current_frame_index = 0
		self.frames_per_step = 20
		self.total_frames = 0
		self.fps = 0.0
		self.cap: cv2.VideoCapture | None = None
		self.current_frame: np.ndarray | None = None

		self._extract_thread: QThread | None = None
		self._extract_worker: ExtractWorker | None = None

		self._build_ui()
		self._load_video_list()
		if self.video_paths:
			self._load_video(0)

	# ---- UI ----------------------------------------------------------------

	def _build_ui(self):
		central = QWidget()
		self.setCentralWidget(central)
		main_layout = QHBoxLayout(central)
		main_layout.setContentsMargins(6, 6, 6, 6)

		left = QWidget()
		left.setFixedWidth(320)
		left.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
		ll = QVBoxLayout(left)
		ll.setSpacing(6)
		ll.setContentsMargins(6, 6, 6, 6)

		scroll = QScrollArea()
		scroll.setWidgetResizable(True)
		scroll.setFrameShape(QScrollArea.Shape.NoFrame)
		scroll_content = QWidget()
		scroll_layout = QVBoxLayout(scroll_content)
		scroll_layout.setSpacing(10)
		scroll_layout.setContentsMargins(0, 0, 0, 0)

		def _polish_button(btn: QPushButton):
			btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
			btn.setMinimumHeight(28)

		def _make_section(title: str, content: QWidget, tight: bool = False) -> QGroupBox:
			box = QGroupBox(title)
			box.setCheckable(True)
			box.setChecked(True)
			box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
			layout = QVBoxLayout(box)
			if tight:
				layout.setContentsMargins(6, 4, 6, 6)
				layout.setSpacing(6)
			layout.addWidget(content)
			content.setVisible(True)
			box.toggled.connect(content.setVisible)
			return box

		sg = QWidget()
		sl = QVBoxLayout(sg)
		sl.setContentsMargins(4, 4, 4, 4)
		sl.setSpacing(6)
		self._input_label = QLabel("")
		self._input_label.setWordWrap(True)
		self._input_label.setStyleSheet("font-size: 10px; color: #aaa;")
		self._output_label = QLabel("")
		self._output_label.setWordWrap(True)
		self._output_label.setStyleSheet("font-size: 10px; color: #aaa;")
		b_in = QPushButton("Open Video Folder...")
		_polish_button(b_in)
		b_in.clicked.connect(self._select_input_folder)
		b_out = QPushButton("Select Output Folder...")
		_polish_button(b_out)
		b_out.clicked.connect(self._select_output_folder)
		sl.addWidget(b_in)
		sl.addWidget(b_out)
		sl.addWidget(self._input_label)
		sl.addWidget(self._output_label)
		section_sources = _make_section("Sources", sg, tight=True)

		vg = QWidget()
		vl = QVBoxLayout(vg)
		self._video_combo = QComboBox()
		self._video_combo.currentIndexChanged.connect(self._on_video_selected)
		self._video_label = QLabel("No video")
		self._video_label.setWordWrap(True)
		self._frame_label = QLabel("Frame: - / -")
		self._step_spin = QSpinBox()
		self._step_spin.setRange(1, 10000)
		self._step_spin.setValue(self.frames_per_step)
		self._step_spin.valueChanged.connect(self._on_step_changed)
		self._time_unit = QComboBox()
		self._time_unit.addItems(["Frames", "Seconds", "Milliseconds"])
		self._time_unit.currentIndexChanged.connect(self._on_time_unit_changed)
		self._step_seconds = QDoubleSpinBox()
		self._step_seconds.setRange(0.001, 3600.0)
		self._step_seconds.setDecimals(3)
		self._step_seconds.setSingleStep(0.1)
		self._step_seconds.setValue(1.0)
		self._step_seconds.valueChanged.connect(self._on_step_seconds_changed)
		self._step_seconds.setEnabled(False)
		vl.addWidget(QLabel("Select Video:"))
		vl.addWidget(self._video_combo)
		vl.addWidget(self._video_label)
		vl.addWidget(self._frame_label)
		vl.addWidget(QLabel("Step unit:"))
		vl.addWidget(self._time_unit)
		vl.addWidget(QLabel("Frames per step:"))
		vl.addWidget(self._step_spin)
		vl.addWidget(QLabel("Time step:"))
		vl.addWidget(self._step_seconds)
		section_video = _make_section("Video", vg)

		ng = QWidget()
		nl = QVBoxLayout(ng)
		row = QHBoxLayout()
		b_prev = QPushButton("◀ Prev Frame")
		b_next = QPushButton("Next Frame ▶")
		_polish_button(b_prev)
		_polish_button(b_next)
		b_prev.clicked.connect(self._prev_frame)
		b_next.clicked.connect(self._next_frame)
		row.addWidget(b_prev)
		row.addWidget(b_next)
		nl.addLayout(row)
		b_prev_v = QPushButton("⏮ Prev Video")
		b_next_v = QPushButton("Next Video ⏭")
		_polish_button(b_prev_v)
		_polish_button(b_next_v)
		b_prev_v.clicked.connect(self._prev_video)
		b_next_v.clicked.connect(self._next_video)
		nl.addWidget(b_prev_v)
		nl.addWidget(b_next_v)
		b_save = QPushButton("Save Current Frame")
		_polish_button(b_save)
		b_save.clicked.connect(self._save_current_frame)
		nl.addWidget(b_save)
		section_nav = _make_section("Navigation", ng)

		eg = QWidget()
		el = QVBoxLayout(eg)
		self._every_spin = QSpinBox()
		self._every_spin.setRange(1, 10000)
		self._every_spin.setValue(30)
		self._range_start = QSpinBox()
		self._range_start.setRange(0, 1_000_000_000)
		self._range_start.setValue(0)
		self._range_end = QSpinBox()
		self._range_end.setRange(0, 1_000_000_000)
		self._range_end.setValue(0)
		self._format_combo = QComboBox()
		self._format_combo.addItems(["PNG (lossless)", "JPG (quality 100)"])
		self._format_combo.currentIndexChanged.connect(self._on_format_changed)
		self._jpg_quality = 100
		self._extract_btn = QPushButton("Extract Every N Frames")
		_polish_button(self._extract_btn)
		self._extract_btn.clicked.connect(self._extract_every_n)
		self._stop_extract = QPushButton("Stop Extraction")
		_polish_button(self._stop_extract)
		self._stop_extract.setEnabled(False)
		self._stop_extract.clicked.connect(self._cancel_extract)
		self._progress = QProgressBar()
		self._progress.setRange(0, 100)
		self._progress.setValue(0)
		el.addWidget(QLabel("Every N frames:"))
		el.addWidget(self._every_spin)
		el.addWidget(QLabel("Start frame (0 = first):"))
		el.addWidget(self._range_start)
		el.addWidget(QLabel("End frame (0 = last):"))
		el.addWidget(self._range_end)
		el.addWidget(QLabel("Save format:"))
		el.addWidget(self._format_combo)
		el.addWidget(self._extract_btn)
		el.addWidget(self._stop_extract)
		el.addWidget(self._progress)
		section_extract = _make_section("Auto Extract", eg)

		scroll_layout.addWidget(section_sources)
		scroll_layout.addWidget(section_video)
		scroll_layout.addWidget(section_nav)
		scroll_layout.addWidget(section_extract)
		scroll_layout.addStretch()
		scroll.setWidget(scroll_content)
		ll.addWidget(scroll)

		self.canvas = GLFrameCanvas(self)
		main_layout.addWidget(left)
		main_layout.addWidget(self.canvas, stretch=1)

		self.setStatusBar(QStatusBar())
		self._update_source_labels()
		self._setup_shortcuts()

	def _setup_shortcuts(self):
		QShortcut(QKeySequence("Left"), self).activated.connect(self._prev_frame)
		QShortcut(QKeySequence("Right"), self).activated.connect(self._next_frame)
		QShortcut(QKeySequence("Up"), self).activated.connect(self._prev_video)
		QShortcut(QKeySequence("Down"), self).activated.connect(self._next_video)
		QShortcut(QKeySequence("S"), self).activated.connect(self._save_current_frame)

	# ---- folders -----------------------------------------------------------

	def _update_source_labels(self):
		self._input_label.setText(f"Input: {self.input_folder}")
		self._output_label.setText(f"Output: {self.output_folder}")

	def _select_input_folder(self):
		folder = QFileDialog.getExistingDirectory(self, "Select Video Folder", self.input_folder)
		if not folder:
			return
		self.input_folder = folder
		self._update_source_labels()
		self._load_video_list()
		if self.video_paths:
			self._load_video(0)
		else:
			self.canvas.set_frame(None)

	def _select_output_folder(self):
		folder = QFileDialog.getExistingDirectory(self, "Select Output Folder", self.output_folder)
		if not folder:
			return
		self.output_folder = folder
		self._update_source_labels()

	# ---- video handling ----------------------------------------------------

	def _load_video_list(self):
		patterns = ["*.mov", "*.MOV", "*.mp4", "*.MP4"]
		paths: list[str] = []
		for p in patterns:
			paths.extend(glob.glob(os.path.join(self.input_folder, p)))
		self.video_paths = sorted(list(dict.fromkeys(paths)))
		self._video_combo.blockSignals(True)
		self._video_combo.clear()
		self._video_combo.addItems([os.path.basename(p) for p in self.video_paths])
		self._video_combo.blockSignals(False)
		self.current_video_index = -1

	def _load_video(self, index: int):
		if not self.video_paths:
			self.statusBar().showMessage("No videos found", 3000)
			return
		index = max(0, min(index, len(self.video_paths) - 1))
		self.current_video_index = index
		self._video_combo.setCurrentIndex(index)
		if self.cap is not None:
			self.cap.release()
		path = self.video_paths[index]
		self.cap = cv2.VideoCapture(path)
		if not self.cap.isOpened():
			self.statusBar().showMessage(f"Failed to open {os.path.basename(path)}", 4000)
			return
		self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
		self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
		self.current_frame_index = 0
		self._read_frame(self.current_frame_index)
		self._update_video_labels()

	def _read_frame(self, idx: int):
		if self.cap is None:
			return
		idx = max(0, min(idx, max(0, self.total_frames - 1)))
		self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
		ok, frame = self.cap.read()
		if not ok:
			self.statusBar().showMessage("Failed to read frame", 2000)
			return
		self.current_frame_index = idx
		self.current_frame = frame
		self.canvas.set_frame(frame)
		self._update_video_labels()

	def _update_video_labels(self):
		if self.current_video_index < 0:
			self._video_label.setText("No video")
			self._frame_label.setText("Frame: - / -")
			return
		name = os.path.basename(self.video_paths[self.current_video_index])
		total = self.total_frames if self.total_frames else "?"
		self._video_label.setText(name)
		if self._time_unit.currentIndex() == 0 or self.fps <= 0:
			self._frame_label.setText(f"Frame: {self.current_frame_index} / {total}")
		elif self._time_unit.currentIndex() == 1:
			cur = self.current_frame_index / self.fps
			end = (self.total_frames - 1) / self.fps if self.total_frames else 0
			self._frame_label.setText(f"Time: {cur:.3f}s / {end:.3f}s")
		else:
			cur = self.current_frame_index / self.fps * 1000.0
			end = (self.total_frames - 1) / self.fps * 1000.0 if self.total_frames else 0
			self._frame_label.setText(f"Time: {cur:.1f}ms / {end:.1f}ms")

	def _on_video_selected(self, index: int):
		if index >= 0:
			self._load_video(index)

	def _on_step_changed(self, v: int):
		self.frames_per_step = v

	def _on_step_seconds_changed(self, v: float):
		# value stored in widget; conversion happens on step
		pass

	def _on_time_unit_changed(self, idx: int):
		is_frames = idx == 0
		self._step_spin.setEnabled(is_frames)
		self._step_seconds.setEnabled(not is_frames)
		self._update_video_labels()

	def _on_format_changed(self):
		# quality fixed to 100 for no visible loss
		self._jpg_quality = 100

	# ---- navigation --------------------------------------------------------

	def _prev_frame(self):
		if self.current_video_index < 0:
			return
		self._step(-1)

	def _next_frame(self):
		if self.current_video_index < 0:
			return
		self._step(1)

	def _step(self, direction: int):
		if self.total_frames <= 0:
			delta = self.frames_per_step if direction > 0 else -self.frames_per_step
			self._read_frame(self.current_frame_index + delta)
			return

		last = self.total_frames - 1
		if self._time_unit.currentIndex() == 0 or self.fps <= 0:
			step = self.frames_per_step
		else:
			seconds = float(self._step_seconds.value())
			if self._time_unit.currentIndex() == 2:
				seconds = seconds / 1000.0
			step = max(1, int(round(seconds * self.fps)))

		if direction > 0:
			if self.current_frame_index == last:
				target = 0
			else:
				target = min(self.current_frame_index + step, last)
		else:
			if self.current_frame_index == 0:
				target = last
			else:
				target = max(self.current_frame_index - step, 0)
		self._read_frame(target)

	def _prev_video(self):
		if self.current_video_index > 0:
			self._load_video(self.current_video_index - 1)

	def _next_video(self):
		if self.current_video_index < len(self.video_paths) - 1:
			self._load_video(self.current_video_index + 1)

	# ---- saving ------------------------------------------------------------

	def _current_video_output_dir(self) -> str:
		if self.current_video_index < 0:
			return self.output_folder
		name = os.path.splitext(os.path.basename(self.video_paths[self.current_video_index]))[0]
		return os.path.join(self.output_folder, name)

	def _save_current_frame(self):
		if self.current_frame is None:
			return
		out_dir = self._current_video_output_dir()
		os.makedirs(out_dir, exist_ok=True)
		fmt = "png" if self._format_combo.currentIndex() == 0 else "jpg"
		ext = ".png" if fmt == "png" else ".jpg"
		out_path = os.path.join(out_dir, f"frame_{self.current_frame_index:06d}{ext}")
		if fmt == "jpg":
			cv2.imwrite(out_path, self.current_frame, [cv2.IMWRITE_JPEG_QUALITY, self._jpg_quality])
		else:
			cv2.imwrite(out_path, self.current_frame)
		self.statusBar().showMessage(f"Saved {os.path.basename(out_path)}", 2000)

	# ---- extraction --------------------------------------------------------

	def _extract_every_n(self):
		if self.current_video_index < 0:
			return
		if self._extract_thread is not None and self._extract_thread.isRunning():
			return
		every_n = int(self._every_spin.value())
		fmt = "png" if self._format_combo.currentIndex() == 0 else "jpg"
		out_dir = self._current_video_output_dir()
		start = int(self._range_start.value())
		end = int(self._range_end.value())
		if self.total_frames > 0:
			last = self.total_frames - 1
			if start <= 0:
				start = 0
			if end <= 0 or end > last:
				end = last
			if start > end:
				start, end = end, start

		options = ExtractOptions(
			every_n=every_n,
			fmt=fmt,
			jpg_quality=self._jpg_quality,
			start_frame=start,
			end_frame=end,
		)
		self._extract_worker = ExtractWorker(self.video_paths[self.current_video_index], out_dir, options)
		self._extract_thread = QThread()
		self._extract_worker.moveToThread(self._extract_thread)
		self._extract_thread.started.connect(self._extract_worker.run)
		self._extract_worker.progress.connect(self._on_extract_progress)
		self._extract_worker.finished.connect(self._on_extract_finished)
		self._extract_worker.error.connect(self._on_extract_error)
		self._extract_worker.finished.connect(self._extract_thread.quit)
		self._extract_worker.finished.connect(self._extract_worker.deleteLater)
		self._extract_thread.finished.connect(self._extract_thread.deleteLater)
		self._progress.setValue(0)
		self._extract_btn.setEnabled(False)
		self._stop_extract.setEnabled(True)
		self.statusBar().showMessage("Extracting frames...")
		self._extract_thread.start()

	def _cancel_extract(self):
		if self._extract_worker is not None:
			self._extract_worker.stop()

	def _on_extract_progress(self, current: int, total: int):
		if total <= 0:
			return
		pct = int(current / total * 100)
		self._progress.setValue(min(100, max(0, pct)))

	def _on_extract_finished(self, msg: str):
		self._progress.setValue(100)
		self.statusBar().showMessage(msg, 4000)
		self._extract_btn.setEnabled(True)
		self._stop_extract.setEnabled(False)

	def _on_extract_error(self, msg: str):
		self.statusBar().showMessage(msg, 5000)
		self._extract_btn.setEnabled(True)
		self._stop_extract.setEnabled(False)

	def closeEvent(self, event):
		if self.cap is not None:
			self.cap.release()
		self.canvas.cleanup()
		super().closeEvent(event)


def main():
	parser = argparse.ArgumentParser(description="OpenGL-accelerated frame extractor")
	parser.add_argument("--input", type=str, default=None, help="Video folder path")
	parser.add_argument("--output", type=str, default=None, help="Output folder path")
	args = parser.parse_args()

	app = QApplication(sys.argv)
	win = FrameExtractor(input_folder=args.input, output_folder=args.output)
	win.show()
	sys.exit(app.exec())


if __name__ == "__main__":
	main()
