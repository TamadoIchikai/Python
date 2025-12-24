import sys
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QGridLayout,
    QSlider, QLabel
)
from PyQt6.QtCore import Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

# -----------------------------
# Simulation parameters
# -----------------------------
dt = 0.001
T  = 5.0
t  = np.arange(0, T, dt)

# Step input 0 -> 1
r = np.ones_like(t)

# Plant parameters (fixed)
wn   = 2.0    # natural frequency
zeta = 0.15   # low damping → overshoot

# -----------------------------
# PID + second-order plant
# -----------------------------
def simulate_pid(Kp, Ki, Kd):
    y = np.zeros_like(t)
    y_dot = 0.0

    ei = 0.0
    e_prev = 0.0

    for k in range(len(t) - 1):
        e = r[k] - y[k]
        ei += e * dt
        ed = (e - e_prev) / dt

        u = Kp * e + Ki * ei + Kd * ed

        # second-order dynamics
        y_ddot = u - 2*zeta*wn*y_dot - wn**2*y[k]

        y_dot += y_ddot * dt
        y[k+1] = y[k] + y_dot * dt

        e_prev = e

    return y

# -----------------------------
# Matplotlib canvas
# -----------------------------
class MplCanvas(FigureCanvasQTAgg):
    def __init__(self):
        fig = Figure(figsize=(9, 7))
        self.ax_y    = fig.add_subplot(2, 2, 1)
        self.ax_iae  = fig.add_subplot(2, 2, 2)
        self.ax_ise  = fig.add_subplot(2, 2, 3)
        self.ax_itae = fig.add_subplot(2, 2, 4)
        super().__init__(fig)

# -----------------------------
# Main Window
# -----------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PID Step Response (0 → 1)")

        self.canvas = MplCanvas()

        slider_cfg = {
            "Kp": (0, 3000, 500),
            "Ki": (0, 500, 50),
            "Kd": (0, 300, 20),
        }

        self.sliders = {}
        self.labels  = {}

        slider_layout = QGridLayout()

        for i, (name, (mn, mx, init)) in enumerate(slider_cfg.items()):
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setMinimum(mn)
            slider.setMaximum(mx)
            slider.setValue(init)
            slider.valueChanged.connect(self.update_plots)

            label = QLabel()
            slider_layout.addWidget(QLabel(name), i, 0)
            slider_layout.addWidget(slider, i, 1)
            slider_layout.addWidget(label, i, 2)

            self.sliders[name] = slider
            self.labels[name]  = label

        layout = QVBoxLayout()
        layout.addLayout(slider_layout)
        layout.addWidget(self.canvas)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.update_plots()

    def update_plots(self):
        Kp = self.sliders["Kp"].value()
        Ki = self.sliders["Ki"].value()
        Kd = self.sliders["Kd"].value()

        self.labels["Kp"].setText(str(Kp))
        self.labels["Ki"].setText(str(Ki))
        self.labels["Kd"].setText(str(Kd))

        y = simulate_pid(Kp, Ki, Kd)
        e = r - y

        iae  = np.cumsum(np.abs(e)) * dt
        ise  = np.cumsum(e**2) * dt
        itae = np.cumsum(t * np.abs(e)) * dt

        for ax in (
            self.canvas.ax_y,
            self.canvas.ax_iae,
            self.canvas.ax_ise,
            self.canvas.ax_itae
        ):
            ax.clear()

        # Output
        self.canvas.ax_y.plot(t, y, label="y(t)")
        self.canvas.ax_y.plot(t, r, "k--", label="r(t)")
        self.canvas.ax_y.set_title("Step Response")
        self.canvas.ax_y.legend()
        self.canvas.ax_y.grid(True)

        self.canvas.ax_iae.plot(t, iae)
        self.canvas.ax_iae.set_title("IAE(t)")
        self.canvas.ax_iae.grid(True)

        self.canvas.ax_ise.plot(t, ise)
        self.canvas.ax_ise.set_title("ISE(t)")
        self.canvas.ax_ise.grid(True)

        self.canvas.ax_itae.plot(t, itae)
        self.canvas.ax_itae.set_title("ITAE(t)")
        self.canvas.ax_itae.grid(True)

        self.canvas.draw()

# -----------------------------
# Run
# -----------------------------
app = QApplication(sys.argv)
window = MainWindow()
window.resize(1100, 800)
window.show()
sys.exit(app.exec())

