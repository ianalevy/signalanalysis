from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
import pyqtgraph as pg
from scipy import signal


def frame_array(data: np.ndarray, frame_length: int) -> np.ndarray:
    """Frame data as a matrix.

    Parameters
    ----------
    data : np.ndarray
        _description_
    frame_length : int
        _description_

    Returns
    -------
    np.ndarray

    """
    _, remainder = (int(_) for _ in divmod(len(data), frame_length))
    if remainder != 0:
        missing_data = np.zeros(frame_length - remainder, dtype=data.dtype)
        data = np.append(data, missing_data)

    num_frames = len(data) / frame_length

    return np.stack(np.array_split(data, num_frames))


def save_as_1000(data, fp, inputs):
    header = bluefile.header(type=1001, format="SB", xunits=1, xdelta=0.01)
    bluefile.write(fp, header, data)


def save_as_2000(data, fp, frame_length, inputs):
    header = bluefile.header(
        type=2000,
        format="SB",
        xunits=1,
        xdelta=0.01,
        subsize=frame_length,
    )
    bluefiile.write(fp, header, frame_array(data, frame_length))


def generate_noise(
    num_samples: int,
    mean: float = 0,
    var: float = 1,
    rng=np.random.default_rng(),
) -> np.ndarray:
    """Generate white noise.

    Parameters
    ----------
    num_samples : int
        _description_
    mean : float, optional
        _description_, by default 0
    var : float, optional
        _description_, by default 1
    rng : _type_, optional
        _description_, by default np.random.default_rng()

    Returns
    -------
    np.ndarray
        _description_

    """
    return rng.normal(mean, var, size=num_samples)


def plotter(win: pg.GraphicsLayout, y: np.ndarray, x: np.ndarray | None = None) -> None:
    """Plot data using pyqtplot.

    Parameters
    ----------
    y : np.ndarray
        _description_
    x : np.ndarray | None, optional
        _description_, by default None
    win : pg.GraphicsLayout | None, optional
        _description_, by default None

    """
    app = pg.mkQApp("Plotting Example")

    win.resize(1000, 600)
    win.setWindowTitle("pyqtgraph example: Plotting")

    pg.setConfigOptions(antialias=True)
    if x is None:
        p1 = win.addPlot(title="Basic array plotting", y=y)
    else:
        p1 = win.addPlot(title="Basic array plotting", y=y, x=x)
    p1.setMouseEnabled(x=True, y=False)


@dataclass
class Pulse:
    analog_shape: callable = np.sinc
    min: float = -10
    max: float = 10

    def __post_init__(self):
        self.hi = 5

    def sample_pulse(self, sample_rate: float):
        num_samples = int((self.max - self.min) / sample_rate)
        x_pts = np.linspace(self.min, self.max, num_samples)
        return self.analog_shape(x_pts)


def make_signal(
    pri_s: float,
    sample_rate_s: float,
    num_pulses: float,
    pw_s: float | None = None,
    duty_cycle: float = 0.001,
    snr: float = 100,
) -> np.ndarray:
    if pw_s is None:
        pw_s = pri_s * duty_cycle

    start = 0
    stop = pri_s * num_pulses + pw_s
    data_times = np.arange(start=start, stop=stop, step=sample_rate_s)
    signal = np.zeros_like(data_times)

    pulse_indices = []
    pulse_starts = pri_s * np.array(list(range(num_pulses)))
    for p_start in pulse_starts:
        start_index = int(p_start / sample_rate_s)
        stop_index = start_index + int(pw_s / sample_rate_s) + 1
        new_indices = list(range(start_index, stop_index))
        pulse_indices.append(new_indices)

        signal[new_indices] = snr

    return (data_times, signal)


def calc_norm(data: np.ndarray) -> float:
    """Norm of data.

    Sum column vectors.

    Parameters
    ----------
    data : np.ndarray

    Returns
    -------
    float

    """
    sums = np.sum(data, axis=0)
    return np.max(np.abs(sums)) / data.shape[1]


def detector(
    data: np.ndarray,
    sample_rate_s: float,
    pw_s: float,
    threshold: float = 10,
) -> np.array:
    """Detect signal.

    Parameters
    ----------
    data : np.ndarray
        _description_
    sample_rate_s : float
        _description_
    pw_s : float
        _description_
    threshold : float, optional
        _description_, by default 10

    Returns
    -------
    np.array
        _description_

    """
    width = int(pw_s / sample_rate_s) + 1
    b_filter = [1 for _ in range(width)]
    detects = signal.lfilter(b=b_filter, a=[1], x=data)

    return np.where(detects >= 400, 1, 0)


def try_pris(data: np.ndarray, sample_rate_s) -> tuple:
    min_pri = 2 * sample_rate_s
    max_pri = 1

    pris_to_test = list(np.arange(min_pri, max_pri, 0.05))
    frame_lengths = (np.array(pris_to_test) / sample_rate_s).astype(int)

    results = [calc_norm(frame_array(data, _)) for _ in frame_lengths]

    return (pris_to_test, results)



@dataclass
class Pdw:
    toa: np.array = fields(default_factory=lambda: np.array([]))
    pw: np.array = fields(default_factory=lambda: np.array([]))
    rf: np.array = fields(default_factory=lambda: np.array([]))


if __name__ == "__main__":
    # data = generate_noise(1000)
    sample_rate_s = 0.0001
    pw_s = 0.002

    (times, data) = make_signal(
        pri_s=0.45,
        sample_rate_s=sample_rate_s,
        num_pulses=8,
        pw_s=pw_s,
    )

    noise = generate_noise(len(data))
    data = data + noise

    detects = detector(data, sample_rate_s, pw_s)

    (pris, norms) = try_pris(detects, sample_rate_s)

    # (times, signal) = make_signal(
    #     pri_s=0.1,
    #     sample_rate_s=0.0001,
    #     num_pulses=12,
    #     pw_s=0.001,
    # )

    win = pg.GraphicsLayoutWidget(show=True, title="Maximum Likelihood")
    win.resize(2000, 2000)
    win.setWindowTitle("PRI estimator")

    p1 = win.addPlot(title="data", y=data, x=times)
    p1.setMouseEnabled(x=True, y=False)
    win.nextRow()
    p2 = win.addPlot(title="detector", y=detects, x=times)
    p2.setMouseEnabled(x=True, y=False)
    win.nextRow()
    p2 = win.addPlot(title="PRI vs nom", y=norms, x=pris)
    p2.setMouseEnabled(x=True, y=False)

    pg.setConfigOptions(antialias=True)

    pg.exec()
