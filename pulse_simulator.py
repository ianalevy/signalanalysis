import numpy as np


def frame_array(data: np.ndarray, frame_length: int):
    quotient, remainder = (int(_) for _ in divmod(len(data), frame_length))

    if remainder != 0:
        missing_data = np.zeros(quotient, dtype=data.dtype)
        data = np.append(data, missing_data)

    num_frames = len(data) / frame_length

    return np.array_split(data, num_frames)


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


if __name__ == "__main__":
    print("hi")
