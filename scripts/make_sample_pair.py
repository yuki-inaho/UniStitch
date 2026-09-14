"""Generate the bundled demo pair (samples/left.png, samples/right.png).

The pair is a deterministic procedural scene and a perspective-shifted view of
it, so the gradio app and the tests have a self-contained, license-free sample.
Regenerate with::

    pixi run python scripts/make_sample_pair.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"
WIDTH, HEIGHT = 720, 540


def make_scene(rng: np.random.Generator) -> np.ndarray:
    image = np.zeros((HEIGHT, WIDTH, 3), np.float32)

    sky = np.linspace(0.85, 1.0, HEIGHT)[:, None]
    image[:, :, 0] = 0.95 * sky  # B
    image[:, :, 1] = 0.75 * sky  # G
    image[:, :, 2] = 0.45 * sky  # R
    horizon = int(HEIGHT * 0.55)
    ground = np.linspace(0.9, 0.55, HEIGHT - horizon)[:, None]
    image[horizon:, :, 0] = 0.45 * ground
    image[horizon:, :, 1] = 0.75 * ground
    image[horizon:, :, 2] = 0.35 * ground

    # "buildings" with textured windows
    for x in range(40, WIDTH - 80, 120):
        top = horizon - rng.integers(60, 150)
        color = rng.uniform(0.4, 0.8, 3)
        cv2.rectangle(image, (x, top), (x + 90, horizon), tuple(float(c) for c in color), -1)
        for wy in range(top + 12, horizon - 10, 22):
            for wx in range(x + 10, x + 80, 20):
                if rng.random() > 0.25:
                    cv2.rectangle(image, (wx, wy), (wx + 10, wy + 12), (0.95, 0.9, 0.6), -1)

    # "trees"
    for _ in range(24):
        cx = int(rng.integers(30, WIDTH - 30))
        cy = int(rng.integers(horizon - 20, HEIGHT - 60))
        radius = int(rng.integers(14, 34))
        cv2.circle(image, (cx, cy), radius, tuple(float(c) for c in rng.uniform(0.15, 0.45, 3)), -1)

    # texture so the keypoint matcher has plenty of corners
    for _ in range(140):
        x0, y0 = (int(v) for v in rng.integers(0, WIDTH - 20, 2))
        x1 = x0 + int(rng.integers(6, 60))
        y1 = y0 + int(rng.integers(4, 40))
        color = tuple(float(c) for c in rng.uniform(0.0, 1.0, 3))
        cv2.rectangle(image, (x0, y0), (x1, y1), color, int(rng.integers(1, 3)))

    noise = rng.normal(0, 6, image.shape[:2])[:, :, None]
    return np.clip(image * 255 + noise, 0, 255).astype(np.uint8)


def main() -> None:
    rng = np.random.default_rng(20260914)
    scene = make_scene(rng)

    shift = 60.0
    matrix = np.array(
        [
            [1.0, 0.0, -shift],
            [0.02, 1.0, -0.06 * shift],
            [4e-5, 1e-5, 1.0],
        ],
        dtype=np.float64,
    )
    shifted = cv2.warpPerspective(scene, matrix, (WIDTH, HEIGHT), borderMode=cv2.BORDER_REPLICATE)

    left = scene[:, : int(WIDTH * 0.78)]
    right = shifted[:, int(WIDTH * 0.22) :]

    SAMPLES.mkdir(exist_ok=True)
    cv2.imwrite(str(SAMPLES / "left.png"), left)
    cv2.imwrite(str(SAMPLES / "right.png"), right)
    print(f"wrote {SAMPLES / 'left.png'} {left.shape[1]}x{left.shape[0]}")
    print(f"wrote {SAMPLES / 'right.png'} {right.shape[1]}x{right.shape[0]}")


if __name__ == "__main__":
    main()
