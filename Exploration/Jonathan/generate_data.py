import numpy as np
import csv

LENGTH = int(1.8e7)

def smooth_noise_generator(length: int, step: float = 0.01, seed: int | None = None):
    """
    Yields (time, value) pairs with smooth noise using exponential smoothing.
    """
    if seed is not None:
        np.random.seed(seed)

    alpha = 0.01  # smoothing factor (lower = smoother)
    value = 0.0

    for i in range(length):
        if i % 1e5 == 0:
            print(f"{round(i / length * 100,2)}%")
        noise = np.random.randn()
        value = alpha * noise + (1 - alpha) * value
        yield i * step, value

def write_csv(filename: str, length: int, chunk_size: int = 1_000_000):
    """
    Write a CSV file with 'length' smooth random points, streamed in chunks.
    """
    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time', 'value'])  # header

        gen = smooth_noise_generator(length)

        chunk = []
        for i, row in enumerate(gen, 1):
            chunk.append(row)
            if i % chunk_size == 0:
                writer.writerows(chunk)
                chunk.clear()
        if chunk:
            writer.writerows(chunk)

if __name__ == "__main__":
    write_csv("smooth_1billion.csv", length=LENGTH)
