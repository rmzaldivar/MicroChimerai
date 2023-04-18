import cupy as cp
import numpy as np
from scipy.optimize import basinhopping


def frft(signal, alpha):
    n = len(signal)
    indices = cp.arange(n)
    M = cp.exp(-1j * cp.pi * alpha * (indices ** 2) / n)
    return cp.fft.ifft(cp.fft.fft(signal) * M)


def mehler_kernel(X, Y, beta=1.0):
    inner_product = cp.sum(X * cp.conj(Y))
    exponent = -beta * cp.abs(inner_product) ** 2
    return cp.exp(exponent)


def cross_correlation(X, Y):
    return cp.abs(cp.sum(cp.conj(X) * Y))


def optimize_alpha(signal1_np, signal2_np, initial_alpha):
    signal1_gpu = cp.array(signal1_np, dtype=cp.float32)
    signal2_gpu = cp.array(signal2_np, dtype=cp.float32)

    def objective_function(alpha):
        X_alpha = frft(signal1_gpu, alpha)
        Y_alpha = frft(signal2_gpu, alpha)
        return -mehler_kernel(X_alpha, Y_alpha).real

    result = basinhopping(objective_function, initial_alpha, niter=50, stepsize=1)
    return result.x[0]


def generate_test_signals_padded(n, alpha, list_length=225):
    base_signal = np.random.uniform(0, 1, list_length)
    transformed_signal = frft(cp.array(base_signal), alpha).get().real

    base_signal_padded = np.pad(base_signal, (0, n - list_length))
    transformed_signal_padded = np.pad(transformed_signal, (0, n - list_length))

    return base_signal_padded, transformed_signal_padded


def generate_batch_data_padded(batch_size, n, list_length=225, alpha_low=0, alpha_high=2):
    signals1 = []
    signals2 = []
    true_alphas = []

    for _ in range(batch_size):
        alpha = np.random.uniform(alpha_low, alpha_high)
        signal1, signal2 = generate_test_signals_padded(n, alpha, list_length)
        signals1.append(signal1)
        signals2.append(signal2)
        true_alphas.append(alpha)

    return signals1, signals2, true_alphas


n = 65536
batch_size = 10
signals1, signals2, true_alphas = generate_batch_data_padded(batch_size, n)

initial_alpha = np.random.uniform(0, 2)
estimated_alphas = []
for i in range(batch_size):
    signal1_np = np.array(signals1[i], dtype=np.float32)
    signal2_np = np.array(signals2[i], dtype=np.float32)
    estimated_alpha = optimize_alpha(signal1_np, signal2_np, initial_alpha)
    estimated_alphas.append(estimated_alpha)

    print(f"Pair {i + 1}:")
    print("Known alpha:", true_alphas[i])
    print("Estimated alpha:", estimated_alpha)
    print("Absolute error:", abs(true_alphas[i] - estimated_alpha))
    print()
