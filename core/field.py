"""
Núcleo MFSU — operadores fraccionales y paso de integración de la SPDE.

Este módulo implementa la física del campo ψ(x,t) y no tiene dependencias
de seguridad ni de UI. Es el único lugar donde vive la matemática MFSU.

Funciones públicas:
    fractional_laplacian(psi, alpha)  → operador (-Δ)^(α/2) vía FFT
    fractional_gaussian_noise(n, h, seed) → ruido η(x,t) con H=0.541
    step_mfsu(psi, h_bytes, step, dt) → un paso de Euler de la SPDE
"""

from __future__ import annotations

import numpy as np
from scipy.fft import fft, ifft, fftfreq

from .constants import DELTA_F, BETA, HURST, GAMMA_NL, SIGMA_ETA


# ── Operador laplaciano fraccional ────────────────────────────────────────────

def fractional_laplacian(psi: np.ndarray, alpha: float) -> np.ndarray:
    """
    Calcula (-Δ)^(α/2) ψ mediante diagonalización espectral (FFT).

    Definición:
        F[(-Δ)^(α/2) f](k) = |k|^α · F[f](k)

    El modo k=0 se fuerza a cero para eliminar la componente constante
    (equivalente a imponer condición de frontera periódica con media cero).

    Args:
        psi:   Campo unidimensional real o complejo.
        alpha: Exponente del operador (β = 1.079 en el MFSU estándar).

    Returns:
        Parte real de la transformada inversa: (-Δ)^(α/2) ψ.
    """
    n = len(psi)
    k = fftfreq(n, d=1.0 / n) * 2.0 * np.pi
    k_alpha = np.abs(k) ** alpha
    k_alpha[0] = 0.0
    return np.real(ifft(k_alpha * fft(psi)))


# ── Ruido gaussiano fraccional ────────────────────────────────────────────────

def fractional_gaussian_noise(n: int, hurst: float, seed: int) -> np.ndarray:
    """
    Genera ruido gaussiano fraccional η(x,t) con exponente de Hurst H.

    El espectro de potencia sigue S(k) ~ |k|^(-(2H+1)), que para H=0.541
    reproduce la firma estadística del fondo cósmico de microondas (CMB).

    El ruido se normaliza a desviación estándar 1 para que SIGMA_ETA
    controle su intensidad de forma predecible.

    Args:
        n:     Número de puntos del campo.
        hurst: Exponente de Hurst (0 < H < 1). H=0.5 → ruido blanco.
        seed:  Semilla determinista derivada de (clave, paso).

    Returns:
        Array de longitud n con el ruido normalizado.
    """
    rng = np.random.default_rng(seed & 0xFFFF_FFFF)
    k = fftfreq(n, d=1.0 / n)
    k[0] = 1.0  # evita división por cero; luego se anula la potencia en k=0
    power = np.abs(k) ** (-(2.0 * hurst + 1.0) / 2.0)
    power[0] = 0.0
    noise_complex = power * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    noise = np.real(ifft(noise_complex))
    std = noise.std()
    return noise / std if std > 0.0 else noise


# ── Paso de integración de la SPDE ───────────────────────────────────────────

def step_mfsu(
    psi: np.ndarray,
    h_bytes: bytes,
    step: int,
    dt: float,
) -> np.ndarray:
    """
    Integra un paso de Euler de la SPDE fractal:

        ψ_{n+1} = ψ_n + dt · [−δF·(-Δ)^(β/2)·ψ + γ|ψ|²ψ + σ·η]

    La semilla del ruido en cada paso se deriva de h_bytes y del número de
    paso, de modo que la trayectoria es completamente determinista dado el
    estado inicial (y por tanto, dado el material de clave).

    Normalización tiempo-constante:
        Se divide siempre por max(|ψ|, 1). No hay branch condicional:
        si max < 1 la división es inocua; si max > 1 evita divergencia.
        Esto elimina el timing-leak que introduciría un 'if max_mod > 1'.

    Args:
        psi:     Campo complejo actual, shape (N,).
        h_bytes: Hash de 64 bytes que ancla la trayectoria a la clave.
        step:    Número de paso actual (para derivar la semilla del ruido).
        dt:      Paso temporal de integración.

    Returns:
        Nuevo estado del campo ψ_{n+1}, shape (N,).
    """
    # Semilla derivada del hash y del paso — cada paso tiene ruido distinto
    seed_s = (
        int.from_bytes(h_bytes[(step * 7) % 56 : (step * 7) % 56 + 8], "big")
        ^ (step * 0x9E3779B97F4A7C15)
    )

    eta = fractional_gaussian_noise(len(psi), HURST, seed_s)

    frac_r = fractional_laplacian(np.real(psi), BETA)
    frac_i = fractional_laplacian(np.imag(psi), BETA)
    diffusion = -DELTA_F * (frac_r + 1j * frac_i)
    nonlinear = GAMMA_NL * (np.abs(psi) ** 2) * psi
    noise_term = SIGMA_ETA * eta

    psi = psi + dt * (diffusion + nonlinear + noise_term)

    # Normalización tiempo-constante — sin branch
    max_mod = max(float(np.max(np.abs(psi))), 1.0)
    return psi / max_mod
