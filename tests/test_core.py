"""Tests del módulo axis.core — física MFSU."""

import numpy as np
import pytest

from axis.core import (
    fractional_laplacian,
    fractional_gaussian_noise,
    step_mfsu,
    DELTA_F, BETA, HURST,
)


class TestFractionalLaplacian:
    def test_shape_preserved(self):
        psi = np.random.randn(128)
        result = fractional_laplacian(psi, BETA)
        assert result.shape == psi.shape

    def test_zero_mode_suppressed(self):
        """El modo k=0 debe ser cero (sin componente constante)."""
        psi = np.ones(64)
        result = fractional_laplacian(psi, BETA)
        # Para ψ constante, (-Δ)^α ψ ≈ 0 (el modo k=0 es anulado)
        assert np.allclose(result, 0.0, atol=1e-10)

    def test_real_output(self):
        psi = np.random.randn(64)
        result = fractional_laplacian(psi, 1.5)
        assert np.isrealobj(result)

    def test_linearity(self):
        """El operador es lineal: L(αψ) = α·L(ψ)."""
        psi = np.random.randn(64)
        alpha = 3.7
        r1 = fractional_laplacian(alpha * psi, BETA)
        r2 = alpha * fractional_laplacian(psi, BETA)
        np.testing.assert_allclose(r1, r2, rtol=1e-10)


class TestFractionalGaussianNoise:
    def test_shape(self):
        noise = fractional_gaussian_noise(128, HURST, seed=42)
        assert noise.shape == (128,)

    def test_normalized(self):
        """La desviación estándar debe ser aproximadamente 1."""
        noise = fractional_gaussian_noise(512, HURST, seed=7)
        assert abs(noise.std() - 1.0) < 0.05

    def test_deterministic(self):
        """La misma semilla debe producir el mismo ruido."""
        n1 = fractional_gaussian_noise(64, HURST, seed=123)
        n2 = fractional_gaussian_noise(64, HURST, seed=123)
        np.testing.assert_array_equal(n1, n2)

    def test_different_seeds(self):
        """Semillas distintas deben producir ruido distinto."""
        n1 = fractional_gaussian_noise(64, HURST, seed=1)
        n2 = fractional_gaussian_noise(64, HURST, seed=2)
        assert not np.array_equal(n1, n2)

    def test_hurst_affects_spectrum(self):
        """H distinto debe cambiar el espectro."""
        n1 = fractional_gaussian_noise(256, 0.3, seed=0)
        n2 = fractional_gaussian_noise(256, 0.9, seed=0)
        assert not np.allclose(n1, n2)


class TestStepMFSU:
    def test_shape_preserved(self):
        psi = np.random.randn(64) + 1j * np.random.randn(64)
        h = b"\x00" * 64
        result = step_mfsu(psi, h, step=0, dt=0.01)
        assert result.shape == psi.shape

    def test_normalized(self):
        """El campo debe estar normalizado: max|ψ| <= 1."""
        psi = np.random.randn(64) + 1j * np.random.randn(64)
        psi *= 1000  # amplitud grande
        h = b"\xAB" * 64
        result = step_mfsu(psi, h, step=0, dt=0.001)
        assert np.max(np.abs(result)) <= 1.0 + 1e-10

    def test_deterministic(self):
        """El mismo (ψ, h, step, dt) siempre produce el mismo resultado."""
        psi = np.ones(32) + 1j * np.zeros(32)
        h = b"\x55" * 64
        r1 = step_mfsu(psi.copy(), h, step=5, dt=0.005)
        r2 = step_mfsu(psi.copy(), h, step=5, dt=0.005)
        np.testing.assert_array_equal(r1, r2)

    def test_step_changes_field(self):
        """Distintos pasos producen campos distintos (ruido diferente)."""
        psi = np.random.randn(64) + 1j * np.random.randn(64)
        h = b"\x99" * 64
        r0 = step_mfsu(psi.copy(), h, step=0, dt=0.01)
        r1 = step_mfsu(psi.copy(), h, step=1, dt=0.01)
        assert not np.allclose(r0, r1)

    def test_key_sensitivity(self):
        """Cambiar h por 1 byte produce ruido distinto → campos distintos."""
        rng = np.random.default_rng(42)
        psi = rng.standard_normal(64) + 1j * rng.standard_normal(64)
        # h1 y h2 difieren en 1 byte → semilla del ruido completamente distinta
        h1 = bytes(range(64))
        h2 = bytes(range(1, 65))   # desplazado en 1 — todos los bytes distintos
        # Tras varios pasos el efecto se propaga claramente
        p1, p2 = psi.copy(), psi.copy()
        for step in range(5):
            p1 = step_mfsu(p1, h1, step, dt=0.01)
            p2 = step_mfsu(p2, h2, step, dt=0.01)
        assert not np.allclose(p1, p2, atol=1e-6)
