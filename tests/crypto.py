"""Tests del módulo axis.crypto — cifrado, descifrado y keystream."""

import os
import numpy as np
import pytest
from scipy.stats import chisquare

from axis.crypto import encrypt, decrypt, keystream
from axis.core import MAGIC, VERSION, HEADER_LEN


class TestKeystream:
    def test_length(self):
        enc_key = os.urandom(64)
        iv = os.urandom(16)
        ks = keystream(enc_key, iv, 1024)
        assert len(ks) == 1024

    def test_dtype(self):
        ks = keystream(os.urandom(64), os.urandom(16), 64)
        assert ks.dtype == np.uint8

    def test_deterministic(self):
        enc_key = b"K" * 64
        iv = b"I" * 16
        ks1 = keystream(enc_key, iv, 256)
        ks2 = keystream(enc_key, iv, 256)
        np.testing.assert_array_equal(ks1, ks2)

    def test_iv_sensitivity(self):
        """IVs distintos → keystreams completamente distintos."""
        enc_key = b"K" * 64
        ks1 = keystream(enc_key, b"I" * 16, 256)
        ks2 = keystream(enc_key, b"J" * 16, 256)
        assert not np.array_equal(ks1, ks2)
        diff_bits = np.sum(np.unpackbits(ks1[:64]) != np.unpackbits(ks2[:64]))
        assert diff_bits > 200  # >39% de bits distintos

    def test_key_sensitivity(self):
        """Claves distintas → keystreams completamente distintos."""
        iv = b"I" * 16
        ks1 = keystream(b"A" * 64, iv, 256)
        ks2 = keystream(b"B" * 64, iv, 256)
        assert not np.array_equal(ks1, ks2)

    def test_uniform_distribution(self):
        """Keystream debe tener distribución uniforme de bytes."""
        ks = keystream(os.urandom(64), os.urandom(16), 4096)
        counts = np.bincount(ks, minlength=256)
        _, p = chisquare(counts)
        assert p > 0.01, f"Distribución no uniforme: p={p:.4f}"

    def test_no_autocorrelation(self):
        """Autocorrelación de Pearson debe ser < 0.05 para todos los lags."""
        ks = keystream(os.urandom(64), os.urandom(16), 4096).astype(float)
        kc = ks - ks.mean()
        var = np.var(ks)
        for lag in range(1, 20):
            r = np.mean(kc[lag:] * kc[:-lag]) / var
            assert abs(r) < 0.05, f"Autocorrelación alta en lag={lag}: r={r:.4f}"

    def test_short_key_raises(self):
        with pytest.raises(ValueError, match="enc_key"):
            keystream(b"short", os.urandom(16), 64)

    def test_short_iv_raises(self):
        with pytest.raises(ValueError, match="iv"):
            keystream(os.urandom(64), b"x", 64)


class TestEncryptDecrypt:
    def test_round_trip_basic(self):
        data = b"Hola MFSU!"
        blob = encrypt(data, "password")
        assert decrypt(blob, "password") == data

    def test_round_trip_empty(self):
        data = b""
        blob = encrypt(data, "password")
        assert decrypt(blob, "password") == data

    def test_round_trip_large(self):
        data = os.urandom(10_000)
        blob = encrypt(data, "password123")
        assert decrypt(blob, "password123") == data

    def test_round_trip_unicode_password(self):
        data = b"secret"
        blob = encrypt(data, "contraseña_🔐")
        assert decrypt(blob, "contraseña_🔐") == data

    def test_blob_starts_with_magic(self):
        blob = encrypt(b"test", "pwd")
        assert blob.startswith(MAGIC + VERSION)

    def test_blob_minimum_size(self):
        blob = encrypt(b"x", "pwd")
        assert len(blob) >= HEADER_LEN + 16  # header + al menos 1 bloque PKCS7

    def test_wrong_password_raises(self):
        blob = encrypt(b"secret", "correct")
        with pytest.raises(ValueError, match="Autenticación fallida"):
            decrypt(blob, "wrong")

    def test_tampered_ciphertext_raises(self):
        blob = encrypt(b"secret", "password")
        tampered = bytearray(blob)
        tampered[-1] ^= 0xFF  # alterar último byte del ciphertext
        with pytest.raises(ValueError, match="Autenticación fallida"):
            decrypt(bytes(tampered), "password")

    def test_tampered_header_raises(self):
        blob = encrypt(b"secret", "password")
        tampered = bytearray(blob)
        tampered[10] ^= 0x01  # alterar 1 byte del IV
        with pytest.raises(ValueError, match="Autenticación fallida"):
            decrypt(bytes(tampered), "password")

    def test_truncated_raises(self):
        with pytest.raises(ValueError):
            decrypt(b"MFSUv3\x03" + b"\x00" * 20, "password")

    def test_wrong_magic_raises(self):
        with pytest.raises(ValueError, match=".fracta"):
            decrypt(b"WRONGMAGIC" + b"\x00" * 100, "password")

    def test_unique_blobs(self):
        """Cada cifrado produce un blob distinto (IV y salts aleatorios)."""
        data = b"mismo contenido"
        blob1 = encrypt(data, "password")
        blob2 = encrypt(data, "password")
        assert blob1 != blob2

    def test_avalanche_on_password(self):
        """Contraseñas distintas → ciphertexts completamente distintos."""
        data = b"A" * 100
        b1 = encrypt(data, "password1")
        b2 = encrypt(data, "password2")
        # Comparar solo el ciphertext (después del header)
        ct1 = np.frombuffer(b1[HEADER_LEN:], dtype=np.uint8)
        ct2 = np.frombuffer(b2[HEADER_LEN:], dtype=np.uint8)
        diff = np.sum(ct1 != ct2) / len(ct1)
        # Con salts distintos los keystreams son distintos → ~50% bytes distintos
        assert diff > 0.3
