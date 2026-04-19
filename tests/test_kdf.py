"""Tests del módulo axis.kdf — KDF memory-hard fractal."""

import os
import pytest

from axis.kdf import derive


class TestDerive:
    def test_output_length_default(self):
        key = derive("password", b"saltsalt12345678")
        assert len(key) == 96

    def test_output_length_custom(self):
        for length in [16, 32, 64, 96, 128]:
            key = derive("password", b"saltsalt12345678", key_len=length)
            assert len(key) == length

    def test_deterministic(self):
        """La misma (contraseña, salt) siempre produce la misma clave."""
        salt = b"A" * 16
        k1 = derive("my_pass", salt)
        k2 = derive("my_pass", salt)
        assert k1 == k2

    def test_password_sensitivity(self):
        """Cambiar 1 carácter de la contraseña cambia completamente la clave."""
        salt = b"B" * 16
        k1 = derive("password", salt)
        k2 = derive("Password", salt)
        assert k1 != k2
        # Verificar efecto avalanche: >30% de bytes distintos
        diff = sum(a != b for a, b in zip(k1, k2))
        assert diff > len(k1) * 0.3

    def test_salt_sensitivity(self):
        """Salts distintos producen claves completamente distintas."""
        k1 = derive("password", b"A" * 16)
        k2 = derive("password", b"B" * 16)
        assert k1 != k2

    def test_unique_per_call_with_random_salt(self):
        """Con salts aleatorios, cada llamada produce clave distinta."""
        k1 = derive("password", os.urandom(16))
        k2 = derive("password", os.urandom(16))
        assert k1 != k2

    def test_invalid_key_len(self):
        with pytest.raises(ValueError, match="key_len"):
            derive("password", b"A" * 16, key_len=0)

    def test_empty_salt_raises(self):
        with pytest.raises(ValueError, match="salt"):
            derive("password", b"")

    def test_empty_password(self):
        """Contraseña vacía es válida (el KDF la acepta)."""
        key = derive("", b"A" * 16)
        assert len(key) == 96

    def test_unicode_password(self):
        """Contraseñas con caracteres unicode deben funcionar."""
        key = derive("パスワード🔑", b"A" * 16)
        assert len(key) == 96
