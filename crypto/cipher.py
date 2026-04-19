"""
Cifrado y descifrado MFSU — axis.crypto.cipher

Formato de archivo .fracta v3:

    ┌──────────────────────────────────────────────────────┐
    │ MAGIC   (6 B)   "MFSUv3"                            │
    │ VERSION (1 B)   0x03                                 │
    │ IV      (16 B)  aleatorio — único por cifrado        │
    │ SALT    (16 B)  KDF salt — único por cifrado         │
    │ MSALT   (16 B)  MAC salt — separado del KDF          │
    │ MAC     (32 B)  HMAC-SHA3-256 (Encrypt-then-MAC)     │
    │ CTEXT   (N B)   XOR keystream + PKCS7                │
    └──────────────────────────────────────────────────────┘
    Header total: 87 bytes

El MAC cubre: IV ‖ SALT ‖ MSALT ‖ CTEXT (Encrypt-then-MAC).
La verificación del MAC usa hmac.compare_digest (tiempo constante).
"""

from __future__ import annotations

import hashlib
import hmac as hmac_mod
import os

import numpy as np

from axis.core import (
    MAGIC, VERSION, IV_LEN, SALT_LEN, MAC_SALT_LEN, MAC_LEN, BLOCK_SIZE, HEADER_LEN,
)
from axis.kdf import derive
from .keystream import generate


# ── Padding PKCS7 ─────────────────────────────────────────────────────────────

def _pkcs7_pad(data: bytes) -> bytes:
    pad_len = BLOCK_SIZE - (len(data) % BLOCK_SIZE)
    return data + bytes([pad_len] * pad_len)


def _pkcs7_unpad(data: bytes) -> bytes:
    if not data:
        raise ValueError("Datos de padding vacíos")
    pad_len = data[-1]
    if pad_len < 1 or pad_len > BLOCK_SIZE:
        raise ValueError("Padding PKCS7 inválido")
    if data[-pad_len:] != bytes([pad_len] * pad_len):
        raise ValueError("Padding PKCS7 corrupto")
    return data[:-pad_len]


# ── Cifrado ───────────────────────────────────────────────────────────────────

def encrypt(data: bytes, password: str) -> bytes:
    """
    Cifra ``data`` con la contraseña dada y devuelve el blob .fracta v3.

    El proceso es:
        1. Generar IV, SALT y MSALT aleatorios.
        2. Derivar material de clave con el KDF memory-hard (96 bytes).
        3. Cifrar con XOR del keystream MFSU + PKCS7.
        4. Calcular HMAC-SHA3-256 sobre (IV ‖ SALT ‖ MSALT ‖ ciphertext).
        5. Ensamblar el blob con el formato .fracta v3.

    Args:
        data:     Plaintext a cifrar (cualquier longitud).
        password: Contraseña en texto plano.

    Returns:
        Blob cifrado en formato .fracta v3.
    """
    iv = os.urandom(IV_LEN)
    salt = os.urandom(SALT_LEN)
    mac_salt = os.urandom(MAC_SALT_LEN)

    # KDF memory-hard → 96 bytes: 64 para cifrado + 32 de base para MAC
    key_material = derive(password, salt, key_len=96)
    enc_key = key_material[:64]
    mac_key = hashlib.sha3_256(key_material[64:] + mac_salt).digest()

    padded = _pkcs7_pad(data)
    ks = generate(enc_key, iv, len(padded))
    ciphertext = (np.frombuffer(padded, dtype=np.uint8) ^ ks).tobytes()

    auth_data = iv + salt + mac_salt + ciphertext
    mac = hmac_mod.new(mac_key, auth_data, hashlib.sha3_256).digest()

    return MAGIC + VERSION + iv + salt + mac_salt + mac + ciphertext


# ── Descifrado ────────────────────────────────────────────────────────────────

def decrypt(blob: bytes, password: str) -> bytes:
    """
    Descifra y verifica la integridad de un blob .fracta v3.

    La verificación del MAC se realiza antes del descifrado (Decrypt-then-verify).
    En caso de MAC inválido el mensaje de error es genérico para no filtrar
    información sobre si la contraseña es correcta o el archivo fue alterado.

    Args:
        blob:     Blob .fracta v3 tal como fue devuelto por ``encrypt``.
        password: Contraseña en texto plano.

    Returns:
        Plaintext original.

    Raises:
        ValueError: Si el blob está truncado, tiene magic/versión incorrectos,
                    o si la verificación del MAC falla.
    """
    if len(blob) < HEADER_LEN + 1:
        raise ValueError("Archivo inválido o truncado")
    if not blob.startswith(MAGIC):
        raise ValueError("No es un archivo .fracta v3")
    if blob[len(MAGIC) : len(MAGIC) + 1] != VERSION:
        ver = blob[len(MAGIC)]
        raise ValueError(f"Versión {ver} no soportada — usa axis-vault v{ver}")

    o = len(MAGIC) + 1
    iv        = blob[o : o + IV_LEN];       o += IV_LEN
    salt      = blob[o : o + SALT_LEN];     o += SALT_LEN
    mac_salt  = blob[o : o + MAC_SALT_LEN]; o += MAC_SALT_LEN
    mac_stored = blob[o : o + MAC_LEN];     o += MAC_LEN
    ctext     = blob[o:]

    if not ctext:
        raise ValueError("Archivo sin cuerpo cifrado")

    key_material = derive(password, salt, key_len=96)
    enc_key = key_material[:64]
    mac_key = hashlib.sha3_256(key_material[64:] + mac_salt).digest()

    auth_data = iv + salt + mac_salt + ctext
    mac_computed = hmac_mod.new(mac_key, auth_data, hashlib.sha3_256).digest()

    # Comparación tiempo-constante — impide timing oracle
    if not hmac_mod.compare_digest(mac_stored, mac_computed):
        raise ValueError(
            "Autenticación fallida — contraseña incorrecta o archivo alterado"
        )

    ks = generate(enc_key, iv, len(ctext))
    padded = (np.frombuffer(ctext, dtype=np.uint8) ^ ks).tobytes()
    return _pkcs7_unpad(padded)
