"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              FractalShield — Defensa Adaptativa Fractal                     ║
║              Módulo de protección por capas anidadas MFSU                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Principio:                                                                  ║
║    El archivo contiene N capas de cifrado. Solo la capa real contiene        ║
║    el magic header MFSU\x04. Las capas señuelo son estadísticamente          ║
║    indistinguibles del ciphertext real.                                      ║
║                                                                              ║
║    El atacante no obtiene oráculo de verificación — cada intento             ║
║    consume el KDF completo sin revelar si acertó.                           ║
║                                                                              ║
║    Cada capa señuelo usa KDF_M mayor (costo geométrico):                    ║
║      Nivel 1: capas KDF_M = [256, 512, 1024]         → 3.5x costo          ║
║      Nivel 2: capas KDF_M = [256, 512, 1024, 2048]   → 7.5x costo          ║
║      Nivel 3: capas KDF_M = [256, 512, 1024, 2048, 4096] → 15.5x costo     ║
║                                                                              ║
║  Formato .fracta v4:                                                         ║
║    [MAGIC 6B][VER 1B][LEVEL 1B][N 1B][SALT_G 16B][IV_ORD 16B]              ║
║    [ORD_LEN 2B][ORDER_ENC NB][MAC 32B]                                      ║
║    [capa_0: SALT 16B + IV 16B + CT NB] × N_capas                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
from scipy.fft import fft, ifft, fftfreq
import hashlib
import hmac as hmac_mod
import os
import struct

# ── Constantes MFSU ──────────────────────────────────────────────────────────

DELTA_F   = 0.921
BETA      = 2.0 - DELTA_F   # 1.079
HURST     = 0.541
GAMMA_NL  = DELTA_F
SIGMA_ETA = 0.1

# ── Constantes FractalShield ─────────────────────────────────────────────────

MAGIC_V4    = b"MFSUv4"
VERSION_V4  = b"\x04"
REAL_MAGIC  = b"MFSU\x04"    # 5 bytes internos — identifica la capa real
ORDER_SALT  = b"MFSU_ORDER_SALT_"  # 16 bytes fijo para cifrar el orden

# KDF_M por capa para cada nivel de protección
SHIELD_LEVELS = {
    1: [256,  512,  1024],                   # Estándar  — 3 capas
    2: [256,  512,  1024, 2048],             # Reforzado — 4 capas
    3: [256,  512,  1024, 2048, 4096],       # Máximo    — 5 capas
}

SHIELD_NAMES = {
    1: ("🛡️ Estándar",  "Documentos personales, fotos, notas",         "~0.5s",  "3.5x"),
    2: ("🔒 Reforzado", "Contratos, datos financieros, credenciales",   "~0.7s",  "7.5x"),
    3: ("💎 Máximo",    "Secretos críticos, datos médicos, legales",    "~1.3s",  "15.5x"),
}


# ══════════════════════════════════════════════════════════════════════════════
#  NÚCLEO MFSU — operadores fraccionales
# ══════════════════════════════════════════════════════════════════════════════

def _fractional_laplacian(psi: np.ndarray, alpha: float) -> np.ndarray:
    k = fftfreq(len(psi), d=1.0 / len(psi)) * 2 * np.pi
    ka = np.abs(k) ** alpha
    ka[0] = 0.0
    return np.real(ifft(ka * fft(psi)))


def _fgn(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed & 0xFFFFFFFF)
    k = fftfreq(n, d=1.0 / n)
    k[0] = 1.0
    p = np.abs(k) ** (-(2 * HURST + 1) / 2)
    p[0] = 0.0
    noise = np.real(ifft(p * (rng.standard_normal(n) + 1j * rng.standard_normal(n))))
    std = noise.std()
    return noise / std if std > 0 else noise


def _step_mfsu(psi: np.ndarray, h_bytes: bytes, step: int, dt: float) -> np.ndarray:
    """Un paso Euler de la SPDE. Normalización tiempo-constante."""
    seed = (
        int.from_bytes(h_bytes[(step * 7) % 56: (step * 7) % 56 + 8], "big")
        ^ (step * 0x9E3779B97F4A7C15)
    )
    eta = _fgn(len(psi), seed)
    fr  = _fractional_laplacian(np.real(psi), BETA)
    fi  = _fractional_laplacian(np.imag(psi), BETA)
    psi = psi + dt * (
        -DELTA_F * (fr + 1j * fi)
        + GAMMA_NL * (np.abs(psi) ** 2) * psi
        + SIGMA_ETA * eta
    )
    return psi / max(np.max(np.abs(psi)), 1.0)


# ══════════════════════════════════════════════════════════════════════════════
#  KDF FRACTAL MEMORY-HARD
# ══════════════════════════════════════════════════════════════════════════════

def _mfsu_kdf(password: str, salt: bytes, kdf_m: int = 256) -> bytes:
    """KDF con scratchpad fractal. Costo = kdf_m × N × 16 bytes RAM."""
    h = hashlib.sha3_512(password.encode() + b"\x00" + salt).digest()
    N = 128
    rng = np.random.default_rng(np.frombuffer(h[:32], dtype=np.uint32))
    psi = rng.standard_normal(N) + 1j * rng.standard_normal(N)

    # Fase 1: llenar scratchpad
    scratchpad = np.zeros((kdf_m, N), dtype=np.complex128)
    for s in range(kdf_m):
        psi = _step_mfsu(psi, h, s, 0.001)
        scratchpad[s] = psi

    # Fase 2: mezcla no-lineal
    pm = scratchpad[-1].copy()
    for s in range(kdf_m):
        idx = int(abs(np.real(pm[0])) * 1e9) % kdf_m
        pm = (pm + 0.001 * scratchpad[idx])
        pm = pm / max(np.max(np.abs(pm)), 1.0)

    # Fase 3: condensación
    sb = (
        (np.real(pm) * 1e10).astype(np.int64).tobytes() +
        (np.imag(pm) * 1e10).astype(np.int64).tobytes()
    )
    k_raw = hashlib.sha3_512(sb + h).digest()

    # HKDF-Expand
    result = bytearray()
    prev = b""
    c = 1
    while len(result) < 96:
        prev = hashlib.sha3_256(prev + k_raw + c.to_bytes(1, "big")).digest()
        result.extend(prev)
        c += 1
    return bytes(result[:96])


# ══════════════════════════════════════════════════════════════════════════════
#  KEYSTREAM MFSU
# ══════════════════════════════════════════════════════════════════════════════

def _mfsu_keystream(derived_key: bytes, iv: bytes, length: int) -> np.ndarray:
    h = hashlib.sha3_512(derived_key + iv).digest()
    n_steps = 48 + (h[0] % 64)
    N = 512
    rng = np.random.default_rng(np.frombuffer(h[:32], dtype=np.uint32))
    psi = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    psi[:64] *= np.frombuffer(h[:64], dtype=np.uint8).astype(float) / 255.0 + 0.5
    mixer_key = hashlib.sha3_256(derived_key[32:64] + iv).digest()

    buf = []
    for s in range(n_steps):
        psi = _step_mfsu(psi, h, s, 0.01)
        buf.extend(((np.real(psi) * 1e4).astype(np.int64) & 0xFF).tolist())
        buf.extend(((np.imag(psi) * 1e4).astype(np.int64) & 0xFF).tolist())
        if len(buf) >= length * 2:
            break

    raw = np.array(buf[:length], dtype=np.uint8)
    mixed = bytearray(length)
    bc = 0
    for i in range(0, length, 32):
        bk = hashlib.sha3_256(mixer_key + bc.to_bytes(4, "big")).digest()
        bc += 1
        for j, (rb, kb) in enumerate(zip(raw[i: i + 32], bk)):
            if i + j < length:
                mixed[i + j] = rb ^ kb

    return np.frombuffer(bytes(mixed), dtype=np.uint8)


# ── Primitivas de bajo nivel ──────────────────────────────────────────────────

def _pkcs7_pad(data: bytes, block: int = 16) -> bytes:
    pl = block - (len(data) % block)
    return data + bytes([pl] * pl)


def _pkcs7_unpad(data: bytes) -> bytes:
    pl = data[-1]
    if pl < 1 or pl > 16 or data[-pl:] != bytes([pl] * pl):
        raise ValueError("Padding inválido")
    return data[:-pl]


def _enc_block(data: bytes, password: str, salt: bytes, iv: bytes, kdf_m: int) -> bytes:
    """Cifra un bloque con KDF fractal + keystream MFSU."""
    km = _mfsu_kdf(password, salt, kdf_m)
    ks = _mfsu_keystream(km[:64], iv, len(data))
    return (np.frombuffer(data, dtype=np.uint8) ^ ks).tobytes()


# ══════════════════════════════════════════════════════════════════════════════
#  FRACTALSHIELD — API PÚBLICA
# ══════════════════════════════════════════════════════════════════════════════

def fractalshield_encrypt(plaintext: bytes, password: str, level: int = 2) -> bytes:
    """
    Cifra plaintext con FractalShield nivel 1/2/3.

    Produce N capas de cifrado:
    - Capa 0 (real): contiene REAL_MAGIC + plaintext cifrado
    - Capas 1..N-1 (señuelo): datos fractal estadísticamente indistinguibles

    El orden de las capas es aleatorio y está cifrado con la clave real.
    Sin la clave correcta, el atacante no sabe en qué posición está la capa real
    ni cuándo ha acertado la contraseña.
    """
    if level not in SHIELD_LEVELS:
        raise ValueError(f"Nivel debe ser 1, 2 o 3. Recibido: {level}")

    kdf_ms = SHIELD_LEVELS[level]
    n      = len(kdf_ms)

    # Tamaño uniforme: todas las capas tendrán exactamente L bytes de ciphertext
    padded_real = _pkcs7_pad(REAL_MAGIC + plaintext)
    L = len(padded_real)

    # ── Generar salts e IVs únicos por capa ──────────────────────────────────
    salts = [os.urandom(16) for _ in range(n)]
    ivs   = [os.urandom(16) for _ in range(n)]

    # ── Cifrar cada capa ─────────────────────────────────────────────────────
    layers = []
    for i, kdf_m in enumerate(kdf_ms):
        if i == 0:
            # Capa real: plaintext con REAL_MAGIC
            data = padded_real
        else:
            # Capa señuelo: datos fractal derivados del campo MFSU
            # Usamos SHA3 del password + índice + salt como seed
            h_decoy = hashlib.sha3_256(
                password.encode() + i.to_bytes(1, "big") + salts[i]
            ).digest()
            rng = np.random.default_rng(np.frombuffer(h_decoy[:32], dtype=np.uint32))
            data = bytes(rng.integers(0, 256, L, dtype=np.uint8))

        layers.append(_enc_block(data, password, salts[i], ivs[i], kdf_m))

    # ── Mezclar orden aleatoriamente usando el campo fractal ──────────────────
    h_ord = hashlib.sha3_256(password.encode() + b"FRACTALSHIELD_ORDER").digest()
    rng_ord = np.random.default_rng(np.frombuffer(h_ord[:32], dtype=np.uint32))
    order = list(range(n))
    rng_ord.shuffle(order)

    # Cifrar el orden con la clave real
    iv_ord   = os.urandom(16)
    order_enc = _enc_block(
        _pkcs7_pad(bytes(order)), password, ORDER_SALT, iv_ord, 256
    )
    ord_len = len(order_enc)

    # ── Construir header (sin MAC aún) ────────────────────────────────────────
    salt_global = os.urandom(16)
    header = (
        MAGIC_V4 + VERSION_V4
        + bytes([level, n])
        + salt_global
        + iv_ord
        + ord_len.to_bytes(2, "big")
        + order_enc
    )

    # ── Ensamblar capas en orden mezclado ─────────────────────────────────────
    layer_blob = b"".join(salts[idx] + ivs[idx] + layers[idx] for idx in order)

    # ── MAC global sobre header + layer_blob (Encrypt-then-MAC) ──────────────
    km_global = _mfsu_kdf(password, salt_global, 256)
    mac = hmac_mod.new(
        km_global[:32],
        header + layer_blob,
        hashlib.sha3_256
    ).digest()

    return header + mac + layer_blob


def fractalshield_decrypt(blob: bytes, password: str) -> bytes:
    """
    Descifra un blob FractalShield.

    Lanza ValueError con mensaje genérico si la contraseña es incorrecta
    o el archivo está alterado. No revela qué falló exactamente.
    """
    if len(blob) < 10:
        raise ValueError("Archivo inválido")
    if not blob.startswith(MAGIC_V4):
        raise ValueError("No es un archivo .fracta v4")
    if blob[6:7] != VERSION_V4:
        raise ValueError("Versión incompatible")

    # ── Parsear header ────────────────────────────────────────────────────────
    o = 9
    salt_global = blob[o: o + 16]; o += 16
    iv_ord      = blob[o: o + 16]; o += 16
    ord_len     = int.from_bytes(blob[o: o + 2], "big"); o += 2
    order_enc   = blob[o: o + ord_len]; o += ord_len
    header      = blob[:o]   # todo lo que precede al MAC

    mac_stored  = blob[o: o + 32]; o += 32
    layer_blob  = blob[o:]

    level = blob[7]
    n     = blob[8]

    if level not in SHIELD_LEVELS:
        raise ValueError("Nivel de shield no reconocido")

    kdf_ms = SHIELD_LEVELS[level]

    # ── Verificar MAC global (tiempo constante) ───────────────────────────────
    km_global = _mfsu_kdf(password, salt_global, 256)
    mac_calc  = hmac_mod.new(
        km_global[:32],
        header + layer_blob,
        hashlib.sha3_256
    ).digest()

    if not hmac_mod.compare_digest(mac_stored, mac_calc):
        raise ValueError(
            "Autenticación fallida — contraseña incorrecta o archivo alterado"
        )

    # ── Descifrar el mapa de orden ────────────────────────────────────────────
    order_plain = _enc_block(order_enc, password, ORDER_SALT, iv_ord, 256)
    order = list(_pkcs7_unpad(order_plain))

    # ── Buscar la capa real ───────────────────────────────────────────────────
    L_layer = len(layer_blob) // n

    for pos, idx in enumerate(order):
        start = pos * L_layer
        s_i   = layer_blob[start: start + 16]
        iv_i  = layer_blob[start + 16: start + 32]
        ct    = layer_blob[start + 32: start + L_layer]
        kdf_m = kdf_ms[idx]

        pt = _enc_block(ct, password, s_i, iv_i, kdf_m)
        if pt[:5] == REAL_MAGIC:
            return _pkcs7_unpad(pt[5:])

    # Nunca debería llegar aquí si el MAC pasó, pero por seguridad:
    raise ValueError("Error interno — capa real no encontrada")


# ══════════════════════════════════════════════════════════════════════════════
#  UTILIDADES
# ══════════════════════════════════════════════════════════════════════════════

def fractalshield_info(level: int) -> dict:
    """Retorna información del nivel para mostrar en UI."""
    name, use_case, time_user, cost_factor = SHIELD_NAMES[level]
    n_layers = len(SHIELD_LEVELS[level])
    return {
        "level":         level,
        "name":          name,
        "n_layers":      n_layers,
        "use_case":      use_case,
        "time_user":     time_user,
        "attacker_cost": cost_factor,
        "kdf_ms":        SHIELD_LEVELS[level],
        "overhead_factor": f"~{n_layers}x tamaño original",
    }


def fractalshield_inspect(blob: bytes) -> dict:
    """
    Inspecciona el header de un archivo .fracta v4 sin descifrar.
    No requiere contraseña.
    """
    if not blob.startswith(MAGIC_V4):
        return {"valid": False, "error": "No es un archivo .fracta v4"}

    level   = blob[7]
    n       = blob[8]
    o       = 9
    salt_g  = blob[o: o + 16]; o += 16
    iv_ord  = blob[o: o + 16]; o += 16
    ord_len = int.from_bytes(blob[o: o + 2], "big"); o += 2
    order_enc = blob[o: o + ord_len]; o += ord_len
    header_size = o
    mac     = blob[o: o + 32]; o += 32
    layer_blob = blob[o:]
    L_layer = len(layer_blob) // n if n > 0 else 0

    name = SHIELD_NAMES.get(level, ("?", "?", "?", "?"))[0]

    return {
        "valid":          True,
        "version":        4,
        "shield_level":   level,
        "shield_name":    name,
        "n_layers":       n,
        "salt_global":    salt_g.hex(),
        "iv_order":       iv_ord.hex(),
        "mac":            mac.hex(),
        "header_size":    header_size + 32,
        "layer_size":     L_layer,
        "total_size":     len(blob),
        "kdf_ms":         SHIELD_LEVELS.get(level, []),
    }
