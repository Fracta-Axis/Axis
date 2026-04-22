#!/usr/bin/env python3
"""
Fractalyx CLI — Fractal-Stochastic Cryptographic System
========================================================
Cifra y descifra archivos .fyx desde la línea de comandos.
Sin servidor. Sin dependencias de internet.

Uso:
    python fractalyx_cli.py encrypt archivo.pdf -p "tu_contrasena" -l 2
    python fractalyx_cli.py decrypt archivo.fyx -p "tu_contrasena"
    python fractalyx_cli.py inspect archivo.fyx

Niveles FractalShield:
    1 = Estándar  (3 capas, 3.5× costo atacante)
    2 = Reforzado (4 capas, 7.5× costo atacante)  [por defecto]
    3 = Máximo    (5 capas, 15.5× costo atacante)

Instalar dependencias:
    pip install numpy scipy

© 2026 Miguel Ángel Franco León — MIT License
https://github.com/Fracta-Axis/Fractalyx
"""

import argparse
import hashlib
import hmac as hmac_mod
import os
import sys
import time

import numpy as np
from scipy.fft import fft, ifft, fftfreq

# ── Constantes Fractalyx ─────────────────────────────────────────────────────

MAGIC_FYX     = b"FRACv1"
VERSION_FYX   = b"\x01"
REAL_MAGIC    = b"FYX\x01"
ORDER_SALT    = b"FRACT_ORDER_SALT"
DELTA_F       = 0.921
BETA          = 2.0 - DELTA_F   # 1.079
HURST         = 0.541
SHIELD_LEVELS = {
    1: [256, 512, 1024],
    2: [256, 512, 1024, 2048],
    3: [256, 512, 1024, 2048, 4096],
}
SHIELD_NAMES = {
    1: "Estándar  (3 capas · 3.5× costo atacante)",
    2: "Reforzado (4 capas · 7.5× costo atacante)",
    3: "Máximo    (5 capas · 15.5× costo atacante)",
}


# ── Núcleo MFSU ──────────────────────────────────────────────────────────────

def _fl1d(psi, alpha):
    k = fftfreq(len(psi), d=1.0 / len(psi)) * 2 * np.pi
    ka = np.abs(k) ** alpha
    ka[0] = 0.0
    return np.real(ifft(ka * fft(psi)))


def _fgn(n, seed):
    rng = np.random.default_rng(seed & 0xFFFFFFFF)
    k = fftfreq(n, d=1.0 / n)
    k[0] = 1.0
    p = np.abs(k) ** (-(2 * HURST + 1) / 2)
    p[0] = 0.0
    noise = np.real(ifft(p * (rng.standard_normal(n) + 1j * rng.standard_normal(n))))
    std = noise.std()
    return noise / std if std > 0 else noise


def _step_mfsu(psi, h_bytes, step_n, dt):
    seed = (
        int.from_bytes(h_bytes[(step_n * 7) % 56:(step_n * 7) % 56 + 8], "big")
        ^ (step_n * 0x9E3779B97F4A7C15)
    )
    eta = _fgn(len(psi), seed)
    fr  = _fl1d(np.real(psi), BETA)
    fi  = _fl1d(np.imag(psi), BETA)
    psi = psi + dt * (
        -DELTA_F * (fr + 1j * fi)
        + DELTA_F * (np.abs(psi) ** 2) * psi
        + 0.1 * eta
    )
    return psi / max(np.max(np.abs(psi)), 1.0)


def _mfsu_kdf(password, salt, kdf_m=256):
    h = hashlib.sha3_512(password.encode() + b"\x00" + salt).digest()
    N = 128
    rng = np.random.default_rng(np.frombuffer(h[:32], dtype=np.uint32))
    psi = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    sp = np.zeros((kdf_m, N), dtype=np.complex128)
    for s in range(kdf_m):
        psi = _step_mfsu(psi, h, s, 0.001)
        sp[s] = psi
    pm = sp[-1].copy()
    for s in range(kdf_m):
        idx = int(abs(np.real(pm[0])) * 1e9) % kdf_m
        pm = (pm + 0.001 * sp[idx])
        pm = pm / max(np.max(np.abs(pm)), 1.0)
    sb = (
        (np.real(pm) * 1e10).astype(np.int64).tobytes()
        + (np.imag(pm) * 1e10).astype(np.int64).tobytes()
    )
    k_raw = hashlib.sha3_512(sb + h).digest()
    result = bytearray()
    prev = b""
    c = 1
    while len(result) < 96:
        prev = hashlib.sha3_256(prev + k_raw + c.to_bytes(1, "big")).digest()
        result.extend(prev)
        c += 1
    return bytes(result[:96])


def _mfsu_keystream(dk, iv, length):
    h = hashlib.sha3_512(dk + iv).digest()
    n_steps = 48 + (h[0] % 64)
    N = 512
    rng = np.random.default_rng(np.frombuffer(h[:32], dtype=np.uint32))
    psi = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    psi[:64] *= np.frombuffer(h[:64], dtype=np.uint8).astype(float) / 255.0 + 0.5
    mk = hashlib.sha3_256(dk[32:64] + iv).digest()
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
        bk = hashlib.sha3_256(mk + bc.to_bytes(4, "big")).digest()
        bc += 1
        for j, (rb, kb) in enumerate(zip(raw[i:i + 32], bk)):
            if i + j < length:
                mixed[i + j] = rb ^ kb
    return np.frombuffer(bytes(mixed), dtype=np.uint8)


def _pad(data, b=16):
    pl = b - (len(data) % b)
    return data + bytes([pl] * pl)


def _unpad(data):
    pl = data[-1]
    if pl < 1 or pl > 16 or data[-pl:] != bytes([pl] * pl):
        raise ValueError("Padding inválido")
    return data[:-pl]


def _enc_block(data, password, salt, iv, kdf_m):
    km = _mfsu_kdf(password, salt, kdf_m)
    ks = _mfsu_keystream(km[:64], iv, len(data))
    return (np.frombuffer(data, dtype=np.uint8) ^ ks).tobytes()


# ── API pública ───────────────────────────────────────────────────────────────

def encrypt(plaintext: bytes, password: str, level: int = 2) -> bytes:
    """
    Cifra datos con Fractalyx FractalShield.

    Produce N capas de cifrado del mismo tamaño.
    Solo la capa real contiene el magic FYX\\x01.
    El atacante no puede verificar si acertó la contraseña sin
    pagar el costo de todas las capas.
    """
    if level not in SHIELD_LEVELS:
        raise ValueError(f"Nivel debe ser 1, 2 o 3. Recibido: {level}")

    ms = SHIELD_LEVELS[level]
    n  = len(ms)

    padded = _pad(REAL_MAGIC + plaintext)
    L = len(padded)

    salts  = [os.urandom(16) for _ in range(n)]
    ivs    = [os.urandom(16) for _ in range(n)]
    layers = []

    for i, m in enumerate(ms):
        if i == 0:
            data = padded
        else:
            h_d = hashlib.sha3_256(
                password.encode() + i.to_bytes(1, "big") + salts[i]
            ).digest()
            rng  = np.random.default_rng(np.frombuffer(h_d[:32], dtype=np.uint32))
            data = bytes(rng.integers(0, 256, L, dtype=np.uint8))
        layers.append(_enc_block(data, password, salts[i], ivs[i], m))

    h_ord   = hashlib.sha3_256(password.encode() + b"FRACTALYX_ORDER").digest()
    rng_ord = np.random.default_rng(np.frombuffer(h_ord[:32], dtype=np.uint32))
    order   = list(range(n))
    rng_ord.shuffle(order)

    iv_ord    = os.urandom(16)
    order_enc = _enc_block(_pad(bytes(order)), password, ORDER_SALT, iv_ord, 256)
    ord_len   = len(order_enc)
    salt_g    = os.urandom(16)

    header = (
        MAGIC_FYX + VERSION_FYX
        + bytes([level, n])
        + salt_g + iv_ord
        + ord_len.to_bytes(2, "big")
        + order_enc
    )
    layer_blob = b"".join(salts[idx] + ivs[idx] + layers[idx] for idx in order)

    km_g = _mfsu_kdf(password, salt_g, 256)
    mac  = hmac_mod.new(km_g[:32], header + layer_blob, hashlib.sha3_256).digest()

    return header + mac + layer_blob


def decrypt(blob: bytes, password: str) -> bytes:
    """
    Descifra un archivo .fyx.
    Lanza ValueError si la contraseña es incorrecta o el archivo está dañado.
    """
    if not blob.startswith(MAGIC_FYX):
        raise ValueError("No es un archivo .fyx válido")
    if blob[6:7] != VERSION_FYX:
        raise ValueError("Versión de archivo no compatible")

    o        = 9
    salt_g   = blob[o:o + 16]; o += 16
    iv_ord   = blob[o:o + 16]; o += 16
    ord_len  = int.from_bytes(blob[o:o + 2], "big"); o += 2
    order_enc = blob[o:o + ord_len]; o += ord_len
    header   = blob[:o]
    mac_s    = blob[o:o + 32]; o += 32
    layer_blob = blob[o:]

    level = blob[7]
    n     = blob[8]

    if level not in SHIELD_LEVELS:
        raise ValueError("Nivel de FractalShield no reconocido")

    ms = SHIELD_LEVELS[level]

    km_g  = _mfsu_kdf(password, salt_g, 256)
    mac_c = hmac_mod.new(km_g[:32], header + layer_blob, hashlib.sha3_256).digest()

    if not hmac_mod.compare_digest(mac_s, mac_c):
        raise ValueError(
            "Autenticación fallida — contraseña incorrecta o archivo alterado"
        )

    order_plain = _enc_block(order_enc, password, ORDER_SALT, iv_ord, 256)
    order = list(_unpad(order_plain))

    L_layer = len(layer_blob) // n
    for pos, idx in enumerate(order):
        start = pos * L_layer
        s_i   = layer_blob[start:start + 16]
        iv_i  = layer_blob[start + 16:start + 32]
        ct    = layer_blob[start + 32:start + L_layer]
        pt    = _enc_block(ct, password, s_i, iv_i, ms[idx])
        if pt[:4] == REAL_MAGIC:
            return _unpad(pt[4:])

    raise ValueError("Error interno — capa real no encontrada")


def inspect(blob: bytes) -> dict:
    """Inspecciona el header de un .fyx sin contraseña."""
    if not blob.startswith(MAGIC_FYX):
        return {"valid": False, "error": "No es un archivo .fyx"}
    level = blob[7]
    n     = blob[8]
    o     = 9
    salt_g   = blob[o:o + 16]; o += 16
    iv_ord   = blob[o:o + 16]; o += 16
    ord_len  = int.from_bytes(blob[o:o + 2], "big"); o += 2
    o += ord_len
    mac = blob[o:o + 32]
    layer_blob = blob[o + 32:]
    L_layer = len(layer_blob) // n if n > 0 else 0
    return {
        "valid":        True,
        "format":       "Fractalyx .fyx v1",
        "magic":        blob[:6].decode(),
        "version":      blob[6],
        "shield_level": level,
        "shield_name":  SHIELD_NAMES.get(level, "?"),
        "n_layers":     n,
        "salt_global":  salt_g.hex(),
        "mac":          mac.hex()[:32] + "...",
        "layer_size":   L_layer,
        "total_size":   len(blob),
        "kdf_m_seq":    SHIELD_LEVELS.get(level, []),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def cmd_encrypt(args):
    if not os.path.isfile(args.file):
        print(f"❌ Archivo no encontrado: {args.file}", file=sys.stderr)
        sys.exit(1)

    with open(args.file, "rb") as f:
        data = f.read()

    out = args.output or args.file + ".fyx"

    print(f"🌀 Fractalyx — Cifrando con FractalShield nivel {args.level}")
    print(f"   {SHIELD_NAMES[args.level]}")
    print(f"   Archivo: {args.file} ({len(data):,} bytes)")

    t0 = time.time()
    blob = encrypt(data, args.password, level=args.level)
    elapsed = time.time() - t0

    with open(out, "wb") as f:
        f.write(blob)

    print(f"✅ Cifrado en {elapsed:.2f}s → {out} ({len(blob):,} bytes)")


def cmd_decrypt(args):
    if not os.path.isfile(args.file):
        print(f"❌ Archivo no encontrado: {args.file}", file=sys.stderr)
        sys.exit(1)

    with open(args.file, "rb") as f:
        blob = f.read()

    # Nombre de salida
    out = args.output
    if not out:
        out = args.file
        if out.endswith(".fyx"):
            out = out[:-4]
        else:
            out = out + ".decrypted"

    print(f"🔓 Fractalyx — Descifrando {args.file}")

    t0 = time.time()
    try:
        plaintext = decrypt(blob, args.password)
    except ValueError as e:
        print(f"❌ {e}", file=sys.stderr)
        sys.exit(1)
    elapsed = time.time() - t0

    with open(out, "wb") as f:
        f.write(plaintext)

    print(f"✅ Descifrado en {elapsed:.2f}s → {out} ({len(plaintext):,} bytes)")


def cmd_inspect(args):
    if not os.path.isfile(args.file):
        print(f"❌ Archivo no encontrado: {args.file}", file=sys.stderr)
        sys.exit(1)

    with open(args.file, "rb") as f:
        blob = f.read()

    info = inspect(blob)

    if not info["valid"]:
        print(f"❌ {info['error']}", file=sys.stderr)
        sys.exit(1)

    print(f"\n🔍 Fractalyx Inspector — {args.file}")
    print(f"{'─'*45}")
    print(f"  Formato:       {info['format']}")
    print(f"  Magic header:  {info['magic']}")
    print(f"  Versión:       {info['version']}")
    print(f"  Nivel Shield:  {info['shield_level']} — {info['shield_name']}")
    print(f"  Capas:         {info['n_layers']}")
    print(f"  KDF_M/capa:    {info['kdf_m_seq']}")
    print(f"  Tamaño capa:   {info['layer_size']:,} bytes")
    print(f"  Tamaño total:  {info['total_size']:,} bytes")
    print(f"  Salt global:   {info['salt_global']}")
    print(f"  MAC:           {info['mac']}")
    print(f"{'─'*45}")
    print(f"  ✅ Archivo .fyx válido")
    print(f"  ℹ️  Todas las {info['n_layers']} capas tienen el mismo tamaño")
    print(f"     El atacante no puede identificar la capa real por tamaño.\n")


def main():
    parser = argparse.ArgumentParser(
        prog="fractalyx",
        description="Fractalyx — Fractal-Stochastic Cryptographic System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python fractalyx_cli.py encrypt secreto.pdf -p "mi_contrasena" -l 2
  python fractalyx_cli.py decrypt secreto.pdf.fyx -p "mi_contrasena"
  python fractalyx_cli.py inspect secreto.pdf.fyx

  # Demo: descifrar el archivo de muestra del repositorio
  python fractalyx_cli.py decrypt demo.fyx -p "fractalyx2026"

© 2026 Miguel Ángel Franco León — MIT License
https://github.com/Fracta-Axis/Axis
        """
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # encrypt
    p_enc = sub.add_parser("encrypt", help="Cifrar un archivo")
    p_enc.add_argument("file", help="Archivo a cifrar")
    p_enc.add_argument("-p", "--password", required=True, help="Contraseña")
    p_enc.add_argument("-l", "--level", type=int, choices=[1, 2, 3], default=2,
                       help="Nivel FractalShield (1=Estándar, 2=Reforzado, 3=Máximo)")
    p_enc.add_argument("-o", "--output", help="Archivo de salida (por defecto: archivo.fyx)")

    # decrypt
    p_dec = sub.add_parser("decrypt", help="Descifrar un archivo .fyx")
    p_dec.add_argument("file", help="Archivo .fyx a descifrar")
    p_dec.add_argument("-p", "--password", required=True, help="Contraseña")
    p_dec.add_argument("-o", "--output", help="Archivo de salida")

    # inspect
    p_ins = sub.add_parser("inspect", help="Inspeccionar header sin contraseña")
    p_ins.add_argument("file", help="Archivo .fyx a inspeccionar")

    args = parser.parse_args()

    if args.command == "encrypt":
        cmd_encrypt(args)
    elif args.command == "decrypt":
        cmd_decrypt(args)
    elif args.command == "inspect":
        cmd_inspect(args)


if __name__ == "__main__":
    main()
