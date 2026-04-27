from __future__ import annotations

import argparse
import getpass
import os
import sys
import time

from fractalyx.crypto import encrypt, decrypt
from fractalyx.hash_mfsu import digest
from fractalyx.totp import generate as totp_generate, verify as totp_verify
from fractalyx.core import MAGIC, VERSION, HEADER_LEN


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_password(prompt: str = "Contraseña: ", confirm: bool = False) -> str:
    """Solicita contraseña interactiva sin mostrarla en pantalla."""
    pwd = getpass.getpass(prompt)
    if confirm:
        pwd2 = getpass.getpass("Confirmar contraseña: ")
        if pwd != pwd2:
            print("Error: las contraseñas no coinciden.", file=sys.stderr)
            sys.exit(1)
    if not pwd:
        print("Error: la contraseña no puede estar vacía.", file=sys.stderr)
        sys.exit(1)
    return pwd


def _output_path(input_path: str, suffix: str, strip: str = "") -> str:
    """Genera el nombre de salida por defecto."""
    if strip and input_path.endswith(strip):
        return input_path[: -len(strip)]
    return input_path + suffix


def _fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024  # type: ignore[assignment]
    return f"{n:.1f} TB"


# ── Subcomandos ───────────────────────────────────────────────────────────────

def cmd_encrypt(args: argparse.Namespace) -> int:
    if not os.path.isfile(args.file):
        print(f"Error: '{args.file}' no encontrado.", file=sys.stderr)
        return 1

    password = args.password or _get_password("Contraseña de cifrado: ", confirm=True)
    out = args.output or _output_path(args.file, ".fracta")

    print(f"Cifrando {args.file!r} → {out!r}")
    print("  KDF memory-hard en curso (~0.5 s)…", end=" ", flush=True)
    t0 = time.perf_counter()

    with open(args.file, "rb") as f:
        data = f.read()

    blob = encrypt(data, password)
    elapsed = time.perf_counter() - t0

    with open(out, "wb") as f:
        f.write(blob)

    print(f"listo en {elapsed:.2f} s")
    print(f"  Original : {_fmt_size(len(data))}")
    print(f"  Cifrado  : {_fmt_size(len(blob))}")
    print(f"  Overhead : {len(blob) - len(data)} bytes (header .fracta v3)")
    return 0


def cmd_decrypt(args: argparse.Namespace) -> int:
    if not os.path.isfile(args.file):
        print(f"Error: '{args.file}' no encontrado.", file=sys.stderr)
        return 1

    password = args.password or _get_password("Contraseña de descifrado: ")
    out = args.output or _output_path(args.file, "", strip=".fracta")

    print(f"Descifrando {args.file!r} → {out!r}")
    print("  Verificando MAC + reconstruyendo ψ…", end=" ", flush=True)
    t0 = time.perf_counter()

    with open(args.file, "rb") as f:
        blob = f.read()

    try:
        plaintext = decrypt(blob, password)
    except ValueError as exc:
        print()
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    elapsed = time.perf_counter() - t0

    with open(out, "wb") as f:
        f.write(plaintext)

    print(f"listo en {elapsed:.2f} s")
    print(f"  Recuperado: {_fmt_size(len(plaintext))}")
    return 0


def cmd_hash(args: argparse.Namespace) -> int:
    if args.text:
        data = args.text.encode("utf-8")
        label = f"texto: {args.text!r}"
    elif args.file and os.path.isfile(args.file):
        with open(args.file, "rb") as f:
            data = f.read()
        label = f"archivo: {args.file!r}"
    else:
        print("Error: proporciona --text o un archivo.", file=sys.stderr)
        return 1

    print(f"Calculando hash MFSU-MDF de {label}…", end=" ", flush=True)
    t0 = time.perf_counter()
    h = digest(data)
    elapsed = time.perf_counter() - t0
    print(f"listo en {elapsed:.2f} s")
    print(h)
    return 0


def cmd_totp(args: argparse.Namespace) -> int:
    code, expires, prev, nxt = totp_generate(args.secret)
    print(f"\nCódigo actual  : {code}")
    print(f"Expira en      : {expires} s")
    print(f"Ventana ant.   : {prev}")
    print(f"Ventana sig.   : {nxt}\n")

    if args.verify:
        ok = totp_verify(args.secret, args.verify)
        status = "✓ válido" if ok else "✗ inválido"
        print(f"Verificación de '{args.verify}': {status}")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    if not os.path.isfile(args.file):
        print(f"Error: '{args.file}' no encontrado.", file=sys.stderr)
        return 1

    with open(args.file, "rb") as f:
        blob = f.read()

    if not blob.startswith(MAGIC):
        print("No es un archivo .fracta v3.", file=sys.stderr)
        return 1

    ver = blob[len(MAGIC)]
    size = len(blob)
    ctext_size = size - HEADER_LEN

    print(f"Archivo   : {args.file}")
    print(f"Formato   : {MAGIC.decode()} v{ver}")
    print(f"Tamaño    : {_fmt_size(size)} total / {_fmt_size(ctext_size)} ciphertext")
    print(f"Header    : {HEADER_LEN} bytes (IV+SALT+MSALT+MAC)")
    return 0


# ── Parser ────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="axis-vault",
        description="MFSU Vault v3 — criptografía fractal",
    )
    parser.add_argument("--version", action="version", version="axis-vault 3.0.0")
    sub = parser.add_subparsers(dest="command", required=True)

    # encrypt
    p_enc = sub.add_parser("encrypt", help="Cifrar un archivo")
    p_enc.add_argument("file", help="Archivo a cifrar")
    p_enc.add_argument("-o", "--output", help="Archivo de salida (default: <file>.fracta)")
    p_enc.add_argument("-p", "--password", help="Contraseña (no recomendado en CLI; mejor interactivo)")
    p_enc.set_defaults(func=cmd_encrypt)

    # decrypt
    p_dec = sub.add_parser("decrypt", help="Descifrar un archivo .fracta")
    p_dec.add_argument("file", help="Archivo .fracta a descifrar")
    p_dec.add_argument("-o", "--output", help="Archivo de salida")
    p_dec.add_argument("-p", "--password", help="Contraseña")
    p_dec.set_defaults(func=cmd_decrypt)

    # hash
    p_hash = sub.add_parser("hash", help="Calcular hash MFSU-MDF")
    p_hash.add_argument("file", nargs="?", help="Archivo a hashear")
    p_hash.add_argument("--text", help="Texto a hashear en lugar de un archivo")
    p_hash.set_defaults(func=cmd_hash)

    # totp
    p_totp = sub.add_parser("totp", help="Generar/verificar código TOTP fractal")
    p_totp.add_argument("secret", help="Secreto compartido")
    p_totp.add_argument("--verify", metavar="CODE", help="Verificar un código")
    p_totp.set_defaults(func=cmd_totp)

    # info
    p_info = sub.add_parser("info", help="Mostrar metadatos de un archivo .fracta")
    p_info.add_argument("file", help="Archivo .fracta")
    p_info.set_defaults(func=cmd_info)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
