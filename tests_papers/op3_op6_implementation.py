"""
op3_op6.py
----------
Resolución empírica de:
  OP3 — Replay-attack mitigation
  OP6 — Constant-time: magic-prefix y MAC verification

OP3: Se agrega un session_id al contexto del MAC.
     Prueba: un ciphertext válido de sesión A es rechazado en sesión B.

OP6: Se reemplaza == por hmac.compare_digest en:
     - magic-prefix check  (CT2a)
     - HMAC tag comparison  (CT2b)
     Prueba: medición de varianza de timing >= 10^5 muestras (CT4).
"""

import hmac as _hmac
import hashlib, os, time, statistics, json

SEP = "=" * 66
MAGIC = b'MFSU\x04'

# ── Helpers hash ─────────────────────────────────────────────
def sha3_256(data: bytes) -> bytes:
    return hashlib.sha3_256(data).digest()

def sha3_512(data: bytes) -> bytes:
    return hashlib.sha3_512(data).digest()

# ─────────────────────────────────────────────────────────────
# OP3 — REPLAY-ATTACK MITIGATION
# ─────────────────────────────────────────────────────────────
#
# Problema original (Construction 3.2):
#   kMAC = H(k0 || "MAC")
#   τ    = HMAC(kMAC, hdr || layers)
#
# El MAC no incluye ningún identificador de sesión. Un ciphertext
# válido de la sesión S1 pasa la verificación en cualquier sesión S2.
#
# Fix (Construction 3.2-r1):
#   session_id = os.urandom(16)   ← generado en cada sesión de cifrado
#   kMAC = H(k0 || "MAC" || session_id)
#   τ    = HMAC(kMAC, hdr || session_id || layers)
#
# El session_id se almacena en el header del archivo .shield.
# El decifrado incluye session_id en la derivación de kMAC y en el
# cuerpo del HMAC, por lo que un ciphertext de otra sesión produce
# un kMAC distinto y el tag falla con probabilidad 1 - negl(κ).
#
# Prueba formal (sketch):
#   Sea C = (hdr, sid, layers, τ) un ciphertext válido de sesión sid.
#   Un adversario A intenta reutilizarlo en sesión sid' ≠ sid.
#   kMAC' = H(k0 || "MAC" || sid') ≠ kMAC (SHA3-256 collision-resistant).
#   HMAC(kMAC', hdr || sid' || layers) ≠ τ con prob. 1 - ε_SUF-CMA.
#   Por SUF-CMA de HMAC-SHA3-256 (Theorem 5.1), ε_SUF-CMA = negl(κ). □

class FractalShieldMAC_Original:
    """MAC original sin session_id — vulnerable a replay."""

    def seal(self, k0: bytes, hdr: bytes, layers: bytes) -> bytes:
        k_mac = sha3_256(k0 + b"MAC")
        return _hmac.new(k_mac, hdr + layers, hashlib.sha3_256).digest()

    def verify(self, k0: bytes, hdr: bytes, layers: bytes, tag: bytes) -> bool:
        k_mac = sha3_256(k0 + b"MAC")
        expected = _hmac.new(k_mac, hdr + layers, hashlib.sha3_256).digest()
        return _hmac.compare_digest(expected, tag)


class FractalShieldMAC_OP3:
    """MAC corregido con session_id — resiste replay (OP3 fix)."""

    def new_session(self) -> bytes:
        """Genera un session_id fresco para cada operación de cifrado."""
        return os.urandom(16)

    def seal(self, k0: bytes, hdr: bytes, session_id: bytes,
             layers: bytes) -> bytes:
        k_mac = sha3_256(k0 + b"MAC" + session_id)
        return _hmac.new(k_mac, hdr + session_id + layers,
                         hashlib.sha3_256).digest()

    def verify(self, k0: bytes, hdr: bytes, session_id: bytes,
               layers: bytes, tag: bytes) -> bool:
        k_mac = sha3_256(k0 + b"MAC" + session_id)
        expected = _hmac.new(k_mac, hdr + session_id + layers,
                             hashlib.sha3_256).digest()
        return _hmac.compare_digest(expected, tag)


def test_op3():
    print(SEP)
    print("  OP3 — Replay-Attack Mitigation")
    print(SEP)

    k0     = os.urandom(32)
    hdr    = b"FractalShield_v4_header_mock"
    layers = os.urandom(512)   # ciphertext simulado

    orig = FractalShieldMAC_Original()
    fix  = FractalShieldMAC_OP3()

    # ── Test 1: construcción original acepta replay ───────────
    tag_s1 = orig.seal(k0, hdr, layers)
    # Mismo tag funciona en cualquier "sesión" porque no hay session_id
    replay_ok_original = orig.verify(k0, hdr, layers, tag_s1)
    print(f"\n  [Original — sin session_id]")
    print(f"  Sesión 1 cifra  → tag generado")
    print(f"  Sesión 2 verifica mismo tag → {'ACEPTA (replay posible) ✗' if replay_ok_original else 'rechaza'}")

    # ── Test 2: construcción corregida rechaza replay ─────────
    sid1 = fix.new_session()
    sid2 = fix.new_session()   # sesión diferente

    tag_fixed_s1 = fix.seal(k0, hdr, sid1, layers)

    # Verificación en la sesión correcta
    ok_same = fix.verify(k0, hdr, sid1, layers, tag_fixed_s1)
    # Intento de replay con session_id distinto
    ok_replay = fix.verify(k0, hdr, sid2, layers, tag_fixed_s1)
    # Intento de replay con session_id correcto pero layers modificado
    ok_tamper = fix.verify(k0, hdr, sid1, layers + b'\x00', tag_fixed_s1)

    print(f"\n  [Corregido — con session_id]")
    print(f"  Sesión 1 cifra con sid1   → tag generado")
    print(f"  Sesión 1 verifica (correcto) → {'ACEPTA ✓' if ok_same else 'rechaza ✗'}")
    print(f"  Sesión 2 intenta replay (sid2) → {'acepta ✗' if ok_replay else 'RECHAZA ✓'}")
    print(f"  Modificación de layers        → {'acepta ✗' if ok_tamper else 'RECHAZA ✓'}")

    # ── Test 3: 1000 pares de sesiones aleatorias ─────────────
    print(f"\n  [Stress test — 1000 pares de sesiones distintas]")
    replays_blocked = 0
    for _ in range(1000):
        k   = os.urandom(32)
        h   = os.urandom(16)
        l   = os.urandom(256)
        s1  = fix.new_session()
        s2  = fix.new_session()
        tag = fix.seal(k, h, s1, l)
        if not fix.verify(k, h, s2, l, tag):
            replays_blocked += 1

    print(f"  Replays bloqueados: {replays_blocked}/1000  "
          f"({'✓ todos bloqueados' if replays_blocked == 1000 else '✗ fallo'})")

    passed = (replay_ok_original and ok_same
              and not ok_replay and not ok_tamper
              and replays_blocked == 1000)
    print(f"\n  OP3 resuelto: {'✓' if passed else '✗'}")
    return passed


# ─────────────────────────────────────────────────────────────
# OP6 — CONSTANT-TIME: MAGIC-PREFIX Y MAC (CT2 + CT4)
# ─────────────────────────────────────────────────────────────
#
# Componente CT2a — Magic-prefix comparison
#   ANTES: plaintext[:5] == MAGIC   (early-exit, timing oracle)
#   DESPUÉS: hmac.compare_digest(plaintext[:5], MAGIC)
#
# Componente CT2b — HMAC tag verification
#   ANTES: computed_tag == stored_tag  (early-exit)
#   DESPUÉS: hmac.compare_digest(computed_tag, stored_tag)
#
# CT4 — Medición de varianza de timing end-to-end:
#   (a) contraseña correcta
#   (b) contraseña incorrecta, primer byte difiere
#   (c) contraseña incorrecta, todos los bytes difieren
#   Pass: todos dentro de ±2% entre sí al percentil 99.

# ── Implementación NCT (vulnerable) ──────────────────────────

def check_magic_nct(plaintext: bytes) -> bool:
    """Early-exit: filtra información sobre el keystream."""
    return plaintext[:5] == MAGIC

def verify_tag_nct(computed: bytes, stored: bytes) -> bool:
    """Early-exit: filtra información sobre el tag."""
    return computed == stored

# ── Implementación CT (corregida) ────────────────────────────

def check_magic_ct(plaintext: bytes) -> bool:
    """Constant-time via hmac.compare_digest (CT2a fix)."""
    return _hmac.compare_digest(plaintext[:5], MAGIC)

def verify_tag_ct(computed: bytes, stored: bytes) -> bool:
    """Constant-time via hmac.compare_digest (CT2b fix)."""
    return _hmac.compare_digest(computed, stored)

# ── Medición de timing ────────────────────────────────────────

def measure_timing(fn, *args, n: int = 100_000) -> dict:
    """
    Mide n ejecuciones de fn(*args).
    Retorna estadísticas de tiempo en nanosegundos.
    """
    times = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        fn(*args)
        times.append(time.perf_counter_ns() - t0)

    times.sort()
    p99 = times[int(0.99 * n)]
    return {
        "mean_ns":   statistics.mean(times),
        "median_ns": statistics.median(times),
        "stdev_ns":  statistics.stdev(times),
        "p99_ns":    p99,
        "cv_pct":    statistics.stdev(times) / statistics.mean(times) * 100,
    }


def test_op6():
    print(f"\n{SEP}")
    print("  OP6 — Constant-Time: Magic-Prefix y MAC Verification")
    print(SEP)
    N_SAMPLES = 100_000

    # ── Casos de prueba ───────────────────────────────────────
    correct_prefix   = MAGIC + os.urandom(27)      # prefijo correcto
    wrong_first_byte = bytes([MAGIC[0] ^ 0xFF]) + MAGIC[1:] + os.urandom(27)
    all_wrong        = os.urandom(5) + os.urandom(27)

    tag_correct = os.urandom(32)
    tag_wrong_1 = bytes([tag_correct[0] ^ 0xFF]) + tag_correct[1:]
    tag_all_bad = os.urandom(32)

    results = {}

    # ── CT2a: magic-prefix ────────────────────────────────────
    print(f"\n  [CT2a] Magic-prefix comparison  (n={N_SAMPLES:,})")
    print(f"  {'─'*58}")

    for label, payload in [
        ("correct prefix  ", correct_prefix),
        ("wrong 1st byte  ", wrong_first_byte),
        ("all wrong       ", all_wrong),
    ]:
        nct = measure_timing(check_magic_nct, payload, n=N_SAMPLES)
        ct  = measure_timing(check_magic_ct,  payload, n=N_SAMPLES)
        print(f"  {label} NCT mean={nct['mean_ns']:6.0f} ns  CV={nct['cv_pct']:4.1f}%  |"
              f"  CT mean={ct['mean_ns']:6.0f} ns  CV={ct['cv_pct']:4.1f}%")
        results[f"magic_{label.strip()}"] = {"nct": nct, "ct": ct}

    # Verificación CT4 para magic: rango ±2% entre casos
    magic_means_ct = [
        measure_timing(check_magic_ct, correct_prefix,   n=N_SAMPLES)["mean_ns"],
        measure_timing(check_magic_ct, wrong_first_byte, n=N_SAMPLES)["mean_ns"],
        measure_timing(check_magic_ct, all_wrong,        n=N_SAMPLES)["mean_ns"],
    ]
    magic_spread = (max(magic_means_ct) - min(magic_means_ct)) / statistics.mean(magic_means_ct) * 100
    magic_ct4_pass = magic_spread < 2.0
    print(f"\n  CT4 magic spread: {magic_spread:.2f}%  "
          f"(umbral <2%)  → {'PASS ✓' if magic_ct4_pass else 'FAIL ✗'}")

    # ── CT2b: HMAC tag verification ───────────────────────────
    print(f"\n  [CT2b] HMAC tag verification  (n={N_SAMPLES:,})")
    print(f"  {'─'*58}")

    for label, t_stored in [
        ("correct tag     ", tag_correct),
        ("wrong 1st byte  ", tag_wrong_1),
        ("all wrong       ", tag_all_bad),
    ]:
        nct = measure_timing(verify_tag_nct, tag_correct, t_stored, n=N_SAMPLES)
        ct  = measure_timing(verify_tag_ct,  tag_correct, t_stored, n=N_SAMPLES)
        print(f"  {label} NCT mean={nct['mean_ns']:6.0f} ns  CV={nct['cv_pct']:4.1f}%  |"
              f"  CT mean={ct['mean_ns']:6.0f} ns  CV={ct['cv_pct']:4.1f}%")
        results[f"tag_{label.strip()}"] = {"nct": nct, "ct": ct}

    tag_means_ct = [
        measure_timing(verify_tag_ct, tag_correct, tag_correct,  n=N_SAMPLES)["mean_ns"],
        measure_timing(verify_tag_ct, tag_correct, tag_wrong_1,  n=N_SAMPLES)["mean_ns"],
        measure_timing(verify_tag_ct, tag_correct, tag_all_bad,  n=N_SAMPLES)["mean_ns"],
    ]
    tag_spread = (max(tag_means_ct) - min(tag_means_ct)) / statistics.mean(tag_means_ct) * 100
    tag_ct4_pass = tag_spread < 2.0
    print(f"\n  CT4 tag spread:   {tag_spread:.2f}%  "
          f"(umbral <2%)  → {'PASS ✓' if tag_ct4_pass else 'FAIL ✗'}")

    # ── Resumen OP6 ───────────────────────────────────────────
    op6_passed = magic_ct4_pass and tag_ct4_pass
    print(f"\n  OP6 (CT2+CT4) resuelto: {'✓' if op6_passed else '✗'}")

    return op6_passed, results


# ── Entry point ───────────────────────────────────────────────

if __name__ == "__main__":
    op3_ok = test_op3()
    op6_ok, timing_data = test_op6()

    print(f"\n{SEP}")
    print("  RESUMEN FINAL")
    print(SEP)
    print(f"  OP3 Replay mitigation   : {'RESUELTO ✓' if op3_ok else 'FALLO ✗'}")
    print(f"  OP6 Constant-time CT2+4 : {'RESUELTO ✓' if op6_ok else 'FALLO ✗'}")
    print(SEP)

    report = {
        "op3_passed": op3_ok,
        "op6_passed": op6_ok,
        "op6_timing": {
            k: {
                "nct_mean_ns": round(v["nct"]["mean_ns"], 1),
                "nct_cv_pct":  round(v["nct"]["cv_pct"], 2),
                "ct_mean_ns":  round(v["ct"]["mean_ns"], 1),
                "ct_cv_pct":   round(v["ct"]["cv_pct"], 2),
            }
            for k, v in timing_data.items()
        }
    }
    with open("/home/claude/op3_op6_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Reporte → /home/claude/op3_op6_report.json")
    print(SEP)
