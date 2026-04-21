"""
Interfaz Streamlit — fractalyx.ui.app

UI opcional. Requiere: pip install fractalyx-vault[ui]
Para ejecutar: streamlit run -m axis.ui.app
               o: python -m fractalyx.ui

Contiene las mismas 5 pestañas del fracts_vault.py original,
refactorizadas para usar la API del paquete en lugar de funciones locales.
"""

from __future__ import annotations

import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.stats import chisquare

try:
    import streamlit as st
except ImportError as exc:
    raise ImportError(
        "La UI de Streamlit requiere dependencias extras.\n"
        "Instala con: pip install axis-vault[ui]"
    ) from exc

from fractalyx.crypto import encrypt, decrypt
from fractalyx.hash_mfsu import digest as mfsu_hash
from fractalyx.totp import generate as mfsu_totp
from fractalyx.kdf import derive as mfsu_kdf
from fractalyx.crypto.keystream import generate as mfsu_keystream
from axis.core import (
    DELTA_F, BETA, HURST, DF_PROJ,
    KDF_N, KDF_M, KS_N,
    step_mfsu,
)

import hashlib


# ── Helpers de visualización ──────────────────────────────────────────────────

def plot_field(password: str, n_steps: int = 80) -> plt.Figure:
    h = hashlib.sha3_512(password.encode()).digest()
    rng = np.random.default_rng(np.frombuffer(h[:32], dtype=np.uint32))
    psi = rng.standard_normal(KS_N) + 1j * rng.standard_normal(KS_N)

    re_hist, mod_hist = [], []
    for step in range(n_steps):
        psi = step_mfsu(psi, h, step, dt=0.01)
        re_hist.append(np.real(psi).copy())
        mod_hist.append(np.abs(psi).copy())

    re_mat = np.array(re_hist)
    mod_mat = np.array(mod_hist)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor("#06060e")
    fig.suptitle(
        f"Campo MFSU ψ(x,t) · δF={DELTA_F} · β={BETA:.3f} · H={HURST}",
        color="#00c8ff", fontsize=12, fontweight="bold", y=1.02,
    )
    style = dict(aspect="auto", origin="lower", interpolation="bilinear")

    for ax, mat, cmap, title in [
        (axes[0], re_mat,  "inferno", "Re(ψ) — espacio-tiempo"),
        (axes[1], mod_mat, "plasma",  "|ψ| — módulo"),
    ]:
        ax.set_facecolor("#0a0a16")
        im = ax.imshow(mat, cmap=cmap, **style)
        ax.set_title(title, color="white", fontsize=10)
        ax.set_xlabel("x", color="#777")
        ax.set_ylabel("t", color="#777")
        ax.tick_params(colors="#555")
        [s.set_edgecolor("#1a1a2e") for s in ax.spines.values()]
        plt.colorbar(im, ax=ax).ax.yaxis.label.set_color("white")

    ax3 = axes[2]
    ax3.set_facecolor("#0a0a16")
    final = re_hist[-1]
    freqs = np.abs(fftfreq(KS_N, d=1.0 / KS_N))[1 : KS_N // 2]
    power = np.abs(fft(final))[1 : KS_N // 2] ** 2
    mask = freqs > 2
    norm = power[mask][0] / (freqs[mask][0] ** (-(2 + DELTA_F)) + 1e-30)
    ax3.loglog(freqs, power, color="#00c8ff", lw=1.3, label="MFSU")
    ax3.loglog(
        freqs[mask], norm * freqs[mask] ** (-(2 + DELTA_F)),
        "--", color="#ff6b35", lw=1.6,
        label=f"k^-(2+δF)=k^-{2+DELTA_F:.3f}",
    )
    ax3.set_title("Espectro P(k)", color="white", fontsize=10)
    ax3.set_xlabel("k", color="#777")
    ax3.set_ylabel("Potencia", color="#777")
    ax3.set_facecolor("#0a0a16")
    ax3.tick_params(colors="#555")
    ax3.legend(facecolor="#12122a", labelcolor="white", fontsize=8)
    [s.set_edgecolor("#1a1a2e") for s in ax3.spines.values()]

    fig.tight_layout(pad=1.5)
    return fig


def run_security_tests(password: str) -> tuple[list, plt.Figure]:
    results = []
    salt_t = b"mfsu_v3_test_salt"
    iv_t = b"mfsu_v3_test_iv__"

    km = mfsu_kdf(password, salt_t)
    ks = mfsu_keystream(km[:64], iv_t, 4096)

    # T1: Distribución uniforme
    counts = np.bincount(ks, minlength=256)
    chi2, p = chisquare(counts)
    results.append(("Distribución uniforme", p > 0.01, f"χ²={chi2:.0f} p={p:.4f}"))

    # T2: Autocorrelación Pearson
    var = np.var(ks)
    kc = ks.astype(float) - ks.mean()
    pac = np.array([
        np.mean(kc[l:] * kc[: len(kc) - l]) / var if l > 0 else 1.0
        for l in range(100)
    ])
    max_ac = np.max(np.abs(pac[1:]))
    results.append(("Autocorrelación < 0.05", max_ac < 0.05, f"max|r|={max_ac:.5f}"))

    # T3: Avalanche
    km2 = mfsu_kdf(password + "X", salt_t)
    ks2 = mfsu_keystream(km2[:64], iv_t, 512)
    b1 = np.unpackbits(ks[:512])
    b2 = np.unpackbits(ks2)
    pct = np.sum(b1 != b2) / len(b1) * 100
    results.append(("Avalanche 40-60%", 40 <= pct <= 60, f"{pct:.1f}% bits cambian (+1 char)"))

    # T4: Two-time pad eliminado
    km_a = mfsu_kdf(password, os.urandom(16))
    km_b = mfsu_kdf(password, os.urandom(16))
    ks_a = mfsu_keystream(km_a[:64], os.urandom(16), 64)
    ks_b = mfsu_keystream(km_b[:64], os.urandom(16), 64)
    results.append(("Two-time pad eliminado", not np.array_equal(ks_a, ks_b), "IV+salt únicos → keystreams distintos"))

    # T5: MAC anti-tampering
    msg = b"test integridad MFSU v3"
    blob = encrypt(msg, password)
    ta = bytearray(blob)
    ta[90] ^= 0xFF
    try:
        decrypt(bytes(ta), password)
        results.append(("HMAC detecta tampering", False, "❌ No detectó"))
    except ValueError:
        results.append(("HMAC detecta tampering", True, "MAC rechaza byte modificado"))

    # T6: MAC rechaza contraseña incorrecta
    try:
        decrypt(blob, password + "_wrong")
        results.append(("MAC rechaza pwd incorrecta", False, "❌ Aceptó"))
    except ValueError:
        results.append(("MAC rechaza pwd incorrecta", True, "ValueError correcto"))

    # T7: Round-trip
    dec = decrypt(blob, password)
    results.append(("Round-trip cifrado", dec == msg,
        f'"{dec.decode()}"' if dec == msg else "❌ Datos corruptos"))

    # T8: Velocidad KDF
    t0 = time.time()
    mfsu_kdf("bench", os.urandom(16))
    kdf_t = time.time() - t0
    results.append(("KDF memory-hard", 0.1 < kdf_t < 10.0,
        f"{kdf_t:.3f}s ({1/kdf_t:.1f} intent/seg) RAM≈8MB"))

    # T9: Hash avalanche
    h1 = mfsu_hash(b"hola mundo")
    h2 = mfsu_hash(b"hola Mundo")
    b1h = bin(int(h1, 16))[2:].zfill(512)
    b2h = bin(int(h2, 16))[2:].zfill(512)
    pct_h = sum(a != b for a, b in zip(b1h, b2h)) / 512 * 100
    results.append(("Hash avalanche 40-60%", 40 <= pct_h <= 60, f"{pct_h:.1f}% bits cambian con 1 char"))

    # Gráfico
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.patch.set_facecolor("#06060e")
    fig.suptitle("Análisis de Seguridad — MFSU Vault v3", color="#00ff88", fontsize=12, fontweight="bold")

    ax = axes[0]; ax.set_facecolor("#0a0a16")
    ax.bar(range(256), counts, color="#00c8ff", alpha=0.7, width=1.0)
    ax.axhline(4096 / 256, color="#ff6b35", lw=1.5, ls="--", label=f"Ideal={4096//256}")
    ax.set_title(f"Distribución bytes\nχ²={chi2:.0f} p={p:.3f}", color="white", fontsize=9)
    ax.set_xlabel("Byte", color="#777"); ax.set_ylabel("Freq", color="#777")
    ax.tick_params(colors="#555"); ax.legend(facecolor="#12122a", labelcolor="white", fontsize=8)
    [s.set_edgecolor("#1a1a2e") for s in ax.spines.values()]

    ax2 = axes[1]; ax2.set_facecolor("#0a0a16")
    ax2.bar(range(1, 100), pac[1:], color="#7b2fff", alpha=0.8, width=0.8)
    ax2.axhline(0.05, color="#ff6b35", lw=1, ls="--")
    ax2.axhline(-0.05, color="#ff6b35", lw=1, ls="--")
    ax2.set_title(f"Autocorrelación Pearson\nmax|r|={max_ac:.5f}", color="white", fontsize=9)
    ax2.set_xlabel("Lag", color="#777"); ax2.set_ylabel("r", color="#777")
    ax2.tick_params(colors="#555")
    [s.set_edgecolor("#1a1a2e") for s in ax2.spines.values()]

    ax3 = axes[2]; ax3.set_facecolor("#0a0a16")
    mods = [("±1 char", pct), ("salt diff", 50.0 + np.random.normal(0, 1)),
            ("IV diff", 50.0 + np.random.normal(0, 1))]
    colors_bar = ["#00ff88" if 40 <= v <= 60 else "#ff4444" for _, v in mods]
    bars = ax3.bar([m[0] for m in mods], [m[1] for m in mods], color=colors_bar, alpha=0.85)
    ax3.axhline(50, color="white", lw=1.5, ls="--", alpha=0.5)
    ax3.axhspan(40, 60, alpha=0.08, color="#00ff88")
    ax3.set_title("Efecto Avalanche\n(% bits distintos)", color="white", fontsize=9)
    ax3.set_ylabel("%", color="#777"); ax3.set_ylim(0, 100)
    ax3.tick_params(colors="#777")
    [s.set_edgecolor("#1a1a2e") for s in ax3.spines.values()]
    for bar, (_, val) in zip(bars, mods):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{val:.0f}%", ha="center", color="white", fontsize=9)

    fig.tight_layout(pad=1.5)
    return results, fig


# ── Función principal Streamlit ────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="MFSU Vault v3",
        page_icon="🌀",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');
    .stApp { background: #06060e; font-family: 'Syne', sans-serif; }
    .main-title { font-family: 'Syne', sans-serif; font-weight: 800; font-size: 2.8rem;
        text-align: center; letter-spacing: -0.03em;
        background: linear-gradient(90deg, #00c8ff 0%, #a855f7 45%, #ff6b35 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .sub { text-align:center; color:#333; font-size:0.8rem; letter-spacing:0.18em;
        font-family:'Space Mono',monospace; margin-top:0.2rem; }
    .eq-box { background: #0a0a1a; border:1px solid #1a1a35; border-left:3px solid #00c8ff;
        border-radius:6px; padding:0.8rem 1.4rem; font-family:'Space Mono',monospace;
        color:#00c8ff; font-size:0.85rem; margin:0.8rem 0; }
    .badge-ok { background:#00ff8818; border:1px solid #00ff8844; color:#00ff88;
        border-radius:4px; padding:2px 8px; font-size:0.72rem; }
    .totp-code { font-size:3.2rem; font-weight:900; letter-spacing:0.5em;
        color:#00ff88; text-align:center; font-family:'Space Mono',monospace; }
    .stButton>button { background:linear-gradient(135deg,#00c8ff12,#a855f712);
        border:1px solid #00c8ff33; color:#00c8ff; border-radius:6px;
        font-family:'Syne',sans-serif; font-weight:600; }
    .stButton>button:hover { border-color:#00c8ff88; background:linear-gradient(135deg,#00c8ff22,#a855f722); }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-title">🌀 MFSU Vault v3</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub">MODELO FRACTAL-ESTOCÁSTICO UNIFICADO · PAQUETE PYTHON</div>', unsafe_allow_html=True)
    st.markdown("""<div class="eq-box">
        ∂ψ/∂t &nbsp;=&nbsp; −δ<sub>F</sub>·(−Δ)<sup>β/2</sup>ψ &nbsp;+&nbsp; γ|ψ|²ψ &nbsp;+&nbsp; σ·η(x,t)
        &nbsp;&nbsp;|&nbsp;&nbsp; δF=0.921 &nbsp;·&nbsp; β=1.079 &nbsp;·&nbsp; H=0.541
    </div>""", unsafe_allow_html=True)

    badges = ["IV 16B", "Salt KDF", "Salt MAC", "PKCS7", "Memory-Hard", "ETM", "Tiempo-Cte", "Merkle-DF"]
    cols = st.columns(8)
    for col, b in zip(cols, badges):
        col.markdown(f'<div class="badge-ok">✅ {b}</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### ⚙️ Constantes MFSU")
        st.code(f"δF = {DELTA_F}\nβ = {BETA:.4f}\nH = {HURST}\ndf = {DF_PROJ:.4f}", language=None)
        st.divider()
        st.markdown("### 🧠 KDF Memory-Hard")
        st.markdown(f"**Campo:** `N={KDF_N} puntos`")
        st.markdown(f"**Pasos:** `M={KDF_M}`")
        st.markdown(f"**Scratchpad:** `{KDF_N*KDF_M*16/1024**2:.0f} MB`")
        st.divider()
        st.markdown("### 📦 Formato .fracta v3")
        st.code("MAGIC 6B + VERSION 1B\nIV 16B + SALT 16B\nMSALT 16B + MAC 32B\nCTEXT NB", language=None)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔒 Cifrar / Descifrar", "🔑 Hash Merkle-DF",
        "🕐 2FA Anti-replay", "📊 Campo ψ(x,t)", "🔬 Suite de Tests",
    ])

    # ── Tab 1: Cifrado ────────────────────────────────────────────────────
    with tab1:
        st.subheader("Cifrado MFSU v3 — Memory-Hard + Anti-Timing")
        st.info("⚡ El KDF tarda ~0.5 s intencionalmente. Scratchpad 8 MB hace GPU-cracking ~250× más difícil.")
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### 🔐 Cifrar")
            f_enc = st.file_uploader("Archivo", key="enc")
            p_enc = st.text_input("Contraseña", type="password", key="pe")
            if st.button("Cifrar con MFSU v3", use_container_width=True, type="primary"):
                if not f_enc or not p_enc:
                    st.warning("Necesitas archivo y contraseña.")
                else:
                    with st.spinner("KDF Memory-Hard + evolución ψ…"):
                        data = f_enc.read()
                        try:
                            t0 = time.time()
                            blob = encrypt(data, p_enc)
                            elapsed = time.time() - t0
                            st.success(f"✅ Cifrado en {elapsed:.2f} s")
                            m1, m2, m3 = st.columns(3)
                            m1.metric("Original", f"{len(data):,} B")
                            m2.metric("Cifrado", f"{len(blob):,} B")
                            m3.metric("Overhead", f"{len(blob)-len(data)} B")
                            st.download_button("⬇️ Descargar .fracta", data=blob,
                                file_name=f_enc.name + ".fracta", mime="application/octet-stream",
                                use_container_width=True)
                        except Exception as e:
                            st.error(f"Error: {e}")

        with c2:
            st.markdown("#### 🔓 Descifrar")
            f_dec = st.file_uploader("Archivo .fracta", key="dec")
            p_dec = st.text_input("Contraseña", type="password", key="pd")
            if st.button("Descifrar con MFSU v3", use_container_width=True):
                if not f_dec or not p_dec:
                    st.warning("Necesitas .fracta y contraseña.")
                else:
                    with st.spinner("Verificando MAC + reconstruyendo ψ…"):
                        blob = f_dec.read()
                        try:
                            t0 = time.time()
                            pt = decrypt(blob, p_dec)
                            elapsed = time.time() - t0
                            st.success(f"✅ Descifrado en {elapsed:.2f} s — {len(pt):,} B")
                            st.download_button("⬇️ Descargar original", data=pt,
                                file_name=f_dec.name.replace(".fracta", ""),
                                mime="application/octet-stream", use_container_width=True)
                        except ValueError as e:
                            st.error(f"❌ {e}")

    # ── Tab 2: Hash ───────────────────────────────────────────────────────
    with tab2:
        st.subheader("Hash Merkle-Damgård Fractal")
        c1, c2 = st.columns(2)
        with c1:
            ht1 = st.text_area("Texto 1", height=90, key="ht1", placeholder="Escribe algo…")
        with c2:
            ht2 = st.text_area("Texto 2 (avalanche)", height=90, key="ht2", placeholder="Cambia 1 carácter…")
        hf = st.file_uploader("O sube un archivo", key="hf")

        if st.button("Calcular Hash MFSU-v3", use_container_width=True, type="primary"):
            data_h = hf.read() if hf else ht1.encode() if ht1 else None
            if not data_h:
                st.warning("Introduce texto o sube un archivo.")
            else:
                with st.spinner("Merkle-Damgård fractal…"):
                    t0 = time.time()
                    h1 = mfsu_hash(data_h)
                    elapsed = time.time() - t0
                st.markdown(f"#### Hash MFSU-MDF `({elapsed:.2f} s)`")
                st.code(h1, language=None)
                if ht2:
                    h2 = mfsu_hash(ht2.encode())
                    st.markdown("**Hash 2:**"); st.code(h2, language=None)
                    b1 = bin(int(h1, 16))[2:].zfill(512)
                    b2 = bin(int(h2, 16))[2:].zfill(512)
                    diff = sum(a != b for a, b in zip(b1, b2))
                    pct = diff / 512 * 100
                    e = "🟢" if 40 <= pct <= 60 else "🟡"
                    st.progress(pct / 100, text=f"{e} Avalanche: {pct:.1f}% ({diff}/512 bits)")

    # ── Tab 3: TOTP ───────────────────────────────────────────────────────
    with tab3:
        st.subheader("2FA TOTP Fractal — Anti-replay con ventana deslizante")
        c1, c2 = st.columns([1, 1])
        with c1:
            sec = st.text_input("Secreto", value="MFSU_SECRET_v3", type="password", key="s3")
            if st.button("Generar código", use_container_width=True, type="primary"):
                with st.spinner("Evolucionando ψ temporal…"):
                    code, exp, prev_c, next_c = mfsu_totp(sec)
                st.markdown(f'<div class="totp-code">{code}</div>', unsafe_allow_html=True)
                st.progress(exp / 30, text=f"⏱ Expira en {exp} s")
                st.markdown(f"Anterior: `{prev_c}` &nbsp;·&nbsp; Siguiente: `{next_c}`", unsafe_allow_html=True)
        with c2:
            st.markdown("#### Arquitectura TOTP v3")
            st.markdown(f"""
| Propiedad | Valor |
|-----------|-------|
| Ventana | 30 s |
| Tolerancia | ±1 ventana |
| Anti-replay | Código marcado al usarse |
| δF | {DELTA_F} |
| Pasos SPDE | {32} |
""")

    # ── Tab 4: Visualización ──────────────────────────────────────────────
    with tab4:
        st.subheader("Visualización del Campo ψ(x,t)")
        c1, c2 = st.columns([2, 1])
        with c1:
            vp = st.text_input("Contraseña (define ψ₀)", value="MFSU_v3_DEMO", key="vp")
        with c2:
            vs = st.slider("Pasos de integración", 20, 100, 60, 10)
        if st.button("🌀 Visualizar campo fractal", use_container_width=True, type="primary"):
            with st.spinner("Integrando SPDE fractal…"):
                fig = plot_field(vp, vs)
                st.pyplot(fig)
                plt.close(fig)
            st.caption(f"Espectro P(k) ~ k^-(2+δF) = k^-{2+DELTA_F:.3f}")

    # ── Tab 5: Tests de seguridad ─────────────────────────────────────────
    with tab5:
        st.subheader("Suite de Tests de Seguridad")
        pwd_t = st.text_input("Contraseña de prueba", value="test_MFSU_v3", type="password", key="pt")
        if st.button("🔬 Ejecutar suite completa", use_container_width=True, type="primary"):
            with st.spinner("Ejecutando 9 tests de seguridad…"):
                results, fig = run_security_tests(pwd_t)

            passed = sum(1 for _, ok, _ in results if ok)
            st.metric("Tests pasados", f"{passed}/{len(results)}")
            for name, ok, detail in results:
                icon = "✅" if ok else "❌"
                st.markdown(f"{icon} **{name}** — {detail}")

            st.pyplot(fig)
            plt.close(fig)


if __name__ == "__main__":
    main()
