"""Compara uma base regerada com a base entregue no pacote.

Uso (na pasta codigo/):
    python conferir_reproducao.py <base_regerada.csv> <base_do_pacote.csv>

Criterio: mesmo numero de linhas e colunas, `bits` e `y` identicos e, em cada
coluna numerica, |dif| <= ATOL + RTOL * |valor do pacote|. O fluxo de potencia
converge com tolerancia 1e-6 (src/dss_builder.py, "Set tolerance=1e-6") e parte
do estado anterior do motor OpenDSS ("a quente"); com varios processos
(--workers) a ordem das amostras em cada processo muda, e grandezas como o
Pprev podem diferir em poucos 1e-6 relativos (medido: ate 2,3e-6). A coluna
`eval_ms` (tempo de execucao) e ignorada.
Codigo de saida: 0 = reproduziu; 1 = divergiu.
"""
import sys

import numpy as np
import pandas as pd

FEATURES = ["Pprev_kW", "Scc_MVA", "Pn_kW", "Ip_In", "cosphi_p", "eta", "C_N1"]
IGNORAR = {"eval_ms"}
ATOL = 1e-6
RTOL = 1e-5


def main(nova: str, ref: str) -> int:
    a = pd.read_csv(nova, dtype={"bits": str})
    b = pd.read_csv(ref, dtype={"bits": str})
    ok = True
    print(f"linhas: regerada {len(a)} | pacote {len(b)}")
    if len(a) != len(b) or list(a.columns) != list(b.columns):
        print("DIVERGE: numero de linhas ou colunas diferente")
        return 1
    for c in ("bits", "y"):
        n = int((a[c].astype(str) != b[c].astype(str)).sum())
        print(f"{c}: {n} linhas diferentes")
        ok &= n == 0
    for c in a.columns:
        if c in IGNORAR or c in ("bits", "y"):
            continue
        if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c]):
            x = a[c].to_numpy(dtype=float)
            r = b[c].to_numpy(dtype=float)
            ok_nan = bool((np.isnan(x) == np.isnan(r)).all())
            m_ = ~np.isnan(r)
            d = np.abs(x[m_] - r[m_])
            lim = ATOL + RTOL * np.abs(r[m_])
            m = float(d.max()) if len(d) else 0.0
            rel = float((d / np.maximum(np.abs(r[m_]), 1e-300)).max()) if len(d) else 0.0
            fora = int((d > lim).sum())
            if c in FEATURES or m > 0:
                print(f"{c}: max|dif| = {m:.3g} (relativa {rel:.2g}); fora da tolerancia: {fora}")
            ok &= ok_nan and fora == 0
        else:
            n = int((a[c].astype(str) != b[c].astype(str)).sum())
            if n:
                print(f"{c}: {n} linhas diferentes (texto)")
                ok = False
    print("REPRODUZIU" if ok else "DIVERGE")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1], sys.argv[2]))
