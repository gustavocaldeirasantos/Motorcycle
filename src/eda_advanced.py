"""Etapa 4 — Análise Exploratória Avançada (EDA).

* Detecção de outliers (IQR e Z-score).
* Correlação entre as métricas (Pearson e Spearman).
* Análise bivariada com o preço e preço médio por marca.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src import config


def detectar_outliers_iqr(df: pd.DataFrame, coluna: str, multiplicador: float = config.IQR_MULTIPLIER) -> pd.Series:
    """Outliers pelo método do Intervalo Interquartil (Tukey)."""
    serie = df[coluna]
    q1, q3 = serie.quantile([0.25, 0.75])
    iqr = q3 - q1
    return (serie < q1 - multiplicador * iqr) | (serie > q3 + multiplicador * iqr)


def detectar_outliers_zscore(df: pd.DataFrame, coluna: str, limite: float = config.ZSCORE_THRESHOLD) -> pd.Series:
    """Outliers pelo Z-score (|z| > limite)."""
    z = np.abs(stats.zscore(df[coluna], nan_policy="omit"))
    return pd.Series(z > limite, index=df.index)


def resumo_outliers(df: pd.DataFrame, colunas: list[str] | None = None) -> pd.DataFrame:
    """Compara a contagem de outliers por IQR e por Z-score."""
    colunas = colunas or ["selling_price", "km_driven", "age"]
    registros: list[dict[str, object]] = []
    for col in colunas:
        iqr_mask = detectar_outliers_iqr(df, col)
        z_mask = detectar_outliers_zscore(df, col)
        registros.append(
            {
                "variavel": col,
                "outliers_iqr": int(iqr_mask.sum()),
                "outliers_iqr_%": round(iqr_mask.mean() * 100, 2),
                "outliers_zscore": int(z_mask.sum()),
                "outliers_zscore_%": round(z_mask.mean() * 100, 2),
            }
        )
    return pd.DataFrame(registros).set_index("variavel")


def matriz_correlacao(df: pd.DataFrame, metodo: str = "pearson", colunas: list[str] | None = None) -> pd.DataFrame:
    """Matriz de correlação (``pearson`` linear ou ``spearman`` monotônica)."""
    colunas = colunas or ["selling_price", "km_driven", "age", "owner_num"]
    return df[colunas].corr(method=metodo)


def analise_bivariada_preco(df: pd.DataFrame) -> pd.DataFrame:
    """Correlação (Pearson e Spearman) de cada métrica com o preço de venda."""
    alvo = "selling_price"
    registros: list[dict[str, object]] = []
    for col in ["age", "km_driven", "owner_num", "km_por_ano"]:
        sub = df[[alvo, col]].dropna()
        r_p, p_p = stats.pearsonr(sub[alvo], sub[col])
        r_s, p_s = stats.spearmanr(sub[alvo], sub[col])
        registros.append(
            {
                "variavel": col,
                "pearson_r": round(float(r_p), 4),
                "pearson_p": float(p_p),
                "spearman_r": round(float(r_s), 4),
                "spearman_p": float(p_s),
            }
        )
    return pd.DataFrame(registros).set_index("variavel")


def preco_por_marca(df: pd.DataFrame, top: int = config.TOP_N_BRANDS) -> pd.DataFrame:
    """Preço médio/mediano e taxa de revenda por marca (mais frequentes)."""
    top_brands = df["brand"].value_counts().head(top).index
    sub = df[df["brand"].isin(top_brands)]
    resumo = (
        sub.groupby("brand")
        .agg(
            n_motos=("selling_price", "size"),
            preco_medio=("selling_price", "mean"),
            preco_mediano=("selling_price", "median"),
            idade_media=("age", "mean"),
            resale_ratio_medio=("resale_ratio", "mean"),
        )
        .sort_values("preco_mediano", ascending=False)
    )
    return resumo


if __name__ == "__main__":
    from src.data_loader import load_raw_data
    from src.preprocessing import clean_data

    df = clean_data(load_raw_data())
    pd.set_option("display.width", 180)
    print("\n=== OUTLIERS ===")
    print(resumo_outliers(df))
    print("\n=== CORRELAÇÃO (SPEARMAN) ===")
    print(matriz_correlacao(df, "spearman").round(3))
    print("\n=== BIVARIADA COM PREÇO ===")
    print(analise_bivariada_preco(df))
    print("\n=== PREÇO POR MARCA ===")
    print(preco_por_marca(df))
