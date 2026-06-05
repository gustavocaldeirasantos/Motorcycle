"""Etapa 2 — Limpeza, tratamento de nulos e engenharia de atributos.

Decisões principais:
    * ``ex_showroom_price`` tem ~41% de ausência e é praticamente o preço de
      tabela (de onde o usado deprecia). Para não "vazar" o alvo, NÃO o usamos
      como preditor — apenas calculamos a taxa de revenda no subconjunto que o tem.
    * extraímos a **marca** do nome e a **idade** do ano; o nº de donos vira ordinal.
    * aplicamos regras de negócio para remover registros implausíveis (ex.: uma
      moto com 880.000 km é quase certamente erro de digitação).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

# Normalização de marcas compostas (primeira palavra não basta).
BRAND_FIX: dict[str, str] = {"Royal": "Royal Enfield"}


def _extrair_brand(nome: pd.Series) -> pd.Series:
    """Extrai a marca a partir da primeira palavra do nome do modelo."""
    brand = nome.str.split().str[0]
    return brand.replace(BRAND_FIX)


def _owner_para_num(owner: pd.Series) -> pd.Series:
    """Converte '1st owner', '2nd owner', ... no número de donos (1..4)."""
    return owner.str.extract(r"(\d)").astype(int).iloc[:, 0]


def _criar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria variáveis derivadas com valor analítico."""
    df = df.copy()
    df["brand"] = _extrair_brand(df["name"])
    df["age"] = config.REFERENCE_YEAR - df["year"]
    df["owner_num"] = _owner_para_num(df["owner"])

    # Taxa de revenda (apenas onde há preço de tabela): preço usado / tabela.
    df["resale_ratio"] = df["selling_price"] / df["ex_showroom_price"]

    # Quilometragem média por ano de uso (intensidade de uso).
    df["km_por_ano"] = df["km_driven"] / df["age"].replace(0, 1)
    return df


def _aplicar_regras_negocio(df: pd.DataFrame) -> pd.DataFrame:
    """Remove registros fora de faixas plausíveis de mercado."""
    n_antes = len(df)
    mask = (
        df["selling_price"].between(config.MIN_PRICE, config.MAX_PRICE)
        & (df["km_driven"] <= config.MAX_KM)
        & (df["age"] >= 0)
    )
    df = df.loc[mask].copy()
    print(
        f"[preprocessing] Regras de negócio: {n_antes:,} -> {len(df):,} linhas "
        f"({n_antes - len(df):,} registros implausíveis removidos)."
    )
    return df


def clean_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Pipeline completo de limpeza e engenharia de atributos."""
    df = _criar_features(df_raw)
    df = _aplicar_regras_negocio(df)
    df = df.reset_index(drop=True)
    print(f"[preprocessing] Base analítica final: {df.shape[0]:,} linhas × {df.shape[1]} colunas.")
    return df


def save_processed(df: pd.DataFrame) -> None:
    """Persiste a base limpa em Parquet (com fallback para CSV)."""
    config.ensure_directories()
    try:
        df.to_parquet(config.PROCESSED_FILE, index=False)
        print(f"[preprocessing] Base salva em {config.PROCESSED_FILE}")
    except Exception as exc:  # pragma: no cover
        fallback = config.PROCESSED_FILE.with_suffix(".csv")
        df.to_csv(fallback, index=False)
        print(f"[preprocessing] Parquet indisponível ({exc}); base salva em {fallback}")


if __name__ == "__main__":
    from src.data_loader import load_raw_data

    df_limpo = clean_data(load_raw_data())
    save_processed(df_limpo)
    print(df_limpo[["name", "brand", "age", "owner_num", "selling_price", "km_driven"]].head())
