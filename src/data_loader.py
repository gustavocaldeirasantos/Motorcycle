"""Etapa 1 — Carregamento dos dados brutos."""
from __future__ import annotations

import pandas as pd

from src import config


def load_raw_data() -> pd.DataFrame:
    """Carrega o dataset bruto de motos usadas.

    Returns:
        DataFrame com as colunas originais (sem tratamento).
    """
    df = pd.read_csv(config.RAW_FILE)
    print(f"[data_loader] {len(df):,} motos carregadas de {config.RAW_FILE.name}.")
    return df


if __name__ == "__main__":
    raw = load_raw_data()
    print(raw.head())
    print(raw.shape)
