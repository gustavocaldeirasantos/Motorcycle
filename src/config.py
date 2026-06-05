"""Configuração central do projeto: caminhos, parâmetros e constantes."""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Caminhos
# --------------------------------------------------------------------------- #
ROOT_DIR: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = ROOT_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

REPORTS_DIR: Path = ROOT_DIR / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"

RAW_FILE: Path = RAW_DATA_DIR / "bike.csv"
PROCESSED_FILE: Path = PROCESSED_DATA_DIR / "motos_limpo.parquet"

# --------------------------------------------------------------------------- #
# Parâmetros de análise
# --------------------------------------------------------------------------- #
RANDOM_STATE: int = 42

IQR_MULTIPLIER: float = 1.5
ZSCORE_THRESHOLD: float = 3.0

ALPHA: float = 0.05
TEST_SIZE: float = 0.25
KMEANS_N_CLUSTERS: int = 4

# Ano de referência da extração (o dado vai até 2020).
REFERENCE_YEAR: int = 2020

# Limites plausíveis de mercado (limpeza por regra de negócio).
MIN_PRICE: float = 3_000.0
MAX_PRICE: float = 2_000_000.0
MAX_KM: float = 200_000.0          # acima disso, provável erro de digitação
TOP_N_BRANDS: int = 6

NUMERIC_COLS: list[str] = ["selling_price", "km_driven", "age", "owner_num"]


def ensure_directories() -> None:
    """Garante que os diretórios de saída existam."""
    for directory in (PROCESSED_DATA_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)
