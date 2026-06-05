"""Etapa 5 — Estatística inferencial.

* Teste t de Welch — preço de motos de 1º dono vs. demais.
* ANOVA — preço entre as principais marcas.
* Correlação de Pearson/Spearman com significância.
* Intervalos de confiança — paramétrico (t) e bootstrap.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from src import config


@dataclass
class ResultadoTeste:
    nome: str
    estatistica: float
    p_valor: float
    significativo: bool
    interpretacao: str

    def __str__(self) -> str:
        return (
            f"[{self.nome}] estatística={self.estatistica:.4f}, "
            f"p-valor={self.p_valor:.4g} -> {self.interpretacao}"
        )


def teste_t_primeiro_dono(df: pd.DataFrame, coluna: str = "selling_price") -> ResultadoTeste:
    """Teste t de Welch: motos de 1º dono valem mais que as de múltiplos donos?

    H0: as médias de preço dos dois grupos são iguais.
    """
    primeiro = df.loc[df["owner_num"] == 1, coluna].dropna()
    demais = df.loc[df["owner_num"] > 1, coluna].dropna()
    stat, p = stats.ttest_ind(primeiro, demais, equal_var=False)
    significativo = bool(p < config.ALPHA)
    interp = (
        f"Diferença significativa (1º dono={primeiro.mean():,.0f} vs "
        f"demais={demais.mean():,.0f})."
        if significativo
        else f"Sem diferença significativa (1º dono={primeiro.mean():,.0f} vs "
        f"demais={demais.mean():,.0f})."
    )
    return ResultadoTeste(f"Teste t 1º dono ({coluna})", float(stat), float(p), significativo, interp)


def anova_marcas(df: pd.DataFrame, coluna: str = "selling_price", top: int = config.TOP_N_BRANDS) -> ResultadoTeste:
    """ANOVA de uma via: o preço difere entre as principais marcas?

    H0: todas as marcas têm o mesmo preço médio populacional.
    """
    top_brands = df["brand"].value_counts().head(top).index
    grupos = [df.loc[df["brand"] == b, coluna].dropna().values for b in top_brands]
    stat, p = stats.f_oneway(*grupos)
    significativo = bool(p < config.ALPHA)
    interp = (
        "Ao menos uma marca tem preço médio distinto (marca tem valor de mercado)."
        if significativo
        else "Não há diferença de preço entre as marcas."
    )
    return ResultadoTeste("ANOVA preço x marca", float(stat), float(p), significativo, interp)


def intervalo_confianca_t(serie: pd.Series, confianca: float = 1 - config.ALPHA) -> tuple[float, float]:
    """Intervalo de confiança paramétrico para a média (t de Student)."""
    dados = serie.dropna()
    margem = stats.sem(dados) * stats.t.ppf((1 + confianca) / 2, df=len(dados) - 1)
    return float(dados.mean() - margem), float(dados.mean() + margem)


def intervalo_confianca_bootstrap(
    serie: pd.Series, confianca: float = 1 - config.ALPHA, n_reamostragens: int = 5_000
) -> tuple[float, float]:
    """Intervalo de confiança via bootstrap (percentil). Não exige normalidade."""
    dados = serie.dropna().to_numpy()
    rng = np.random.default_rng(config.RANDOM_STATE)
    medias = rng.choice(dados, size=(n_reamostragens, len(dados)), replace=True).mean(axis=1)
    alpha = 1 - confianca
    low, high = np.percentile(medias, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(low), float(high)


def tabela_intervalos_confianca(df: pd.DataFrame, colunas: list[str] | None = None) -> pd.DataFrame:
    """Compara IC paramétrico (t) e bootstrap para a média de cada variável."""
    colunas = colunas or ["selling_price", "km_driven", "age"]
    registros: list[dict[str, object]] = []
    for col in colunas:
        t_low, t_high = intervalo_confianca_t(df[col])
        b_low, b_high = intervalo_confianca_bootstrap(df[col])
        registros.append(
            {
                "variavel": col,
                "media": float(df[col].mean()),
                "ic95_t_inferior": t_low,
                "ic95_t_superior": t_high,
                "ic95_boot_inferior": b_low,
                "ic95_boot_superior": b_high,
            }
        )
    return pd.DataFrame(registros).set_index("variavel")


if __name__ == "__main__":
    from src.data_loader import load_raw_data
    from src.preprocessing import clean_data

    df = clean_data(load_raw_data())
    print(teste_t_primeiro_dono(df))
    print(anova_marcas(df))
    print("\n=== IC (95%) ===")
    pd.set_option("display.width", 180)
    pd.set_option("display.float_format", lambda v: f"{v:,.2f}")
    print(tabela_intervalos_confianca(df))
