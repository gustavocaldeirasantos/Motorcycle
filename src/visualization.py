"""Módulo de visualização — funções de plotagem reutilizáveis."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import config

sns.set_theme(style="whitegrid", palette="flare")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.titleweight"] = "bold"


def _salvar(fig: plt.Figure, nome: str | None) -> None:
    if nome:
        config.ensure_directories()
        caminho = config.FIGURES_DIR / nome
        fig.savefig(caminho, bbox_inches="tight")
        print(f"[visualization] Figura salva em {caminho}")


def plot_distribuicao_preco(df: pd.DataFrame, salvar_como: str | None = None) -> plt.Figure:
    """Distribuição do preço em escala original e logarítmica."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    serie = df["selling_price"].dropna()
    sns.histplot(serie, kde=True, ax=axes[0], color="#b5446e")
    axes[0].set_title("Preço de venda (escala original)")
    axes[0].set_xlabel("Preço (₹)")
    sns.histplot(np.log10(serie), kde=True, ax=axes[1], color="#6d597a")
    axes[1].set_title("log10(Preço de venda)")
    axes[1].set_xlabel("log10(Preço)")
    fig.tight_layout()
    _salvar(fig, salvar_como)
    return fig


def plot_boxplots_outliers(df: pd.DataFrame, salvar_como: str | None = None) -> plt.Figure:
    """Boxplots padronizados (z-score) para visualizar outliers lado a lado."""
    cols = ["selling_price", "km_driven", "age", "owner_num"]
    padronizado = df[cols].apply(lambda s: (s - s.mean()) / s.std())
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=padronizado, ax=ax, orient="h")
    ax.set_title("Boxplots padronizados (z-score) — detecção de outliers")
    ax.set_xlabel("Desvios-padrão em relação à média")
    fig.tight_layout()
    _salvar(fig, salvar_como)
    return fig


def plot_matriz_correlacao(matriz: pd.DataFrame, titulo: str, salvar_como: str | None = None) -> plt.Figure:
    """Heatmap de uma matriz de correlação."""
    fig, ax = plt.subplots(figsize=(6.5, 5))
    mascara = np.triu(np.ones_like(matriz, dtype=bool))
    sns.heatmap(matriz, mask=mascara, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, square=True, linewidths=0.5, ax=ax)
    ax.set_title(titulo)
    fig.tight_layout()
    _salvar(fig, salvar_como)
    return fig


def plot_preco_vs_idade(df: pd.DataFrame, salvar_como: str | None = None) -> plt.Figure:
    """Dispersão preço × idade (preço em log) com reta de tendência."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.regplot(data=df, x="age", y="selling_price",
                scatter_kws={"alpha": 0.3, "s": 22}, line_kws={"color": "#b5446e"}, ax=ax)
    ax.set(yscale="log")
    ax.set_title("Preço × idade da moto (preço em escala log)")
    ax.set_xlabel("Idade (anos)")
    ax.set_ylabel("Preço (₹)")
    fig.tight_layout()
    _salvar(fig, salvar_como)
    return fig


def plot_preco_por_marca(resumo: pd.DataFrame, salvar_como: str | None = None) -> plt.Figure:
    """Barras do preço mediano por marca (saída de ``preco_por_marca``)."""
    dados = resumo.sort_values("preco_mediano")
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(x=dados["preco_mediano"], y=dados.index.astype(str), ax=ax, palette="flare")
    ax.set_title("Preço mediano por marca")
    ax.set_xlabel("Preço mediano (₹)")
    ax.set_ylabel("")
    fig.tight_layout()
    _salvar(fig, salvar_como)
    return fig


def plot_importancias(importancias: pd.Series, salvar_como: str | None = None) -> plt.Figure:
    """Barras horizontais da importância das variáveis (Random Forest)."""
    dados = importancias.sort_values()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(dados.index.astype(str), dados.values, color="#6d597a")
    ax.set_title("Importância das variáveis — Random Forest")
    ax.set_xlabel("Importância relativa")
    fig.tight_layout()
    _salvar(fig, salvar_como)
    return fig


def plot_clusters(df_clusterizado: pd.DataFrame, salvar_como: str | None = None) -> plt.Figure:
    """Dispersão idade × preço colorida pelos clusters do K-Means."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df_clusterizado, x="age", y="selling_price",
                    hue="cluster", palette="flare", alpha=0.6, s=28, ax=ax)
    ax.set(yscale="log")
    ax.set_title("Segmentos de motos identificados por K-Means")
    ax.set_xlabel("Idade (anos)")
    ax.set_ylabel("Preço (₹)")
    ax.legend(title="Cluster")
    fig.tight_layout()
    _salvar(fig, salvar_como)
    return fig


if __name__ == "__main__":
    from src.data_loader import load_raw_data
    from src.eda_advanced import matriz_correlacao
    from src.preprocessing import clean_data

    df = clean_data(load_raw_data())
    plot_distribuicao_preco(df, salvar_como="dist.png")
    plot_matriz_correlacao(matriz_correlacao(df), "Correlação", salvar_como="corr.png")
    print("Figuras de exemplo geradas.")
