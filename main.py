"""Pipeline completo — orquestra todas as etapas da análise.

Uso:
    python main.py
"""
from __future__ import annotations

import pandas as pd

from src import config
from src.data_loader import load_raw_data
from src.descriptive_stats import resumo_categorico, tabela_descritiva, teste_normalidade
from src.eda_advanced import (
    analise_bivariada_preco,
    matriz_correlacao,
    preco_por_marca,
    resumo_outliers,
)
from src.inferential_stats import (
    anova_marcas,
    tabela_intervalos_confianca,
    teste_t_primeiro_dono,
)
from src.modeling import segmentar_kmeans, treinar_random_forest, treinar_regressao_linear
from src.preprocessing import clean_data, save_processed
from src import visualization as viz


def _cabecalho(titulo: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {titulo}")
    print("=" * 72)


def executar_pipeline() -> pd.DataFrame:
    pd.set_option("display.width", 180)
    pd.set_option("display.float_format", lambda v: f"{v:,.2f}")
    config.ensure_directories()

    _cabecalho("ETAPA 1-2 | CARREGAMENTO E LIMPEZA")
    df = clean_data(load_raw_data())
    save_processed(df)

    _cabecalho("ETAPA 3 | ESTATÍSTICA DESCRITIVA")
    print(tabela_descritiva(df))
    print("\n-- Normalidade --")
    print(teste_normalidade(df))
    print("\n-- Marcas --")
    print(resumo_categorico(df, "brand"))

    _cabecalho("ETAPA 4 | EDA AVANÇADA")
    print(resumo_outliers(df))
    print("\n-- Correlação com o preço --")
    print(analise_bivariada_preco(df))
    print("\n-- Preço por marca --")
    resumo_marcas = preco_por_marca(df)
    print(resumo_marcas)

    _cabecalho("ETAPA 5 | ESTATÍSTICA INFERENCIAL")
    print(teste_t_primeiro_dono(df))
    print(anova_marcas(df))
    print("\n-- Intervalos de confiança (95%) --")
    print(tabela_intervalos_confianca(df))

    _cabecalho("ETAPA 6 | MODELAGEM")
    reg = treinar_regressao_linear(df)
    print(f"Regressão OLS (log do preço) | R² treino={reg.r2_treino:.3f} | teste={reg.r2_teste:.3f}")
    print(reg.coeficientes.round(4))
    rf = treinar_random_forest(df)
    print(f"\nRandom Forest | R² teste={rf.r2_teste:.3f}")
    print(rf.importancias)
    clu = segmentar_kmeans(df)
    print(f"\nK-Means | silhouette={clu.silhouette:.3f}")
    print(clu.perfil_clusters)

    _cabecalho("GERAÇÃO DE FIGURAS")
    viz.plot_distribuicao_preco(df, salvar_como="01_distribuicao_preco.png")
    viz.plot_boxplots_outliers(df, salvar_como="02_boxplots_outliers.png")
    viz.plot_matriz_correlacao(matriz_correlacao(df, "spearman"),
                               "Matriz de correlação (Spearman)", salvar_como="03_correlacao.png")
    viz.plot_preco_vs_idade(df, salvar_como="04_preco_vs_idade.png")
    viz.plot_preco_por_marca(resumo_marcas, salvar_como="05_preco_por_marca.png")
    viz.plot_importancias(rf.importancias, salvar_como="06_importancias_rf.png")
    viz.plot_clusters(clu.df_clusterizado, salvar_como="07_clusters_kmeans.png")

    _cabecalho("PIPELINE CONCLUÍDO")
    print(f"Figuras em: {config.FIGURES_DIR}")
    return df


if __name__ == "__main__":
    executar_pipeline()
