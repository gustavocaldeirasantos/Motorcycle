"""Etapa 6 — Modelagem estatística e de machine learning.

* Regressão Linear (OLS / statsmodels) sobre ``log(preço)`` — *explicativa*:
  coeficientes interpretáveis, p-valores e R².
* Random Forest — *preditiva*: captura não-linearidades e fornece a importância
  das variáveis; comparamos seu R² ao da regressão linear.
* K-Means — *não supervisionada*: segmenta as motos em perfis de mercado.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src import config

NUM_FEATURES: list[str] = ["age", "km_driven", "owner_num"]


def _montar_matriz(df: pd.DataFrame, top_brands: int = config.TOP_N_BRANDS) -> pd.DataFrame:
    """Monta a matriz de features: numéricas + marca one-hot (top marcas)."""
    base = df.copy()
    principais = base["brand"].value_counts().head(top_brands).index
    base["brand_grp"] = base["brand"].where(base["brand"].isin(principais), "Outras")
    num = base[NUM_FEATURES]
    cat = pd.get_dummies(base["brand_grp"], prefix="brand", drop_first=True, dtype=float)
    return pd.concat([num, cat], axis=1)


# --------------------------------------------------------------------------- #
# 1. Regressão Linear (OLS sobre log do preço)
# --------------------------------------------------------------------------- #
@dataclass
class ResultadoRegressao:
    r2_treino: float
    r2_teste: float
    coeficientes: pd.DataFrame
    sumario: str = field(repr=False)


def treinar_regressao_linear(df: pd.DataFrame) -> ResultadoRegressao:
    """Ajusta uma Regressão Linear (OLS) para explicar log(preço de venda).

    O log lineariza a relação e estabiliza a variância (preço é assimétrico).
    """
    X = _montar_matriz(df)
    y = np.log(df["selling_price"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    X_train_c = sm.add_constant(X_train)
    X_test_c = sm.add_constant(X_test)
    modelo = sm.OLS(y_train, X_train_c).fit()

    coefs = pd.DataFrame(
        {
            "coeficiente": modelo.params,
            "p_valor": modelo.pvalues,
            "ic_inferior": modelo.conf_int()[0],
            "ic_superior": modelo.conf_int()[1],
        }
    )
    return ResultadoRegressao(
        r2_treino=float(modelo.rsquared),
        r2_teste=float(r2_score(y_test, modelo.predict(X_test_c))),
        coeficientes=coefs,
        sumario=modelo.summary().as_text(),
    )


# --------------------------------------------------------------------------- #
# 2. Random Forest (preditivo, não-linear)
# --------------------------------------------------------------------------- #
@dataclass
class ResultadoRandomForest:
    r2_teste: float
    importancias: pd.Series = field(repr=False)


def treinar_random_forest(df: pd.DataFrame) -> ResultadoRandomForest:
    """Random Forest sobre log(preço); reporta R² de teste e importâncias."""
    X = _montar_matriz(df)
    y = np.log(df["selling_price"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    modelo = RandomForestRegressor(n_estimators=300, random_state=config.RANDOM_STATE)
    modelo.fit(X_train, y_train)
    importancias = pd.Series(modelo.feature_importances_, index=X.columns).sort_values(
        ascending=False
    )
    return ResultadoRandomForest(
        r2_teste=float(r2_score(y_test, modelo.predict(X_test))),
        importancias=importancias.round(3),
    )


# --------------------------------------------------------------------------- #
# 3. Clusterização
# --------------------------------------------------------------------------- #
@dataclass
class ResultadoClusterizacao:
    df_clusterizado: pd.DataFrame
    perfil_clusters: pd.DataFrame
    silhouette: float


def segmentar_kmeans(df: pd.DataFrame, n_clusters: int = config.KMEANS_N_CLUSTERS) -> ResultadoClusterizacao:
    """Segmenta as motos por preço, idade e quilometragem."""
    features = ["selling_price", "age", "km_driven", "owner_num"]
    dados = df[features].dropna().copy()
    X_scaled = StandardScaler().fit_transform(dados)
    kmeans = KMeans(n_clusters=n_clusters, random_state=config.RANDOM_STATE, n_init=10)
    dados["cluster"] = kmeans.fit_predict(X_scaled)

    perfil = (
        dados.groupby("cluster")
        .agg(
            n_motos=("selling_price", "size"),
            preco_medio=("selling_price", "mean"),
            idade_media=("age", "mean"),
            km_medio=("km_driven", "mean"),
        )
        .round(1)
    )
    return ResultadoClusterizacao(dados, perfil, float(silhouette_score(X_scaled, dados["cluster"])))


if __name__ == "__main__":
    from src.data_loader import load_raw_data
    from src.preprocessing import clean_data

    df = clean_data(load_raw_data())

    print("\n=== REGRESSÃO LINEAR (log do preço) ===")
    reg = treinar_regressao_linear(df)
    print(f"R² treino={reg.r2_treino:.3f} | teste={reg.r2_teste:.3f}")
    print(reg.coeficientes.round(4))

    print("\n=== RANDOM FOREST ===")
    rf = treinar_random_forest(df)
    print(f"R² teste={rf.r2_teste:.3f}")
    print(rf.importancias)

    print("\n=== K-MEANS ===")
    clu = segmentar_kmeans(df)
    print(f"Silhouette={clu.silhouette:.3f}")
    print(clu.perfil_clusters)
