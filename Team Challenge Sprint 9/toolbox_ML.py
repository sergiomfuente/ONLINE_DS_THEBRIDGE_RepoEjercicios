import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


def describe_df(df):
    """Describe el DataFrame con métricas de tipo, nulos, cardinalidad y uniques."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("describe_df solo acepta DataFrame")

    total = len(df)
    cols = []

    for col in df.columns:
        n_null = df[col].isna().sum()
        unique_values = df[col].nunique(dropna=False)
        cardinality_pct = unique_values / total * 100 if total else 0

        cols.append({
            "variable": col,
            "dtype": str(df[col].dtype),
            "n_null": n_null,
            "pct_null": n_null / total * 100 if total else 0,
            "n_unique": unique_values,
            "pct_cardinalidad": cardinality_pct
        })

    resultado = pd.DataFrame(cols)
    return resultado.set_index("variable")


def tipifica_variables(df, umbral_categoria=10, umbral_continua=0.5):
    """Clasifica columnas: Binaria, Categórica, Numerica Continua, Numerica Discreta."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("tipifica_variables solo acepta DataFrame")
    if not isinstance(umbral_categoria, int) or umbral_categoria < 2:
        print("umbral_categoria debe ser un entero >=2")
        return None
    if not isinstance(umbral_continua, (int, float)) or not (0 <= umbral_continua <= 1):
        print("umbral_continua debe ser un float entre 0 y 1")
        return None

    rows = []
    nrows = len(df)

    for col in df.columns:
        card = df[col].nunique(dropna=False)
        pct_card = card / nrows if nrows else 0

        if card == 2:
            tipo = "Binaria"
        elif card < umbral_categoria:
            tipo = "Categórica"
        else:
            if pct_card >= umbral_continua:
                tipo = "Numerica Continua"
            else:
                tipo = "Numerica Discreta"

        rows.append({"nombre_variable": col, "tipo_sugerido": tipo})

    return pd.DataFrame(rows)


def get_features_num_regression(df, target_col, umbral_corr=0.3, pvalue=None):
    """Devuelve features numéricas correlacionadas con target y, opcionalmente, pvalue estadístico."""
    if not isinstance(df, pd.DataFrame):
        print("df debe ser un DataFrame")
        return None
    if target_col not in df.columns:
        print("target_col no existe en el DataFrame")
        return None
    if not np.issubdtype(df[target_col].dtype, np.number):
        print("target_col debe ser numérica")
        return None
    if not (0 <= umbral_corr <= 1):
        print("umbral_corr debe estar entre 0 y 1")
        return None
    if pvalue is not None and (pvalue <= 0 or pvalue >= 1):
        print("pvalue debe estar entre 0 y 1")
        return None

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != target_col]

    if len(numeric_cols) == 0:
        return []

    ret = []
    for col in numeric_cols:
        corr = df[target_col].corr(df[col])
        if pd.isna(corr):
            continue
        if abs(corr) >= umbral_corr:
            if pvalue is not None:
                try:
                    _, pv = stats.pearsonr(df[target_col].dropna(), df[col].dropna())
                except Exception:
                    continue
                if pv <= pvalue:
                    ret.append(col)
            else:
                ret.append(col)

    return ret


def plot_features_num_regression(df, target_col="", columns=None, umbral_corr=0, pvalue=None):
    """Pinta pairplots con las variables numéricas correlacionadas con target."""
    if not isinstance(df, pd.DataFrame):
        print("df debe ser un DataFrame")
        return None
    if target_col == "" or target_col not in df.columns:
        print("target_col debe ser válido")
        return None
    if not np.issubdtype(df[target_col].dtype, np.number):
        print("target_col debe ser numérico")
        return None

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if columns is None or len(columns) == 0:
        columns = [c for c in numeric_cols if c != target_col]

    selected = []
    for col in columns:
        if col not in df.columns:
            continue
        corr = df[target_col].corr(df[col])
        if pd.isna(corr):
            continue
        if abs(corr) >= umbral_corr:
            if pvalue is not None:
                try:
                    _, pv = stats.pearsonr(df[target_col].dropna(), df[col].dropna())
                except Exception:
                    continue
                if pv <= pvalue:
                    selected.append(col)
            else:
                selected.append(col)

    if len(selected) == 0:
        print("No hay columnas que cumplan el criterio")
        return []

    max_cols = 5
    for i in range(0, len(selected), max_cols - 1):
        chunk = selected[i:i + (max_cols - 1)]
        cols_to_plot = [target_col] + chunk
        sns.pairplot(df[cols_to_plot].dropna())
        plt.show()

    return selected


def get_features_cat_regression(df, target_col, pvalue=0.05):
    """Devuelve categóricas con relación significativa a target numérico."""
    if not isinstance(df, pd.DataFrame):
        print("df debe ser DataFrame")
        return None
    if target_col not in df.columns:
        print("target_col no existe")
        return None
    if not np.issubdtype(df[target_col].dtype, np.number):
        print("target_col debe ser numérico")
        return None
    if pvalue <= 0 or pvalue >= 1:
        print("pvalue debe estar entre 0 y 1")
        return None

    cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    resultado = []

    for col in cat_cols:
        if df[col].nunique(dropna=True) < 2:
            continue

        groups = [df[df[col] == cat][target_col].dropna() for cat in df[col].dropna().unique()]
        if len(groups) < 2:
            continue

        try:
            if len(groups) == 2:
                stat, pv = stats.ttest_ind(groups[0], groups[1], equal_var=False)
            else:
                stat, pv = stats.f_oneway(*groups)
        except Exception:
            continue

        if pv <= pvalue:
            resultado.append(col)

    return resultado


def plot_features_cat_regression(df, target_col="", columns=None, pvalue=0.05, with_individual_plot=False):
    """Pinta histogramas/boxplots por categorías con test de significación estadística."""
    if not isinstance(df, pd.DataFrame):
        print("df debe ser DataFrame")
        return None
    if target_col == "" or target_col not in df.columns:
        print("target_col debe ser válido")
        return None
    if not np.issubdtype(df[target_col].dtype, np.number):
        print("target_col debe ser numérico")
        return None

    candidate_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    if columns is None or len(columns) == 0:
        columns = candidate_cols

    selected = []
    for col in columns:
        if col not in df.columns or col not in candidate_cols:
            continue
        groups = [df[df[col] == cat][target_col].dropna() for cat in df[col].dropna().unique()]
        if len(groups) < 2:
            continue
        try:
            if len(groups) == 2:
                _, pv = stats.ttest_ind(groups[0], groups[1], equal_var=False)
            else:
                _, pv = stats.f_oneway(*groups)
        except Exception:
            continue

        if pv <= pvalue:
            selected.append(col)

    for col in selected:
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=col, y=target_col, data=df)
        plt.title(f"{target_col} vs {col}")
        plt.show()

        if with_individual_plot:
            for valor in df[col].dropna().unique():
                plt.figure(figsize=(6, 3))
                subset = df[df[col] == valor]
                sns.histplot(subset[target_col], kde=True)
                plt.title(f"{target_col} para {col} = {valor}")
                plt.show()

    return selected
