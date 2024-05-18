import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import pearsonr
from scipy.stats import chi2_contingency


def describe_df (data: pd.DataFrame) -> pd.DataFrame: 
    '''
    Describe el dtype de cada columna, los valores nulos en %, quantos valores únicos hay en la columna y el % de cardinalidad.

    Argumentos:
    data(pd.DataFrame): DataFrame de Pandas inicial

    Retorna:
    pd.DataFrame: Data inicial transformado con los valores descritos   
    '''

    dic_describe = {
        'DATA_TYPE' : [data[x].dtype for x in data],
        'MISSINGS (%)' : [round(data[x].isnull().sum()/len(data[x])*100,2) for x in data],
        'UNIQUE_VALUES' : [data[x].nunique() for x in data],
        'CARDIN (%)' : [round(data[x].nunique() / len(data[x]) * 100, 2) for x in data]
    }
    
    return pd.DataFrame(dic_describe, index=[x for x in data]).T

###############################################################################################

# Simple funcion para clasificar y que el código quede mas bonito
def _classify(data: pd.DataFrame, key: str,  umbral_categoria:int, umbral_continua:float) -> str: 
    cardi = data[key].nunique() 
    if cardi == 2: # ¿La cardinalidad es igual que dos?
        return "Binaria"
    elif cardi < umbral_categoria: # ¿La cardinalidad es mas pequeña que el número que escogemos para considerar una variable categórica?
        return "Categórica"
    elif cardi/len(data[key])*100 >= umbral_continua: # ¿El % de la cardinalidad es mayor o igual que el número que escogemos para delimitar cuando és Continua o Discreta?
        return "Numérica Continua"
    else:
        return "Numérica Discreta"
        

def tipifica_variable (data:pd.DataFrame, umbral_categoria:int, umbral_continua:float) -> pd.DataFrame:
    '''
    Tipo de variable de cada columna según su cardinalidad.

    Argumentos:
    data(pd.DataFrame): DataFrame inicial
    umbral_categoria(int): Número que escogemos para delimitar a partir de cuanto consideramos que es una variable categorica
    umbral_continua(float): Número que escogemos para delimitar a partir de cuanto una variable numérica es discreta
    
    Retorna:
    pd.DataFrame: Data inicial transformado   
    '''
    # Diccionario con los resultados de las preguntas sobre la cardinalidad
    dic_tip_var = {
        "tipo_sugerido": [_classify(data, key, umbral_categoria, umbral_continua) for key in data]
    }
    # Añadimos un extra, simple print para tener en cuenta si hay valores nulos no tratados en el dataframe
    for x in data:
        hay_nulos = data[x].isnull().sum()
        if hay_nulos != 0:
            print(f'OJO! En la columna "{x}" hay valores nulos.')

    return pd.DataFrame(dic_tip_var, index=[x for x in data])

###############################################################################################

def get_features_num_regression (df:pd.DataFrame, target_col:str, umbral_corr:float, pvalue = None):

    """
    Está función debe devolver una lista con las columnas numéricas del dataframe cuya correlación con la columna designada por "target_col" 
    sea superior en valor absoluto al valor dado por "umbral_corr". Además si la variable "pvalue" es distinta de None, sólo devolverá las 
    columnas numéricas cuya correlación supere el valor indicado y además supere el test de hipótesis con significación mayor o igual a 1-pvalue.

    Argumentos:
    - df (DataFrame): un dataframe de Pandas.
    - target_col (string): el nombre de la columna del Dataframe objetivo.
    - umbral_corr (float): un valor de correlación arbitrario sobre el que se elegirá como de correlacionadas queremos que estén las columnas elegidas (por defecto 0).
    - pvalue (float): con valor "None" por defecto.

    Retorna:
    - Lista de las columnas correlacionadas que cumplen el test en caso de que se haya pasado p-value.
    """

    if target_col not in df.columns:
        print(f"Error: La columna {target_col} no está en el DataFrame.")
        return None
    
    if not np.issubdtype(df[target_col].dtype, np.number):
        print(f"Error: La columna {target_col} no es numérica.")
        return None
    
    if umbral_corr > 1 or umbral_corr < 0:
        print("Error: 'umbral_corr' debe ser un valor entre 0 y 1.")
        return None
    
    if pvalue is not None and (pvalue <= 0 or pvalue >= 1):
        print("Error: 'pvalue' debe ser un valor entre 0 y 1.")
        return None
        
    features = []
    for col in df.select_dtypes(include=[np.number]).columns:
        if col != target_col:
            corr, pval = stats.pearsonr(df[col].dropna(), df[target_col].dropna())
            if abs(corr) >= umbral_corr and (pvalue is None or pval <= pvalue):
                features.append(col)
    return features

###############################################################################################

def plot_features_num_regression(df, target_col:str, umbral_corr=0, pvalue=None):
    """
    Visualiza gráficos de dispersión para características numéricas en relación con una columna objetivo de un modelo de regresión.

    Argumentos:
    df (pandas.DataFrame): El DataFrame del que se obtienen las características.
    target_col (str, opcional): El nombre de la columna que debe ser el objetivo del modelo de regresión. Por defecto es "".
    umbral_corr (float, opcional): Umbral de correlación para considerar una característica como significativa. Por defecto es 0.
    pvalue (float, opcional): Valor de p para realizar el test de hipótesis. Por defecto es None.

    Retorna:
    list or None: Una lista con las características que cumplen con los criterios especificados, o None si hay errores.
    """
    selected_features = get_features_num_regression(df, target_col, umbral_corr, pvalue)

    num_plots = (len(selected_features) // 4) + 1
    for i in range(num_plots):
        cols_to_plot = [target_col] + selected_features[i*4:(i+1)*4]
        fig = sns.pairplot(df[cols_to_plot], hue = target_col)
        plt.show()
    return fig

###############################################################################################

def get_features_cat_regression(df, target_col, p_value=0.05):
    """
    Identifica características categóricas relevantes para un modelo de regresión.

    Argumentos:
    - df (DataFrame de pandas): El DataFrame sobre el que se realizará el análisis.
    - target_col (str): La columna objetivo que se utilizará en el análisis.
    - p_value (float): El valor de p máximo aceptable para considerar una característica como relevante.
      Por defecto, es 0.05.

    Retorna:
    - Lista con las columnas categóricas consideradas relevantes para el modelo de regresión.
      Tipo lista compuesto por cadenas de texto.
    """

    if df.empty:
        print("El dataframe esta vacío")
        return None
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print("La columna que has puesto no es una columna numerica")
        return None
    if not isinstance(p_value, float) or 0 > p_value or 1 < p_value:
        print("El p_value no tiene un valor valido, recuerda que tiene que estar entre 0 y 1")
        return None
    if target_col not in df:
        print("La columna no esta en el Dataframe, cambiala por una valida")
        return None
    
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    relevant_columns = []
    
    for col in categorical_columns:
        grouped = df.groupby(col)[target_col].apply(list).to_dict()
        f_vals = []
        for key, value in grouped.items():
            f_vals.append(value)
        f_val, p_val = stats.f_oneway(*f_vals)
        if p_val <= p_value:
            relevant_columns.append(col)

    return relevant_columns

###############################################################################################

def plot_features_cat_regression(df:pd.DataFrame, target_col:str, columns=[], pvalue=0.05):
    """
    Realiza un análisis de las características categóricas en relación con una columna objetivo para un modelo de regresión.

    Argumentos:
    - df (DataFrame de pandas): El DataFrame que contiene los datos.
    - target_col (str): La columna objetivo para el análisis.
    - columns (list): Lista de columnas categóricas a considerar. Si está vacía, se considerarán todas las columnas categóricas del DataFrame.
    - pvalue (float): El nivel de significancia para determinar la relevancia estadística de las variables categóricas. Por defecto, es 0.05.
    - with_individual_plot (bool): Indica si se debe mostrar un histograma agrupado para cada variable categórica significativa. Por defecto, es False.

    Retorna:
    - Lista de las columnas categóricas que muestran significancia estadística con respecto a la columna objetivo.
    """

    # Verificar que target_col esté en el dataframe
    if target_col != "" and target_col not in df.columns:
        raise ValueError("La columna 'target_col' no existe en el DataFrame")

    # Verificar que las columnas en columns estén en el dataframe
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"La columna '{col}' no existe en el DataFrame")

    # Si columns está vacío, seleccionar todas las variables categóricas
    if not columns:
        columns = df.select_dtypes(include=['object']).columns.tolist()

    # Lista para almacenar las variables categóricas que cumplen con las condiciones
    significant_categorical_variables = []

    # Iterar sobre las columnas seleccionadas
    for col in columns:
        # Verificar si la columna es categórica
        if df[col].dtype == 'object':
            # Calcular el test de chi-cuadrado entre la columna categórica y target_col
            contingency_table = pd.crosstab(df[col], df[target_col])
            chi2, p_val, _, _ = chi2_contingency(contingency_table)
            
            # Verificar si el p-value es menor que el umbral de significancia
            if p_val < pvalue:
                # Agregar la columna a la lista de variables categóricas significativas
                significant_categorical_variables.append(col)

                sns.histplot(data=df, x=col, hue=target_col, multiple="stack")
                plt.title(f"Histograma agrupado de {col} según {target_col}")
                plt.show()
            else:
                print(f"No se encontró significancia estadística para la variable categórica '{col}' con '{target_col}'")

    # Si no se encontró significancia estadística para ninguna variable categórica
    if not significant_categorical_variables:
        print("No se encontró significancia estadística para ninguna variable categórica")

    # Devolver las variables categóricas que cumplen con las condiciones
    return significant_categorical_variables