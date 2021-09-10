import numpy as np
import pandas as pd

def normalize(X):
    """
    Normaliza os atributos em X
    
    Esta função retorna uma versao normalizada de X onde o valor da
    média de cada atributo é igual a 0 e desvio padrao é igual a 1. Trata-se de
    um importante passo de pré-processamento quando trabalha-se com 
    métodos de aprendizado de máquina.
    """
    try:
        X_np = X.to_numpy()
    except AttributeError:
        X_np = X
    
    m, n = X.shape # m = qtde de objetos e n = qtde de atributos por objeto
    
    # Inicializa as variaves de saída
    X_norm = np.zeros( (m,n) ) #inicializa X_norm (base normalizada)
    mu = 0 # inicializa a média
    sigma = 1 # inicializa o desvio padrão

    mu = np.mean(X_np,axis=0)
    sigma = np.std(X_np,axis=0, ddof=1)
       
    for j in range(n):
        for i in range(m):
            X_norm[i][j] = (X_np[i][j] - mu[j])/sigma[j]
    
    if isinstance(X,pd.DataFrame):
        X_norm = pd.DataFrame(X_norm, columns = X.columns)

    return X_norm, mu, sigma

