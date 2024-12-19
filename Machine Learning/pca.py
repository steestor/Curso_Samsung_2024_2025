import pandas as pd
import sklearn
import matplotlib.pyplot as plt 
 
# Importamos los módulos específicos
 
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
 
 # Para clasificar
from sklearn.linear_model import LogisticRegression

# Escalador de 0 a 1 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
 
if __name__ == "__main__":
 
   # Cargamos los datos del dataframe de pandas
   dt_heart = pd.read_csv('./data/heart.csv')
   print(dt_heart.head(5))
 
   # Guardamos nuestro dataset sin la columna de target
   dt_features = dt_heart.drop(['target'], axis=1)
   # Este será nuestro dataset, pero sin la columna
   dt_target = dt_heart['target']
 
   # Normalizamos los datos
   dt_features = StandardScaler().fit_transform(dt_features)
  
   # Partimos el conjunto de entrenamiento. Para añadir replicabilidad usamos el random state
   X_train, X_test, y_train, y_test = train_test_split(dt_features, dt_target, test_size=0.3, random_state=42)