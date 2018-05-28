import pickle
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split

data = pd.read_hdf('..//..//data//model//data_model_clus.h5')

X = data[[
    
 'Tenderloin',
 'South of Market',
 'Downtown',
 'Inner Mission',
 'Van Ness/Civic Center',
 'Duboce Triangle',
 'Haight Ashbury',
 'Miraloma Park',
 'Financial District/Barbary Coast',
 'Silver Terrace', 
    
 'winter',
 'spring',
 'summer',
 'fall',
 'hour_0',
 #'hour_6',
 'hour_12',
 #'hour_18',
 'dow_0',
 'dow_1',
 'dow_2',
 'dow_3',
 'dow_4',
 'dow_5',
 'dow_6',
 'UNIVPROX',
 'SIGNALIZED',
 'PKGMETERS',
 'MAXPCTSLPE',
 'MODEL6_VOL',
 'HH_PEDMODE',
 'PCOL_04_09',
 'PCOL_RATE',
 'HH_income',
 'total_pop',
 'unemp_20_24',
 'unemp_25_44',
 'unemp_45_54',
 'unemp_55_64',
 'white',
 'black',
 'asian',
 'other'
]]

y = data['poop']

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)

# impute and scale
X_train_imp = Imputer().fit_transform(X_train)
X_train_scale = StandardScaler().fit_transform(X_train_imp)

X_test_imp = Imputer().fit_transform(X_test)
X_test_scale = StandardScaler().fit_transform(X_test_imp)

scoring = {'AUC': 'roc_auc',
          'F1': make_scorer(f1_score),
          'Accuracy': make_scorer(accuracy_score),
          'Precision': make_scorer(precision_score),
          'Recall': make_scorer(recall_score)
          }

# K nearest neighbors
gs_knn = GridSearchCV(KNeighborsClassifier(),
                     param_grid = {'n_neighbors':range(2,20,2),
                                   'weights':['uniform','distance'],
                                   'metric':['minkowski','manhattan','euclidean']},
                     scoring=scoring,
                     cv=StratifiedKFold(),
                     refit='AUC')
gs_knn.fit(X_train_scale, y_train)
pickle.dump(gs_knn, open('knn.gs', 'wb'))