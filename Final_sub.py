{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 101,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import sklearn\n",
    "import seaborn as sns\n",
    "from sklearn.experimental import enable_iterative_imputer\n",
    "from sklearn.impute import IterativeImputer\n",
    "from sklearn.model_selection import RandomizedSearchCV\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "import matplotlib.pyplot as plt\n",
    "import warnings;\n",
    "warnings.filterwarnings('ignore');\n",
    "from imblearn.over_sampling import RandomOverSampler, BorderlineSMOTE "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 102,
   "metadata": {},
   "outputs": [],
   "source": [
    "df=pd.read_csv('train_input.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 103,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_copy = df.copy()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 104,
   "metadata": {},
   "outputs": [],
   "source": [
    "lastcolumn_np = df.iloc[:,-1:].to_numpy()\n",
    "lastcolumn_op=np.ravel(lastcolumn_np)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 105,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_copy.drop(['Target Variable (Discrete)','Feature 16','Feature 17'], axis=1, inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 106,
   "metadata": {},
   "outputs": [],
   "source": [
    "imp_mean = IterativeImputer(random_state=0)\n",
    "imp_mean.fit(df_copy)\n",
    "IterativeImputer(random_state=0)\n",
    "data = imp_mean.transform(df_copy)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 108,
   "metadata": {},
   "outputs": [],
   "source": [
    "corr_features = set()\n",
    "correlation_matrix = pd.DataFrame(data).corr()\n",
    "for i in range(len(correlation_matrix.columns)):\n",
    "    for j in range(i):\n",
    "        if abs(correlation_matrix.iloc[i, j]) > 0.8:\n",
    "            colname = correlation_matrix.columns[i]\n",
    "            corr_features.add(colname)\n",
    "to_drop=np.array(list(corr_features))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 109,
   "metadata": {},
   "outputs": [],
   "source": [
    "np_drop = np.delete(data,to_drop,1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 110,
   "metadata": {},
   "outputs": [],
   "source": [
    "Dict = {17:10, 12:10, 11:10, 16:10, 10:10, 9:10}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 111,
   "metadata": {},
   "outputs": [],
   "source": [
    "over = RandomOverSampler(random_state=0, sampling_strategy=Dict)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 112,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_resampled, y_resampled = over.fit_resample(np_drop, lastcolumn_op)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 113,
   "metadata": {},
   "outputs": [],
   "source": [
    "sm = BorderlineSMOTE(random_state=0, k_neighbors=2, kind='borderline-2')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_resampled, y_resampled = sm.fit_resample(X_resampled, y_resampled)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'n_estimators': 1200,\n",
       " 'min_samples_split': 2,\n",
       " 'min_samples_leaf': 2,\n",
       " 'max_features': 'auto',\n",
       " 'max_depth': 50,\n",
       " 'bootstrap': False}"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]\n",
    "max_features = ['auto', 'sqrt']\n",
    "max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]\n",
    "max_depth.append(None)\n",
    "min_samples_split = [2, 5, 10]\n",
    "min_samples_leaf = [1, 2, 4]\n",
    "bootstrap = [True, False]\n",
    "params = {'n_estimators': n_estimators,\n",
    "               'max_features': max_features,\n",
    "               'max_depth': max_depth,\n",
    "               'min_samples_split': min_samples_split,\n",
    "               'min_samples_leaf': min_samples_leaf,\n",
    "               'bootstrap': bootstrap}\n",
    "cv_result = RandomizedSearchCV(RandomForestClassifier(),params,cv=3,scoring='accuracy',random_state = 5)\n",
    "cv_result.fit(X_resampled, y_resampled)\n",
    "cv_result.best_params_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 114,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "RandomForestClassifier(bootstrap=False, max_depth=70, n_estimators=2000,\n",
       "                       random_state=42)"
      ]
     },
     "execution_count": 114,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "classifier_op = RandomForestClassifier(n_estimators = 2000, min_samples_split=2,min_samples_leaf=1,max_features='auto',\n",
    "                               max_depth=70, random_state = 42,bootstrap=False)\n",
    "classifier_op.fit(X_resampled, y_resampled)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 115,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_2=pd.read_csv('iith_foml_2020_test.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 122,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_2_copy = df_2.copy()\n",
    "df_2_copy.drop(['Feature 16','Feature 17'], axis=1, inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 123,
   "metadata": {},
   "outputs": [],
   "source": [
    "data_2 = imp_mean.transform(df_2_copy)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 124,
   "metadata": {},
   "outputs": [],
   "source": [
    "np_drop_2 = np.delete(data_2, to_drop,1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 125,
   "metadata": {},
   "outputs": [],
   "source": [
    "data_pred = classifier_op.predict(np_drop_2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 126,
   "metadata": {},
   "outputs": [],
   "source": [
    "col_id = np.linspace(1,426,426,dtype=int)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 127,
   "metadata": {},
   "outputs": [],
   "source": [
    "Final_df = pd.DataFrame(columns=['Id','Category'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 128,
   "metadata": {},
   "outputs": [],
   "source": [
    "Final_df['Id'] = col_id"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 129,
   "metadata": {},
   "outputs": [],
   "source": [
    "Final_df['Category'] = data_pred"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 130,
   "metadata": {},
   "outputs": [],
   "source": [
    "Final_df.to_csv('test_ouput.csv', index=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
