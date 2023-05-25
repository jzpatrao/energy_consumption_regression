#!/usr/bin/env python
# coding: utf-8

# C'est le final de trois notebooks.<br> 
# Ce notebook se concentre sur la construction et l'interprétation de modèles de régression pour la ciblé TotalGHGEmissions.<br> 
# Il examine également l'efficacité de la variable ENERGYStarScore comme prédicteur des émissions totales.

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Preprocessing" data-toc-modified-id="Preprocessing-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Preprocessing</a></span></li><li><span><a href="#Models" data-toc-modified-id="Models-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Models</a></span></li><li><span><a href="#Model-interpretation" data-toc-modified-id="Model-interpretation-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Model interpretation</a></span></li><li><span><a href="#ENERGYStarScore" data-toc-modified-id="ENERGYStarScore-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>ENERGYStarScore</a></span></li></ul></div>

# In[1]:


# Importer les packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Charger le jeu des données
df = pd.read_csv("C:\\Users\\zheng\\Documents\\Openclassrooms\\Data Scientist\\Project_04\\df_final.csv")


# In[3]:


# Changer les variables catégorielles de interger en string.
df['CouncilDistrictCode'] = df['CouncilDistrictCode'].astype(str)
df['ZipCode'] = df['ZipCode'].astype(str)


# In[4]:


# Définir le X et le y
X = df[[
    'PrimaryPropertyType', 'ZipCode', 'Neighborhood', 'CouncilDistrictCode',
       'Latitude', 'Longitude', 'NumberofBuildings', 'NumberofFloors',
       'PropertyGFAParking', 'LargestPropertyUseType',  'ENERGYSTARScore', 'BuildingAge',
       'parking_area_prcnt', 'PropertyGFATotal_log', 'PropertyGFABuilding(s)_log', 'LargestPropertyUseTypeGFA_log'
       ]].copy()

y = df['TotalGHGEmissions_log']


# ## Preprocessing

# In[5]:


# Initier le columns transformer (avec one hot encoder).
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
col_transform = make_column_transformer(
    (OneHotEncoder(sparse=False, handle_unknown="ignore"),
     ["PrimaryPropertyType", "ZipCode", 'Neighborhood', 'CouncilDistrictCode', 'LargestPropertyUseType'],),
    remainder="passthrough")


# Initier StandardScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# ## Models

# In[6]:


# Partitionnement le dataset (taille du test 30 %)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[7]:


# Importer les packages 
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor


# In[8]:


# Définir les paramètres candidats pour les modèles choisis
params_ridge = {}
params_ridge["ridge__alpha"] = np.logspace(-5, 5, 5)

params_net = {}
params_net["elasticnet__alpha"] = np.logspace(-5, 5, 5)

params_knn = {}
params_knn["kneighborsregressor__n_neighbors"] = [3, 10, 30]

params_tree = {}
params_tree["decisiontreeregressor__min_samples_leaf"] = [10, 30, 50]
params_tree["decisiontreeregressor__max_depth"] = [10, 30, 50]

params = [params_ridge, params_net, params_knn, params_tree]


# In[9]:


# Trouver les meilleurs paramètres avec GridSearchCV et pipeline
models = [
    Ridge(random_state=42, tol=0.1),
    ElasticNet(random_state=42, tol=1),
    KNeighborsRegressor(),
    DecisionTreeRegressor(random_state=42)
]

best_params = []
for model, param in zip(models, params):
    pipe = make_pipeline(col_transform, scaler, model)
    grid = GridSearchCV(pipe, param, cv=5, scoring="neg_mean_absolute_error")
    grid.fit(X_train, y_train)
    best_params.append(grid.best_params_)


# In[10]:


# Les meilleurs paramètres...
best_params


# In[11]:


# Branchez les meilleurs paramètres.
names = ['ridge_regression', 'elastic_net_regression',
         'knn_regression', 'decision_tree_regression']

models_best_params = [
    Ridge(random_state=42, tol=0.1,
          alpha=best_params[0]['ridge__alpha']),

    ElasticNet(random_state=42, tol=0.1,
               alpha=best_params[1]['elasticnet__alpha']),

    KNeighborsRegressor(
        n_neighbors=best_params[2]['kneighborsregressor__n_neighbors']),

    DecisionTreeRegressor(random_state=42,
                          max_depth=best_params[3]['decisiontreeregressor__max_depth'],
                          min_samples_leaf=best_params[3]['decisiontreeregressor__min_samples_leaf'])
]


# In[12]:


# Créer un dataset pour plot les prédictions
result_viz = pd.DataFrame(y_test)
result_viz.columns=['target']


# In[13]:


# Obtenir les scores train et test, ansi que les MAE (mean absolute error) pour chaque model
train_score_r2 = []
test_score_r2 = []
train_mae = []
test_mae = []

for model, name in zip(models_best_params, names):
    pipe = make_pipeline(col_transform, scaler, model)
    score_train = cross_val_score(
        pipe, X_train, y_train, cv=5, scoring="r2").mean()
    score_test = cross_val_score(
        pipe, X_test, y_test, cv=5, scoring="r2").mean()
    mae_train = np.abs(cross_val_score(pipe, X_train, y_train,
                       cv=5, scoring="neg_mean_absolute_error").mean())
    mae_test = np.abs(cross_val_score(pipe, X_test, y_test,
                      cv=5, scoring="neg_mean_absolute_error").mean())

    test_score_r2.append(score_test)
    train_score_r2.append(score_train)
    train_mae.append(mae_train)
    test_mae.append(mae_test)

    pipe.fit(X_train, y_train)
    predict = pipe.predict(X_test)
    result_viz[name] = predict


# In[14]:


results = {'model':names}
results['train_score_r2']=train_score_r2
results['test_score_r2']=test_score_r2
results['train_mean_absolute_error']=train_mae
results['test_mean_absolute_error']=test_mae


# In[15]:


# métriques de résultats dans un tableau
pd.set_option('display.float_format', '{:.2f}'.format)
model_scores = pd.DataFrame(results)
model_scores = round(model_scores,2)
model_scores.sort_values(by='test_score_r2',ascending=False)


# Sur la base des mesures ci-dessus, le regression elastic net donne le meilleur score de détermination ainsi que la plus petite mean absolute error. C'est le meilleur modèle des quatre.

# In[16]:


# Un tableau avec les valeurs prédites sur le test set
result_viz = result_viz.sort_values(by="target")
result_viz.reset_index(drop=True, inplace=True)
result_viz.head()


# In[17]:


colors = ["sandybrown", "royalblue", "salmon", "darkseagreen"]
cols = ['ridge_regression', 'elastic_net_regression',
        'knn_regression', 'decision_tree_regression']

plt.style.use('ggplot')

fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True,
                         sharey=True, figsize=(7, 7))
for i, ax in enumerate(axes.flatten()):
    sns.scatterplot(data=result_viz, x=result_viz.index,
                    y=result_viz[cols[i]], color=colors[i], label=cols[i], ax=ax)
    sns.scatterplot(data=result_viz, x=result_viz.index,
                    y=result_viz["target"], color="black", label="Target", marker="+", ax=ax)

    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('Index')
    ax.set_yscale("log")

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.suptitle("Comparison between target and model predictions (emission)")
plt.savefig('scatter_plot_results_compare_emissionn',bbox_inches='tight',dpi=300)
plt.show()


# Les prédictions faites par la régression ridgesont les plus cohérentes avec les valeurs cibles

# In[18]:


# Distribution des prévisions et des valeurs cibles
fig, axes = plt.subplots(nrows=2, ncols=2, sharey=True, figsize=(7, 7))
for i, ax in enumerate(axes.flatten()):
    sns.kdeplot(result_viz[cols[i]], shade=True, log_scale=True,
                    bw_adjust=0.6, color=colors[i], ax=ax, label=cols[i], linewidth=0.5, alpha=0.7)
    sns.kdeplot(result_viz['target'], shade=True, log_scale=True,
                    bw_adjust=0.6, color='black', ax=ax, label='target', linewidth=0.3)
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel(None)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.suptitle(
    "Distribution of predicted and real values (emission) on test set")
plt.savefig('kde_plot_results_compare_emission',bbox_inches='tight',dpi=300)
plt.show()


# Encore une fois, les prédictions faites par la régression ridge sont les plus cohérentes avec les valeurs cibles.<br>
# Le modèle KNN a sous-prédit beaucoup de valeurs (le pic évident autour des petites valeurs).

# ## Model interpretation

# In[19]:


import shap


# In[33]:


X_one_hot_encoded = pd.DataFrame(col_transform.fit_transform(
    X), columns=col_transform.get_feature_names_out())


# In[34]:


X_ohe_scaled = pd.DataFrame(scaler.fit_transform(
    X_one_hot_encoded), columns=X_one_hot_encoded.columns)


# In[35]:


alpha_net = best_params[1]['elasticnet__alpha']

elastic_net = ElasticNet(alpha=alpha_net, random_state=42,tol=0.1).fit(X_ohe_scaled, y)


# In[36]:


explainer_net = shap.Explainer(elastic_net.predict, X_ohe_scaled)


# In[37]:


shap_values_net = explainer_net(X_ohe_scaled)


# In[46]:


fig=shap.plots.beeswarm(shap_values_net,max_display=10,show=False)
plt.title('Model:ElasticNet regression, target:TotalGHGEmissions_log')
plt.savefig('shap_plot_net_beeswarm_emission',dpi=300,bbox_inches='tight')
plt.show()


# In[47]:


fig = shap.plots.bar(shap_values_net, max_display=10, show=False)
plt.title('Model:ridge regression, target:TotalGHGEmissions')
plt.savefig('shap_plot_net_bar_emission',dpi=300,bbox_inches='tight')
plt.show()


# Résultat au niveau de l'échantillon

# In[40]:


df.iloc[1,:]


# In[48]:


shap.plots.waterfall(shap_values_net[2], show=False)
plt.title(f"Emission prediction of {df.loc[2, 'PropertyName']}")
plt.savefig('shap_plot_bar_indiv_emission', bbox_inches='tight',dpi=300)
plt.show()


# In[42]:


shap.initjs()
fig = shap.plots.force(shap_values_net[1], matplotlib=True)


# ## ENERGYStarScore

# In[43]:


X_with_var = X.copy()
X_without_var = X.drop(columns='ENERGYSTARScore')
y = df['TotalGHGEmissions_log']


# In[44]:


pipe = make_pipeline(col_transform, scaler,ElasticNet(alpha=alpha_net, random_state=42,tol=0.1))
score_r2_with_var = cross_val_score(pipe, X_with_var, y, cv=5, scoring="r2").mean()
score_r2_without_var = cross_val_score(pipe, X_without_var, y, cv=5, scoring="r2").mean()   


# In[45]:


print('Score R² ridge regression avec ENERGYSTARScore:',
      np.around(score_r2_with_var, 4))
print('Score R² ridge regression sans ENERGYSTARScore:',
      np.around(score_r2_without_var, 4))
print('Difference between the results:', np.around(
    np.diff([score_r2_with_var, score_r2_without_var]), 4))


# Le résultat ci-dessus montre qu'ENERGY Star Score améliore les performances du modèle de 4,42 %.

# Fin de projet. Merci.
