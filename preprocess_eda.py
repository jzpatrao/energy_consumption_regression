#!/usr/bin/env python
# coding: utf-8

# # Context
# 
# Pour atteindre l'objectif de la ville de Seattle, neutre en émissions de carbone en 2050, s’intéresse de près à la consommation et aux émissions des bâtiments non destinés à l’habitation. Veuillez tenter de prédire les émissions de CO2 et la consommation totale d’énergie de bâtiments non destinés à l’habitation pour lesquels elles n’ont pas encore été mesurées.<br>
# 
# Vous cherchez également à évaluer l’intérêt de l’"ENERGY STAR Score" pour la prédiction d’émissions, qui est fastidieux à calculer avec l’approche utilisée actuellement par votre équipe. Vous l'intégrerez dans la modélisation et jugerez de son intérêt.

# C'est le premier des trois notebooks.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv("C:\\Users\\zheng\\Documents\\Openclassrooms\\Data Scientist\\Project_04\\2016_Building_Energy_Benchmarking.csv")


# In[3]:


pd.set_option('display.max_columns', None)


# In[4]:


df.head()


# # Nettoyage des données

# In[5]:


df.shape


# In[6]:


df.columns


# In[7]:


df.info()


# In[8]:


print("Nombres des variables categorielles:", len(df.select_dtypes(include='object').dtypes))
print("Nombres des variables numeriques:", len(df.select_dtypes(exclude='object').dtypes))


# In[9]:


# Variables qui contient un seul valeur porte peu de information pour l'analyse. Ils sont suprimés.
single_value_columns = df.columns[df.nunique() == 1].to_list()
df.drop(columns=single_value_columns, inplace=True)


# Plusieurs variables décrivent le type de propriété. En sélectionnera un et supprimera le reste.

# In[10]:


# Vérification du nombre de valeurs pour chaque variable de type de bâtiment
df[["BuildingType", "PrimaryPropertyType", "ListOfAllPropertyUseTypes"]].nunique()


# La variable *BuildingType* contient peu de valeurs différentes et peut donc fournir moin d'informations pour l'analyse et la modelisation. La variable *ListOfAllPropertyUseTypes* contient par contre trop de valeurs.

# In[11]:


# Suprimer variable BuildingType et ListOfAllPropertyUseTypes
df.drop(columns=["BuildingType", "ListOfAllPropertyUseTypes"], inplace=True)


# ## Batiments residentiels
# Le projet ne s'intéresse qu'aux bâtiments non résidentiels. Les immeubles résidentiels doivent étre enlever.

# In[12]:


# Recherche de types de bâtiments résidentiels selon le variable PrimaryPropertyType
df.PrimaryPropertyType.value_counts()


# In[13]:


# Créer une liste avec des types de bâtiments résidentiels
residential = [
    "Low-Rise Multifamily",
    "Mid-Rise Multifamily",
    "High-Rise Multifamily",
    "Hotel",
    "Senior Care Community",
    "Residence Hall",
]

# Supprimer les bâtiments résidentiels selon PrimaryPropertyType.
df_non_residential = df[~df["PrimaryPropertyType"].isin(residential)].copy()


# In[14]:


# Recherche de types de bâtiments résidentiels selon le variable LargestPropertyUseType
df_non_residential.LargestPropertyUseType.value_counts()


# In[15]:


# Supprimer les bâtiments résidentiels selon LargestPropertyUseType.
residential2 = [
    "Multifamily Housing",
    "Hotel",
    "Senior Care Community",
    "Residence Hall/Dormitory",
]
df_non_residential = df_non_residential[
    ~df_non_residential["LargestPropertyUseType"].isin(residential2)
]


# ## Group property type values
# Il y a trop d'entrées dans les variables *PrimaryPropertyType* et *LargestPropertyUseType*. Regroupant certains d'entre eux pour faciliter une analyse plus facile par la suite.

# In[16]:


# Verifier les nombres de chaque valeur dans PrimaryPropertyType
plt.style.use('ggplot')
plt.figure(figsize=(6, 4))
plt.barh(y=(
    df_non_residential.PrimaryPropertyType.value_counts()).index,
    width=(df_non_residential.PrimaryPropertyType.value_counts()).values,
)
plt.xlabel('Value counts')
plt.title('Value counts for variable PrimaryPropertyType', fontsize=10)
plt.savefig("barh_plot_primarypropertytype_value_counts_before",dpi=300,bbox_inches = 'tight')
plt.show()


# In[17]:


# Remplacer les valeurs manuellement
df_non_residential.PrimaryPropertyType.replace(
    {
        "Distribution Center": "Warehouse",
        "Self-Storage Facility": "Warehouse",
        "University": "K-12 School",
        "Refrigerated Warehouse": "Warehouse",
        "Restaurant": "Other",
        "Laboratory": "Medical Office",
        "Hospital": "Medical Office",
        "Office": "Small- and Mid-Sized Office",
    },
    inplace=True,
)


# In[18]:


# Les nombres de chaque valeur dans PrimaryPropertyType
plt.style.use('ggplot')
plt.figure(figsize=(6, 4))
plt.barh(y=(
    df_non_residential.PrimaryPropertyType.value_counts()).index,
    width=(df_non_residential.PrimaryPropertyType.value_counts()).values,
)
plt.xlabel('Value counts')
plt.title('Value counts for variable PrimaryPropertyType after grouping', fontsize=10)
plt.savefig("barh_plot_primarypropertytype_value_counts_after",dpi=300,bbox_inches = 'tight')
plt.show()


# In[19]:


# Verifier les nombres de chaque valeur dans LargestPropertyUseType
plt.figure(figsize=(6, 4))
plt.barh(y=(
    df_non_residential.LargestPropertyUseType.value_counts()).index,
    width=(df_non_residential.LargestPropertyUseType.value_counts()).values,
)
plt.xlabel('Value counts')
plt.title('Value counts for variable LargestPropertyUseType', fontsize=10)
plt.savefig("barh_plot_LargestPropertyUseType_value_counts_before",dpi=300,bbox_inches = 'tight')
plt.show()


# In[20]:


# Trouver des valeurs dans LargestPropertyUseType qui apparaît moins de 30 fois
count_over_30 = df_non_residential.LargestPropertyUseType.value_counts() < 30
count_over_30 = count_over_30.loc[count_over_30.values == True].index.to_list()

# Remplacez les valeurs ci-dessus par "Other"
df_non_residential.LargestPropertyUseType.replace(count_over_30, 'Other', inplace=True)


# In[21]:


# Les nombres de chaque valeur dans LargestPropertyUseType
plt.figure(figsize=(6, 4))
plt.barh(y=(
    df_non_residential.LargestPropertyUseType.value_counts()).index,
    width=(df_non_residential.LargestPropertyUseType.value_counts()).values,
)
plt.xlabel('Value counts')
plt.title('Value counts for variable LargestPropertyUseType after grouping', fontsize=10)
plt.savefig("barh_plot_LargestPropertyUseType_value_counts_after",dpi=300,bbox_inches = 'tight')
plt.show()


# ## Parking GFA
# Comme la consommation d'énergie et les émissions de CO2 sont faibles dans les parkings. Les bâtiments qui ont de grandes surfaces de parking fausseront l'analyse. Ils seront supprimés.

# In[22]:


# Calculer la proportion de surface de parking
parking_proportion = df_non_residential.PropertyGFAParking / df_non_residential.PropertyGFATotal

# Supprimer les bâtiments dont plus de 60 % de la surface  de parking.
df_non_residential.drop(index=parking_proportion[parking_proportion > 0.6].index, inplace=True)


# ## Valeurs aberrantes

# La variable *ComplianceStatus* indique si le benchmarking du bâtiment est conforme à la conformité. Les bâtiments qui ne sont pas conformes donnent un enregistrement de données peu fiable et doivent donc être supprimés.

# Selon le site Seattle Energy Benchmarking :
# > *Reports with unusually low or high (outlier) EUIs or other errors will be flagged for accuracy and required to make corrections.* <br>
# 

# In[23]:


# Vérifier les valeurs dans la variable ComplianceStatus
df_non_residential.ComplianceStatus.value_counts(dropna=False)


# In[24]:


# Enlever les bâtiments Non-Compliant
non_compliant = df_non_residential[df_non_residential["ComplianceStatus"] == "Non-Compliant"]
df_non_residential.drop(index=non_compliant.index, inplace=True)


# Recherche des bâtiments restants marqués comme outlier.

# In[25]:


df_non_residential.Outlier.value_counts(dropna=False)


# In[26]:


# Supprimer les lignes signalées comme outliers
df_non_residential.drop(
    index=df_non_residential[~df_non_residential["Outlier"].isna()].index, inplace=True
)


# Étant donné que cet jeu de données enregistre la consommation d'énergie et les émissions de CO2, toute valeur inférieure à zéro n'est pas normale.

# In[27]:


# Recherche de valeurs négatives.
(df_non_residential.select_dtypes(exclude=["object"]) < 0).sum()


# Outre la variable *Longitude*, cinq variables contiennent des valeurs négatives. Ils seront supprimés.

# In[28]:


# Eliminer les bâtiments à consommation et émission négatives
negative = df_non_residential.loc[
    (df_non_residential["SourceEUIWN(kBtu/sf)"] < 0)
    | (df_non_residential["Electricity(kWh)"] < 0)
    | (df_non_residential["Electricity(kBtu)"] < 0)
    | (df_non_residential["TotalGHGEmissions"] < 0)
    | (df_non_residential["GHGEmissionsIntensity"] < 0)
]
negative


# In[29]:


df_non_residential.drop(index=negative.index, inplace=True)


# ## Outliers
# Ici, je recherche des valeurs trop élevées et les traite en fonction de la nature de leurs valeurs élevées.

# In[30]:


# Plot ces variables avec des plots de moustache pour repérer les valeurs aberrantes potentielles.
columns = ['NumberofBuildings', 'NumberofFloors', 'PropertyGFATotal',
           'PropertyGFAParking', 'PropertyGFABuilding(s)',
           'LargestPropertyUseTypeGFA', 'SecondLargestPropertyUseTypeGFA',
           'ThirdLargestPropertyUseTypeGFA']

figure, axes = plt.subplots(
    round(len(columns) / 4), 4, sharex=False, figsize=(12, 6)
)
y = 0
for col in columns:
    i, j = divmod(y, 4)
    sns.boxplot(x=df_non_residential[col], ax=axes[i, j])
    y = y + 1
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.suptitle("Distribution of numeric variables",
             fontsize=14, fontweight="bold")
plt.savefig('boxplot_outlier', dpi=300)
plt.show()


# Il y a des valeurs extrêmement larges dans les variables *NumberofBuildings*, *NumberofFloors*, *PropertyGFATotal*, etc.

# In[31]:


# Rechercher des outlier dans le variable NumberofBuildings
df_non_residential.loc[df_non_residential.NumberofBuildings ==
                       df_non_residential.NumberofBuildings.max()]


# L'Université de Washington, bien que grande, peut contient 111 bâtiments.

# In[32]:


# Rechercher des bâtiments de plus de 50 étages
df_non_residential[df_non_residential.NumberofFloors > 50]


# In[33]:


# Le nombre d'étages de 'Seattle Chinese Baptist Church' devrait être de 2, selon la recherche en ligne.
df_non_residential.loc[
    df_non_residential["PropertyName"] == "Seattle Chinese Baptist Church",
    "NumberofFloors",
] = 2


# In[34]:


# Chercher la plus grande valeur en surface de propriété.
df_non_residential.loc[df_non_residential['PropertyGFATotal']
                       == df_non_residential['PropertyGFATotal'].max()]


# L'Université de Washington - Campus de Seattle est une grande propriété. Bien qu'il s'agisse d'une valeur élevée, il s'agit d'une entrée valide. Après quelques essais, il s'avère que la suppression de ce  batiment n'améliore pas le résultat.

# ## Valeurs atypiques

# In[35]:


# Rechercher des valeurs atypiques pour la variable YearBuilt
np.sort(df_non_residential.YearBuilt.unique())


# In[36]:


df_non_residential['Neighborhood'].value_counts()


# Le quartier delridge est représenté deux fois. Corriger l'un d'eux

# In[37]:


df_non_residential['Neighborhood'].replace({"delridge neighborhoods":'delridge'},inplace=True)


# ## Valeurs a zero

# In[38]:


# Nombre des valeurs nulles par variable
df_nulle = (df_non_residential == 0).sum().to_frame().reset_index()
df_nulle.columns=['variable','counts']


# In[39]:


plt.figure(figsize=(6, 4))
zero_plot = df_nulle[df_nulle.counts != 0].sort_values(
    by='counts', ascending=True)

zero_plot.plot(kind='barh', y="counts", x="variable")
plt.xlabel(None)
plt.ylabel(None)
plt.title('Value counts for values at zeros per variable', fontsize=10)
plt.savefig("barh_plot_zero_value_counts",dpi=300,bbox_inches = 'tight')
plt.show()


# Pour les variables PropertyGFAParking, SteamUse, NaturalGas et Electricity, il est plausible qu'ils aient des valeurs nulles, puisqu'un bâtiment peut ne pas avoir d'aire de stationnement, ou ne pas consommer de vapeur/gaz/électricité. Je laisserai ces valeurs zéro intactes.
# 
# La présence de valeurs nulles est anormale pour ces variables: 
# - NumberofBuildings
# - SiteEUI(kBtu/sf)
# - SiteEUIWN(kBtu/sf)
# - SourceEUI(kBtu/sf)
# - SourceEUIWN(kBtu/sf)
# - SiteEnergyUse(kBtu)
# - SiteEnergyUseWN(kBtu)
# - TotalGHGEmissions 
# - GHGEmissionsIntensity

# In[40]:


# Remplacement des valeurs nulles par les valeurs moyennes des variables NumberofBuildings et NumberofFloors
df_non_residential[["NumberofBuildings", "NumberofFloors"]] = df_non_residential[
    ["NumberofBuildings", "NumberofFloors"]
].replace(0, df_non_residential[["NumberofBuildings", "NumberofFloors"]].mean())


# In[41]:


# Chercher des ligns avec zero consumption d'energie et zero emmision
consumption_zero = df_non_residential[
    df_non_residential["SiteEUI(kBtu/sf)"] == 0
].copy()

emission_zero = df_non_residential[
    df_non_residential["TotalGHGEmissions"] == 0
].copy()


# Les bâtiments qui n'ont pas de consommation et/ou d'émission n'aideront pas les performances de notre modèle. 
# Ils seront éliminés.
df_non_residential.drop(index=np.concatenate([consumption_zero.index,emission_zero.index]),inplace=True)


# ## Valeurs manquantes

# In[42]:


# Chercher des valeurs manquantes
nan = df_non_residential.isnull().sum().to_frame().reset_index()
nan.columns = ['variable', 'counts']
nan = nan[nan.counts != 0].sort_values(by='counts')


# In[43]:


plt.figure(figsize=(6, 4))
nan.plot(kind='barh', y='counts', x='variable')
plt.xlabel(None)
plt.ylabel(None)
plt.title('Value counts for NaN values per variable')
plt.savefig("barh_plot_nan_values", dpi=300, bbox_inches='tight')
plt.show


# Les seules variables à traiter ici sont ZipCode, LargestPropertyUseTypeGFA et LargestPropertyUseType. Le reste ne sera pas traité car ils ne seront pas utilisés pour notre analyse ou notre modélisation (en raison de la quantité de valeur manquante dont ils disposent).

# ### ZipCode
# Les codes postaux manquants sont remplacés par le code postal le plus fréquent du même quartier

# In[44]:


# Unifier le format string de la variable Neighborhood
df_non_residential.Neighborhood = df_non_residential.Neighborhood.str.lower()


# In[45]:


# Chercher les zip code le plus fréquent du chacque quartier
neighborhood_zipcode = (
    df_non_residential.groupby(["Neighborhood", "ZipCode"])["Neighborhood"]
    .count()
    .to_frame()
)
neighborhood_zipcode.columns = ["count"]
neighborhood_zipcode.reset_index("ZipCode", inplace=True)
neighborhood_zipcode.max(level=0)


# In[46]:


# Chercher les zip code manquantes et leurs quartiers
df_non_residential[df_non_residential["ZipCode"].isna()]["Neighborhood"]


# In[47]:


# Créer un dictionnaire pour ZipCode/Neighborhood.
zip_dict = {
    "north": 98165,
    "central": 98144,
    "ballard": 98134,
    "magnolia / queen anne": 98199,
    "east": 98136,
    "southeast": 98178,
    "delridge neighborhoods": 98146,
    "greater duwamish": 98199,
    "downtown": 98191,
}


# In[48]:


# Remplir le code postal manquant
df_non_residential.ZipCode = df_non_residential.ZipCode.fillna(
    df_non_residential.Neighborhood.map(zip_dict)
)


# In[49]:


df_non_residential.ZipCode = df_non_residential.ZipCode.astype("int").astype("str")


# ### Property Use Type

# In[50]:


# Chercher des valeurs manquantes dans la variable LargestPropertyUseType
df_non_residential[df_non_residential.LargestPropertyUseType.isna()]


# In[51]:


# saisir manuellement les valeurs
df_non_residential.loc[353, "LargestPropertyUseType"] = "Non-Refrigerated Warehouse"
df_non_residential.loc[2414, "LargestPropertyUseType"] = "Office"
df_non_residential.loc[2459, "LargestPropertyUseType"] = "Other"


# In[52]:


# Remplir la plus grande surface de propriété manquante avec la surface totale
df_non_residential.LargestPropertyUseTypeGFA.fillna(
    df_non_residential["PropertyGFATotal"], inplace=True
)


# ## ENERGYStarScore
# La variable ENERGYStarScore est le plus influencé par le type d'utilisation du bâtiment (selon leur site Web). Je compléterai donc les valeurs manquantes par la moyenne par type de bâtiment.

# In[53]:


df_non_residential['ENERGYSTARScore'] = df_non_residential['ENERGYSTARScore'].fillna(
    df_non_residential.groupby('PrimaryPropertyType')['ENERGYSTARScore'].transform('mean'))


# # Feature enginerring

# In[54]:


df_final = df_non_residential.copy()


# ## Ajouter de nouvelles variables

# In[55]:


# New variable: buildings' age
from datetime import datetime

df_final["BuildingAge"] = datetime.now().year - df_final["YearBuilt"]
df_final.drop(columns=["YearBuilt"], inplace=True)

# Nouvelle variable : pourcentage de parking dans la surface totale
df_final["parking_area_prcnt"] = (
    df_final["PropertyGFAParking"] / df_final["PropertyGFATotal"]
)

# Nouvelle variable : pourcentage de la plus grande surface de propriété dans la surface totale du bâtiment
df_final["largest_property_GFA_prcnt"] = (
    df_final["LargestPropertyUseTypeGFA"] / df_final["PropertyGFABuilding(s)"]
)


# In[56]:


# Les variables Electricity(kWh) et NaturalGas(therms) seront enlevées
# car ces informations sont déjà présentes dans les données (en unité kBtu).
df_final.drop(columns=["Electricity(kWh)", "NaturalGas(therms)"], inplace=True)


# ## Log transformation

# In[57]:


var_numeric = df_final[['NumberofBuildings', 'NumberofFloors', 'PropertyGFATotal', 'PropertyGFAParking',
                        'PropertyGFABuilding(s)', 'LargestPropertyUseTypeGFA', 'ENERGYSTARScore',
                        'SiteEnergyUse(kBtu)', 'TotalGHGEmissions']].copy()

figure, axes = plt.subplots(3, 3,
                            sharex=False, figsize=(9, 9))
y = 0
for var in var_numeric.columns:
    i, j = divmod(y, 3)
    variable = df_final[[var]]
    sns.histplot(variable, ax=axes[i, j])
    y = y + 1
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.suptitle('Distribution of numeric features', fontsize=10)
plt.savefig('histplot_skewed_features',bbox_inches='tight',dpi=300)
plt.show()


# In[58]:


# Transformez certains variables asymétriques en logarithme naturel.
skewed_features = ["PropertyGFATotal",
                   "PropertyGFABuilding(s)",
                   'LargestPropertyUseTypeGFA',
                   'TotalGHGEmissions',
                   'SiteEnergyUse(kBtu)'
                   ]

features_log = ["PropertyGFATotal_log",
                "PropertyGFABuilding(s)_log",
                'LargestPropertyUseTypeGFA_log',
                'TotalGHGEmissions_log',
                'SiteEnergyUse(kBtu)_log'
                ]

for feature, logged in zip(skewed_features, features_log):
    df_final[logged] = np.log(df_final[feature])


# In[59]:


df_final.drop(index=df_final[df_final.TotalGHGEmissions_log < 0].index,inplace=True)


# In[60]:


#Examinez les relations entre les caractéristiques et les variables cibles, 
# en prévision de la sélection d'un type de modèle.
var_corr = [
    'BuildingAge',
    'parking_area_prcnt',
    'largest_property_GFA_prcnt',
    'PropertyGFATotal_log',
    'PropertyGFABuilding(s)_log',
    'LargestPropertyUseTypeGFA_log',
    'TotalGHGEmissions_log',
    'SiteEnergyUse(kBtu)_log']

sns.pairplot(df_final[var_corr],kind='reg')
plt.suptitle('Correlations between variables',fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.98])
plt.savefig('pairplot', bbox_inches='tight', dpi=300)
plt.show()


# In[61]:


df_final.shape


# In[62]:


# df_final.to_csv('df_final.csv', index=False)

