#import des librairies numpy et pandas
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#chargement du fichier csv dans df notre dataframe
df = pd.read_csv("adult.csv")

#suppression des caracteres non voulu par des case vide avec numpy
df = df.replace("?", np.nan)

#print de verification
print(df.head())
print(df.info())

#nettoyage des colonnes ici on s'assure que ce soit bien des strings meme si ca ne me semble pas obligatoire
df["workclass"] = df["workclass"].astype(str)
df["education"] = df["education"].astype(str)
df["marital-status"] = df["marital-status"].astype(str)

print(df.info())

#suppression des lignes incomplete
df = df.dropna()
df = df.replace("?", np.nan)

print(df["income"].unique())

#cible en binaire : 1 si >50k sinon 0 (y represente les données a prédire)
y = (df["income"].str.contains(">50K")).astype(int)

noms_uniques = y.unique().astype(str)

#nos features sans la colonne cible y car x sont les données qui vont nous permettre de pedire y
x = df.drop(columns=["income"])

#encodage des colonnes texte en valeur numérique
x = pd.get_dummies(x, drop_first=True, dtype=np.float32)
x = x.astype("float32")
print(x.shape)

#Question 2
#Equilibre des classes
print("Répartition brute de income :")
print(df["income"].value_counts())

print("\nRépartition en pourcentage :")
print(df["income"].value_counts(normalize=True) * 100)

#Y
print("\nRépartition de y (0/1) :")
print(y.value_counts())
print("\nRépartition de y (%) :")
print(y.value_counts(normalize=True) * 100)

#Statistiques descriptives simples
print("\nStatistiques descriptives des variables numériques :")
print(df.describe())

cat_cols = ["workclass", "education", "marital-status", "gender"]
for col in cat_cols:
    print(f"\nTop modalités pour {col} :")
    print(df[col].value_counts().head(5))

#Visualisation de l'équilibre des classes
import matplotlib.pyplot as plt

df["income"].value_counts().plot(kind="bar")
plt.title("Répartition des classes de revenu")
plt.xlabel("Classe de revenu")
plt.ylabel("Nombre d'observations")
plt.show()

from sklearn.model_selection import train_test_split, GridSearchCV

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, random_state = 42
)

print(x.dtypes.unique())

print("Colonnes non numériques dans X :")
print(x.select_dtypes(exclude=[np.number]).columns)

import torch
from torch.utils.data import TensorDataset, DataLoader

#transformation des data en tensors
x_train_tensor = torch.tensor(x_train.values, dtype= torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype= torch.float32).view(-1, 1)

x_test_tensor  = torch.tensor(x_test.values, dtype=torch.float32)
y_test_tensor  = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

#implementation de ces datas dans des datasets
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)


#modele MLP -> Multi-Layer Perceptron
import torch.nn as nn

#recupération de la dimension des donnée "pour du calcule matriciel"
input_dim = x_train_tensor.shape[1]

class IncomeNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)
model = IncomeNet(input_dim)


criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)



#boucle d'entrainement
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.00

    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()* batch_x.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)

    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}")

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        outputs = model(batch_X)
        preds = torch.sigmoid(outputs)  # probas
        predicted = (preds >= 0.5).float()
        correct += (predicted == batch_y).sum().item()
        total   += batch_y.size(0)

accuracy = correct / total
print(f"Accuracy: {accuracy:.4f}")



#Question 4
#arbre de décision
from sklearn.tree import DecisionTreeClassifier, export_text

tree_clf = DecisionTreeClassifier(
    criterion="gini",          # ou "entropy"
    max_depth=None,            # profondeur illimitée au début
    min_samples_split=20,      # analogue à minsplit
    ccp_alpha=0.0,             # paramètre de pruning (analogue à cp)
    random_state=42
)

tree_clf.fit(x_train, y_train)

#affichage des règles
tree_rules = export_text(tree_clf, feature_names=list(x_train.columns), max_depth=3)
print(tree_rules)

#Question 5
#prédiction sur l'arbre
y_pred_tree = tree_clf.predict(x_test)

#erreur de test
accuracy_tree = accuracy_score(y_test, y_pred_tree)
print(f"Accuracy arbre de décision : {accuracy_tree:.4f}")
print(f"Erreur de test arbre de décision : {1 - accuracy_tree:.4f}")


#matrice de confusion
cm_tree = confusion_matrix(y_test, y_pred_tree)
print("Matrice de confusion (arbre de décision) :")
print(cm_tree)

print("\nRapport de classification (arbre de décision) :")
print(classification_report(y_test, y_pred_tree, target_names=noms_uniques))

#visualisatiion matrice
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(tree_clf, x_test, y_test)
plt.title("Matrice de confusion - Arbre de décision")
plt.show()

# implémentation de random forest avec sklearn



parametre_grid={
    "n_estimators" : [50, 100, 200],
    "criterion" : ["gini", "entropy"],
    'max_depth' : [None, 10, 20]
}

#initialisation du model RandomForest avec grid search pour trouver les meilleurs parametres
grid_Search = GridSearchCV(estimator = RandomForestClassifier(),

                           param_grid = parametre_grid,
                           cv = 5,
                           verbose = 1)

#initialisation du model RandomForest
model = RandomForestClassifier(
    criterion = "gini",
    n_estimators = 100,
    max_depth = None,
    random_state = 42
)


grid_Search.fit(x_train, y_train)

print("Meilleurs parametres trouvé:", grid_Search.best_params_)

#meilleursModele contiens les meilleurs estimateur selon la grid Search
meilleursModele = grid_Search.best_estimator_

#prediction avec predict du model
prediction = meilleursModele.predict(x_test)

print(f"Précision (Accuracy): {accuracy_score(y_test, prediction):.2f}")
print("\nRapport de classification :")
print(classification_report(y_test, prediction, target_names=noms_uniques))


import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

importance = meilleursModele.feature_importances_
noms_colonnes = x.columns

feat_importances = pd.Series(importance, index=noms_colonnes)

#preparation des données pour affichage optimisé
feat_importances.nlargest(10).sort_values().plot(kind='barh', color='skyblue')

#ajout du contexte
plt.title("Top 10 des variables les plus importantes")
plt.xlabel("Score d'importance")
plt.ylabel("Variables")

#matrice de confusion
ConfusionMatrixDisplay.from_estimator(meilleursModele, x_test, y_test)

plt.show()


from sklearn.preprocessing import StandardScaler

# On utilise ton 'x' (qui contient déjà les dummies)
scaler = StandardScaler()

# On "fit" (calcule moyenne/écart-type) et "transform" (applique la formule)
x_scaled = scaler.fit_transform(x)


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

wcss = []
# On teste de 1 à 10 clusters
for i in range(1, 11):
    # init='k-means++' aide l'algo à converger plus vite
    kmeans_test = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
    kmeans_test.fit(x_scaled)
    wcss.append(kmeans_test.inertia_)

# On affiche le graphique
plt.figure(figsize=(10,5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Méthode du Coude (Elbow Method)')
plt.xlabel('Nombre de clusters')
plt.ylabel('Inertie (WCSS)')
plt.show()



# Choix du nombre de clusters
k = 3

kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)

# Entrainement sur les données mises à l'échelle
kmeans.fit(x_scaled)

# Récupération des étiquettes (0, 1, 2...) pour chaque ligne de ton tableau
cluster_labels = kmeans.labels_


# On crée une copie de x pour ne pas casser tes variables pour la suite
df_analyse = x.copy()

# On ajoute la colonne des clusters
df_analyse['Cluster'] = cluster_labels

# On ajoute aussi ton 'y' (income) juste pour voir comment les groupes se comportent face au revenu
# (Rappel : le K-Means n'a PAS utilisé cette info pour créer les groupes)
df_analyse['Revenu_Target'] = y

# Analyse des profils moyens par cluster
print("\n--- MOYENNES PAR CLUSTER ---")
# On groupe par cluster et on prend la moyenne
grouped_means = df_analyse.groupby('Cluster').mean()

# Affichons quelques colonnes intéressantes (les numériques pures sont plus parlantes)
# 'age', 'education-num', 'hours-per-week' sont dans ton x d'origine ou encodé ?
# Comme tu as fait un get_dummies, 'age' est intact, mais 'education' est éclaté.
# Regardons tout :
print(grouped_means.T) # .T transpose pour une lecture plus facile verticale