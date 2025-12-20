import numpy as np
import pandas as pd

df = pd.read_csv(r"C:\Users\Utilisateur\OneDrive\Documents\archive\adult.csv")


df = df.replace("?", np.nan)

print(df.head())
print(df.info())

df["workclass"] = df["workclass"].astype(str)
df["education"] = df["education"].astype(str)
df["marital-status"] = df["marital-status"].astype(str)

print(df.info())

#suppression des lignes incomplete
df = df.dropna()
df = df.replace("?", np.nan)

print(df["income"].unique())

#cible en binaire : 1 si >50k sinon 0
y = (df["income"].str.contains(">50K")).astype(int)

noms_uniques = y.unique().astype(str)

#nos features sans la colonne cible y
x = df.drop(columns=["income"])

#encodage des colonnes texte en valeur numérique
x = pd.get_dummies(x, drop_first=True, dtype=np.float32)
x = x.astype("float32")
print(x.shape)


from sklearn.model_selection import train_test_split, GridSearchCV

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, random_state = 42
)

print(x.dtypes.unique())

print("Colonnes non numériques dans X :")
print(x.select_dtypes(exclude=[np.number]).columns)

import torch
from torch.utils.data import TensorDataset, DataLoader

x_train_tensor = torch.tensor(x_train.values, dtype= torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype= torch.float32).view(-1, 1)

x_test_tensor  = torch.tensor(x_test.values, dtype=torch.float32)
y_test_tensor  = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)


#modele MLP
import torch.nn as nn

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




# implémentation de random forest avec sklearn

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

parametre_grid={
    "n_estimators" : [50, 100, 200],
    "criterion" : ["gini", "entropy"],
    'max_depth' : [None, 10, 20]
}

grid_Search = GridSearchCV(estimator = RandomForestClassifier(),

                           param_grid = parametre_grid,
                           cv = 5,
                           verbose = 1)

model = RandomForestClassifier(
    criterion = "gini",
    n_estimators = 100,
    max_depth = None,
    random_state = 42
)


grid_Search.fit(x_train, y_train)

print("Meilleurs parametres trouvé:", grid_Search.best_params_)

meilleursModele = grid_Search.best_estimator_
prediction = meilleursModele.predict(x_test)

print(f"Précision (Accuracy): {accuracy_score(y_test, prediction):.2f}")
print("\nRapport de classification :")
print(classification_report(y_test, prediction, target_names=noms_uniques))


import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

importance = meilleursModele.features_importances_
noms_colonnes = x.columns

feat_importances = pd.Series(importance, index=noms_colonnes)

feat_importances.nlargest(10)
feat_importances.plot(kind='barh')

ConfusionMatrixDisplay.from_estimator(meilleursModele, y_test, x_test)

plt.show()