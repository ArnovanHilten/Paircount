import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.ensemble import RandomForestClassifier


data_path = "/media/avanhilten/pHDD1TB/Paircount/datasets/Animals.csv"


x = pd.read_csv(data_path)
y = np.arange(x.shape[0])


# Model (can also use single decision tree)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=10)

# Train
model.fit(x.drop(x.columns[0], axis=1),y)
# Extract single tree
estimator = model.estimators_[5]

from sklearn.tree import export_graphviz
# Export as dot file
export_graphviz(estimator, out_file='tree.dot',
                feature_names = x.columns[1:],
                class_names = x['ylabel'],
                rounded = True, proportion = False,
                precision = 2, filled = True)

# Convert to png using system command (requires Graphviz)
from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree_animals.png', '-Gdpi=600'])

# Display in jupyter notebook
