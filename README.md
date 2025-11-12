# Iris_Data
Training a model to tell which species a flower belongs to based on its measurements.

The Iris dataset contains information about flowers — specifically, iris flowers of three different species: Setosa, Versicolor and Virginica.
Each flower is described by 4 features: Sepal length, width and petal length, width.

The model uses a decision tree classifier to classify the flowers based on their measurement. The root node would be the petal length. If petal length <2.45 cm, then the flower is likely to be setosa. Else, it moves onto the internal nodes, which are the width and the length (>2.45 cm). The leaf nodes are the flower species themselves.

The dataset is part of Scikit-Learn's vast dataset repository. 
