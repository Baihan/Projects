# Import pyplot
from matplotlib import pyplot as plt
# Import PCA
from sklearn.decomposition import PCA
# Import numpy
import numpy as np

# Use numpy (loadtxt) to read csv file
with open("Unsupervised learning/interpretable_features/lcd-digits.csv") as lcd:
    samples = np.loadtxt(lcd, delimiter=",")


def show_as_image(sample):
    bitmap = sample.reshape((13, 8))
    plt.figure()
    plt.imshow(bitmap, cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.show()


# Create a PCA instance: model
model = PCA(n_components=7)

# Apply fit_transform to samples: features
features = model.fit_transform(samples)

# Call show_as_image on each component
for component in model.components_:
    show_as_image(component)
