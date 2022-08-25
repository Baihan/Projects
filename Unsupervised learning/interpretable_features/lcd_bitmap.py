# Import pyplot
from matplotlib import pyplot as plt
# Import NMF
from sklearn.decomposition import NMF
# Import numpy
import numpy as np

# Use numpy (loadtxt) to read csv file
with open("Unsupervised learning/interpretable_features/lcd-digits.csv") as lcd:
    samples = np.loadtxt(lcd, delimiter=",")


# Select the 0th row: digit
digit = samples[0, :]

# Print digit
print(digit)

# Reshape digit to a 13x8 array: bitmap
bitmap = digit.reshape(13, 8)

# Print bitmap
print(bitmap)

# Use plt.imshow to display bitmap
plt.imshow(bitmap, cmap='gray', interpolation='nearest')
plt.colorbar()
plt.show()


# Create an NMF model: model
model = NMF(n_components=7)

# Apply fit_transform to samples: features
features = model.fit_transform(samples)

# Function to display bitmap as an image


def show_as_image(sample):
    bitmap = sample.reshape((13, 8))
    plt.figure()
    plt.imshow(bitmap, cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.show()


# Call show_as_image on each component
for component in model.components_:
    show_as_image(component)

# Select the 0th row of features: digit_features
digit_features = features[0, :]

# Print digit_features
print(digit_features)
