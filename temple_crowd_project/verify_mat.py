import scipy.io
import numpy as np

mat = scipy.io.loadmat(
    "dataset/ground_truth/img_00000.mat"
)

print("MAT Keys:")
print(mat.keys())

points = mat['annPoints']

print("\nShape:")
print(points.shape)

print("\nFirst 10 Points:")
print(points[:10])

print("\nPeople Count:")
print(len(points))