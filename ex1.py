import matplotlib.pyplot as plt
import numpy as np
import sys
import os

image_fname, centroids_fname, out_fname = sys.argv[1], sys.argv[2], sys.argv[3]
centroids = np.loadtxt(centroids_fname)
orig_pixels = plt.imread(image_fname)
pixels = orig_pixels.astype(float) / 255.
pixels = pixels.reshape(-1, 3)


def Kmeans():
    output_file = open(out_fname, "w")
    iteration = 0
    prev_centroids = centroids.copy()

    while iteration < 20:
        counter = 0
        clusters = [[] for centroid in centroids]

        # clustering
        for pixel in pixels:
            distances = []
            # calculate the distances between the current pixel to each centroid
            for centroid in centroids:
                distance = np.linalg.norm(pixel - centroid)
                distances.append(distance)
            min_distance = min(distances)
            # the index of the closest centroid
            index = distances.index(min_distance)
            clusters[index].append(pixel)

        # update each centroid according to the avarage of its cluster
        for i in range(len(clusters)):
            average = np.average(clusters[i], axis=0)
            centroids[i] = centroids[i].round(4)
            average = average.round(4)
            centroids[i] = average.copy()

        output_file.write(
            f"[iter {iteration}]:{','.join([str(i.round(4)) for i in centroids])}\n"
        )

        # check convegence
        if np.array_equal(centroids, prev_centroids):
            break
        else:
            prev_centroids = centroids.copy()
            iteration += 1


if __name__ == "__main__":
    Kmeans()