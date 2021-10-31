import matplotlib.pyplot as plt
import numpy as np
import sys

image_fname, centroids_fname, out_fname = sys.argv[1], sys.argv[2], sys.argv[3]
centroids = np.loadtxt(centroids_fname)  # load centroids
orig_pixels = plt.imread(image_fname)
pixels = orig_pixels.astype(float) / 255.
# Reshape the image(128x128x3) into an Nx3 matrix where N = number of pixels.
pixels = pixels.reshape(-1, 3)
print(centroids)


def Kmeans():
    print(len(pixels))
    output_file = open(out_fname, "w")
    iteration = 0
    while iteration < 20:
        counter = 0
        clusters = [[] for centroid in centroids]

        # clustering
        for pixel in pixels:
            distances = []
            # calculate the distances between the current pixel to each centroid
            for centroid in centroids:
                distance = np.sqrt(np.sum(np.square(centroid - pixel)))
                distances.append(distance)
            min_distance = min(distances)

            # the index of the closest centroid
            index = distances.index(min_distance)
            clusters[index].append(pixel)

        # update each centroid according to the avarage of its cluster
        for i in range(len(clusters)):
            sum_of_pixels = sum(clusters[i])
            average = sum_of_pixels / len(clusters[i])

            if (centroids[i] == average).all():
                counter += 1
            centroids[i] = average

        # convegence
        if counter == len(centroids):
            break
        else:
            print(
                f"[iter {iteration}]:{','.join([str(i.round(4)) for i in centroids])}\n"
            )
            output_file.write(
                f"[iter {iteration}]:{','.join([str(i) for i in centroids])}\n"
            )
        iteration += 1


if __name__ == "__main__":
    Kmeans()