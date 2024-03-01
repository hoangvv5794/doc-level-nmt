import codecs
import time

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


def load_lines_special(file_name):
    with codecs.open(file_name, 'r', 'utf-8') as fin:
        lines = [line.strip() for line in fin.read().split('\n')]
        return lines


def plot_example(distances):
    plt.plot(distances);

    y_upper_bound = 1.2
    plt.ylim(0, y_upper_bound)
    plt.xlim(0, len(distances))

    breakpoint_percentile_threshold = 80
    breakpoint_distance_threshold = np.percentile(distances,
                                                  breakpoint_percentile_threshold)
    plt.axhline(y=breakpoint_distance_threshold, color='r', linestyle='-');
    # Then we'll see how many distances are actually above this one
    num_distances_above_theshold = len([x for x in distances if
                                        x > breakpoint_distance_threshold])
    plt.text(x=(len(distances) * .01), y=y_upper_bound / 50, s=f"{num_distances_above_theshold + 1} Chunks");
    plt.show()

    time.sleep(15)


def calculate_cosine_distances(list_embedding):
    distances = []
    for i in range(len(list_embedding) - 1):
        embedding_current = list_embedding[i]
        embedding_next = list_embedding[i + 1]

        # Calculate cosine similarity
        similarity = cosine_similarity([embedding_current], [embedding_next])[0][0]

        # Convert to cosine distance
        distance = 1 - similarity

        # Append cosine distance to the list
        distances.append(distance)

    # Optionally handle the last sentence
    # sentences[-1]['distance_to_next'] = None  # or a default value

    return distances


def calculate_cosine_distances_dictionary(data_set):
    distances = []
    for i in range(len(data_set) - 1):
        embedding_current = data_set[i]['combined_sentence_embedding']
        embedding_next = data_set[i + 1]['combined_sentence_embedding']

        # Calculate cosine similarity
        similarity = cosine_similarity([embedding_current], [embedding_next])[0][0]

        # Convert to cosine distance
        distance = 1 - similarity

        # Append cosine distance to the list
        distances.append(distance)

        # Store distance in the dictionary
        data_set[i]['distance_to_next'] = distance

    # Optionally handle the last sentence
    # sentences[-1]['distance_to_next'] = None  # or a default value

    return distances, data_set


def load_lines(file_name):
    with codecs.open(file_name, 'r', 'utf-8') as fin:
        lines = [line.strip() for line in fin.readlines()]
        return lines


def read_file(file_name):
    with codecs.open(file_name, 'r', 'utf-8') as fin:
        return fin.read()


def save_lines(file_name, lines):
    with codecs.open(file_name, 'w', 'utf-8') as fout:
        for line in lines:
            print(line, file=fout)


def remove_seps(text):
    sents = [s.strip() for s in text.replace('<s>', '').split('</s>')]
    sents = [s for s in sents if len(s) > 0]
    return sents
