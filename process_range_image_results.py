import yaml
import matplotlib.pyplot as plt
import operator

def plot_overlaps_over_frameid():

    with open('output\overlaps_results_2019_02_27_0046_labeled.yml', 'r') as file:
        labeled_overlaps = yaml.safe_load(file)

    with open('output\overlaps_results_2019_02_27_0046_raw.yml', 'r') as file:
        raw_overlaps = yaml.safe_load(file)

    with open('output\overlaps_results_2019_02_27_0046_corrected.yml', 'r') as file:
        corrected_overlaps = yaml.safe_load(file)

    frame_count = min(len(labeled_overlaps), len(raw_overlaps))

    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(list(labeled_overlaps.keys())[:frame_count], list(labeled_overlaps.values())[:frame_count], label='Labeled data')
    ax.plot(list(raw_overlaps.keys())[:frame_count], list(raw_overlaps.values())[:frame_count], label='Raw data')
    ax.plot(list(corrected_overlaps.keys())[:frame_count], list(corrected_overlaps.values())[:frame_count], label='Corrected data')
    plt.xlabel("Frame ID")
    plt.ylabel("Number of overlapping points")
    ax.legend()
    plt.title("Overlapping points for Labeled, Corrected, and Raw data (2019_02_27 - 0046)")
    plt.show()


def sort_by_decreasing_overlaps():
    with open('output\overlaps_results_2019_02_27_0004_raw.yml', 'r') as file:
        raw_overlaps = yaml.safe_load(file)

    sort = sorted(raw_overlaps.items(), key=operator.itemgetter(1), reverse=True)
    print(dict(sort))
    
plot_overlaps_over_frameid()
sort_by_decreasing_overlaps()