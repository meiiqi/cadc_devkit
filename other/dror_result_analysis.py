import json
import numpy as np
import matplotlib.pyplot as plt

with open('dror_results.json', 'r') as f:
  data = json.load(f)

clear = 25
light = 250
medium = 500
heavy = 750
extreme = 1500

npdata = np.empty([7000, 5], dtype=int)

i = 0
for drive, sequences in data.items():
    for seq, frame_data in sequences.items():
        for frame, snow_points in enumerate(frame_data):
            if snow_points < clear:
                npdata[i] = [drive, seq, frame, snow_points, 0]
            elif snow_points < light:
                npdata[i] = [drive, seq, frame, snow_points, 1]
            elif snow_points < medium:
                npdata[i] = [drive, seq, frame, snow_points, 2]
            elif snow_points < heavy:
                npdata[i] = [drive, seq, frame, snow_points, 3]
            elif snow_points < extreme:
                npdata[i] = [drive, seq, frame, snow_points, 4]
            else:
                npdata[i] = [drive, seq, frame, snow_points, 5]

            i += 1


statistics = {
    "clear_count": len(npdata[npdata[:, 4] == 0]),
    "light_count": len(npdata[npdata[:, 4] == 1]),
    "medium_count": len(npdata[npdata[:, 4] == 2]),
    "heavy_count": len(npdata[npdata[:, 4] == 3]),
    "extreme_count": len(npdata[npdata[:, 4] == 4]),
    "blizzard_count": len(npdata[npdata[:, 4] == 5]),
}

print(npdata[npdata[:, 4] == 5])
print(statistics)

fig = plt.figure(figsize = (10, 5))

# creating the bar plot
plt.bar(statistics.keys(), statistics.values(), color ='blue',
        width = 0.4)

plt.xlabel("Snowfall intensity")
plt.ylabel("No. of snow points ")
plt.title("Snowfall statistics of CADC")
# plt.show()

plt.savefig("statistics.png")