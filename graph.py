import os
from matplotlib import pyplot as plt

tests_dir = 'stored'
tot_episodes = 1001
# directory name is the key and files list is the value
filesdict = dict()

for dirpath,_,files in os.walk(os.getcwd() + os.path.sep + tests_dir):
    # extract *.csv in folders
    filename = [s for s in files if ".csv" in s]
    if len(filename) > 0:
        filesdict[dirpath] = filename[0]

#print(filesdict)
data_dict = dict()

x = list(range(0, tot_episodes))
y = [0] * tot_episodes
for item in filesdict.items():
    filepath = item[0] + os.path.sep + item[1]
    with open(filepath, 'r') as file:
        # skip header
        next(file)
        for line in file:
            # split episode number from result
            n_episode, reward = line.split(',')
            n_episode = int(float(n_episode))
            y[n_episode] = float(reward)
        test_name = item[0].split(os.path.sep)[-1]
        data_dict[test_name] = y.copy()

# data_dict contains the interesting data
# underlying plot
for title in data_dict.keys():
    plt.plot(x, data_dict[title], label=title)

plt.legend()
plt.axis([0,1000,-750, 150])
plt.show()

# divided plot
i=0
x_limits = (0,1000)
y_limits = (-750, 150)
for title in data_dict.keys():
    i+=1
    ax = plt.subplot(3,3,i)
    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)
    plt.plot(x, data_dict[title], label=title)
    plt.title(title)
plt.show()
