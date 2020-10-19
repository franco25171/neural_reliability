# PLOT PUPIL MEAN SIZE ACROSS TRIALS FOR ALL MOVIE CLIPS AS THEY WERE PRESENTED

# Fetch pupil traces
pupil_traces = (PupilOraclesV2() & 'pupil_filter_method=3').fetch('pupil_trace')

# Calculate mean for every pupil trace
pupil_means = []
for i in range(len(pupil_traces)):
    pupil_means.append(np.mean(pupil_traces[i]))

colors = ['#EA1CD7','#A91CEA','#1C2CEA','#1CEADA','#6DEA1C','#EAC31C'] * 100

# Fetch movie names in the order in which they were presented
clip_names, clip_number = (PupilOraclesV2()).fetch('movie_name', 'clip_number')
clip_names = [(x,y) for x, y in zip(clip_names, clip_number)]
clip_names = unique_clips(clip_names)

# Plot scatter plot
x = np.linspace(0,len(pupil_means),len(pupil_means))

for i in range(len(pupil_means)):
    plt.scatter(x[i], pupil_means[i], c=colors[i])
plt.legend(clip_names, ncol=3, frameon=False)
plt.plot(x, pupil_means, c='gray', alpha=0.5)
plt.xlabel('Trial')
plt.ylabel('Mean pupil size')
plt.title('')