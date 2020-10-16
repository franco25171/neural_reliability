# PLOT PUPIL TRACES HEAT MAPS FOR ALL CLIPS

# Fetch movie names in the order in which they were presented
clip_names, clip_number = (PupilOraclesV2()).fetch('movie_name', 'clip_number')
clip_names = [(x,y) for x, y in zip(clip_names, clip_number)]
clip_names = unique_clips(clip_names)

fig, axs = plt.subplots(3, 2, figsize=(12,8))
jet = cm = plt.get_cmap('jet')

all_pupil_traces = []
for ax, (clip_name, clip_number) in zip(axs.flat, clip_names):
    
    pupil_traces = []
    
    movie_name = {'movie_name': clip_name}
    number_of_clip = {'clip_number': clip_number}
    data = (PupilOraclesV2() & 'pupil_filter_method=3' & movie_name & number_of_clip).fetch('pupil_trace_diff')
    # ********** Uncomment this section if you want to normalize the entire matrix  *************
#     data = np.transpose(np.squeeze(np.dstack(data)))
#     data = data - np.mean(data, axis=0)
#     xmax, xmin = np.array(data).max(), np.array(data).min()
#     data = (data)/(xmax - xmin)
    # *******************************************************************************************
    for i in range(len(data)):
        pupil_traces.append((data[i]))

    all_pupil_traces.append(pupil_traces)

    im = ax.imshow(pupil_traces, aspect='auto', cmap=jet, vmin=-1, vmax=1)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('dilation/constriction\nrate', fontsize=10)
    ax.set(xlabel='sample', ylabel='trial #', title=f'{clip_name}_{clip_number}')

for ax in axs.flat:
    ax.label_outer()
fig.tight_layout(pad=1.0)