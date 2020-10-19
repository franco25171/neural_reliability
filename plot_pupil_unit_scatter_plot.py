# PLOT SCATTER PLOTS FOR ALL 100 UNITS AND PUPIL TRIALS

t = time.time()
#************************************************

# Fetch movie names in the order in which they were presented
clip_names, clip_number = (PupilOraclesV2()).fetch('movie_name', 'clip_number')
clip_names = [(x,y) for x, y in zip(clip_names, clip_number)]
clip_names = unique_clips(clip_names)
number_of_clips = len(clip_names)
number_of_trials = int(len((PupilOraclesV2() & 'pupil_filter_method=3').fetch('trial_idx'))/number_of_clips)

# Fetch pupil traces
all_pupil_traces = []
for clip_name, clip_number in clip_names:
    
    pupil_traces = []
    
    movie_name = {'movie_name': clip_name}
    number_of_clip = {'clip_number': clip_number}
    data = (PupilOraclesV2() & 'pupil_filter_method=3' & movie_name & number_of_clip).fetch('pupil_trace')
    # ********** Uncomment this section if you want to normalize the entire matrix  *************
#     data = np.transpose(np.squeeze(np.dstack(data)))
#     data = data - np.mean(data, axis=0)
#     xmax, xmin = np.array(data).max(), np.array(data).min()
#     data = (data)/(xmax - xmin)
    # *******************************************************************************************
    for i in range(len(data)):
        pupil_traces.append((data[i]))

    all_pupil_traces.append(pupil_traces)
    

# Get units oracles for all 100 trials

number_of_units = len(np.unique((UnitOraclesV4() & 'unit_trace_method=3').fetch('unit_id')))

oracles = []
for clip in clip_names:
    movie_name = {'movie_name': clip[0]}
    clip_number = {'clip_number': clip[1]}
    oracles.append((UnitOraclesV4() & 'unit_trace_method=3' & movie_name & clip_number).fetch('unit_oracle'))
print(np.shape(oracles))


    
fig, axs = plt.subplots(3, 2, figsize=(12,8))

for ax, clip_name, pupil_traces, unit_oracle in zip(axs.flat, clip_names, all_pupil_traces, oracles):
    
    units_oracles = np.reshape(unit_oracle, (number_of_units, number_of_trials))
    mean_unit_oracle = np.mean(units_oracles, axis=0)
    
    pupil_mean = np.mean(pupil_traces, axis=1)
    
    # build linear model
    
    x = np.linspace(0, 1, num=number_of_trials , endpoint=True)
    mean_unit_oracle = mean_unit_oracle.reshape(-1,1)
    model = LinearRegression().fit(mean_unit_oracle, pupil_mean)
    r_sq = model.score(mean_unit_oracle, pupil_mean)
    #print('coefficient of determination:', r_sq)
    #print('intercept:', model.intercept_)
    #print('slope:', model.coef_)
    y_pred = model.predict(mean_unit_oracle)
    #print('predicted response:', y_pred, sep='\n')
    y = model.coef_*x + model.intercept_
    
    ax.scatter(mean_unit_oracle, pupil_mean)
    ax.text(0.95, 0.01, 'slope:' "{:.5f}".format(model.coef_[0]),
            verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes,
            color='k', fontsize=10)
    ax.text(0.70, 0.01, 'Rsq:' "{:.3f}".format(r_sq),
            verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes,
            color='k', fontsize=10)
    ax.plot(x,y, c='gray')
    ax.set_xlabel('Units mean oracle', fontsize=14)
    ax.set_ylabel('Pupil mean size', fontsize=14)
    ax.set_title(f'{clip_name}', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=14)
    ax.set(xlim=(0.0,0.5))#, ylim=(0,1))
    
# for ax in axs.flat:
#     ax.label_outer()
fig.tight_layout(pad=1.0)

#************************************************
elapsed = time.time() - t
print('elapsed_time:' , elapsed, 'seconds')
