# PLOT INDIVIDUAL CLIPS MEAN UNIT ORACLES

def unique_clips(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

# Fetch movie names in the order in which they were presented
clip_names, clip_number = (PupilOraclesV2()).fetch('movie_name', 'clip_number')
clip_names = [(x,y) for x, y in zip(clip_names, clip_number)]
clip_names = unique_clips(clip_names)
number_of_clips = len(clip_names)
number_of_trials = int(len((PupilOraclesV2() & 'pupil_filter_method=3').fetch('trial_idx'))/number_of_clips)

# Get units oracles for all 100 trials

number_of_units = len(np.unique((UnitOraclesV4() & 'unit_trace_method=3').fetch('unit_id')))

oracles = []
for clip in clip_names:
    movie_name = {'movie_name': clip[0]}
    clip_number = {'clip_number': clip[1]}
    oracles.append((UnitOraclesV4() & 'unit_trace_method=3' & movie_name & clip_number).fetch('unit_oracle'))
print(np.shape(oracles))

fig, axs = plt.subplots(3, 2, figsize=(12,8))

for ax, clip_name, unit_oracle in zip(axs.flat, clip_names, oracles):
    
    units_oracles = np.reshape(unit_oracle, (number_of_units, number_of_trials))
    mean_unit_oracle = np.mean(units_oracles, axis=0)
    
    # build linear model
    
    x = np.linspace(0, len(mean_unit_oracle), num=len(mean_unit_oracle) , endpoint=True)
    x = x.reshape(-1,1)
    model = LinearRegression().fit(x, mean_unit_oracle)
    r_sq = model.score(x, mean_unit_oracle)
    #print('coefficient of determination:', r_sq)
    #print('intercept:', model.intercept_)
    #print('slope:', model.coef_)
    y_pred = model.predict(x)
    #print('predicted response:', y_pred, sep='\n')
    y = model.coef_*x + model.intercept_
    
    ax.plot(x, mean_unit_oracle)
    ax.text(0.95, 0.01, 'slope:' "{:.5f}".format(model.coef_[0]),
            verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes,
            color='k', fontsize=10)
    ax.text(0.70, 0.01, 'Rsq:' "{:.3f}".format(r_sq),
            verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes,
            color='k', fontsize=10)
    ax.plot(x,y, c='gray')
    ax.set_xlabel('Trial', fontsize=14)
    ax.set_ylabel('Units mean oracle', fontsize=14)
    ax.set_title(f'{clip_name}', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=14)
    
# for ax in axs.flat:
#     ax.label_outer()
fig.tight_layout(pad=1.0)