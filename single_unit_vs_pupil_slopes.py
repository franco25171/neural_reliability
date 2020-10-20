# CALCULATE SLOPE FOR EVERY UNIT VS PUPIL MEAN 

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

slopes = []
negative_slopes = []
positive_slopes = []
for clip_name, pupil, oracle in zip(clip_names, all_pupil_traces, oracles):
    
    pupil_mean = np.mean(pupil, axis=1)
    oracle = np.reshape(oracle,(number_of_units, number_of_trials))
    
    # build linear model
    
    x = np.linspace(0, number_of_trials, num=number_of_trials, endpoint=True)
    
    slope = []
    negative_slope = []
    positive_slope = []
    for i in range(number_of_units): 
        #unit_oracles = np.array(unit_oracles)
        unit_oracles = oracle[i]
        unit_oracles = unit_oracles.reshape(-1,1)
        model = LinearRegression().fit(unit_oracles, pupil_mean)
        r_sq = model.score(unit_oracles, pupil_mean)
        #print('coefficient of determination:', r_sq)
        #print('intercept:', model.intercept_)
        #print('slope:', model.coef_)
        y_pred = model.predict(unit_oracles)
        #print('predicted response:', y_pred, sep='\n')
        y = model.coef_*x + model.intercept_
        slope.append(model.coef_)
        negative_slope.append([i, model.coef_]) if model.coef_ < 0 else positive_slope.append([i, model.coef_])
    slopes.append(slope)
    negative_slopes.append(negative_slope)
    positive_slopes.append(positive_slope)

# Plot histograms of slopes

plt.rcParams['figure.figsize'] = [16, 12]
plt.rcParams.update({'font.size':18})

fig, axs = plt.subplots(3, 2, figsize=(16,12))

x_bins = np.linspace(-40, 40, num=81, endpoint=True)

for ax, slopes_, clip_name in zip(axs.flat, slopes, clip_names):
    slopes_ = np.squeeze(slopes_)
    media = np.median(slopes_)
    ax.hist(slopes_, bins = x_bins)
    ax.axvline(x=media, color='k', ls='--')
    ax.set(xlabel='slope', ylabel='frequency', title=f'{clip_name}', ylim=(0,400))#, xlim=(-10,10))
    ax.annotate('median:' + str(round(media,3)), xy=(10,200), fontsize=20)
fig.tight_layout()