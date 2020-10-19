# PLOT UNITS MEAN ORACLE ACROSS TRIALS FOR ALL MOVIE CLIPS AS THEY WERE PRESENTED

def unique_clips(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

# Fetch mean unit oracle per trial
unit_oracles, clip_names, clip_number = (UnitOraclesV4() & 'unit_trace_method=3').fetch('unit_oracle','movie_name','clip_number')
units = np.unique((UnitOraclesV4() & 'unit_trace_method=3').fetch('unit_id'))
trials = np.unique((UnitOraclesV4() & 'unit_trace_method=3').fetch('trial_idx'))
unit_oracles = np.reshape(unit_oracles, (len(units),len(trials)))
unit_oracles = np.mean(unit_oracles, axis=0)

colors = ['#EA1CD7','#A91CEA','#1C2CEA','#1CEADA','#6DEA1C','#EAC31C'] * 100

# Fetch movie names in the order in which they were presented
clip_names = [(x,y) for x, y in zip(clip_names, clip_number)]
clip_names = unique_clips(clip_names)

# Plot scatter plot
x = np.linspace(0,len(pupil_means),len(pupil_means))

for i in range(len(unit_oracles)):
    plt.scatter(x[i], unit_oracles[i], c=colors[i])
plt.legend(clip_names, ncol=3, frameon=False)
plt.plot(x, unit_oracles, c='gray', alpha=0.5)
plt.xlabel('Trial')
plt.ylabel('Mean unit oracle')
plt.title('')