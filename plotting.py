from matplotlib import pyplot
import json
import numpy as np



samples = 698.0

with open('computed_results.json') as f:
        print("test", f.read())
        data = json.loads(f.read())
        

        
means = ['google_linear', 'google_radial', 'google_rf', 'own_linear', 'own_radial', 'own_rf']
measures = ['folds_google_linear', 'folds_google_radial', 'folds_google_rf', 'folds_own_linear', 'folds_own_radial', 'folds_own_rf']

variances = {}
for mean_key, measure_key in zip(means, measures):
    mean = data[mean_key]
    measure = data[measure_key]
    variances[mean_key] = np.sum([np.power((x-mean),2) for x in measure])/(samples-1)
    
models_own = ['own_linear', 'own_radial', 'own_rf']
models_google = ['google_linear', 'google_radial', 'google_rf']

t_values = {}
for model_own, model_google in zip(models_own, models_google):
    t_values[model_own] = np.abs(data[model_own]-data[model_google])/np.sqrt(variances[model_own]+variances[model_google])
    
with open('T_scors.json', 'w') as write_file:
    json.dump(t_values, write_file, indent=4, sort_keys=True)
    print("dumped")