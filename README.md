# Modelling of k-growing graphs and privacy for eta-AD

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/vs-uulm/eta-adaptive/main?filepath=Distribution%20Model.ipynb)

## Structure

 * Distribution Model Notebook: Followalong with the Distribution Model section of the paper. (You can run the interactive versin on [Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/vs-uulm/eta-adaptive/main?filepath=Distribution%20Model.ipynb))
 * 'models.py': implementation of the fittable and fitted models we use for calculating mu and sigma.
 * 'data.py': Provides the 3 datasets as canonical objects base, val1 and val2.
 * 'helpers.py': Provide some internal helper methods to process, load, transform etc. data.
 * Datafiles in 'data/': 'normal_approx_' 1 through 3 are pickle files of normal distribution fits for various experimental results. Number 1 was used to fit the models, while 2 and 3 are used for validation.
 * 'generate.py': Generate a pickle file of normal distribution fits for random graphs. Results will be of compatible format to be loaded via 'load_data' from 'helpers.py'
