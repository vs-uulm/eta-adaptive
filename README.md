# Modelling of k-growing graphs and privacy for eta-AD

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/vs-uulm/eta-adaptive/main?filepath=Distribution%20Model.ipynb)

## Models

The models we fit in this example are built in the following way:

![
\begin{align*}
M_1(n,k) &= \alpha \ln(\beta n) + \frac{\gamma}{e^{\delta k}} + \epsilon, \\
M_2(n,k) &= \frac{\alpha \ln(\beta n)}{e^{\gamma k}} + \delta \ln(\eta n) + \frac{\zeta}{e^{\gamma k}} + \epsilon, \\
M_3(n,k) &= \frac{\alpha \ln(\beta n)}{e^{\gamma k}} + \epsilon, \\
M_4(n,k) &= \frac{\alpha \ln(\beta n)}{e^{\gamma k}} - \frac{\alpha \delta k}{e^{\gamma k}} + \frac{\zeta}{e^{\eta k}} + \frac{\theta \ln(\iota n)}{(\kappa n)^{\lambda n}} +\nu \ln(\xi n).
\end{align*}](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%0A%5Cbegin%7Balign%2A%7D%0AM_1%28n%2Ck%29+%26%3D+%5Calpha+%5Cln%28%5Cbeta+n%29+%2B+%5Cfrac%7B%5Cgamma%7D%7Be%5E%7B%5Cdelta+k%7D%7D+%2B+%5Cepsilon%2C+%5C%5C%0AM_2%28n%2Ck%29+%26%3D+%5Cfrac%7B%5Calpha+%5Cln%28%5Cbeta+n%29%7D%7Be%5E%7B%5Cgamma+k%7D%7D+%2B+%5Cdelta+%5Cln%28%5Ceta+n%29+%2B+%5Cfrac%7B%5Czeta%7D%7Be%5E%7B%5Cgamma+k%7D%7D+%2B+%5Cepsilon%2C+%5C%5C%0AM_3%28n%2Ck%29+%26%3D+%5Cfrac%7B%5Calpha+%5Cln%28%5Cbeta+n%29%7D%7Be%5E%7B%5Cgamma+k%7D%7D+%2B+%5Cepsilon%2C+%5C%5C%0AM_4%28n%2Ck%29+%26%3D+%5Cfrac%7B%5Calpha+%5Cln%28%5Cbeta+n%29%7D%7Be%5E%7B%5Cgamma+k%7D%7D+-+%5Cfrac%7B%5Calpha+%5Cdelta+k%7D%7Be%5E%7B%5Cgamma+k%7D%7D+%2B+%5Cfrac%7B%5Czeta%7D%7Be%5E%7B%5Ceta+k%7D%7D+%2B+%5Cfrac%7B%5Ctheta+%5Cln%28%5Ciota+n%29%7D%7B%28%5Ckappa+n%29%5E%7B%5Clambda+n%7D%7D+%2B%5Cnu+%5Cln%28%5Cxi+n%29.%0A%5Cend%7Balign%2A%7D)

## Structure

### Distribution Model Notebook
 
Followalong with the Distribution Model section of the paper. (You can run the interactive versin on Binder here: https://mybinder.org/v2/gh/vs-uulm/eta-adaptive/main?filepath=Distribution%20Model.ipynb )

### Prepared Datafiles

There are three files provided in 'data/': 'normal_approx_' 1 through 3.
These files are pickle files of normal distribution fits for various experimental results.
Number 1 was used to fit the models, while 2 and 3 are used for validation.

These files are loaded via 'load_data' from 'helpers.py'.
Their contents are also available convieniently via 'data.py' as prepared objects base, val1 and val2.
 
### Python Files
 
 * 'models.py': implementation of the fittable and fitted models we use for calculating mu and sigma.
 * 'data.py': Provides the 3 datasets as canonical objects base, val1 and val2.
 * 'helpers.py': Provide some internal helper methods to process, load, transform etc. data.
 * 'generate.py': Generate a pickle file of normal distribution fits for random graphs. Results will be of compatible format to be loaded via 'load_data' from 'helpers.py'
