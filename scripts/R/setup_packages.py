import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector

utils = rpackages.importr('utils')
utils.chooseCRANmirror(ind=1)
packnames = ['dplyr', 'deSolve', 'mvtnorm', 'pracma', 'reshape', 'ggplot2',
             'tidyr', 'ggthemes', 'latex2exp']
names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
if len(names_to_install) > 0:
    print(f" installing {len(names_to_install)} packages")
    utils.install_packages(StrVector(names_to_install))
