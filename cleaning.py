from preprocessing_functions import preprocessing_functions as pf
import h5py as h5
import glob
# from hdfchain import HDFChain

blacklist = pf.blacklist
colblacklist = pf.colblacklist
dataPath = '~/sync/bachelorarbeit/daten/sim/'
#hardcoded nomissingvalues can be replaced by variable

#different types of data need to be in different directories

def prepare_corsika_from_blacklist(path, oldEnding='', newEnding='nomissing'):
    """Takes the path of the corsika files (there should be only corsika files in a single directory)
    and removes unuseable and too short keys and writes new hdf5 files"""
    oldCorsika = [h5.File(file, 'r') for file in sorted(glob.glob(path+'*'+oldEnding))]
    uselessKeys = []
    for corsika in oldCorsika:
        uselessKeys.extend(pf.attrs_that_arent_simple_arrays(corsika))
    tooShortKeys = []
    for corsika in oldCorsika:
        tooShortKeys.extend(pf.too_short_list(corsika, keyBlacklist=uselessKeys))
    for corsika in oldCorsika:
        pf.corsika_generate_new_arrays(corsika, fileEnding=newEnding)
    newCorsika = [h5.File(file, 'r') for file in sorted(glob.glob(path+'*'+newEnding+'*'))]
    for new, old in zip(newCorsika, oldCorsika):
        pf.corsika_assign_arrays_and_weights(new, old)
        pf.setid(new)
        new.flush()
        new.close()
        old.close()
    return uselessKeys, tooShortKeys


def prepare_nu_from_blacklist(path, oldEnding, newEnding):
    """Takes the path of the nu files (there should be only nu files in a single directory)
    and removes unuseable and too short keys and writes new hdf5 files
    Returns the useless and too short keys"""
    oldNU = [h5.File(file, 'r') for file in sorted(glob.glob(path+'*'+oldEnding))]
    uselessKeys = []
    for nu in oldNU:
        uselessKeys.extend(pf.attrs_that_arent_simple_arrays(nu))
    tooShortKeys = []
    for nu in oldNU:
        tooShortKeys.extend(pf.too_short_list(nu, keyBlacklist=uselessKeys))
    for nu in oldNU:
        pf.generate_new_arrays(nu, fileEnding=newEnding)
    newNU = [h5.File(file, 'r') for file in sorted(glob.glob(path+'*'+newEnding+'*'))]
    for new, old in zip(newNU, oldNU):
        pf.assign_arrays(new, old)
        pf.setid(new)
        new.flush()
        new.close()
        old.close()
    return uselessKeys, tooShortKeys
