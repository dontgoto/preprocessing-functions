# from hdfchain import HDFChain
# import glob
# from matplotlib import pyplot as plt
import h5py as h5
import numpy as np
import pickle

"""almost all of these functions take h5Files as data parameter"""

path = '/home/fongo/sync/bachelorarbeit/daten/sim/'
colblacklist = ['Event', 'SubEventStream', 'SubEvent', 'exists', 'pdg_encoding','type','shape',
               'x', 'y', 'z', 'location', 'azimuth']
blacklist = set(['EHEATWDPulseSeriesSRT',
              'EHEFADCPulseSeriesSRT',
             'EHEOpheliaParticleBTWSRT',
             'EHEOpheliaParticleSRT',
             'EHEOpheliaParticleSRT_ImpLF',
             'I3EventHeader',
             'I3MCWeightDict'
             'FilterMask',
             'QFilterMask',
             'FilterMask_NullSplit0',
             # 'I3MCPrimary',
             'IC2013_LE_L3',
             'IC2013_LE_L3_No_RTVeto',
             'IC2013_LE_L3_Vars',
             'TimeShift',
             '__I3Index__',
             'CorsikaMoonMJD'
             'CorsikaSunMJD',
             'splittedDOMMapSRT',
             'CorsikaMoonMJD',
             'CorsikaSunMJD',
#useless
    'CascadeSplitPulses_L21',
                'CascadeSplitPulses_L22',
                'CascadeTopoSplitPulses_Doubles_IC0CleanedKeys',
                'CascadeTopoSplitPulses_Doubles_IC1CleanedKeys',
                'CascadeTopoSplits0',
                'CascadeTopoSplits1',
                'CascadeTopoSplits2',
                'CascadeTopoSplits3',
                'CorsikaWeightMap',
                'FilterMask',
                'FilterMask_NullSplit0',
                'GaisserH3a',
                'I3MCWeightDict',
                'I3SuperDST',
                'I3TriggerHierarchy',
                'IC2012_ExpLE_L3',
                'IC2012_ExpLE_L3_Vars',
                'IC2012_LE_L3',
                'IC2012_LE_L3_No_RTVeto',
                'IC2012_LE_L3_Vars',
                'IC2013_LE_L3',
                'IC2013_LE_L3_No_RTVeto',
                'IC2013_LE_L3_Vars',
                'InIceDSTPulses',
                'InIcePulses',
                'InIcePulses_STW_ClassicRT_NoiseEnginess',
                'InIcePulses_STW_NoiseEnginess',
                'InIceRawData',
                'LowEnergy_2012L3_Bool',
                'LowEnergy_2013L3_Bool',
                'OfflinePulsesHLC',
                'OfflinePulsesSLC',
                'OnlineL2_CleanedMuonPulses',
                'QFilterMask',
                'RTTWOfflinePulses_FR_WIMP',
                'SRTInIcePulses',
                'SRTInIcePulses_IC_Singles_L2CleanedKeys',
                'SRTTWOfflinePulsesDC',
                'SRTTWOfflinePulsesExpDC',
                'SplitInIceDSTPulses',
                'SplitInIcePulses',
                'TWOfflinePulsesDC',
                'TWOfflinePulsesHLC',
                'TWOfflinePulses_FR_WIMP',
                'TWRTVetoSeries',
                'TimeShift',
                'ToI_EDCEval2',
                'ToI_EDCEval3',
#
    'CascadeTopoSplitPulses_Doubles_IC1CleanedKeys',
                  'CascadeTopoSplitPulses_Doubles_IC0CleanedKeys'
                 'EHEATWDPulseSeriesSRT', 'EHEFADCPulseSeriesSRT',
       'EHEOpheliaParticleBTWSRT', 'EHEOpheliaParticleSRT',
       'EHEOpheliaParticleSRT_ImpLF', 'HuberFit', 'LargestOMKey',
       'LineFitEHE', 'SPEFit12EHE', 'SPEFitSingleEHE',
       'SRTInIcePulses_WODCCleanedKeys', 'splittedDOMMap',
       'splittedDOMMapSRT'
        'OnlineL2_CleanedMuonPulses', 'InIceRawData',
        'I3TriggerHierarchy', 'SRTInIcePulses_IC_Singles_L2CleanedKeys',
        'I3SuperDST',
#attributes with missing values
    'AtmCscdEnergyReco_L2',
    'CascadeDipoleFit_L2',
    'CascadeImprovedLineFit_L2',
     'CascadeLast_IC_Singles_L2',
     'CascadeLast_L2',
     'CascadeLast_TopoSplit_IC0',
     'CascadeLast_TopoSplit_IC1',
     'CascadeLineFitSplit1_L2',
     'CascadeLineFitSplit2_L2',
     'CascadeLineFit_L2',
     'CascadeLlhVertexFitSplit1_L2',
     'CascadeLlhVertexFitSplit2_L2',
     'CascadeLlhVertexFit_IC_Singles_L2',
     'CascadeLlhVertexFit_L2',
     'CascadeLlh_TopoSplit_IC0',
     'CascadeLlh_TopoSplit_IC1',
     'CascadeToISplit1_L2',
     'CascadeToISplit2_L2',
     'CascadeTopo_CscdSplitCount',
     'CorsikaMoonMJD',
     'CorsikaSunMJD',
     'OnlineL2_BayesianFit',
     'OnlineL2_BestFit',
     'OnlineL2_BestFitCharacteristics',
     'OnlineL2_BestFitCharacteristicsWithSAll',
     'OnlineL2_BestFitDirectHitsA',
     'OnlineL2_BestFitDirectHitsB',
     'OnlineL2_BestFitDirectHitsC',
     'OnlineL2_BestFitDirectHitsD',
     'OnlineL2_BestFit_MuE',
     'OnlineL2_BestFit_MuEx',
     'OnlineL2_BestFit_TruncatedEnergy_AllBINS_MuEres',
     'OnlineL2_BestFit_TruncatedEnergy_AllBINS_Muon',
     'OnlineL2_BestFit_TruncatedEnergy_AllBINS_Neutrino',
     'OnlineL2_BestFit_TruncatedEnergy_AllDOMS_MuEres',
     'OnlineL2_BestFit_TruncatedEnergy_AllDOMS_Muon',
     'OnlineL2_BestFit_TruncatedEnergy_AllDOMS_Neutrino',
     'OnlineL2_BestFit_TruncatedEnergy_BINS_MuEres',
     'OnlineL2_BestFit_TruncatedEnergy_BINS_Muon',
     'OnlineL2_BestFit_TruncatedEnergy_BINS_Neutrino',
     'OnlineL2_BestFit_TruncatedEnergy_DOMS_MuEres',
     'OnlineL2_BestFit_TruncatedEnergy_DOMS_Muon',
     'OnlineL2_BestFit_TruncatedEnergy_DOMS_Neutrino',
     'OnlineL2_BestFit_TruncatedEnergy_ORIG_Muon',
     'OnlineL2_BestFit_TruncatedEnergy_ORIG_Neutrino',
     'OnlineL2_BestFit_TruncatedEnergy_ORIG_dEdX',
     'OnlineL2_CramerRao_BestFit_cr_azimuth',
     'OnlineL2_CramerRao_BestFit_cr_zenith',
     'OnlineL2_CramerRao_MPEFit_cr_azimuth',
     'OnlineL2_CramerRao_MPEFit_cr_zenith',
     'OnlineL2_CramerRao_SPE2itFit_cr_azimuth',
     'OnlineL2_CramerRao_SPE2itFit_cr_zenith',
     'OnlineL2_HitMultiplicityValues',
     'OnlineL2_HitStatisticsValues',
     'OnlineL2_MPEFit',
     'OnlineL2_MPEFitDirectHitsA',
     'OnlineL2_MPEFitDirectHitsB',
     'OnlineL2_MPEFitDirectHitsC',
     'OnlineL2_MPEFitDirectHitsD',
     'OnlineL2_SPE2itFit',
     'OnlineL2_SPE2itFitDirectHitsA',
     'OnlineL2_SPE2itFitDirectHitsB',
     'OnlineL2_SPE2itFitDirectHitsC',
     'OnlineL2_SPE2itFitDirectHitsD',
     'OnlineL2_SplitGeo1_BayesianFit',
     'OnlineL2_SplitGeo1_Linefit',
     'OnlineL2_SplitGeo1_SPE2itFit',
     'OnlineL2_SplitGeo2_BayesianFit',
     'OnlineL2_SplitGeo2_Linefit',
     'OnlineL2_SplitGeo2_SPE2itFit',
     'OnlineL2_SplitTime1_BayesianFit',
     'OnlineL2_SplitTime1_Linefit',
     'OnlineL2_SplitTime1_SPE2itFit',
     'OnlineL2_SplitTime2_BayesianFit',
     'OnlineL2_SplitTime2_Linefit',
     'OnlineL2_SplitTime2_SPE2itFit',
     'PoleCascadeFilter_CscdLlh',
     'PoleCascadeFilter_LFVel',
     'PoleCascadeFilter_ToiVal',
     'PoleMuonLlhFitDirectHitsA',
     'PoleMuonLlhFitDirectHitsB',
     'PoleMuonLlhFitDirectHitsC',
     'PoleMuonLlhFitDirectHitsD',
# remove attributes that arent in every dataset
    'EHEATWDPulseSeriesSRT', 'EHEFADCPulseSeriesSRT',
       'EHEOpheliaParticleBTWSRT', 'EHEOpheliaParticleSRT',
       'EHEOpheliaParticleSRT_ImpLF', 'HuberFit', 'LargestOMKey',
       'LineFitEHE', 'SPEFit12EHE', 'SPEFitSingleEHE',
       'SRTInIcePulses_WODCCleanedKeys', 'splittedDOMMap',
       'splittedDOMMapSRT',
        'CascadeTopoSplitPulses_Doubles_IC1CleanedKeys',
        'CascadeTopoSplitPulses_Doubles_IC0CleanedKeys',
        'CascadeTopoSplitPulses_Doubles_IC0CleanedKeys',
        'CascadeTopoSplitPulses_Doubles_IC1CleanedKeys'])




def assign_arrays(data, olddata):
    """fills the arrays in a new empty hdf5 file"""
    keynames = [key for key in data.keys()]
    for key in keynames:
        print(key)
        names = data[key].dtype.names
        for col in names:
            data[key][col] = olddata[key][col]


def corsika_assign_arrays(data, olddata):
    """fills the arrays in a new empty hdf5 file"""
    keynames = [key for key in data.keys()]
    for key in keynames:
        print(key)
        names = data[key].dtype.names
        for col in names:
            data[key][col] = olddata[key][col]
        data['honda2014_spl_solmin']['value'] = 1/(7000*4.0143923434173443)


def generate_new_arrays_from_blacklist(data, fileEnding='nomissingvalues.hdf5', blacklist=[], whitelist=None):
    """generates an empty hdf5 file without the keys in blacklist and colblacklist"""
    writefile = h5.File(data.filename+fileEnding, 'w')
    try:
        keynames = [key for key in data.keys() if key not in blacklist]
        if whitelist:
            keynames = [key for key in data.keys() if key in whitelist]

        for key in keynames:
            # print('########'+key)
            types = data[key].dtype
            names = types.names
            datatypes = [(names[i],data[key][col][1].dtype )
                for i,col in enumerate(names) if col not in colblacklist]
            stacked = np.vstack((
                    [data[key][cols]
                     for cols in names if cols not in colblacklist]
                ))
            newarray = np.zeros((len(stacked[0]),), dtype=datatypes)
            writefile.create_dataset(key, data=newarray)
    finally:
        writefile.flush()
        writefile.close()


def flatten_to_one_dataset(data, fileEnding='nomissingvalues.hdf5', idKey='Run',
                           idType=np.uint32, blacklist=blacklist, colblacklist=colblacklist):
    """generates an empty hdf5 file with only a single dataset without the keys in blacklist and colblacklist
    takes a keyid so the id isn't written multiple times"""
    writefile = h5.File(data.filename+fileEnding, 'w')
    colblacklist.append(idKey)
    allTypes = [(idKey, np.uint32)]
    allNames = [idKey]
    try:
        keynames = [key for key in data.keys() if key not in blacklist]
        allStack = [data[keynames[0]][idKey]]
        for key in keynames:
            types = data[key].dtype
            names = [key+'__'+name for name in types.names if name not in colblacklist]
            allNames.append(names)
            datatypes = [(names[i],data[key][col][1].dtype)
                for i,col in enumerate(names)]
            allTypes.append(datatypes)
            stack = [data[key][cols] for cols in names]
            allStack.append(stack)

        stacked = np.vstack(allStack)
        newarray = np.zeros((len(stacked[0]),), dtype=allTypes)
        writefile.create_dataset('data', data=newarray)
    finally:
        writefile.flush()
        writefile.close()


def corsika_generate_new_arrays_from_blacklist(data, fileEnding='nomissingvalues.hdf5'):
    """generates an empty hdf5 file without the keys in blacklist and colblacklist"""
    writefile = h5.File(data.filename+fileEnding, 'w')
    try:
        keynames = [key for key in data.keys() if key not in blacklist]
        for key in keynames:
            # print('########'+key)
            types = data[key].dtype
            names = types.names
            datatypes = [(names[i],data[key][col][1].dtype )
                for i,col in enumerate(names) if col not in colblacklist]
            stacked = np.vstack((
                    [data[key][cols]
                     for cols in names if cols not in colblacklist]
                ))
            newarray = np.zeros((len(stacked[0]),), dtype=datatypes)
            writefile.create_dataset(key, data=newarray)
        datatypes = [('Run', np.uint32), ('value', np.float64)]
        newarray = np.zeros((len(stacked[0]),), dtype=datatypes)
        writefile.create_dataset('honda2014_spl_solmin', data=newarray)
    finally:
        writefile.flush()
        writefile.close()


def setnans(data, keyblacklist=''):
    keynames = [key for key in data.keys() if key not in keyblacklist]
    for key in keynames:
        print(key)
        for col in data[key].dtype.names:
            if data[key][col].dtype in [np.float32, np.float64, np.int64, np.int32, np.int8]:
                data[key][col][np.isnan(data[key][col])] =  -9000
            else:
                data[key][col][np.isnan(data[key][col])] =  11111


def remover(data, blacklist=''):
    for keys in blacklist:
        del data[keys]


def remove_blacklisted_keys(data, blacklist=''):
    mykeys = [keys for keys in data.keys()]
    myblacklist = np.intersect1d(blacklist, mykeys)
    print(myblacklist)
    remover(data, myblacklist)


def corr_to_label(data, label):
    names = np.array([])
    corrs = np.array([])
    for key in data._tables.keys():
        colnames = [col for col in data.getNode('/'+key).colnames
                if col not in colblacklist]
        for col in colnames:
            corr = (np.corrcoef(data.getNode('/'+key).col(col),
                    label))[0][1]
            corr = np.abs(corr)
            corrs = np.append(corrs, corr)
            names = np.append(names, key+'__'+col)
    return names, corrs


def constant_attributes(data):
    blacklist = np.array([])
    # keynames = [key for key in data._tables.keynames() if key not in keyblacklist]
    keynames = [key for key in data._tables.keynames()]
    for key in keynames:
        colnames = [col for col in data.getNode('/'+key).colnames if col not in colblacklist]
        for col in colnames:
            if data.getNode('/'+key).col(col).value(0) != data.getNode('/'+key).col(col).value(1):
                continue
            elif np.nanstd(data.getNode('/'+key).col(col)) == 0:
                blacklist = np.append(blacklist, key+'__'+col)
    return blacklist


def correlated_attributes(data, blacklist=np.array([]), namesblacklist = np.array([])):
    #blacklist = np.array([])
    colblacklist = ['Run', 'Event', 'SubEvent', 'SubEventStream',
                    'pdg_encoding', 'exists']
    oldnames = np.array([])
    for key in data._tables.keys():
        colnames = [col for col in data.getNode('/'+key).colnames
                    if key+'__'+col not in blacklist]
        for col in colnames:
            oldnames = np.append(oldnames, key+'__'+col)
            # calculates corr of current col with all cols in file
            corr = [np.corrcoef(data.getNode('/'+key).col(col),
                    data.getNode('/'+keys).col(cols))[0][1]
                    for keys in data._tables.keys()
                    for cols in data.getNode('/'+key).colnames
                    if keys+'__'+cols not in oldnames
                    if cols not in colblacklist]
            corr = np.abs(corr)
            names = [keys+'__'+cols for keys in data._tables.keys()
                    for cols in data.getNode('/'+key).colnames
                    if keys+'__'+cols not in oldnames
                    if cols not in colblacklist]

            for corr, name in zip(corr, names):
                # use 0.9 as cutoff, see Tim Ruhe's "Data Mining on Ice"
                if corr >0.90:
                    print(key+'__'+col+' with '+ name+' :'+corr)
                    blacklist = np.append(blacklist, name)
    return blacklist


def nan_list(data, keyBlacklist=''):
    """takes a hdfChain and returns a list of all attributes and their fraction of NaNs """
    nanList = []
    attrList = []
    keynames = [key for key in data._tables.keys() if key not in keyBlacklist]
    for key in keynames:
        colnames = [col for col in data.getNode('/'+key).colnames]
        for col in colnames:
            nans = np.isnan(data.getNode('/'+key).col(col))
            attrList.append(key+'__'+col)
            nanList.append(np.sum(nans)/nans.size)
    return attrList, nanList


def too_short_list(data, targetKey='I3MCPrimary', keyBlacklist=''):
    """takes a h5File and returns a list of all keys that are shorter than the traget key"""
    keynames = [key for key in data.keys() if key not in keyBlacklist]
    keyLengths = [data[key].size for key in keynames]
    targetLength = [data[targetKey].size for key in keynames]
    tooShortAttrs = keynames[np.not_equal(keyLengths, targetLength)]
    return tooShortAttrs


def attrs_that_arent_simple_arrays(data):
    """takes a h5File and returns a list of all keys that cant be read as a simple array
    these need to be excluded as a first step for further data cleaning"""
    keynames = [key for key in data.keys()]
    unreadableAttrs = []
    for key in keynames:
        try:
            data[key].size
            data[key]['Run'].size
        except:
            unreadableAttrs.append(key)
    return unreadableAttrs


def setid(data, i=1):
    length = len(data['MPEFit']['Run'])
    for key in data.keys():
        print(key)
        data[key]['Run'] = range(length*(i-1), length*i)


def resize_and_insert_missing_values(data, lognans=False, oldData=None):
    if oldData == None:
        oldData = data
    normal = data["honda2014_spl_solmin"]["Event"]
    normalLength = len(normal)
    keynames = [key for key in data.keys()]

    for key in keynames:
        missingvalues = np.array([])
        colnames = data[key].dtype.names
        colarray = data[key]["Event"]
        length = len(colarray)
        if length == normalLength:
            continue
        #which values are missing?
        missingvalues = find_missing_values(colarray,
                                            length, normal, normalLength)
        data[key].resize(normalLength, axis=0)
        if lognans == True:
            log_nans(data.file, missingvalues, key)

        #insert the missing values
        print(key)
        for col in colnames:
            print(col)
            print(data[key][col].dtype)
            if col in ['Event', 'SubEvent', 'SubEventStream', 'Run']:
                data[key][col] = data["honda2014_spl_solmin"][col]
            else:
                #array = construct_inserted_array( data[key][col][:-k], missingvalues)
                nanvalue = -9000
                if col not in ['exists'] and data[key][col].dtype in [np.float32, np.float64]:
                    # nanvalue = np.nan
                    nanvalue = -9000
                array = np.insert(oldData[key][col][:-(normalLength-length)], missingvalues, nanvalue)
                data[key][col] = array


def find_missing_values(colarray, length, normal,
                        normalLength, lognans=False):
    """ colarray is the Event column with missing values, length is its length, """
    append = np.append
    k = 0
    missingvalues = np.array([])

    for i in range(normalLength):
        if normal[i] != colarray[i-k]:
            missingvalues = append(missingvalues, i-k)
            k +=1
    length += k
    return missingvalues


def log_nans(fileEnding, missingvalues, key, path='/home/fongo/sync/bachelorarbeit/daten/pickle/'):
    fileEnding = str(fileEnding)
    with open(path+"_NaNs_"+fileEnding+key ,'wb') as newfile:
        pickle.dump(missingvalues, newfile)


def generate_col_with_inserted_missing_values(dataKeyCol, missingvalues, nanvalue=-9000):
    """ takes a np.array, dont use the h5array directly from disk or else
        it cant be resized automatically  by np.insert
        uint values have to be low enough to fit into their int variants"""

    dtype = dataKeyCol.dtype
    if dtype in [np.uint8]:
        dataKeyCol = dataKeyCol.astype(np.int8)
    elif dtype in [np.uint16]:
        dataKeyCol = dataKeyCol.astype(np.int16)
    elif dtype in [np.uint32]:
        dataKeyCol = dataKeyCol.astype(np.int32)
    elif dtype in [np.uint64]:
        dataKeyCol = dataKeyCol.astype(np.int64)
    dataKeyCol = np.insert(dataKeyCol, missingvalues, nanvalue)
    return dataKeyCol


def generate_new_arrays(data, keyblacklist='', namesblacklist=''):
    writefile = h5.File(data.filename+'nonans.hdf5', 'w')
    try:
        keynames = [key for key in data.keys() if key not in keyblacklist]
        for key in keynames:
            print(key)
            types = data[key].dtype
            names = types.names
            datatypes = [(names[i],data[key][col][1].dtype )
                for i,col in enumerate(names) if col not in colblacklist
                        if key+'__'+col not in namesblacklist]
            stacked = np.vstack(( [ data[key][cols]
                for cols in names if cols not in colblacklist
                        if key+'__'+cols not in namesblacklist]) )
            newarray = np.zeros((len(stacked[0]),), dtype=datatypes)
            writefile.create_dataset(key, data=newarray)
    finally:
        writefile.flush()
        writefile.close()


def flatten_h5(data, newEnding='new.hdf5'):
    """data : HDFChain object"""
    writefile = h5.File(data.filename+newEnding, 'w')
    try:
        keynames = [key for key in data.keys()]
        for key in keynames:
            colnames = [col for col in data[key].dtype.names
                        if col not in colblacklist]
            for col in colnames:
                writefile.create_dataset(key+'__'+col, data=data[key][col])
    finally:
        writefile.flush()
        writefile.close()


def generate_new_dataset(dataKey, key, writefile, namesblacklist=''):
    types = dataKey.dtype
    print(types)
    names = types.names
    # datatypes = [(names[i], types[i] )
                # for i,col in enumerate(names) if col not in blacklist
                # if key+'__'+col not in namesblacklist]
    datatypes = []
    typeNames = np.array([types[typeName] for typeName in names])
    for i, dtype in enumerate(typeNames):
        if dtype in [np.uint8]:
            newdtype = np.int8
        elif dtype in [np.uint16]:
            newdtype = np.int16
        elif dtype in [np.uint32]:
            newdtype = np.int32
        elif dtype in [np.uint64]:
            newdtype = np.int64
        else:
            newdtype = dtype
        datatypes.append((names[i], newdtype))
    stacked = np.vstack(([dataKey[cols] for cols in names
                          if key+'__'+cols not in namesblacklist]
                         ))
    newarray = np.zeros((len(stacked[0]),), dtype=datatypes)
    writefile.create_dataset(key, data=newarray)


def generate_label(signal, background, lenKey='MPEFit'):
    false = np.full(len(background.getNode('/'+lenKey).col('energy')),
                    False, dtype=np.bool)
    true = np.full(len(signal.getNode('/'+lenKey).col('energy')),
                   True, dtype=np.bool)
    label = np.concatenate(true, false, axis=0)
    return label


