import json
import numpy as np
import os
import pickle
import h5py


## JSON
class NumpyAwareEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyAwareEncoder, self).default(obj)

## H5 handling

def save_dict_to_hdf5(dic, filename):
    """
    ....
    """
    with h5py.File(filename, 'w') as h5file:
        recursively_save_dict_contents_to_group(h5file, '/', dic)

def recursively_save_dict_contents_to_group(h5file, path, dic):
    """
    ....
    """
    for key, item in dic.items():
        if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes)):
            h5file[path + key] = item
        elif isinstance(item, dict):
            recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
        else:
            raise ValueError('Cannot save %s type'%type(item))

def load_dict_from_hdf5(filename):
    """
    ....
    """
    with h5py.File(filename, 'r') as h5file:
        return recursively_load_dict_contents_from_group(h5file, '/')

def recursively_load_dict_contents_from_group(h5file, path):
    """
    ....
    """
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item.value
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return ans

## GENERAL

def save_file(obj, path, file_name, file_type='pickle'):

    if len(file_name) > 240:
        print('Warning: Truncating file name to avoid errors')
        file_name = file_name[:240]

    # Put the file type at the end if needed
    if not(file_name.endswith('.' + file_type)):
        file_name = file_name + '.' + file_type

    # Make sure path is provided otherwise do not save
    if path == '':
        print(('WARNING: Saving \'{0}\' cancelled, no path given.'.format(file_name)))
        return False

    if file_type == 'json':
        assert os.path.exists(path), 'Directory {} does not exist'.format(path)
        f = open(os.path.join(path, file_name), 'w')
        json.dump(obj, f, indent=4, sort_keys=True, cls=NumpyAwareEncoder)
        f.close()
    elif file_type == 'pickle':
        f = open(os.path.join(path, file_name), 'wb')
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()
    elif file_type == 'h5':
        assert isinstance(obj,dict), 'h5 saving enabled only for dict'
        save_dict_to_hdf5(obj,os.path.join(path, file_name))

    return True

def load_file(path,file_name,file_type=None):

    if file_type is None:
        if file_name.endswith('.json'):
            file_type = 'json'
        elif file_name.endswith('.h5'):
            file_type = 'h5'
        elif file_name.endswith('.pickle'):
            file_type = 'pickle'
        else:
            raise ValueError('Unkwon data type for {}'.format(file_name))
    else:

        # Put the file type at the end if needed
        if not (file_name.endswith('.' + file_type)):
            file_name = file_name + '.' + file_type

    if path == '':
        print(('Saving \'{0}\' cancelled, no path given.'.format(file_name)))
        return False

    if file_type == 'json':
        f = open(os.path.join(path, file_name), 'r')
        obj = json.load(f)
    elif file_type == 'pickle':
        f = open(os.path.join(path, file_name), 'rb')
        obj = pickle.load(f)
    elif file_type == 'h5':
        obj = load_dict_from_hdf5(os.path.join(path, file_name))
    else:
        raise ValueError('Not understanding file type: type requested {}, file name {}'.format(file_type,file_name))

    return obj

def compute_or_load(function,path,file_name,file_type='pickle',verbose=True):

    file_path = os.path.join(path, file_name + '.' + file_type)

    if os.path.exists(file_path):
        if verbose: print('File {} loaded'.format(file_name))
        return load_file(path,file_name,file_type= file_type)

    else:
        obj = function()
        save_file(obj, path, file_name, file_type=file_type)
        if verbose: print('File {} saved'.format(file_name))

    return obj

