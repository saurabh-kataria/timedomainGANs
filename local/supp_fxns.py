import os
import sys
import numpy as np
from pathlib import Path
import random
import scipy
import scipy.stats
import re
from warnings import warn
import subprocess
import shutil
import datetime
import yaml
import pickle

import torch


def read_file_as_dict(filepath, idx1=0, idx2=1, delimiter=' '):
    d = {}
    with open(filepath, 'rt') as f:
        for line in f:
            d[line.split(delimiter)[idx1]] = line.split(delimiter)[idx2].strip()
    return d

def utt2spk_from_segments(filepath):
    assert filepath.split('/')[-1] == 'segments.csv'
    dir_of_interest = str(Path(filepath).parent)
    assert Path(filepath).exists()
    utt2spk = os.path.join(dir_of_interest, 'utt2spk')
    unsorted_dict = {}
    with open(filepath, 'rt') as f2r:
        idx = 0
        for line in f2r:
            if idx != 0:
                segment_id, speaker_id, *_ = line.split(',')
                unsorted_dict[segment_id] = speaker_id
            idx += 1
    sorted_dict = dict(sorted(unsorted_dict.items()))   # not sure how useful is this 
    with open(utt2spk, 'wt') as f2w:
        for key in list(sorted_dict):
            l2w = f'{key} {sorted_dict[key]}\n'
            f2w.write(l2w)
    print(f'written: {utt2spk}')


def dict_reverser(my_dict):
    ''' taken from https://therenegadecoder.com/code/how-to-invert-a-dictionary-in-python/
    '''
    my_inverted_dict = {}
    for key, value in my_dict.items():
        my_inverted_dict.setdefault(value, list()).append(key)
    return my_inverted_dict


def nthlineoffile(filepath, n):
    with open(filepath, "r") as fp:
       lines = fp.readlines()
    return lines[n-1]


def numlinesinfile(fname):
    def _make_gen(reader):
        while True:
            b = reader(2 ** 16)
            if not b: break
            yield b

    with open(fname, "rb") as f:
        count = sum(buf.count(b"\n") for buf in _make_gen(f.raw.read))
    return count


def get_indices(my_list, element):
    indices = [i for i, x in enumerate(my_list) if x == element]
    return indices


def mergeDicts(dict1, dict2):
    res = {**dict1, **dict2}
    return res


def dict_rename(d, startIDX=1, remove=True, token='module.'):
    keys = list(d)
    assert len(keys) > 0
    for key in keys:
        if remove:
            d['.'.join(key.split('.')[startIDX:])] = d.pop(key)
        else:
            d[token + key] = d.pop(key)
    return d


def get_objsizeonRAM(obj, seen=None, ver=2):
    """
    Recursively finds size of objects and answer is reported in Bytes
    https://goshippo.com/blog/measure-real-size-any-python-object/
    https://stackoverflow.com/questions/449560/how-do-i-determine-the-size-of-an-object-in-python/450034
    """
    if ver == 1:
        size = sys.getsizeof(obj)
        if seen is None:
            seen = set()
        obj_id = id(obj)
        if obj_id in seen:
            return 0
        # Important mark as seen *before* entering recursion to gracefully handle
        # self-referential objects
        seen.add(obj_id)
        if isinstance(obj, dict):
            size += sum([get_size(v, seen) for v in obj.values()])
            size += sum([get_size(k, seen) for k in obj.keys()])
        elif hasattr(obj, '__dict__'):
            size += get_size(obj.__dict__, seen)
        elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
            size += sum([get_size(i, seen) for i in obj])
        return size
    elif ver == 2:
        return len(pickle.dumps(obj))/1e6   # MB
    else:
        raise NotImplementedError


def read_yaml(yaml_file):
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)
    return data


def write_yaml(d, filepath, overwrite=True):
    if not overwrite:
        backup_if_exists(filepath)
    with open(filepath, 'w') as f:
        yaml.dump(d, f, default_flow_style=False)
    print(f'written to disk: {filepath}')


def getcurrtimestamp():
    return "-".join(datetime.datetime.now().strftime("%Y-%m-%d %a %H:%M:%S").split())


def get_latestFileInDir(pattern):
    return subprocess.run(f'ls -1tv {pattern}', shell=True, check=True, stdout=subprocess.PIPE).stdout.decode("UTF-8").split()[-1]


def subtractlists(lst1, lst2):
    '''subtract two lists; lst1-lst2
    can return empty lists too
    '''
    return list(set(lst1)-set(lst2))


def copyfilewithreplace(src_path, dest_path, makesymlink=False, verbose=False):
    """
        copy a file w/ replace NOT BACKUP
        bad code, change line3
    """
#    if not pathexists(src_path):
#        raise Exception(f"copyfilewithreplace() failed: {src_path} does not exist, cannot copy")
    if not os.path.isfile(src_path):
        raise Exception(f"copyfilewithreplace() failed: file {src_path} does not exist, cannot copy")
    if os.path.isfile(dest_path):
        os.unlink(dest_path)    # unlink/delete is required because both os.symlink() and shutil.copy2() requires file to not be present
        if makesymlink:
            os.symlink(src_path, dest_path)
        else:
            shutil.copy2(src_path, dest_path)
    else:
        if makesymlink:
            os.symlink(src_path, dest_path)
        else:
            shutil.copy2(src_path, dest_path)
    if verbose:
        print(f"copied '{src_path}' -> '{dest_path}'")


def print_shell_cmd_output(cmd):
    assert isinstance(cmd, str)
    print(subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode("UTF-8"))   # removed check=True


def listpatternnatsorted(filepattern):
    """
        get filenames with matching pattern in form of list, naturally sorted, uses bash ls
    """
    try:
        p = subprocess.run(f"ls -v {filepattern}", shell=True, check=True, stdout=subprocess.PIPE)
        return p.stdout.decode("UTF-8").split()
    except:
        return []


def rename_verbose(src, dest, verbose=True):
    """
        renames a file or directory using os.rename() but more verbose
        try to rename else warn
    """
    try:
        os.rename(src, dest)
        print(f"renamed '{src}' -> '{dest}'")
    except:
        warn(f"cannot rename '{src}' -> '{dest}'")


def ensure_list(obj):
    'if input is list, pass; else enclose the object in list form'
    if type(obj) is not list:
        return [obj]   # if type is not list, make it a list
    else:
        return obj     # already a list so no worries


def check_type(obj, valid_types, return_=False):
    """
        checks if type of input is among valid_types
        works with None also
    """
    valid_types = ensure_list(valid_types)
#    if obj in [None, "None"] and None in valid_types:
#        obj = None
    if type(obj) not in valid_types:
        raise TypeError(f"{type(obj)} not in {valid_types}")
    if return_ is True: return obj


def backup_if_exists(filenames):
    """
        utility for making backup of file(s)
        silent execution of this fxn means it did nothing
        this may give output like "ls: cannot access filename.backup*': No such file or directory", ignore it
    """
    check_type(filenames, [str, list])
    filenames = ensure_list(filenames)
    for filename in filenames:
        if Path(filename + ".backup").is_file():  # this is done to cover up for the last imperfect version of this function
            rename_verbose(filename + ".backup", filename + ".backup1")
        if Path(filename).is_file():
            backups = listpatternnatsorted(filename + ".backup*")   # check for backups
            if len(backups) == 0:   # if no backup exists, make the first backup
                idx = 1
            else:   # look for latest backup
                idx = int(re.findall(r'\d+', backups[-1].split(".")[-1])[0]) + 1
                if idx < 0 or type(idx) != int:
                    raise Exception("error is trying to get latest backup ID")
            rename_verbose(filename, filename + f".backup{idx}")


def chain_generators(*iterables):
    '''
    https://stackoverflow.com/questions/3211041/how-to-join-two-generators-in-python
    '''
    for iterable in iterables:
        yield from iterable


def file_exists_and_notempty(path):
    return os.path.isfile(path) and os.stat(path).st_size != 0


def dir_exists_and_notempty(path):
    return os.path.isdir(path) and any(os.scandir(path))


def format2str(x, commas=True):
    'convert to (pretty) string'
    if commas:
        return f"{x:,}"
    else:
        return str(x)


def get_stats(data):
    """
        get common statistics for all elements in "data" array
        currently, clubs data from all dimensions
    """
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    data = np.asarray(data).flatten()
    numel = format2str(len(data))   # number of elements
    MIN = np.min(data)
    MAX = np.max(data)
    MEAN = np.mean(data)
    MEDIAN = np.median(data)
    MODE = scipy.stats.mode(data)
    print(f"numel={numel}\nMIN={MIN}\nMAX={MAX}\nMEAN={MEAN}\nMEDIAN={MEDIAN}\nMODE={MODE}")


def getcol(filename, n=1, linesplitdelimiter="", suffix="", prefix="", asarray=False, asfloat=False):
    """
        extension of get_first_col
        get the n-th column of filename in form of list
        n = 1,2,...
        by default, split line with space delimiter
        this can apply suffix or prefix also
        bug corrected: splitting a line w/ space delimiter keeps newline character for the last split element,
        which is usually undesirable
    """
    if type(n) is tuple:
        outlist = []
        for ii in range(len(n)):
            # recursive call
            outlist.append(getcol(filename, n=n[ii], linesplitdelimiter=linesplitdelimiter, suffix=suffix, prefix=prefix, asarray=asarray, asfloat=asfloat))
        return outlist
    else:
        if not os.path.isfile(filename):
            raise Exception(f"getcol() failed to get column {n} from {filename}: {filename} does not exists")
        out_list = []
        try:
            with open(filename, "rt") as fid:
                for line in fid:
                    if linesplitdelimiter:
                        word = prefix + line.split(linesplitdelimiter)[n - 1] + suffix
                    else:
                        word = prefix + line.split()[n - 1] + suffix
                    out_list.append(word)
        except:
            raise Exception(f"getcol() failed to get column {n} from {filename}")
        if asfloat == True:
            return [float(_) for _ in out_list]
        elif asarray == True:
            return np.asarray([float(_) for _ in out_list])
        else:
            return out_list


def split_file(file,out1,out2,percentage=0.85,isShuffle=True,seed=0,mode=1):
    """Splits a file in 2 given the `percentage` to go in the large file.
    https://stackoverflow.com/a/48347284/3342981
    """
    if mode == 1:
        random.seed(seed)
        with open(file, 'r',encoding="utf-8") as fin, \
             open(out1, 'w') as foutBig, \
             open(out2, 'w') as foutSmall:

            nLines = sum(1 for line in fin) # if didn't count you could only approximate the percentage
            fin.seek(0)
            nTrain = int(nLines*percentage) 
            nValid = nLines - nTrain

            i = 0
            for line in fin:
                r = random.random() if isShuffle else 0 # so that always evaluated to true when not isShuffle
                if (i < nTrain and r < percentage) or (nLines - i > nValid):
                    foutBig.write(line)
                    i += 1
                else:
                    foutSmall.write(line)
    elif mode == 2:
        random.seed(seed)
        with open(file, 'r',encoding="utf-8") as fin, \
             open(out1, 'w') as foutBig, \
             open(out2, 'w') as foutSmall:

            for line in fin:
                r = random.random() 
                if r < percentage:
                    foutBig.write(line)
                else:
                    foutSmall.write(line)
    else:
        raise NotImplementedError(f'{mode=}')


def set_randomseed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
#    torch.backends.cudnn.deterministic = True   # false by default
    print(f"executed set_randomseed(seed={seed})")


def format2str(x, commas=True):
    'convert to (pretty) string'
    if commas:
        return f"{x:,}"
    else:
        return str(x)


def ensure_list(obj):
    'if input is list, pass; else enclose the object in list form'
    if type(obj) is not list:
        return [obj]   # if type is not list, make it a list
    else:
        return obj     # already a list so no worries


def check_type(obj, valid_types, return_=False):
    """
        checks if type of input is among valid_types
        works with None also
    """
    valid_types = ensure_list(valid_types)
#    if obj in [None, "None"] and None in valid_types:
#        obj = None
    if type(obj) not in valid_types:
        raise TypeError(f"{type(obj)} not in {valid_types}")
    if return_ is True: return obj


def mkdirsafe(dirnames, v=True):
    """
        safely makes specified directories
    """
    check_type(dirnames, [str, list])
    dirnames = ensure_list(dirnames)
    for dir_curr in dirnames:
        direxists = os.path.isdir(dir_curr)
        Path(dir_curr).mkdir(parents=True, exist_ok=True)
        if v and not direxists:
            print(f"made directory: {dir_curr}")

mkdir_safe = mkdirsafe
