import os
import tarfile


def extract_tar(src, dest_dir):
    '''
    Function to open a tar file.
    '''
    f = tarfile.open(src)
    f.extractall(dest_dir)
    f.close()

    os.remove(src)


def list_dirs(dir):
    '''
    List out all the directories, files and directories of directories
    in the directory specified.

    '''

    out = []

    def inner_loop(dir_):

        ds = {}
        for d in os.listdir(dir_):
            try:
                ds[os.path.join(dir_, d)] = inner_loop(os.path.join(dir_, d))

            except NotADirectoryError:
                out.append(os.path.join(dir_, d))
                ds[os.path.join(dir_, d)] = None

            except Exception as e:
                print(f'[!] Error occured : {e}')

        return ds

    inner_loop(dir)

    return out


def extract_tar(src, dest_dir):
    '''
    Function to open a tar file.
    '''
    f = tarfile.open(src)
    f.extractall(dest_dir)
    f.close()

    os.remove(src)


def list_dirs(dir):
    '''
    List out all the directories, files and directories of directories
    in the directory specified.

    '''

    out = []

    def inner_loop(dir_):

        ds = {}
        for d in os.listdir(dir_):
            try:
                ds[os.path.join(dir_, d)] = inner_loop(os.path.join(dir_, d))

            except NotADirectoryError:
                out.append(os.path.join(dir_, d))
                ds[os.path.join(dir_, d)] = None

            except Exception as e:
                print(f'[!] Error occured : {e}')

        return ds

    inner_loop(dir)

    return out