import os.path
import sys

# generate root dir of a batch of results
# e.g. Seq/DNA
t = sys.argv[1]
c = '../results/batch/' + str(t) + '/'


def get_model_params_result(result_path):
    with open('batch_params.txt', 'w') as pf, open('batch_results.txt', 'w') as rf:
        for target_root, dirs, files in os.walk(result_path):  # (root, dirs, files)
            if len(files) == 0:
                continue
            assert len(dirs) == 0
            print('extracting result target ' + target_root + '...')
            if os.path.exists(target_root + '/Opt_params.txt'):
                # recognize result target
                # final_results.txt, Opt_params.txt, <model_name>.model
                for f in files:
                    if f.rfind('.model') != -1:
                        model_name = f.split('.')[0]
                        write_header(pf, model_name, target_root)
                        write_header(rf, model_name, target_root)
                        break

                for f in files:
                    if f == 'Opt_params.txt':
                        write_file(pf, target_root, f)
                    elif f == 'final_results.txt':
                        write_file(rf, target_root, f)
                    else:
                        pass

        print('\nderive batch_results here:')
        os.system('pwd')
        os.system('ls batch*.txt')
        print('done.')


def write_file(io, target_path, file_name):
    with open(target_path + '/' + file_name) as r:
        r.readline()
        io.writelines(r.readlines())
    io.writelines('\n')


def write_header(io, tag, root):
    io.writelines(''.center(100, '+') + '\n')
    io.writelines(root[len('../results/batch/'):].center(50, ' ').center(100, '+') + '\n')
    io.writelines(tag.center(20, ' ').center(100, '+') + '\n')


get_model_params_result(c)
