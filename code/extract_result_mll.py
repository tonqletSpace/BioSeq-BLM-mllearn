import os.path
import sys
from pathlib import Path

p1 = sys.argv[1]  # e.g. Seq/DNA
p2 = sys.argv[2]  # br_rf
result_dir = '../results'

# generate root dir of a batch of results
p11 = result_dir + '/batch/' + str(p1) + '/'


def get_model_params_result(result_path, method_name):
    extracted_dir = result_dir + '/extracted'
    ex_p = extracted_dir + '/' + method_name + '_parameter.txt'
    ex_m = extracted_dir + '/' + method_name + '_evaluation.txt'

    if not Path(extracted_dir).exists():
        Path(extracted_dir).mkdir(parents=True)

    with open(ex_p, 'w') as pf, open(ex_m, 'w') as rf:
        for target_root, dirs, files in os.walk(result_path):  # (root, dirs, files)
            if len(files) == 0:
                continue
            if os.path.exists(target_root + '/Opt_params.txt'):
                print('extracting result target ' + target_root + '...')
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

        print('done.')


def write_file(io, target_path, file_name):
    with open(target_path + '/' + file_name) as r:
        r.readline()
        io.writelines(r.readlines())
    io.writelines('\n')


def write_header(io, tag, root):
    # io.writelines(''.center(100, '+') + '\n')
    prefix_len=len('../results/batch/')
    dlen=len(root) - prefix_len
    io.writelines(root[prefix_len:].center(dlen+2, ' ').center(100, '+') + '\n')
    io.writelines(tag.center(len(tag)+2, ' ').center(100, '+') + '\n')


get_model_params_result(p11, p2)
