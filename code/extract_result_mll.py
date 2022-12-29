import os.path
import sys
from pathlib import Path

p1 = sys.argv[1]  # the directory(relative to results/) to collect. e.g. batch/Seq/DNA
p2 = sys.argv[2]  # custom method name of output files. e.g. br_rf, br_rf_v0.1
p3 = sys.argv[3]  # True to collect independent test results. False to collect cross_validation results.
result_dir = '../results'  # directory of results

# generate root dir of a batch of results
p1_ = result_dir + '/' + str(p1) + '/'


def get_model_params_result(target_dir_to_collect, method_name, ind=False):
    ind = True if ind.startswith('t') or ind.startswith('T') else False
    print('set extracting mode to {} and start working...'.format('independent' if ind else 'cross_validation'))

    # generate output directory: ${target_dir_to_collect}/extracted
    extracted_dir = target_dir_to_collect + '/extracted'
    ex_p = extracted_dir + '/' + method_name + '{}parameter.txt'.format('_ind_' if ind else '_')
    ex_e = extracted_dir + '/' + method_name + '{}evaluation.txt'.format('_ind_' if ind else '_')
    if not Path(extracted_dir).exists():
        Path(extracted_dir).mkdir(parents=True)

    with open(ex_p, 'w') as pf, open(ex_e, 'w') as rf:
        for target_root, dirs, files in os.walk(target_dir_to_collect):  # (root, dirs, files)
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
                    elif ind and f == 'ind_final_results.txt':
                        write_file(rf, target_root, f)
                    elif not ind and f == 'final_results.txt':
                        write_file(rf, target_root, f)
                    else:
                        pass

        print('you can find extracted results in directory {}'.format(extracted_dir))
        print('done.')
        print()


def write_file(io, target_path, file_name):
    with open(target_path + '/' + file_name) as r:
        r.readline()
        io.writelines(r.readlines())
    io.writelines('\n')


def write_header(io, tag, root):
    """
     e.g. tag=br_rf_tree_300, root=../results/batch/Seq/DNA/WE/fastText/Kmer
     ++++++ batch/Seq/DNA/WE/fastText/Kmer ++++++
     ++++++++++ br_rf_tree_300 ++++++++++++
    :param io: file io
    :param tag: model_name
    :param root: directory where the model and result lie in
    """

    prefix_len = len('../results/')  # 消除和model无关的目录信息
    dir_len = len(root) - prefix_len
    io.writelines(root[prefix_len:].center(dir_len+2, ' ').center(100, '+') + '\n')
    io.writelines(tag.center(len(tag)+2, ' ').center(100, '+') + '\n')


get_model_params_result(p1_, p2, p3)
