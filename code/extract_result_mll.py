import os.path
import sys

# generate root dir of a batch of results
# e.g. Seq/DNA
t = sys.argv[1]
c = '../results/batch/' + str(t) + '/'


def get_model_params_result(root_dir):
    for _, result_dirs, _ in os.walk(root_dir):  # (root, dirs, files)
        with open('params_batch_file', 'w') as pf, open('result_batch_file', 'w') as rf:
            for result_dir in result_dirs:
                if os.path.exists(root_dir + result_dir + '/Opt_params.txt'):
                    # recognize result target
                    # final_results.txt, Opt_params.txt, <model_name>.model
                    for b in os.walk(root_dir + result_dir):
                        assert len(b[0]) == 0 and len(b[1]) == 0

                        for f in b[2]:
                            if f.rfind('.model') != -1:
                                model_name = f.split('.'[0])
                                write_file(pf, model_name)
                                write_file(rf, model_name)
                                break
                        for f in b[2]:
                            if f == 'Opt_params.txt':
                                write_file(pf, root_dir + result_dir, f)
                            elif f == 'final_results.txt':
                                write_file(rf, root_dir + result_dir, f)
                            else:
                                pass


def write_file(io, file_dir, file_name):
    with open(file_dir + '/' + file_name) as r:
        io.writelines(r.readlines())
    io.writelines('\n')
        # if os.path.exists(c + dirname + '/param.txt'):
        #     if os.path.exists(c + dirname + '/cv_model.model'):
        #         with open(c + dirname + '/final_results.txt') as f, open(c + dirname + '/cv_model.model') as m,\
        #                 open(c + dirname + '/param.txt') as p, open(c + 'results.txt', 'a') as w:
        #             lines = f.readlines()
        #             m_lines = m.readlines()
        #             p_lines = p.readlines()
        #             for line in lines[1:6]:
        #                 dirname += ' ' + line.strip().split(' ')[-1]
        #             if t.split('/')[1] == 'svm':
        #                 param = m_lines[0].strip().split(',')
        #                 dirname += ' ' + param[1] + ' ' + param[3]
        #             else:
        #                 param = m_lines[-1].strip().split(',')
        #                 dirname += ' ' + param[0]
        #             dirname += ' ' + p_lines[0]
        #             w.writelines(dirname + '\n')
        #     else:
        #         with open(c + dirname + '/cv_eval_results.txt') as f, open(c + dirname + '/param.txt') as p, \
        #                 open(c + 'results.txt', 'a') as w:
        #             lines = f.readlines()
        #             p_lines = p.readlines()
        #             for line in lines[1:6]:
        #                 dirname += ' ' + line.strip().split(' ')[-1]
        #             if t.split('/')[1] == 'rf':
        #                 dirname += ' ' + 'None'
        #             dirname += ' ' + p_lines[0]
        #             w.writelines(dirname + '\n')
        # else:
        #     with open(c + dirname + '/cv_eval_results.txt') as f, open(c + dirname + '/cv_model.model') as m,\
        #             open(c + 'results.txt', 'a') as w:
        #         lines = f.readlines()
        #         m_lines = m.readlines()
        #         for line in lines[1:6]:
        #             dirname += ' ' + line.strip().split(' ')[-1]
        #         if t.split('/')[1] == 'svm':
        #             param = m_lines[0].strip().split(',')
        #             dirname += ' ' + param[1] + ' ' + param[3]
        #         else:
        #             param = m_lines[-1].strip().split(',')
        #             dirname += ' ' + param[0]
        #         dirname += ' ' + 'None'
        #         w.writelines(dirname + '\n')


def write_file(io, tag):
    io.writelines(''.center(100, '+') + '\n')
    io.writelines((' ' + tag + ' ').center(100, '+') + '\n')


get_model_params_result(c)
