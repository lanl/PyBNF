import subprocess
import re


def gen_config(filename, outmodel, outact):
    next_index = dict()
    outm = open(outmodel, 'w')
    outa = open(outact, 'w')
    with open(filename) as f:
        for line in f:
            if line[0:4]=='with':
                line = line.strip('\n')
                muts = line.split(' ')
                muts = [re.sub(r'(?<=[+\-*/])=', '', x) for x in muts if x not in ('with',)]
                    
            elif line[0]=='$':
                phen = line[1]
                if phen not in next_index:
                    next_index[phen] = 0
                model_line = 'mutant = %s %s%s ' + ' '.join(muts) + ': examples/Tyson/constraints/r%s%s.con\n'
                base = 'yeast'
                outm.write(model_line % (base, phen, next_index[phen], phen, next_index[phen],))
                action_line = 'time_course = model:%s%s, time:1000, step:1, suffix:r\n' % (phen, next_index[phen])
                outa.write(action_line)
                next_index[phen] += 1
                
                
    outm.close()
    outa.close()
    for phen in next_index:
        for i in range(next_index[phen]):
            subprocess.run(['cp', 'constraints/%s.con'%phen, 'constraints/r%s%s.con' % (phen, i)])
                
if __name__ == '__main__':
    gen_config('tyson13-skeleton.con','mutmodels.txt','mutactions.txt')
