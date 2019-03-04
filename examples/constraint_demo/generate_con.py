import numpy as np

def generate(nsamples):
    f = open('par.prop','w')
    for x in np.linspace(0., 10., nsamples):
        sign = '<' if 0.5*x**2 - 3*x + 5 < x + 1.5 else '>'
        f.write('par%sline at x=%s\n' % (sign, x))
    f.close()
    
if __name__=='__main__':
    generate(2**7+1)
