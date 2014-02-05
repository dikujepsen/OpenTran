import os, os.path
import subprocess
import shutil
import sys
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--make", help="run make clean && make on all files",
                    action="store_true")
parser.add_argument("-p", "--printresult", help="Compiles the code with printing of the result enabled",
                    action="store_true")
parser.add_argument("-b", "--backcopy", help="Compiles the code so that the result is copied back from the GPU",
                    action="store_true")
parser.add_argument("-t", "--tag", help="tag this benchmark with a string")
parser.add_argument("-r", "--run", help="run all binary files for the given device", choices=['CPU', 'GPU'])
parser.add_argument("-i", "--input", help="input choice for the binarys", choices=['basic', 'K20Max'])

parser.add_argument("-n", "--numberofiterations", help="the number of iterations we benchmark a given binary.", type=int, default=1)

args = parser.parse_args()
   
benchmark = ['MatMul',
             'Jacobi',
             'KNearest',
             'NBody'
             ]

cmdlineoptsbasic = {'MatMul'  		    : '-n 1024' ,
            		'Jacobi'  		    : '-n 1024' ,
               		'KNearest'		    : '-n 1024 -k 16' ,
               		'NBody'   		    : '-n 1024' ,
               		'Laplace' 		    : '-n 256 -k 3' ,
               		'GaussianDerivates' : '-n 256 -m 256 -k 3'}

cmdlineoptsK20Max = {'MatMul'  		    : '-n 12544' ,
            		'Jacobi'  		    : '-n 16384' ,
               		'KNearest'		    : '-n 16384 -k 16' ,
               		'NBody'   		    : '-n 1081600' ,
               		'Laplace' 		    : '-n 215296 -k 5' ,
               		'GaussianDerivates' : '-n 4608 -m 4608 -k 3'}

## benchmark = ['MatMul']

# Check all folder are actually there
for n in benchmark:
    if not os.path.exists(n):
        raise Exception('Folder ' + n + 'does not exist')

if args.make:
    # run the makefile in each folder
    command = "make clean && make"
    if args.printresult:
        command += " DEF=PRINT"
    if args.backcopy:
        command += " DEF=READBACK"
    for n in benchmark:
        os.chdir(n)
        p1 = subprocess.Popen(command, shell=True,\
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)
        erracc = ''
        while True:
            line = p1.stdout.readline()
            if not line:
                line = p1.stderr.readline()
                if not line: break
                erracc += line
                if line[0:9] == 'make: ***':
                    raise Exception('Program ' + n + ' did not compile: ' + erracc)
        os.chdir('..')
    
if args.run is not None:
    dev = args.run
    # run each exe in benchmark
    if args.input == 'K20Max':
        cmdlineopts = cmdlineoptsK20Max
    else:
        cmdlineopts = cmdlineoptsbasic

    tag = ''
    if args.tag:
        tag = args.tag + '_'
        
    for n in benchmark:
        m = n + dev
        uniqueid = open('logs/.uniqueid.txt','r')
        uid = uniqueid.readline()
        uniqueid.close()
        uniqueid = open('logs/.uniqueid.txt','w')
        uniqueid.write(str(int(uid) + 1))
        log = open('logs/' + uid + '_' + tag + m + cmdlineopts[n].replace(" ", "_") \
                   .replace("-", "_") + '.txt','w')
        os.chdir(n)
        for k in xrange(args.numberofiterations):
            p1 = subprocess.Popen('./' + m +'.exe ' + cmdlineopts[n], shell=True,\
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE)
            acc = '$Func ' + m + ', $Defines ' + cmdlineopts[n]
            while True:
                line = p1.stdout.readline()
                if not line:
                    line = p1.stderr.readline()
                    if not line: break
                acc += ', ' + line[:-1]
            log.write(acc + '\n')
            log.flush()
            os.fsync(log)
            #print acc + '\n'
        os.chdir('..')
        log.close()
        uniqueid.close()
	

