import os
import sys

if len(sys.argv) != 2 or sys.argv[1] == '--help':
    print 'usage: python your/path/syncModel.py <model_name>'

model_name = sys.argv[1]
#src_url = 'tianxingli24@instance-1:/home/tianxingli24/ml/face-alignment/models/' + model_name
src_url = 'modidev@76.64.78.165:/home/tian/fun/face-alignment/models/' + model_name
targ_dir = '~/Desktop/machine-learning/face-alignment/models/' + model_name

os.system('scp -r ' + src_url + ' ' + targ_dir)
#os.system('gcloud compute scp --recurse '
#          + src_url + ' ' + targ_dir
#          )