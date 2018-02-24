import os
import sys

if len(sys.argv) != 2 or sys.argv[1] == '--help':
    print 'usage: python your/path/syncModel.py <model_name>'

model_name = sys.argv[1]
src_url = 'tianxingli24@instance-1:/home/tianxingli24/ml/face-alignment/models/' + model_name
targ_dir = '~/Desktop/tf_testing/face-alignment/models/'
os.system('gcloud compute scp --recurse '
          + src_url + ' ' + targ_dir
          )