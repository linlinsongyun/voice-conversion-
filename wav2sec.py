import os
import sys
file_dir = sys.argv[1]
#'../corpus/arctic_test' 
out_16 = sys.argv[2]
# '../corpus/arc_test16'

out_32 = sys.argv[3]
#'../corpus/arc_test32'
if not os.path.exists(out_16):
    os.makedirs(out_16)
if not os.path.exists(out_32):
    os.makedirs(out_32)
for files in os.listdir(file_dir):
    file_name = files.split('.wav')[0]
    print("filename",file_name)
    new = os.path.join(out_16,"%s.s16"%file_name)     
    print("new16",new)
    new_s = os.path.join(out_32,"%s.32"%file_name)
    print("new32",new_s)
    file_path = os.path.join(file_dir, files)
    print("file",file_path)
    os.system('/usr/bin/sox %s -r 16000 -c 1 -t sw %s' % (file_path, new))
    os.system('./dump_data -test %s %s' %(new, new_s))

#/usr/bin/sox ./tts_aishell/BAC009S0764W0124.wav -r 16000 -c 1 -t sw BAC009S0764W0124.s16
#./dump_data -test BAC009S0764W0124.s16 BAC009S0764W0124.32
