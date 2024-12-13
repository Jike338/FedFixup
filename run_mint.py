import os 

header = "#!/bin/bash \n#PBS -l walltime=80:15:00 \n#PBS -l nodes=1:ppn=2:gpus=1,mem=100GB \n#PBS -A PAS2099 \n#PBS -N jike338 \n#PBS -j oe \n#PBS -m be \n#SBATCH --output=script_output1/R-%x.%j.out \nmodule load python \nmodule load cuda \n\ncd /fs/ess/scratch/PAS2099/jike/DelayBN\n"


with open('run_temp.sh') as topo_file:
    for idx, line in enumerate(topo_file):

        f = open("run_mint/"+str(idx)+".sh", "w")
        f.write(header)
        f.write(line)

