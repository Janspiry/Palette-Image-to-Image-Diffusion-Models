#!/bin/ksh 
#$ -q gpu
#$ -o result.out
#$ -j y
#$ -N cDDPM
cd $WORKDIR
cd /beegfs/data/work/imvia/in156281/cDDPM
source /beegfs/data/work/imvia/in156281/cDDPM/venv/bin/activate
module load python
export PYTHONPATH=/work/imvia/in156281/cDDPM/venv/lib/python3.9/site-packages:$PYTHONPATH
export MPLCONFIGDIR=/work/imvia/in156281/.cache/matplotlib

python run.py -c config/config.json -p train