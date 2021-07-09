# Obtain required files
cp -r ../../Python/{aux_routines,qmat}.py ./

# Experiment parameters
M=80000
N=50000
PROBLEM=dense1

# Execute experiment
rm -rf output
mkdir output
python exp1.py $M $N $PROBLEM