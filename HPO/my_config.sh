# Model settings
export CANDLE_MODEL_TYPE="SINGULARITY"
export MODEL_NAME=${/PATH/TO/SINGULARITY/IMAGE/FILE.sif}
export PARAM_SET_FILE=${/PATH/TO/GA/PARAMETER/FILE.json}
# If you have write access to /lambda_stor, you can save on the shared
# filesystem. If not, make a directory in /tmp
export CANDLE_DATA_DIR=/tmp/my_username

# System settings
export PROCS=3 # minimum 3

