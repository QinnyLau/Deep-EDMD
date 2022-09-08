from configparser import NoOptionError
from configparser import ConfigParser

parser = ConfigParser()
#parser.read("./data/new_sub_traj/IDM_sub_traj.ini")
parser.read("./data/NGSIM/ngsim.ini")

section = 'experiment'
NAME = parser.get(section, 'name')

# load files
section = 'files'
STATE_VALUE = None
exec("STATE_VALUE = " + parser.get(section, 'state_value'))
#CONTROL_VALUE = parser.get(section, 'control_value')
#exec("DIM_EACH_STATE = " + parser.get(section, 'dim_each_state'))
DIM_STATE = parser.getint(section, 'dim_state')
#DIM_CTRL = parser.getint(section, 'dim_ctrl')
try:
    exec("STATE_SEQS = " + parser.get(section, 'state_seqs'))
except NoOptionError:
    STATE_SEQS = None
# try:
#     exec("CTRL_SEQS = " + parser.get(section, 'ctrl_seqs'))
# except NoOptionError:
#     CTRL_SEQS = None
DEVICE = "cpu"

# train parameters
section = "train-parameters"
EPOCH = parser.getint(section, 'epoch')
BATCH_SIZE = parser.getint(section, 'batch_size')
NUM_TRAIN_TRAJ = parser.getint(section, 'num_train_traj')
NUM_TEST_TRAJ = parser.getint(section, 'num_test_traj')
SNAPSHOT_TRAIN_LEN = parser.getint(section, 'snapshot_train_len')
SNAPSHOT_TEST_LEN = parser.getint(section, 'snapshot_test_len')
LR = parser.getfloat(section, 'lr')
R_CO = parser.getfloat(section, 'r_co')
P_CO = parser.getfloat(section, 'p_co')
A_CO = parser.getfloat(section, 'a_co')
PRED_LENGTH = parser.getint(section, 'pred_length')
deltaT = parser.getfloat(section, 'deltaT')

# net structure
section = 'net-parameters'
encoder_params = parser.get(section, 'encoder').split(',')
decoder_params = parser.get(section, 'decoder').split(',')
EN_IN = int(encoder_params[0])
EN_HID1 = int(encoder_params[1])
EN_HID2 = int(encoder_params[2])
EN_HID3 = int(encoder_params[3])
EN_OUT = int(encoder_params[-1])
DE_IN = int(decoder_params[0])
DE_OUT = int(decoder_params[-1])
AE_ACT_FUN = parser.get(section, 'ae_act_fun')

# test config
section = 'test-parameters'
TEST_BATCH = parser.getint(section, 'test_batch')
TEST_MODEL_NAME = parser.get(section, 'test_model_name')

# train files save dir
NET_SAVE_DIR = 'saved/' + NAME
