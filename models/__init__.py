from models import encoder
from models import losses
from models import resnet
from models import ssl
from models import solvers

REGISTERED_MODELS = {
    'sim-clr': ssl.SimCLR,
    'eval': ssl.SSLEval,
    'semi-supervised-eval': ssl.SemiSupervisedEval,
    'mmcl_inv': ssl.MMCL_INV,
    'mmcl_pgd': ssl.MMCL_PGD,
}
