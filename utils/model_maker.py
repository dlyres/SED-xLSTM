from model.plfa_model import make_model as PLFA
from model.VGT import make_model as VGT
from model.VIT import make_model as VIT
from model.SSGFormer import make_model as SSGFormer
from model.MS1D_CNN import make_model as MS1D_CNN
from model.CCNN import make_model as CCNN


from model.VIT_SAM_BiGRU import make_model as VIT_SAM_BiGRU
from model.VIT_BiLSTM import make_model as VIT_BiLSTM
from model.VIT_CBAM import make_model as VIT_CBAM


def make_model(args):
    model = None
    if args.model_name == 'plfa':
        model = PLFA(args)
    if args.model_name == 'VIT+Bi-GRU':
        model = VGT(args)
    if args.model_name == 'SSGFormer':
        model = SSGFormer(args)
    if args.model_name == 'MS1D_CNN':
        model = MS1D_CNN(args)
    if args.model_name == 'CCNN':
        model = CCNN(args)
    if args.model_name == 'VIT':
        model = VIT(args)


    if args.model_name == 'VIT+SAM+Bi-GRU':
        model = VIT_SAM_BiGRU(args)
    if args.model_name == 'VIT+Bi-LSTM':
        model = VIT_BiLSTM(args)
    if args.model_name == 'VIT+CBAM':
        model = VIT_CBAM(args)
    return model