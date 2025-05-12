from model.PLFA import make_model as PLFA
from model.MS1D_CNN import make_model as MS1D_CNN
from model.CCNN import make_model as CCNN
from model.SED_xLSTM import make_model as SED_xLSTM



def make_model(args):
    model = None
    if args.model_name == 'PLFA':
        model = PLFA(args)
    if args.model_name == 'MS1D_CNN':
        model = MS1D_CNN(args)
    if args.model_name == 'CCNN':
        model = CCNN(args)
    if args.model_name == 'SED_xLSTM':
        model = SED_xLSTM(args)


    return model