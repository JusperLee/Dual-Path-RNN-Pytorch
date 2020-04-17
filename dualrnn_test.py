import os
import torch
from data_loader.AudioReader import AudioReader, write_wav
import argparse
from torch.nn.parallel import data_parallel
from model.model_rnn import Dual_RNN_model
from logger.set_logger import setup_logger
import logging
from config.option import parse
import tqdm


class Separation():
    def __init__(self, mix_path, yaml_path, model, gpuid):
        super(Separation, self).__init__()
        self.mix = AudioReader(mix_path, sample_rate=8000)
        opt = parse(yaml_path)
        net = Dual_RNN_model(**opt['Dual_Path_RNN'])
        dicts = torch.load(model, map_location='cpu')
        net.load_state_dict(dicts["model_state_dict"])
        setup_logger(opt['logger']['name'], opt['logger']['path'],
                            screen=opt['logger']['screen'], tofile=opt['logger']['tofile'])
        self.logger = logging.getLogger(opt['logger']['name'])
        self.logger.info('Load checkpoint from {}, epoch {: d}'.format(model, dicts["epoch"]))
        self.net=net.cuda()
        self.device=torch.device('cuda:{}'.format(
            gpuid[0]) if len(gpuid) > 0 else 'cpu')
        self.gpuid=tuple(gpuid)

    def inference(self, file_path):
        with torch.no_grad():
            for key, egs in tqdm.tqdm(self.mix):
                #self.logger.info("Compute on utterance {}...".format(key))
                egs=egs.to(self.device)
                norm = torch.norm(egs,float('inf'))
                if len(self.gpuid) != 0:
                    if egs.dim() == 1:
                        egs = torch.unsqueeze(egs, 0)
                    ests=self.net(egs)
                    spks=[torch.squeeze(s.detach().cpu()) for s in ests]
                else:
                    if egs.dim() == 1:
                        egs = torch.unsqueeze(egs, 0)
                    ests=self.net(egs)
                    spks=[torch.squeeze(s.detach()) for s in ests]
                index=0
                for s in spks:
                    s = s[:egs.shape[1]]
                    s = s - torch.mean(s)
                    s = s/torch.max(torch.abs(s))
                    #norm
                    #s = s*norm/torch.max(torch.abs(s))
                    s = s.unsqueeze(0)
                    index += 1
                    os.makedirs(file_path+'/spk'+str(index), exist_ok=True)
                    filename=file_path+'/spk'+str(index)+'/'+key
                    write_wav(filename, s, 8000)
                break
            self.logger.info("Compute over {:d} utterances".format(len(self.mix)))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument(
        '-mix_scp', type=str, default='../create_scp/tt_mix.scp', help='Path to mix scp file.')
    parser.add_argument(
        '-yaml', type=str, default='./config/train_rnn_opt.yml', help='Path to yaml file.')
    parser.add_argument(
        '-model', type=str, default='./checkpoint/Dual_Path_RNN_opt/best.pt', help="Path to model file.")
    parser.add_argument(
        '-gpuid', type=str, default='0', help='Enter GPU id number')
    parser.add_argument(
        '-save_path', type=str, default='./result/dual-rnn/', help='save result path')
    args=parser.parse_args()
    gpuid=[int(i) for i in args.gpuid.split(',')]
    separation=Separation(args.mix_scp, args.yaml, args.model, gpuid)
    separation.inference(args.save_path)


if __name__ == "__main__":
    main()