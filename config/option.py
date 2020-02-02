import yaml

def parse(opt_path):
    with open(opt_path, mode='r') as f:
        opt = yaml.load(f,Loader=yaml.FullLoader)
    
    opt['resume']['path'] = opt['resume']['path']+'/'+opt['name']
    opt['logger']['path'] = opt['logger']['path']+'/'+opt['name']
    return opt


if __name__ == "__main__":
    parse('train.yml')