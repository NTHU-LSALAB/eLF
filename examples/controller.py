import argparse
import fix_paths
import elf


parser = argparse.ArgumentParser()
parser.add_argument('listen_address')
args = parser.parse_args()
ctrl = elf.Controller(args.listen_address)
