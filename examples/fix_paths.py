import os
import sys


here = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(here, '..', 'python')))
ext_dir = os.path.abspath(os.path.join(here, '..', 'build'))
sys.path.append(ext_dir)
os.environ.setdefault('ELF_LIB_DIR', ext_dir)
