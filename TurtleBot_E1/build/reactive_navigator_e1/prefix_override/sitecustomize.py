import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/lee/robotica/guia5/TurtleBot_E1/install/reactive_navigator_e1'
