# import sys
# sys.path.insert(0, './tools')
# import tools

import sys
#from phonomials import phonomialsBase


if (sys.version_info > (3, 0)):
    # Python 3 
    from phonesse import phonesse

else:
    # Python 2
    import phonesse 