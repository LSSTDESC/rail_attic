import os


def _all_python_files():
    root_dir = os.path.dirname(__file__)
    names = []
    for filename in os.listdir(root_dir):
        if filename.endswith('.py') and filename != '__init__.py':
            name = filename[:-3]
            names.append(name)
    return names


for name in _all_python_files():
    try:
        __import__(name, globals(), locals(), level=1)
    except ModuleNotFoundError:  #pragma: no cover
        print("estimator {} not installed".format(name))
