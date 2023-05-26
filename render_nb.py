
import os
import glob

nb_files = glob.glob('examples/*_examples/*.ipynb')

command = "jupyter nbconvert"
options = "--to html"

status = {}

for nb_file in nb_files:

    subdir = os.path.dirname(nb_file).split('/')[-1]
    basename = os.path.splitext(os.path.basename(nb_file))[0]
    outfile = f"../../docs/{subdir}/{basename}.html"
    relpath = f"docs/{subdir}"
    try:
        os.makedirs(relpath)
    except FileExistsError:
        pass
    
    comline = f"{command} {options} --output {outfile} --execute {nb_file}"
    print(comline)
    #render = 0
    render = os.system(comline)
    status[nb_file] = render


for key, val in status.items():
    print(f"{key} {val}")
            

