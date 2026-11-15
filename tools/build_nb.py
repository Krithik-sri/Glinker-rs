import json, os, sys, nbformat
root, tpl, outp = sys.argv[1], sys.argv[2], sys.argv[3]
files = {}
for d, dirs, fs in os.walk(root):
    dirs[:] = [x for x in dirs if x != "target"]
    for f in fs:
        rel = os.path.relpath(os.path.join(d, f), root)
        if rel != "Cargo.lock": files[rel] = open(os.path.join(d, f)).read()
files = dict(sorted(files.items()))
print(len(files), list(files))
src = open(tpl).read().replace("FILES = __FILES__", "FILES = " + json.dumps(files, indent=0, ensure_ascii=False))
compile(src, "cell", "exec")
nb = nbformat.v4.new_notebook()
nb.metadata = {"colab": {"provenance": [], "name": os.path.basename(outp)}, "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}}
nb.cells = [nbformat.v4.new_code_cell(src)]
nbformat.validate(nb)
nbformat.write(nb, outp)
print(os.path.getsize(outp), "bytes,", len(nb.cells), "cell")
