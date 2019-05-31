from __future__ import print_function
import os
import sys
import xlsxwriter
import pandas as pd
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('output-file')
parser.add_argument('input-files', nargs='+')
args = parser.parse_args()

dfs = []
stopped_at = []
names = []
output_filename = sys.argv[1]
for f in sys.argv[2:]:
    sheets = pd.read_excel(f, index_col=0, sheet_name=[0,1])
    dfs.append(sheets[0].transpose())
    stopped_at.append(sheets[1].median()[1])
    names.append(os.path.basename(f[:f.rfind('.')]))

df = pd.DataFrame(dict(zip(names, map(lambda x: x.median(), dfs))))
ax = df.plot()
ax.set(xlabel="Training set size", ylabel="Test set error")
for i,d in enumerate(dfs):
    ax.fill_between(d.min().index, d.min(), d.max(), facecolor = 'C' + str(i), alpha = 0.5)
    ax.axvline(x = stopped_at[i], color = 'C' + str(i), linestyle=':')

plt.savefig(output_filename, dpi=150)
plt.close()
