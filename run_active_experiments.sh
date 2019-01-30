# Binary classification - Classes - rec.sport.baseball and rec.sport.hockey. Min-Max-Median plot for test error saved to 'figs/active/baseball_vs_hockey_entropy.png'
python2 active.py --categories rec.sport.baseball,rec.sport.hockey --err_filename baseball_vs_hockey_entropy.png --N 650

# Binary classification - Classes - rec.sport.baseball and rec.motorcycles. Min-Max-Median plot for test error saved to 'figs/active/baseball_vs_motorcycles_entropy.png'
python2 active.py --categories rec.sport.baseball,rec.motorcycles --err_filename baseball_vs_motorcycles_entropy.png --N 650

# Binary classification - Classes - rec.autos and rec.motorcycles. Min-Max-Median plot for test error saved to 'figs/active/autos_vs_motorcycles_entropy.png'
python2 active.py --categories rec.autos,rec.motorcycles --err_filename autos_vs_motorcycles_entropy.png --N 650

# Multi-class Classification - with all 20 news groups. Min-Max-Median plot for test error saved to 'figs/active/twentynewsgroups_entropy.png'
python2 active.py --err_filename twentynewsgroups_entropy.png --N 1000