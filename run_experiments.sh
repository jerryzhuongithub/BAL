# Binary classification - Classes - rec.sport.baseball and rec.sport.hockey. Box-and-whisker plot for test error saved to 'figs/baseball_vs_hockey.png'
python passive.py --categories "rec.sport.baseball,rec.sport.hockey" --growth geometric --err_filename baseball_vs_hockey.png

# Binary classification - Classes - rec.autos and rec.sport.hockey. Box-and-whisker plot for test error saved to 'figs/autos_vs_hockey.png'
python passive.py --categories "rec.autos,rec.sport.hockey" --growth geometric --err_filename autos_vs_hockey.png

# Binary classification - Classes - rec.autos and rec.motorcycles. Box-and-whisker plot for test error saved to 'figs/autos_vs_motorcycles.png'
python passive.py --categories "rec.autos,rec.motorcycles" --growth geometric --err_filename autos_vs_motorcycles.png

# Multiclass Classification - with all 20 news groups. Box-and-whisker plot for test error saved to 'figs/twentynewsgroups.png'
python passive.py --growth geometric --err_filename twentynewsgroups.png