all: baseball_vs_hockey/err.png 20newsgroups/err.png

TRIALS=20
N=100

.PHONY: all clean

# === 20 NEWSGROUPS MULTICLASS EXAMPLE ===

20newsgroups/err.png: 20newsgroups/passive.xlsx 20newsgroups/active.xlsx 20newsgroups/weak-teaching.xlsx
	python ../plot.py $@ $^

20newsgroups/training.npy:
	python prepare-20-newsgroups.py

20newsgroups/passive.xlsx: 20newsgroups/training.npy
	python ../bal.py passive 20newsgroups/training.npy 20newsgroups/test.npy $@ --trials $(TRIALS) --training-set-max-size $(N) --batch 10

20newsgroups/active.xlsx: 20newsgroups/training.npy
	python ../bal.py active 20newsgroups/training.npy 20newsgroups/test.npy $@ --trials $(TRIALS) --training-set-max-size $(N) --batch 10

20newsgroups/weak-teaching.xlsx: 20newsgroups/training.npy
	python ../bal.py weak-teaching 20newsgroups/training.npy 20newsgroups/test.npy $@ --trials $(TRIALS) --training-set-max-size $(N) --batch 10

# === 20 NEWSGROUPS TWO-CLASS EXAMPLE ===

baseball_vs_hockey/err.png: baseball_vs_hockey/passive.xlsx baseball_vs_hockey/active.xlsx baseball_vs_hockey/weak-teaching.xlsx baseball_vs_hockey/minipool.xlsx
	python ../plot.py $@ $^

baseball_vs_hockey/training.npy:
	python prepare-20-newsgroups.py baseball_vs_hockey rec.sport.baseball rec.sport.hockey

baseball_vs_hockey/passive.xlsx: baseball_vs_hockey/training.npy
	python ../bal.py passive baseball_vs_hockey/training.npy baseball_vs_hockey/test.npy $@ --trials $(TRIALS) --training-set-max-size $(N)

baseball_vs_hockey/active.xlsx: baseball_vs_hockey/training.npy
	python ../bal.py active --uncertainty entropy baseball_vs_hockey/training.npy baseball_vs_hockey/test.npy $@ --trials $(TRIALS) --training-set-max-size $(N)

baseball_vs_hockey/weak-teaching.xlsx: baseball_vs_hockey/training.npy
	python ../bal.py weak-teaching baseball_vs_hockey/training.npy baseball_vs_hockey/test.npy $@ --trials $(TRIALS) --training-set-max-size $(N)

baseball_vs_hockey/minipool.xlsx: baseball_vs_hockey/training.npy
	python ../bal.py minipool --candidates 10 --validation-set baseball_vs_hockey/validation.npy baseball_vs_hockey/training.npy baseball_vs_hockey/test.npy $@ --trials $(TRIALS) --training-set-max-size $(N)

clean:
	rm -rf 20newsgroups baseball_vs_hockey
