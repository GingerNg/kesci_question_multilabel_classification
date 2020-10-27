
lines = ["121", "4545", "34343"]
lines = list(map(lambda x: x.strip(), lines))
vocab = dict(zip(lines, range(len(lines))))
print(vocab)