import gzip

with gzip.open('input.gz','rt') as f:
    for line in f:
        print('got line', line)

