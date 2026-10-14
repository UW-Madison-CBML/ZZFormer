# Zigzag homology
We used the [Dionysus2](https://mrzv.org/software/dionysus2/) package for our zig zag homology module.

# Running scripts
To compute the apex representatives via zigzag homology:

```
python preML.py $alphabet $kmer $memory $filename
```

Edit `preML.py` to accept your specific sequence file. It is constructed to accept a text file where each row is an individual sequence.

