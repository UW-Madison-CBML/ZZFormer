# ZZFormer

ZFormer provides a general framework for incorporating zigzag topological structure into sequence models, enabling learning across a wide range of biological sequence analysis problems involving tandem repeats, regulatory motifs, RNA structures, and other recurrent or cyclic patterns. 

# Usage
### Clone repo
```
git clone https://github.com/UW-Madison-CBML/ZZFormer.git
```

### Environment

Use requirements.txt to build environment or pull docker image `lsvaren/transformers`

```


```

### Zigzag homology computation

Prepare all sequences in a text file, with one sequence per line. Run `preML.py` passing a string of alphabets, memory, k-mer size, and input text file name. The output will be a `pkl` file `results_{input_file}_{k}mer.pkl`

```
cd zigzag_homology

python preML.py "A,T,G,C,X" 100 4 "dna_preML.txt"

# Output file: results_dna_preML_4mer.pkl
```

### Persistence image computation
```

```



### Train
```
Edit
```

### Test
```
Test
```

# Potential use case
ZZFormer was tested with Repbase, RepetDB and MnTEdb on classifying transposable elements at the order and superfamily levels.


# Citation
```
@article{ZZFormer,
  author=,
  title=
}
```




