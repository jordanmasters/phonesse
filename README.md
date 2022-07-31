# Phonesse 
Phonesse is a toolkit for dealing with verbal sound patterns. It supports visualizing, extracting, searching, and analyzing sound patterns from text data.

## Documentation

ReadTheDocs [walkthrough and documentation](https://jordanmasters.github.io/phonomials/_static/overviews/phonesse.html) is available online.

## Disclaimer

Licence, and notice of work in progress 

## Setup
will be available soon through `pip install phonesse` for download & install 

until then... follow the instructions below

### Download

```
$ git clone https://github.com/jordanmasters/phonesse
```

### Local pip install

This is so you don't have to be in the phonesse folder to 'import phonesse'

```
$ cd phonesse

$ pip install ./
or 
$ pip install --user ./ --upgrade
```

Note: package still under development. Phonesse will be available for download and install through pip and the online PyPi repo soon.


## Basic Usage

```
from phonesse import phonesse

# get segments

>>> phonesse.get_segments("It is this easy to get vowel segments from text", segments='vowels')
['IH', 'IH', 'IH', 'IY', 'IY', 'UW', 'EH', 'AW', 'EH', 'AH', 'AH', 'EH']

>>> phonesse.get_segments("Just as easy to get the underlying stress pattern", segments='stress')
[1, 1, 1, 0, 1, 1, 0, 2, 0, 1, 0, 1, 1, 0]

>>> phonesse.get_segments("Or the first sound of each word", segments='word_initial')
['AO1', 'DH', 'F', 'S', 'AH1', 'IY1', 'W']

>>> phonesse.n_grams(['IH', 'IH', 'IH', 'IY', 'IY', 'UW', 'EH', 'AW', 'EH', 'AH', 'AH', 'EH'],2)

```

Neural-Net Back-up
```
# a neural network deals with misspelled or unknown words.
>>> phonesse.get_segments("Badd Sppellling is fine", segments='consonants')
['B', 'D', 'S', 'P', 'L', 'NG', 'Z', 'F', 'N'
```

Plotting

Visualize language sounds in MIDI and grids  using psychologically grounded color mappings. 
```
# Plot sounds as MIDI
phonesse.plot_as_MIDI("It is this easy to see sound segments from text")

# Plot sounds as Grid
phonesse.plot_as_grid(phonesse.data.get_sample('limerick_1'))

phonesse.plot_as_grid(phonesse.data.get_sample('rap_eminem'))

```

Many other functions are outlined in the [documentation](https://jordanmasters.github.io/phonomials/_static/overviews/phonesse.html).