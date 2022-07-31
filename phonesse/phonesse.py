############################################################################
############################################################################
#					LOADING
############################################################################
############################################################################

import pronouncing
from phonomials import get_cmu
from tqdm import tqdm
import itertools
import copy
from nltk import sent_tokenize
from nltk.tokenize import WhitespaceTokenizer
from pathlib2 import Path
from g2p_en import G2p
import plotly.figure_factory as ff
from collections import defaultdict, deque, Counter
import math
g2p = G2p()
from ipywidgets import GridspecLayout
from ipywidgets import Button, Layout, jslink, Dropdown, IntText, IntSlider, Text, HTML, ToggleButtons, Select, Checkbox 
from ipywidgets import widgets, interactive
search_counter = 0


# SOME GLOBAL VARIABLES

vowel_digit_map = { 0:'XX',
                 .066:'IY',
                 .133:'IH',
                 .2:'EY',
                 .266:'EH',
                 .333:'AH',
                 .4:'ER',
                 .466:'AY',
                 .533:'UW',
                 .6:'UH',
                 .666:'OW',
                 .733:'AO',
                 .8:'AE',
                 0.866:'OY',
                 .933:'AW',
                 1:'AA'}
# Swap keys and values
vowel_digit_map_inv = {v: k for k, v in vowel_digit_map.items()}
vowel_color_map = { 'XX':'rgb(255,255,255)',
                    'IY':'rgb(240, 219, 116)',
                    'IH':'rgb(182, 201, 103)',
                    'EY':'rgb(165, 196, 125)',
                    'EH':'rgb(146,190,152)',
                    'AH':'rgb(127, 186, 178)',
                    'ER':'rgb(103, 152, 173)',
                    'AY':'rgb(84,120,165)',
                    'UW':'rgb(67,90,158)',
                    'UH':'rgb(79,78,137)',
                    'OW':'rgb(86,62,117)',
                    'AO':'rgb(91,64,100)',
                    'AE':'rgb(43, 29, 67)',
                    'OY':'rgb(32, 22, 49)',
                    'AW':'rgb(21, 13, 32)',
                    'AA':'rgb(0, 0, 0)'}
# Define the order of colors
font_colors = ['black', 'white']
# Define vowel segment to 0-1 scaling
vowel_colorscale = [[0.0, vowel_color_map['XX']],        # XX  IY white (yellow)
              [.066, vowel_color_map['IY']],     # IY
              [.133, vowel_color_map['IH']],     # IH
              [.2, vowel_color_map['EY']],       # EY
              [.266, vowel_color_map['EH']],       # EH
              [.333, vowel_color_map['AH']],     # AH
              [.4, vowel_color_map['ER']],       # ER
              [.466, vowel_color_map['AY']],        # AY
              [.533, vowel_color_map['UW']],         # UW
              [.6, vowel_color_map['UH']],           # UH
              [.666, vowel_color_map['OW']],         # OW
              [.733, vowel_color_map['AO']],       # AO
              [.8, vowel_color_map['AE']],         # AE
              [.866,vowel_color_map['OY']],       # OY
              [.933, vowel_color_map['AW']],       # AW
              [1., vowel_color_map['AA']]]        # AA
# Define stress mapping
stress_digit_map = { 0.5:0, 0.75:2, 1:1, 0:10}
stress_digit_map_inv = {v: k for k, v in stress_digit_map.items()}
# Define stress mapping
stress_colorscale = [[0, 'rgb(255,255,255)'],        # XX  empty white
              [.5, 'rgb(167, 199, 231)'],     # IY
              [.75, 'rgb(111, 143, 175)'],     # IH
              [1, 'rgb(25,25,112)'], ]      # EY


############################################################################
############################################################################
#					PHONESSE USER FUNCTIONS
############################################################################
############################################################################

def get_segments(user_input, segments='vowels'):
	if type(user_input) == str:
		p = phonomial.from_string(user_input)
		user_str = user_input
	elif str(type(user_input)) == "<class 'phonomials.phonomialsBase.phonomial'>": # change when swap package naming
		p = user_input
	if segments=='vowels':
		ret = p.get_vowels()
	elif segments=='stress':
		ret = p.get_vowels(stress=True)
	elif segments=='consonants':
		ret = p.get_cons(flat="yes",removeDelims="yes", placeholder="no")
	elif segments=='consonant_clusters':
		ret = p.get_cons(flat="no",removeDelims="yes", placeholder="no")
	elif segments=='all':
		ret = [seg for seg in phonesse.flatten(p.blocks) if seg not in [0,1,2,'.']]
	elif segments=='words': # elif words orthography
		ret = p.cmu_word_phones
		if [] in p.syllables:
			print("Words")
	elif segments=='syllables':
		ret = p.syllables
		if [] in p.syllables:
			empty_counter = 0
			for syll in p.syllables:
				if syll == []:
					empty_counter+=1
			print("There are",empty_counter,"syllables missing from the encoding")
	elif segments=='word_initial':
		# ret = p.syllables
		first_sounds = []
		for word in p.cmu_word_prons:
			try:
				first_sounds.append(word[0].split(' ')[0])
			except:
				pass
		ret = first_sounds
	elif segments=='vowels_stress':
		vowels = p.get_vowels()
		stress = p.get_vowels(stress=True)
		return [m+str(n) for m,n in zip(vowels,stress)]

		syllables = []
		temp = copy.deepcopy(p.syllables)
		for syllable in temp:
		    if syllable !=[]: # not empty
		        #onset
		        for j, item in enumerate(syllable[0]):
		            # was hitting all sorts fow rods in cons streams so added this last check as tmep
		            if item == "." or item == "-" or any(char for char in item if char.islower()):
		                del syllable[0][j]
		        #coda
		        for j, item in enumerate(syllable[2]):
		            # was hitting all sorts fow rods in cons streams so added this last check as tmep
		            if item == "." or item == "-" or any(char for char in item if char.islower()):
		                del syllable[2][j]
		    syllables.append(syllable)
		ret = syllables
	return ret

    
def set_element(p, block_location='', element=''):
    """
    Take a phonomial, a block location to populate, and an element to put in the location. 

    :param p: phonomial
    :param block_location: str (e.g. 'C1', 'C2', 'V1', 'V2', etc...)
    :param element: ARPABET str (C positions only take consonants, V positions only take vowels) Block counts start at 1, left to right.
    :returns: The input phonomial updated with new element in block location
    
    """    
    element_type = block_location[0]
    element_location = block_location[1]
    if element_type == 'V':
        if element in vowels(): # make sure element is a vowel
            v_block_index = int(element_location)*2-1
            p.blocks[v_block_index] = [element]
    if element_type == 'C':
        if element in consonants(): # make sure element is a consonant
            c_block_index = int(element_location)*2-2
            p.blocks[c_block_index] = [element]
    return p


def flatten(list_of_lists):
	"""
	Flattens a list of lists into a list
	
	:param param1: list of lists
	:returns: list

	>>> phonomial.flatten([[1,2],[3,4]])
	[1, 2, 3, 4]

	"""

	# only supports lists of lists where each item is itself definately a list
	flattened = [val for sublist in list_of_lists for val in sublist]
	return flattened


def vowels():
	"""
	Returns a list oll English vowels (APRABET)

	:returns: list of strings (vowels)

	>>> phonomial.vowels()
	['IY','IH','EY','EH','AH','ER','AY','UW','UH','OW','AO','AE','OY','AW','AA']
	"""
	vowels = ['IY','IH','EY','EH','AH','ER','AY','UW','UH','OW','AO','AE','OY','AW','AA']
	return vowels


def consonants():
	"""
	Returns a list oll English consonants (APRABET)

	:returns: list of strings (consonants)

	>>> phonomial.consonants()
	['B','CH','D','DH','F','G','HH','JH','K','L','M','N','NG','P','R','S','SH','T','TH','W','V','Y','Z','ZH']

	"""
	consonants = ['B','CH','D','DH','F','G','HH','JH','K','L','M','N','NG','P','R','S','SH','T','TH','W','V','Y','Z','ZH']
	return consonants



def naturalclass_2_ARPABET(natural_classes, segments='all',logic='or'):

    """
    Takes an list of strings (natural classses) and returns all the segments that belong all classes (and logic). 
    If you give ['high', 'front'], it will only returns segments that are both high and front
    Default mode returns both consonants and vowels, but if mode is set to 'vowels' or 'consonants', it will only return that type of segment

    :param segment: an natural class (string)
    :param mode: 'all', 'vowels', or 'consonants' (string)
    :param logic: 'and', 'or' (string)

    :returns: list of strings or list of lists of strings

	>>> naturalclass_2_ARPABET(['voiceless'], segments='all',logic='or')
	[['CH', 'F', 'HH', 'K', 'P', 'S', 'SH', 'T', 'TH']]
																																			
    >>> phonesse.naturalclass_2_ARPABET(['stop','voiceless'], segments='all',logic='and')
	['K', 'P', 'T']

    """

    if type(natural_classes) == str: # catch for passing string of single natural class
        natural_classes = [natural_classes]
    segment_matches = []
    condition_threshold = len(natural_classes)
    if segments == 'all':
        v = vowels_naturalclasses_dict()
        c = consonants_naturalclasses_dict()
        v.update(c)
    elif segments == 'vowels':
        v = vowels_naturalclasses_dict()
    elif segments == 'consonants':
        v = consonants_naturalclasses_dict()
    
    for natural_class in natural_classes: # for each natural class condition
        condition_count = 0
        matched = []
        for segment in v: # for each item in phoneme_naturalclass_dictionary
            
            if natural_class in v[segment]:
                matched.append(segment)
        segment_matches.append(matched)        

    if logic == 'or':
        return segment_matches    
    elif logic == 'and':
        if len(natural_classes) > 1:
            try: # move this try upthe stack
                set1 = segment_matches[0]
                passed = []

                for item in set1:
                    cond_threshold = len(natural_classes)
                    cond_count = 0
                    for matched_list in segment_matches:
                        if item in matched_list:
                            cond_count+=1
                    if cond_count >= cond_threshold:
                        passed.append(item)
            except: print("Error")
        else:
            print("Can't use and condition unless you provide more than 1 natural classs")
            return
        return passed


def ARPABET_2_naturalclasses(segment):
	"""
	Takes an ARPABET segment and returns the list of associated natural classes (includes both consonant and vowel items)
	
	:param segment: an ARPABET segment (string)
	:returns: list of strings (natural classes)

	>>> ARPABET_2_naturalclasses('IY')
	['high','front','unrounded']

	"""
	# https://en.wikipedia.org/wiki/Diphthong
	# https://en.wikipedia.org/wiki/ARPABET
	# slow to merge every time its called...
	v = vowels_naturalclasses_dict()
	c = consonants_naturalclasses_dict()
	v.update(c)
	return v[segment]


############################################################################
############################################################################
#					INFO FUNCTIONS
############################################################################
############################################################################


def show_APRABET_examples():
	"""
	Basic Tables of Information about the sound segements in this package
	
	:returns: Two tables, vowel and consonant examples.

	"""
	v_data_matrix = [['ARPABET', 'IPA', 'Example'],
					['IY', 'i', 'beat'],
					['IH', 'I', 'bit'],
					['EY', 'eI', 'bait'],
					['EH', 'ɛ', 'bet'],
					['AH', 'ʌ', 'butt'],
					['ER', 'ɝ', 'bird'],
					['AY', 'aI', 'bite'],
					['UW', 'u', 'boot'],
					['UH', 'ʊ', 'book'],
					['OW', 'oʊ', 'boat'],
					['AO', 'ɔ', 'story'],
					['AE', 'æ', 'bat'],
					['OY', 'ɔI', 'boy'],
					['AW', 'aʊ', 'bout'],
					['AA', 'ɑ', 'balm, bot']]
	fig = ff.create_table(v_data_matrix)
	fig.show()

	c_data_matrix = [['ARPABET', 'IPA', 'Example'],
	                ['B', 'b', 'buy'],
	                ['CH', 'tʃ', 'China'],
	                ['D', 'd', 'die'],
	                ['DH', 'ð', 'thy'],
	                ['F', 'f', 'fight'],
	                ['G', 'ɡ', 'guy'],
	                ['HH', 'h', 'high'],
	                ['JH', 'dʒ', 'jive'],
	                ['K', 'k', 'kite'],
	                ['L', 'l', 'lie'],
	                ['M', 'm', 'my'],
	                ['N', 'n', 'nigh'],
	                ['NG', 'ŋ', 'sing'],
	                ['P', 'p', 'pie'],
	                ['R', 'ɹ', 'rye'],
	                ['S', 's', 'sigh'],
	                ['SH', 'ʃ', 'shy'],
	                ['T', 't', 'tie'],
	                ['TH', 'θ', 'thigh'],
	                ['V', 'v', 'vie'],
	                ['W', 'w', 'wise'],
	                ['Y', 'j', 'yatch'],
	                ['Z', 'z', 'zoo'],
	               ['ZH', 'ʒ', 'pleasure']]
	fig = ff.create_table(c_data_matrix)
	fig.show()


def arpabet_2_ipa_dict():
	"""
	Returns a dictionary where keys are ARPABET and values are IPA symbols

	:returns: dict

	>>> arpabet_2_ipa_dict()
		{'IY': 'i',
		 'IH': 'I',
		 'EY': 'eI',
		 'EH': 'ɛ',
		 'AH': 'ʌ',
		 'ER': 'ɝ',
		 'AY': 'aI',
		 'UW': 'u',
		 'UH': 'ʊ',
		 'OW': 'oʊ',
		 'AO': 'ɔ',
		 'AE': 'æ',
		 'OY': 'ɔI',
		 'AW': 'aʊ',
		 'AA': 'ɑ',
		 'B': 'b',
		 'CH': 'tʃ',
		 'D': 'd',
		 'DH': 'ð',
		 'F': 'f',
		 'G': 'ɡ',
		 'HH': 'h',
		 'JH': 'dʒ',
		 'K': 'k',
		 'L': 'l',
		 'M': 'm',
		 'N': 'n',
		 'NG': 'ŋ',
		 'P': 'p',
		 'R': 'ɹ',
		 'S': 's',
		 'SH': 'ʃ',
		 'T': 't',
		 'TH': 'θ',
		 'V': 'v',
		 'W': 'w',
		 'Y': 'j',
		 'Z': 'z',
		 'ZH': 'ʒ'}

	"""
	return {'IY': 'i',
	'IH': 'I',
	'EY': 'eI',
	'EH': 'ɛ',
	'AH': 'ʌ',
	'ER': 'ɝ',
	'AY': 'aI',
	'UW': 'u',
	'UH': 'ʊ',
	'OW': 'oʊ',
	'AO': 'ɔ',
	'AE': 'æ',
	'OY': 'ɔI',
	'AW': 'aʊ',
	'AA': 'ɑ',
	'B': 'b',
	'CH': 'tʃ',
	'D': 'd',
	'DH': 'ð',
	'F': 'f',
	'G': 'ɡ',
	'HH': 'h',
	'JH': 'dʒ',
	'K': 'k',
	'L': 'l',
	'M': 'm',
	'N': 'n',
	'NG': 'ŋ',
	'P': 'p',
	'R': 'ɹ',
	'S': 's',
	'SH': 'ʃ',
	'T': 't',
	'TH': 'θ',
	'V': 'v',
	'W': 'w',
	'Y': 'j',
	'Z': 'z',
	'ZH': 'ʒ'}

def consonants_naturalclasses_dict():
	"""
	Returns a dictionary where keys are ARPABET and values are lists of natural classes associated with each segment
	
	:returns: dict

	>>>
	cons_naturalclasses['B'] =  ['voiced','bilabial','stop']
	cons_naturalclasses['CH'] = ['voiceless','postalveolar','sibilant','affricate']
	cons_naturalclasses['D'] = ['voiced','alveolar','dental','postalveolar','stop']
	cons_naturalclasses['DH'] = ['voiced','dental','fricative']
	cons_naturalclasses['F'] = ['voiceless','llabiodental','fricative']
	cons_naturalclasses['G'] = ['voiced','velar','stop']
	cons_naturalclasses['HH'] = ['voiceless','glottal','fricative']
	cons_naturalclasses['JH'] = ['voiced','postalveolar','sibilant','affricate']
	cons_naturalclasses['K'] = ['voiceless','velar','stop']
	cons_naturalclasses['L'] = ['voiced','alveolar','lateral','approximant']
	cons_naturalclasses['M'] = ['voiced','bilabial','nasal']
	cons_naturalclasses['N'] = ['voiced','alveolar','nasal']
	cons_naturalclasses['NG'] = ['voiced','velar','nasal']
	cons_naturalclasses['P'] = ['voiceless','bilabial','stop']
	cons_naturalclasses['R'] = ['voiced','alveolar','approximant']
	cons_naturalclasses['S'] = ['voiceless','alveolar','fricative']
	cons_naturalclasses['SH'] = ['voiceless','postalveolar','fricative']
	cons_naturalclasses['T'] = ['voiceless','alveolar','posstalveolar','dental','stop']
	cons_naturalclasses['TH'] = ['voiceless','dental','non-sibilant','fricative']
	cons_naturalclasses['V'] = ['voiced','labiodental','fricative']
	cons_naturalclasses['W'] = ['voiced','labial-velar','approximant']
	cons_naturalclasses['Y'] = ['voiced','palatal','approximant']
	cons_naturalclasses['Z'] = ['voiced','alveolar','fricative']
	cons_naturalclasses['ZH'] = ['voiced','postalveolar','fricative']



	"""
	# https://en.wikipedia.org/wiki/Diphthong
	# https://en.wikipedia.org/wiki/ARPABET

	# Add obstruents etc
	cons_naturalclasses = {}
	cons_naturalclasses['B'] =  ['voiced','bilabial','stop']
	cons_naturalclasses['CH'] = ['voiceless','postalveolar','sibilant','affricate']
	cons_naturalclasses['D'] = ['voiced','alveolar','dental','postalveolar','stop']
	cons_naturalclasses['DH'] = ['voiced','dental','fricative']
	cons_naturalclasses['F'] = ['voiceless','llabiodental','fricative']
	cons_naturalclasses['G'] = ['voiced','velar','stop']
	cons_naturalclasses['HH'] = ['voiceless','glottal','fricative']
	cons_naturalclasses['JH'] = ['voiced','postalveolar','sibilant','affricate']
	cons_naturalclasses['K'] = ['voiceless','velar','stop']
	cons_naturalclasses['L'] = ['voiced','alveolar','lateral','approximant','sonorant']
	cons_naturalclasses['M'] = ['voiced','bilabial','nasal','sonorant']
	cons_naturalclasses['N'] = ['voiced','alveolar','nasal','sonorant']
	cons_naturalclasses['NG'] = ['voiced','velar','nasal','sonorant']
	cons_naturalclasses['P'] = ['voiceless','bilabial','stop']
	cons_naturalclasses['R'] = ['voiced','alveolar','approximant','sonorant']
	cons_naturalclasses['S'] = ['voiceless','alveolar','fricative']
	cons_naturalclasses['SH'] = ['voiceless','postalveolar','fricative']
	cons_naturalclasses['T'] = ['voiceless','alveolar','posstalveolar','dental','stop']
	cons_naturalclasses['TH'] = ['voiceless','dental','non-sibilant','fricative']
	cons_naturalclasses['V'] = ['voiced','labiodental','fricative']
	cons_naturalclasses['W'] = ['voiced','labial-velar','approximant','sonorant']
	cons_naturalclasses['Y'] = ['voiced','palatal','approximant','sonorant']
	cons_naturalclasses['Z'] = ['voiced','alveolar','fricative']
	cons_naturalclasses['ZH'] = ['voiced','postalveolar','fricative']
	return cons_naturalclasses

def vowels_naturalclasses_dict():
	"""
	Returns a dictionary where keys are ARPABET and values are lists of natural classes associated with each segment
	
	:returns: dict

	>>>
	vowels_naturalclasses['IY'] = ['high','front','unrounded']
	vowels_naturalclasses['IH'] = ['near-high','front','unrounded']
	vowels_naturalclasses['EY'] = ['mid','central','near-high','front','unrounded','diphthong'] # dipthong
	vowels_naturalclasses['EH'] = ['low-mid','front','unrounded']
	vowels_naturalclasses['AH'] = ['near-low','central']
	vowels_naturalclasses['ER'] = ['rhotic']
	vowels_naturalclasses['AY'] = ['low','front','unrounded','near-high','diphthong'] # dipthong
	vowels_naturalclasses['UW'] = ['high','back','rounded']
	vowels_naturalclasses['UH'] = ['near-high','back','rounded']
	vowels_naturalclasses['OW'] = ['high-mid','near-high','back','rounded','diphthong'] # dipthong
	vowels_naturalclasses['AO'] = ['low-mid','back','rounded']
	vowels_naturalclasses['AE'] = ['near-low','front','unrounded']
	vowels_naturalclasses['OY'] = ['low-mid','back','rounded','near-high','front','unrounded','diphthong'] # dipthong
	vowels_naturalclasses['AW'] = ['low','front','unrounded','near-high','back','rounded','diphthong'] # dipthong
	vowels_naturalclasses['AA'] = ['low','central','unrounded']  y, w, l, r, m, n, and ng
	"""
	vowels_naturalclasses = {}

	vowels_naturalclasses['IY'] = ['high','front','unrounded','sonorant']
	vowels_naturalclasses['IH'] = ['near-high','front','unrounded','sonorant']
	vowels_naturalclasses['EY'] = ['mid','central','near-high','front','unrounded','diphthong','sonorant'] # dipthong
	vowels_naturalclasses['EH'] = ['low-mid','front','unrounded','sonorant']
	vowels_naturalclasses['AH'] = ['near-low','central','sonorant']
	vowels_naturalclasses['ER'] = ['rhotic','sonorant']
	vowels_naturalclasses['AY'] = ['low','front','unrounded','near-high','diphthong','sonorant'] # dipthong
	vowels_naturalclasses['UW'] = ['high','back','rounded','sonorant']
	vowels_naturalclasses['UH'] = ['near-high','back','rounded','sonorant']
	vowels_naturalclasses['OW'] = ['high-mid','near-high','back','rounded','diphthong','sonorant'] # dipthong
	vowels_naturalclasses['AO'] = ['low-mid','back','rounded','sonorant']
	vowels_naturalclasses['AE'] = ['near-low','front','unrounded','sonorant']
	vowels_naturalclasses['OY'] = ['low-mid','back','rounded','near-high','front','unrounded','diphthong','sonorant'] # dipthong
	vowels_naturalclasses['AW'] = ['low','front','unrounded','near-high','back','rounded','diphthong','sonorant'] # dipthong
	vowels_naturalclasses['AA'] = ['low','central','unrounded','sonorant']
	return vowels_naturalclasses


def show_vowel_colors(mode='plot'):
	"""
	Plots the colors associated with vowels, optionally returns the dictionary of associations
	
	:returns: Plotly plot

	>>> show_vowel_colors(mode='dict')
		{ 'XX':'rgb(255,255,255)',
		'IY':'rgb(240, 219, 116)',
		'IH':'rgb(182, 201, 103)',
		'EY':'rgb(165, 196, 125)',
		'EH':'rgb(146,190,152)',
		'AH':'rgb(127, 186, 178)',
		'ER':'rgb(103, 152, 173)',
		'AY':'rgb(84,120,165)',
		'UW':'rgb(67,90,158)',
		'UH':'rgb(79,78,137)',
		'OW':'rgb(86,62,117)',
		'AO':'rgb(91,64,100)',
		'AE':'rgb(43, 29, 67)',
		'OY':'rgb(32, 22, 49)',
		'AW':'rgb(21, 13, 32)',
		'AA':'rgb(0, 0, 0)'}
	"""
	if mode == 'dict':
		return vowel_color_map

	elif mode == 'plot':
		z = [vowel[0] for vowel in vowel_colorscale]
		annot = list(vowel_color_map.keys())
		hover = list(vowel_color_map.values())
		fig = ff.create_annotated_heatmap([z], colorscale=vowel_colorscale,font_colors=font_colors,showscale=False,zmin=0, zmax =1,annotation_text=[annot],hovertext=[hover],hovertemplate = "%{hovertext}")
		fig.update_layout(
			autosize=True,
			height=100,
			margin=dict(
				# l=50,
				# r=50,
				b=25,
				t=30,
				# pad=4
			),
		)
		fig.show()


############################################################################
############################################################################
#					PLOTTING
############################################################################
############################################################################


def plot_as_grid(user_input,alignment='right',mode='vowels',delimiter='lines'):
	"""
	This function plots color-coded vowel and stress segments in a grid configuration. It requires 1 param, either a string of orthographic text or a list of phonomial objects. 
	There are also 3 other parameters, alignment, mode, and delimiter.	
	:param user_input: (str) or [phonomial1,phonomial2,...]
	:param alignment: alignment
		right: right-aligns 
		left: left-aligns 
	:param mode: mode
		vowels: vowels
		stress: stress
	:param delimiter: 'lines' or 'sentences'	
	:returns: Plotly Plot

	"""

	if type(user_input) == str:
		ps = phonomial.from_string(user_input, split=delimiter)
		text = user_input 
	elif str(type(user_input[0])) == "<class 'phonomials.phonomialsBase.phonomial'>":
		ps = user_input
		text = '\n'.join([p.ortho for p in ps])
	else:
		print("Error: This function only takes a string of orthographic text or a phonomial set")
		return

	# encode text as phonomial set
	if mode == 'vowels':
		seg_set_lol = [p.get_vowels() for p in ps]
		colorscale = vowel_colorscale
		pad_for_grid_mode = 'IPA'
	if mode == 'stress':
		seg_set_lol = [p.get_vowels(stress=True) for p in ps]
		colorscale = 'Greys'
		pad_for_grid_mode = 'Stress'

	# create annotation text (segments)
	seg_set_matrix = phonomial.pad_for_grid(seg_set_lol, alignment=alignment, mode=pad_for_grid_mode)

	if mode == 'stress': # replace -1s with 10s - done here instead of in phonomials
		for i, row in enumerate(seg_set_matrix):
			for j, item in enumerate(row):
					if item == -1:
						seg_set_matrix[i][j] = 10 

	seg_set_matrix_nums = phonomial.segment_2_num(seg_set_matrix,seg_type=mode)
	word_rel_grid = phonomial.text_2_phonomial_2_wordmatrix(text,alignment=alignment) #lines assumed to be \n seperated
	seg_set_matrix_nums = seg_set_matrix_nums[::-1]
	# annotation matrix (vowels)
	m = seg_set_matrix[::-1]
	# remove XXs from 
	for i, row in enumerate(m):
		for j, item in enumerate(row):
			if item=='XX' or item==10:
				m[i][j] = ''
	# swap with IPA for display
	for i, row in enumerate(m):
		for j, item in enumerate(row):
			if item != '':
				m[i][j] = arpabet_2_ipa_dict()[item]

	# add responsive font from Web App
	fig = ff.create_annotated_heatmap(seg_set_matrix_nums, colorscale=colorscale,font_colors=font_colors,showscale=False,zmin=0, zmax =1,annotation_text=m,hovertext=word_rel_grid,hovertemplate = "%{hovertext}")
	fig.update_layout(
		autosize=True,
		margin=dict(
			b=25,
			t=30,
		),
	)
	fig.show()


def plot_as_MIDI(user_input):
	"""
	This function plots color-coded vowel segments in a MIDI configuration. It requires 1 param, either a string of orthographic text or a single phonomial object. 
		
	:param user_input: str or phonomial
	:returns: Plotly plot

	"""
	# detect if input is a text string or a phonomial
	# add another condition where to allow transforming a pstack into a p 
	# - should just use the function to join them
	if type(user_input) == str:
		# print("Type is str")
		user_input = user_input.replace('\n',' ') # make sure there are no line breaks in the text - otherwise wll fail on collecting word_list
		p = phonomial.from_string(user_input)
		user_str = user_input
	elif str(type(user_input)) == "<class 'phonomials.phonomialsBase.phonomial'>":
		# print("Type is phonomial")
		p = user_input
		user_str = p.ortho.replace('\n',' ')
	else:
		print("Error: This function only takes a string of orthographic text or a single phonomial")
		return
	vs = p.get_vowels()
	# hover
	# make sure there are not lineebreaks in the text 
	word_list = phonomial.text_2_phonomial_2_wordmatrix(user_str)[0] #lines assumed to be \n seperated

	ordered_vowels = list(vowel_digit_map.values())[1:]
	seg_matrix = []
	words_list_matrix = []
	for item in ordered_vowels:
		seg_row = []
		words_row = []
		for j, seg in enumerate(vs):
			if item == seg:
				seg_row.append(seg)
				words_row.append(word_list[j])
			else:
				seg_row.append("")
				words_row.append("")

		seg_matrix.append(seg_row)
		words_list_matrix.append(words_row)

	# z
	seg_matrix_nums = phonomial.segment_2_num(seg_matrix,seg_type='vowels')

	# hover - turn vector to matrix with annotations transform


	# transform stuff for plotting (z)
	seg_matrix_nums = seg_matrix_nums[::-1]
	# annotation matrix (vowels)
	m = seg_matrix[::-1]
	# flip hover
	words_list_matrix = words_list_matrix[::-1]


	# remove XXs from 
	for i, row in enumerate(m):
		for j, item in enumerate(row):
			if item=='XX' or item==10:
				m[i][j] = ''


	fig = ff.create_annotated_heatmap(seg_matrix_nums, colorscale=vowel_colorscale,font_colors=font_colors,showscale=False,zmin=0, zmax =1,annotation_text=m,hovertext=words_list_matrix,hovertemplate = "%{hovertext}")
	fig.update_layout(
		autosize=True,
		# title="Sound Item Grid",
		height=200,
		margin=dict(
			# l=50,
			# r=50,
			b=25,
			t=30,
			# pad=4
		),
	)

	# Make text size smaller
	for i in range(len(fig.layout.annotations)):
		fig.layout.annotations[i].font.size = 1

	fig.show()




def prhyme_heatmap(z_sym,show_symbols=True,color=13): 

    # define matricies
    
#     if show_symbols:
#         symbols = [["Vowels", "", "", ".2", "", ".3","", ".2", ""],
#                    ["Stress", "", "", ".2", "", ".3","", ".2", ""],
#                    ["Cons","", ".8", "", ".4", "","0.5", "", "0.7"]]
#     else:    
#         symbols = [["", "", "", "","", "", "", "", "", ""],
#                     ["", "", "", "","", "", "", "", "", ""],          
#                     ["", "", "", "","", "", "", "", "", ""]]        

    z = z_sym[0]
    symbols = z_sym[1]
    x = z_sym[2]



    # define layout
    colorscale = [[0, '#f4cccc'], [0.99999, '#cc0000'], [1, 'white']]


    colorscale = [[0, 'darkgreen'], [0.99999, 'white'], [1, 'white']]


    font_colors = ['black', 'white']
    fig = ff.create_annotated_heatmap(z,x=x,y=["<b>Stress","<b>Vowels","<b>Cons"],font_colors=font_colors, colorscale=colors[14],annotation_text=symbols,showscale=True)
    fig.update_layout(
        autosize=False,
        width=950,
        height=350,
        margin=dict(
            l=0,
            r=50,
            b=100,
            t=100,
        ),
    )

    fig.show()



def plot_phrase_set_summary(ps, take_n_columns,alignment):
    """
    This function plots color-coded vowel segments in a MIDI configuration. It requires 1 param, either a string of orthographic text or a single phonomial object. 
    If you pass it text is will convert it into a single phonomials. 
    	
    :param user_input: str or phonomial
    :returns: Plotly plot
    """

    v_grid = get_rect_grid(ps,take_n_columns=take_n_columns,alignment=alignment,mode="vowels")
    s_grid = get_rect_grid(ps,take_n_columns=take_n_columns,alignment=alignment,mode="stress")
    c_grid = get_rect_grid(ps,take_n_columns=take_n_columns+1,alignment=alignment,mode="cons")

    v_pos_ents = get_col_entropy(v_grid[1])
    s_pos_ents = get_col_entropy(s_grid[1])
    c_pos_ents = get_col_entropy(c_grid[1])


    v_pos_ents = [round(num,2) for num in v_pos_ents]
    s_pos_ents = [round(num,2) for num in s_pos_ents]
    c_pos_ents = [round(num,2) for num in c_pos_ents]

    x = build_heatmap_matricies(v_pos_ents,s_pos_ents,c_pos_ents)
    prhyme_heatmap(x)
    
    return x



############################################################################
############################################################################
#					ANALYTICS - General
############################################################################
############################################################################





def markov_model(stream, model_order):
  model, stats = defaultdict(Counter), Counter()
  circular_buffer = deque(maxlen = model_order)
  for token in stream:
    prefix = tuple(circular_buffer)
    circular_buffer.append(token)
    if len(prefix) == model_order:
      stats[prefix] += 1
      model[prefix][token] += 1
  return model, stats


def entropy(stats, normalization_factor):
  return -sum(proba / normalization_factor * math.log2(proba / normalization_factor) for proba in stats.values())
 
def entropy_rate(model, stats):
  try:
    return sum(stats[prefix] * entropy(model[prefix], stats[prefix]) for prefix in stats) / sum(stats.values())
  except: 
    return "NaN"

def n_grams(items, n_orders=2):
	"""
	Take a number of n-gram orders and a list of items, and returns list of n dictionaries. Each dictionary will contain the freequency counts for each size n-gram up to order n
	
	:param items: list of items  e.g. ['A','A','B','C','D','E','E','F','G','H']
	:param n_orders: int (should not be longer than numbers of items)

	:returns: list of dictionaries
	:raises keyError: raises an exception

	This has a bug: it doesn't seem to to including the very last element in any count 

	>>> phonesse.n_grams(3,['A','A','B','A','C','A','A','A','B','A','C'])[1]
	[Counter({('A',): 7, ('B',): 2, ('C',): 1}),
	Counter({('A', 'A'): 3,
			('A', 'B'): 2,
			('B', 'A'): 2,
			('A', 'C'): 1,
			('C', 'A'): 1}),
	Counter({('A', 'A', 'B'): 2,
			('A', 'B', 'A'): 2,
			('B', 'A', 'C'): 1,
			('A', 'C', 'A'): 1,
			('C', 'A', 'A'): 1,
			('A', 'A', 'A'): 1})]
	"""	
	models = []
	ret_stats = []
	for i in range(1,n_orders+1):    
		if type(items) == list:
			model, stats = markov_model(items, i)
		else: # assumes a files 
			model, stats = markov_model(words(items), i)
		# entropies.append(entropy_rate(model, stats))
		models.append(model)
		ret_stats.append(stats)
	# return models,ret_stats #, entropies
	return ret_stats





############################################################################
############################################################################
#					ANALYTICS - Prhyme Phrasses
############################################################################
############################################################################


from scipy import stats
import statistics

colors= ['Blackbody','Bluered','Blues','Earth','Electric','Greens',
'Greys',
'Hot',
'Jet',
'Picnic',
'Portland',
'Rainbow',
'RdBu',
'Bluered',
'Reds',
'Viridis',
'YlGnBu',
'YlOrRd']
def counts_2_prob(counter):
    new_dict = {}
    denom = sum(counter.values())
    for key in counter:
        new_dict[key] = counter[key]/denom
    return new_dict  

def position_entropy(counter):
    probs = counts_2_prob(counter)
    probs_list = list(probs.values())
    position_entropy = stats.entropy(probs_list)
    return position_entropy
    
def get_col_entropy(colsT):
    column_entropys = [] 
    for column in colsT:
        column_freqs = Counter(column)
        probs = counts_2_prob(column_freqs)
        entropy = position_entropy(column_freqs)
        column_entropys.append(entropy)
    return column_entropys

def get_rect_grid(ps,mode="vowels",alignment="right",take_n_columns=1):
    # make this robust to getting empty phonomials, 
    # still fails if empty line at end of rhyme phrase data
    
    if mode=="vowels":
        p_stack = [p.get_vowels() for p in ps]
        if alignment == "right": vowel_grid = [p[-take_n_columns:] for p in p_stack]
        if alignment == "left":  vowel_grid = [p[:take_n_columns] for p in p_stack]
        column_listsT = list(map(list, zip(*vowel_grid)))

        return vowel_grid, column_listsT

    if mode=="stress":
        p_stack = [p.get_vowels(stress=True) for p in ps]
        if alignment == "right": vowel_grid = [p[-take_n_columns:] for p in p_stack]
        if alignment == "left":  vowel_grid = [p[:take_n_columns] for p in p_stack]
        column_listsT = list(map(list, zip(*vowel_grid)))

        return vowel_grid, column_listsT    
    
    if mode=="cons":
        p_stack = [p.get_cons(removeDelims="yes") for p in ps]
        if alignment == "right": cons_grid = [p[-take_n_columns:] for p in p_stack]
        if alignment == "left":  cons_grid = [p[:take_n_columns] for p in p_stack]
        column_listsT = list(map(list, zip(*cons_grid)))
        column_listsT = [flatten(column) for column in column_listsT]
        return cons_grid, column_listsT



# this can take any single values cvc mapped entries
def build_heatmap_matricies(v_pos_ents,s_pos_ents,c_pos_ents):

    empty_grid_cell_color = 0 # 4.65

    v = []
    v_labeled = []
    for i, item in enumerate(v_pos_ents):
        v.append(empty_grid_cell_color)
        v.append(item)
        v_labeled.append("")
        v_labeled.append(str(item))
    v.append(empty_grid_cell_color)
    v_labeled.append("")
    
    s = []
    s_labeled = []
    for i, item in enumerate(s_pos_ents):
        s.append(empty_grid_cell_color)
        s.append(item)
        s_labeled.append("")
        s_labeled.append(str(item))
    s.append(empty_grid_cell_color)
    s_labeled.append("")


    c = []
    c_labeled = []
    for i, item in enumerate(c_pos_ents):
        c.append(item)
        c.append(empty_grid_cell_color)
        c_labeled.append(str(item))
        c_labeled.append("")
    c.pop()    
    c_labeled.pop()    

    # should be optional based on flag for entropy vs MC display
    v.append(empty_grid_cell_color) #filler
    v.append(3.91)
    s.append(empty_grid_cell_color) #filler
    s.append(1.58)
    c.append(empty_grid_cell_color) #filler
    c.append(4.64)
    
    v_labeled.append("")
    v_labeled.append("3.91")
    s_labeled.append("")
    s_labeled.append("1.58")
    c_labeled.append("")
    c_labeled.append("4.64")
    
    x = []
    x_c_counter = 1
    x_s_counter = 1
    for i,item in enumerate(range(1,len(v_labeled)-2+1)):
        if i % 2 != 0: # is nucleus pos
            x.append("<b>V"+str(x_c_counter))
            x_c_counter+=1
        else: # is cons position
            x.append("C"+str(x_s_counter))
            x_s_counter+=1
    x.append("") # gap spacer
    x.append("<b>Max Ent") # max ent column
    return [[s,v,c],[s_labeled,v_labeled,c_labeled],x]



############################################################################
############################################################################
#					PHONOMIAL CLASS
############################################################################
############################################################################


class phonomial:
	"""
	A phonomial a simple class used to encode ARPABET representation in a form more conducive to modular coding than the simple string repressentation. 

	
	:param param1: phonomial
	:returns: int of block or syllable count
	:raises keyError: raises an exception

	>>> type(phonomial(1))
	<type 'instance'>

	"""	
	#print "TESTING!"

	def __init__(self, syll_length):
		"""
		A phonomial contains data on a given set of consecutive sounds 
			-A phonomial usually represents a sentence / row / line of information

			0 should create a random length empty phonomial. up to 5

		>>> type(phonomial(1))
		<type 'instance'>

		"""

		# EMPTY PHONOMIAL INIT

		# Init an Empty Phonomial by Syllable Length
		# again, abstract this out to a drived function of the blocks or syllables
		syll_count = syll_length
		block_count = syll_count*2+1	# this converts syllable count to CVBlock count

		# Attribute structures derived from Syllable Length
		# self.stress = [None]*self.syll_count # Derived from one of the 


		self.blocks = [ [] for i in range(block_count) ] # CV Alternating Block
		self.syllables = [ [[]]*3 for i in range(syll_count) ]



	######################## PHONOMIAL USER FUNCTIONS ########################

	# simplify this to only run as a string conversion, abstract the rest into a from_word_list phonomial creation method
	@classmethod
	def from_string(cls,ortho,altPronSave='no',split="none", ignore_missing=False,replace_missing=True, save_missing_to=False):
		""" 
		Takes a string of text and returns a phonomial
		can return multiple phonomials by specifing a delimiter


		:param ortho: str (orthographic text)
		:param altPronSave: coming soon 
		:param split: str ('lines','sents')
		:param ignore_missing: bool (if true, orthographic text will show up with returned sounds where sounds are not available)
		:param replace_missing: bool (If true, uses neural network to predict pron. of unknown word)
		:param save_missing_to: bool coming soon

		:returns: one or more phonomial objects

		>>> type(phonomial.from_string("this"))
		<type 'instance'>
		"""

		# Sanitize: remove all but letters, numbers, and spaces
		if split == "none":
			ortho = list([i for i in ortho if i.isalpha() or i.isnumeric() or i==' ']) 
			ortho = "".join(ortho)  

		words = phonomial.phrase_2_words(ortho)
		# 
		cmu_word_phones = 	phonomial.get_phrase_pronunciations_compressed(words, replace_missing=replace_missing)									# Word Phones Compressed
		cmuS_word_phones =	phonomial.get_phrase_pronunciations_compressed(words, mode='syllables', replace_missing=replace_missing)									# Word Phones Compressed
		

		# we need a more thorough system of cleansing and checking / dealing with errors in the stream
		if ignore_missing:
			cmu_word_phones = [item for item in cmu_word_phones if not item[0][0]=="*"]
			cmuS_word_phones = [item for item in cmuS_word_phones if not item[0][0]=="*"]
		if save_missing_to != False: 
			missing_cmu_word_phones = [item for item in cmu_word_phones if item[0][0]=="*"]
			missing_cmuS_word_phones = [item for item in cmuS_word_phones if item[0][0]=="*"]




			script_location = Path(__file__).absolute().parent
			file_location = save_missing_to
			clean_words = [entry[0].strip("*").upper() for entry in missing_cmu_word_phones]
			print('Saving',len(clean_words), 'Missing Words To:',file_location)
			print(clean_words)

			with open(file_location, 'a+') as file:
				file.write('\n'.join(clean_words)+'\n')
			file.close()



		if split == 'none': # don't expand alts, expand default pron

			cmu_word_phones_permutation = []
			for word in cmu_word_phones:
				cmu_word_phones_permutation.append(word[0])
			cmuS_word_phones_permutation = []
			for word in cmuS_word_phones:
				cmuS_word_phones_permutation.append(word[0])

			# NOTE: Fix data formats: issue - have to feed from_phrase_pronunciations_to_phone_items 
			# last temp data wrapped inside a list for function to work, remove list dependency intonal to function.

			cmu_phone_item_list = phonomial.from_phrase_pronunciations_to_phone_items([cmu_word_phones_permutation])
			cmuS_phone_item_list = phonomial.from_phrase_pronunciations_to_phone_items([cmuS_word_phones_permutation])	

			blocks = phonomial.from_phrase_phone_items(cmu_phone_item_list[0])
			syllables = phonomial.from_phrase_phone_items(cmuS_phone_item_list[0],mode='syllables')

			# Init Phonomial
			p = cls(len(syllables))

			# Override Constructor assignments.
			p.blocks = blocks
			p.syllables = syllables

			p.cmu_word_prons = cmu_word_phones
			p.cmuS_word_prons = cmuS_word_phones


			p.ortho = ortho

			return p
		# Multi Phonomial recursion functionality
		else:
			ps = []	
	
			# Prep Text
			if split=="sents":
				sents = sent_tokenize(ortho)
				print(sents)
			elif split =="lines":
				sents = ortho.split("\n")

			for i,sent in enumerate(sents):
				# clean the texts after spliting them by delimiter
				sent = list([i for i in phonomial.from_string(sent).ortho if i.isalpha() or i.isnumeric() or i==' ']) 
				sent = "".join(sent)  
				ps.append(phonomial.from_string(sent))
			
			return ps 


	def __len__(p):			
		"""
		Take a phonomial of some kind and return the number of blocks or syllables, depending.
		
		:param param1: phonomial
		:returns: int of block or syllable count
		:raises keyError: raises an exception

		>>> len(phonomial.from_string("this is important"))
		4

		"""
	
		#careful with namespaces (make shorthard for phonomial - maybe)
		# this acts as a simple mask
		# this is stupid, and real use cases should employ normal len dunder, this is only a catch all for syllable usage (dodn't like it)
		return len(p.syllables)





	def generate_blocks_permutations(p):
		# add the Arpabet string versions of permutations to attribute blocks_permutations
		p.add_phrase_pron_permutations(mode='blocks')
		for permutation in range(len(p.blocks_permutations)):
			tempBlock = p.blocks_permutations[permutation]
			tempBlock = phonomial.from_phrase_pronunciations_to_phone_items([tempBlock])
			tempBlock = phonomial.from_phrase_phone_items(tempBlock[0])
			# reattach the final product to whence it came
			p.blocks_permutations[permutation] = tempBlock
	

	def generate_syllables_permutations(p):
		# add the Arpabet string versions of permutations to attribute blocks_permutations
		p.add_phrase_pron_permutations(mode='syllables')
		for permutation in range(len(p.syllables_permutations)):
			tempSyll = p.syllables_permutations[permutation]
			tempSyll = phonomial.from_phrase_pronunciations_to_phone_items([tempSyll])
			tempSyll = phonomial.from_phrase_phone_items(tempSyll[0],mode='syllables')
			# reattach the final product to whence it came
			p.syllables_permutations[permutation] = tempSyll

	# maybe change to concat
	@staticmethod
	def join(ps, delim='\n'):
		# Join (serially concatanate) a list of phonomials
		"""		
		:param param1: List of Phonomial objects
		:returns: single Phonomial object  
		:raises keyError: raises an exception
		"""	    


		# delim options are \n and . for other side of reconstructing either line or sentence seperated data
		
		# SETUP EMPTY NEW PHONOMIAL OBJECT
		new = phonomial(0)
		new.ortho = ''
		new.blocks = []
		new.syllables = []
		new.cmu_word_prons = []
		new.cmuS_word_prons = []
		
		# LOOP PHONOMIAL STACK AND POPULATE NEW PHONOMIAL WITH THEIR CONTENTS

		for i, p in enumerate(ps):
			new.ortho+=p.ortho
			new.ortho+=delim

			# Block join note
			# we always join some CVC with some CVCVC, which mean we always \
			# have to internally thread the contents of bounding Cs so not to throw
			# off the block CVC rotation dependency
			for j, block in enumerate(p.blocks):
				# We enumerate both loops here because we 
				# need to know when it is the first phonomial
				if j == 0 and i != 0:
					for item in block:
						new.blocks[-1].append(item)
				else:
					new.blocks.append(block)
			for syllable in p.syllables:
				new.syllables.append(syllable)
			for pron in p.cmu_word_prons:
				new.cmu_word_prons.append(pron)
			for pronS in p.cmuS_word_prons:
				new.cmuS_word_prons.append(pronS)
	
		return new



	# USER METHOD
	@staticmethod
	def phoneme_distinctive_feats(mode='plot'):
		"""
		Flatten a 2D list into a 1D list - Use recursion on more than 2D lists if needed.
		
		:param param1: phonomial
		:returns: int of block or syllable count
		:raises keyError: raises an exception

		>>> phonomial.flatten([[1,2],[3,4]])
		[1, 2, 3, 4]

		"""
		data_matrix = [['ARPABET', 'IPA', 'Example'],
						['IH', 2000, 'Example'],
						['IY', 2000, 'Example'],
						['United States', 2000, 'Example'],
						['Canada', 2000, 'Example']]




	######################## PHONOMIAL SUPPORT FUNCTIONS ########################

	@staticmethod
	def phrase_2_words(sent):
		"""
		Filter and extract word from a sentence,
		
		:param param1: phonomial
		:returns: int of block or syllable count
		:raises keyError: raises an exception

		>>> phonomial.phrase_2_words("This is Dr. door, don't deny it.")
		['This', 'is', 'Dr', 'door', "don't", 'deny', 'it']
		>>> phonomial.phrase_2_words("blacksmith launchpad")
		['blacksmith', 'launchpad']


		"""
		# cleanse with decode and then encode
		# sent.decode('utf-8').encode('utf-8')
		# tthi should take a dictionary and replace all the values i'ded as possible unicode with asci acceptable for use with cmu
		# write and efficient loop to take care of this cleaning with a replace dict



		sent = sent.replace("\xe2\x80\x99","'")





		sent_words = WhitespaceTokenizer().tokenize(sent)
		sent_words_clean = []
		for word in sent_words:
			# this wrongly removes stuff from words like like 'DR.'
			# also, could store the original sentences before sanitization for client comparison and fixes.
			sent_words_clean.append(word.rstrip('.?!@#$%^&*(),/\'\":;-,\'').lstrip('.?!@#$%^&*(),/\'\":;-').rstrip('.?!@#$%^&*(),/\'\":;-,\'').lstrip('.?!@#$%^&*(),/\'\":;-'))

		return sent_words_clean


	# should be a normal class method - the list form
	@staticmethod
	def from_phrases_phone_items(phrases_phone_items,mode='blocks'):
		"""
		Take a list of list of tuples of word pronunciations and return a list of list of phonomials

		from_phrase_pronunciation
		text: 'this is the end'
		"""
		phonomials_out = []
		for phrase in phrases_phone_items:
			phonomials_out.append(phonomial.from_phrase_phone_items(phrase,mode))
		return phonomials_out


	@staticmethod
	def from_phrase_phone_items(phrase_pronunciation_permutation, mode='blocks'):
		"""
		Take a list of tuples of word pronunciations and return 

		text: 'this is the end'
		"""
		# missing empty consonant positions - fails on empty cons without syll mark
		if mode=="blocks":
			cv = []
			# stress = []
			temp_con_block = []
			for i, item in enumerate(phrase_pronunciation_permutation):
				# if item is a vowel/nucleus
				if item[-1] == '0' or item[-1] == '1' or item[-1] == '2':
					# append trailing Consonant Cluster
					cv.append(temp_con_block)
					# reset the cons cluster block to empty.
					temp_con_block = []
					# append new vowel and stress items
					cv.append([item[:2],int(item[-1])])
				# if item is a consonant or word boundary
				else:
					temp_con_block.append(item)
				# terminal item condition upon which to append final consonant cluster.
				if i == len(phrase_pronunciation_permutation)-1:
					cv.append(temp_con_block)
			return cv

		if mode=="syllables":
			# break the 3 for-loops into smaller support functions
			split_index_type = [] # ". or "-""
			syllables = []
			split_index = []
			messy_syllables = []
			stress = []
			# both '.'' and '-' count as syllable delimiters, '-' is also a word delimiter 

			for i, item in enumerate(phrase_pronunciation_permutation):
				if item == '.' or item == '-':
					split_index.append(i)
					split_index_type.append(item)

			last_index = 0
			for i, item in enumerate(split_index):
				# remove '-' delimiter, only leaving '.' word delimiter inside syllables.  
				if split_index_type[i] == '-':
					messy_syllables.append(phrase_pronunciation_permutation[last_index:item])
				else:
					messy_syllables.append(phrase_pronunciation_permutation[last_index:item+1])

		
				last_index = item + 1
		
			# temrinal condition
			# reason: indexing is based on '.', '-' delimiters
			# we dont get them at the start of a phonomial, 
			# and sometimes we do not get one at the end, which throws off the trailing indexing. 
			# here we simply return any elements after the last index if those elements are not nothing. 
			if phrase_pronunciation_permutation[last_index:] != []:
				# print phrase[last_index:]
				syllables.append(phrase_pronunciation_permutation[last_index:])

			# CLEAN UP LOOP
			for messy_syllable in messy_syllables:
				# are these variables used?
				syllable = []
				onset = []
				nucleus = []
				coda = []
				stress = []
				for i, item in enumerate(messy_syllable):
					if item[-1] == '0' or item[-1] == '1' or item[-1] == '2':
						# print
						# print "onset:",messy_syllable[:i]
						# print "nucle:",[messy_syllable[i][:-1],int(messy_syllable[i][-1])]
						# print "codas:",messy_syllable[i+1:]
						syllable.append(messy_syllable[:i])
						syllable.append([messy_syllable[i][:-1],int(messy_syllable[i][-1])])
						syllable.append(messy_syllable[i+1:])
				syllables.append(syllable)
			return syllables

	@staticmethod	
	def from_phrase_pronunciations_to_phone_items(phrase_pronunciations_permutations):
		"""
		Take a list of phrase_pronunciations_permutations and return a list of phone items 		
		
		"""
		phrases_phon_items = []
		for phrase in phrase_pronunciations_permutations:
			phone_items_out = []
			for word in phrase:
				phones_items = word.split(" ")
				phones_items.append('.')
				# print phones_items
				for item in phones_items:
					phone_items_out.append(item)
			phrases_phon_items.append(phone_items_out)
		return phrases_phon_items


	@staticmethod	
	def get_word_pronunciations(word, mode='phones',ignore_missing=True, replace_missing=True):
		""" 
		Takes a word (orthography) and returns a list of strings, 
		each a possible pronunciation of the word.
	
		:param param1: word (text)
		:param param2: mode (text)
			phones:	Sequence of phones in a string
			sylls: Sequence of phones in a string, syllables delimited by '-' 
		:returns: list of word pronunciations
		:raises keyError: raises an exception (FINISH)
		"""
		if mode == 'phones':
			pron = pronouncing.phones_for_word(word.lower())
			if pron != []: # if the word has a cmu (phonetic) entry
				return pron
			elif replace_missing == True:
				# this is super slow, loading every time, move the dict call up as many lvls as poss
				phone_items = g2p(word) # only works on this cause i am feeding it words only. 
				return [" ".join(phone_items)] # returning in list - akin to list of list for possible multi-pron cmuu pull
			else:
				if ignore_missing:
					return ['***'+word+'***'] # returning an emtpy list, lets see you it proigates
				else:
					return [word.lower()]
		if mode == 'syllables':
			try:			 
				return get_cmu.cmuSyllDict[word.upper()]
			except:
				if ignore_missing:
					return ['***'+word+'***']
				else:
					return [word.lower()]

	@staticmethod
	def get_phrase_pronunciations_compressed(words, mode='phones',from_data="string", replace_missing=True):
		""" 
		Takes a string of text and returns a list of lists of 
		all possible  pronunciations of each word in the phrase.
		
		:param param1: phrase (text)
			words delimiter by spaces
			lines delimited by '.' or '\\n'
		:param param2: mode (text)
			phones:	Sequence of phones in a string
			sylls: Sequence of phones in a string, syllables delimited by '-' 
		:returns: list of list of possible pronunciations of the phrase
		:raises keyError: raises an exception

		>>> phonomial.phrase_2_words("the blacksmith")
		['the', 'blacksmith']
		>>> phonomial.get_phrase_pronunciations_compressed(['the', 'blacksmith'])
		[[u'DH AH0', u'DH AH1', u'DH IY0'], [u'B L AE1 K S M IH2 TH']]
		>>> phonomial.get_phrase_pronunciations_compressed(['the', 'blacksmith'], mode='syllables')
		[[u'DH AH0', u'DH AH1', u'DH IY0'], [u'B L AE1 K - S M IH2 TH']]
		"""

		# change and clean out this from_data function, its paired with loading in text in the from_text from ortho
		if mode == 'phones':
			word_possibities = []
			for word in words:
				word_possibities.append(phonomial.get_word_pronunciations(word, replace_missing=replace_missing))
			return word_possibities
		if mode == 'syllables':
			word_possibities = []
			for word in words:
				word_possibities.append(phonomial.get_word_pronunciations(word,mode='syllables', replace_missing=replace_missing))
			return word_possibities

	@staticmethod
	def get_phrases_pronunciations_compressed(phrases, mode='phones'):
		""" 
		Takes a string of text and returns a list of lists of 
		all possible  pronunciations of each word in the phrase.
		
		:param param1: phrase (text)
			words delimiter by spaces
			lines delimited by '.' or '\\n'
		:param param2: mode (text)
			phones:	Sequence of phones in a string
			sylls: Sequence of phones in a string, syllables delimited by '-' 
		:returns: list of list of possible pronunciations of the phrase
		:raises keyError: raises an exception
		"""
		phrases_compressed = []
		# DRY this crap
		if mode == 'phones':
			for phrase in phrases:
				phrases_compressed.append(phonomial.get_phrase_pronunciations_compressed(phrase))
		if mode == 'syllables':
			for phrase in phrases:
				phrases_compressed.append(phonomial.get_phrase_pronunciations_compressed(phrase, mode='syllables'))
		return phrases_compressed

	def add_phrase_pron_permutations(self, mode='blocks'):
		"""
		Add attribute to a phonomial - calculate alt pronunciations
		
		:param param1: phonomial
		:returns: int of block or syllable count
		:raises keyError: raises an exception
		"""
		if mode == 'blocks':
			self.blocks_permutations = phonomial.get_phrase_pron_permutations(self.cmu_word_prons)
		else:
			self.syllables_permutations = phonomial.get_phrase_pron_permutations(self.cmuS_word_prons)
		# return self

	@staticmethod
	def get_phrase_pron_permutations(word_possibities):
		""" 
		Takes a string of text and returns a list of lists of 
		all possible permutations of pronunciations of the phrase.

		**make one of these for phonomials as well, this should be considered preprocessing**

		This function is not run by default. There should be flags to have this added at the from_ level, as well as examples of adding this attr to a phonomial
	
		:param param1: word (text)
			text:
			words delimiter by spaces
		:param param2: mode (text)
			phones:	Sequence of phones in a string
				EX: blacksmith...
			sylls: Sequence of phones in a string, syllables delimited by '-' 
				EX: blacksmith...
		:returns: list of list of possible pronunciations of the phrase
		:raises keyError: raises an exception

		>>> phonomial.phrase_2_words("the blacksmith")
		['the', 'blacksmith']
		>>> phonomial.get_phrase_pronunciations_compressed(['the', 'blacksmith'])
		[[u'DH AH0', u'DH AH1', u'DH IY0'], [u'B L AE1 K S M IH2 TH']]
		>>> phonomial.get_phrase_pronunciations_compressed(['the', 'blacksmith'], mode='syllables')
		[[u'DH AH0', u'DH AH1', u'DH IY0'], [u'B L AE1 K - S M IH2 TH']]
		>>> phonomial.get_phrase_pron_permutations([[u'DH AH0', u'DH AH1', u'DH IY0'], [u'B L AE1 K - S M IH2 TH']])
		[(u'DH AH0', u'B L AE1 K - S M IH2 TH'), (u'DH AH1', u'B L AE1 K - S M IH2 TH'), (u'DH IY0', u'B L AE1 K - S M IH2 TH')]
		"""



		pron_paths = list(itertools.product(*word_possibities))
		return pron_paths
	

	@staticmethod
	def get_phrases_pron_permutations(phrases_compressed):
		""" 
		Takes a string of text and returns a list of lists of 
		all possible permutations of pronunciations of the phrase.

		**make one of these for phonomials as well, this should be considered preprocessing**

	
		:param param1: word (text)
			text:
			words delimiter by spaces
		:param param2: mode (text)
			phones:	Sequence of phones in a string
				EX: blacksmith...
			sylls: Sequence of phones in a string, syllables delimited by '-' 
				EX: blacksmith...
		:returns: list of list of possible pronunciations of the phrase
		:raises keyError: raises an exception
		"""

		# use get_phrase_pron_permutations() in here
		phrases_perms = []
		for phrase in phrases_compressed:
			pron_paths = list(itertools.product(*phrase))
			phrases_perms.append(pron_paths)
		return phrases_perms
	



	######################## SUPPORT FUNCTIONS ########################
	@staticmethod
	def segment_2_num(lol_padded,seg_type='vowels'):
		v_grid_nums = []
		for row in lol_padded:
			v_grid_num_row = []
			if seg_type == 'vowels':
				for item in row:
					if item == '': # patch for init_intermediate
						item = 'XX'
					v_grid_num_row.append(vowel_digit_map_inv[item])
			if seg_type == 'stress':
				for item in row:

					v_grid_num_row.append(stress_digit_map_inv[item])
			v_grid_nums.append(v_grid_num_row)
		return v_grid_nums
		
		
	# convert to extract ortho text from given phonomial stack
	@staticmethod
	def text_2_phonomial_2_wordmatrix(text,alignment='left'):
		vowel_matrix = []
		word_rel_matrix = []
		lines = text.split('\n')

		for line in lines:
			line_vowels = []
			line_words = []
			words = line.split(' ')

			for word in words:
				p = phonomial.from_string(word)
				vowels = p.get_vowels()
				line_vowels.extend(vowels)
				line_words.extend(['<b>'+p.ortho+'</b> '+str(i+1)+'/'+str(len(vowels)) +'<br>'+vowels[i] for i, rep in enumerate(vowels)])
			vowel_matrix.append(line_vowels)
			word_rel_matrix.append(line_words)


		padded_word_rel_matrix = phonomial.pad_for_grid(word_rel_matrix,alignment=alignment)
		ps = phonomial.from_string(text,split='lines')

		ret = ""
		for p in ps:
			ret += " ".join([prons[0] for prons in p.cmu_word_prons])+"\n"
		ret = ret.rstrip("\n")




		return padded_word_rel_matrix[::-1]



	@staticmethod
	def pad_for_grid(lol, alignment='right', mode="IPA"):
		"""
		Flatten a 2D list into a 1D list - Use recursion on more than 2D lists if needed.
		
		:param param1: phonomial
		:returns: int of block or syllable count
		:raises keyError: raises an exception

		>>> phonomial.flatten([[1,2],[3,4]])
		[1, 2, 3, 4]

		"""
		ps = copy.deepcopy(lol)
		# detect mode and set automatically
		try:
			if type(ps[0][0]) == int:
				mode="numbers"
		except:
			pass

		maxlen = len(max(ps,key=len))

		if alignment == "right":
			for row in ps:
				if len(row) < maxlen:
					for gap in range(maxlen-len(row)):
						if mode == "IPA":
							row.insert(0,"XX")
						else: 
							row.insert(0,-1)

		elif alignment == "left":
			for row in ps:
				if len(row) < maxlen:
					for gap in range(maxlen-len(row)):
						if mode == "IPA":
							row.append("XX")
						else: 
							row.append(-1)
		return ps


	@staticmethod
	def map_to_nums(list_list_vowels):
	
		vowel_digit_map = {
		-1:'XX', 0:'AA', 1:'AE', 2:'AH', 3:'AO', 4:'AW', 5:'AY', 6:'EH', 7:'ER', 8:'EY', 9:'IH', 10:'IY', 11:'OW', 12:'OY', 13:'UH', 14:'UW'}
		# Flip keys with values
		vowel_digit_map_inverse = {v: k for k, v in vowel_digit_map.items()}
	
		ps_nums = []
		
		for row in list_list_vowels:
			n_row = []
			for item in row:
				n_row.append(vowel_digit_map_inverse[item])
			ps_nums.append(n_row)
		return ps_nums



	######################## REMOVE ########################
	@staticmethod
	def get_serial_constituents(text, mode = 'vowel_coda'):
		"""Possible constituents modes for this are listed below

			Takes raw text as input -> a list of constituents concats 

			vowel_coda
			vowel_stress
			vowel_stress_coda
		"""
		# this is needed for all these parts (accidental atm - some (cons only or cons-stress - may violate)
		p = phonomial.from_string(text)
		vs= p.get_vowels()
		ss= p.get_vowels(stress=True)
		cs= p.get_cons(removeDelims='yes')
		
		if mode == "vowel_coda":
			vowel_coda = []
			for loc, i in enumerate(vs):
			    try:
			        v_c = vs[loc]+cs[loc+1][0]
			        vowel_coda.append(v_c)
			    except:
			        v_c = vs[loc]
			        vowel_coda.append(v_c)
			return vowel_coda
		if mode == "vowel_stress":
			vowel_stress = [vs[loc]+str(ss[loc]) for loc,i in enumerate(vs)] 
			return vowel_stress
		if mode == "vowel_stress_coda":
			vowel_stress_coda = []
			for loc, i in enumerate(vs):
			    try:
			        word = [vs[loc]+str(ss[loc])+cs[loc+1][0]]
			        vowel_stress_coda.append(word[0])
			    except:
			        # exception because we were hitting empty lists [] - empty cons clusters - 
			        # and trying to take the [][0] of it.
			        word = [vs[loc]+str(ss[loc])]
			        vowel_stress_coda.append(word[0])

			return vowel_stress_coda
		if mode == "all_phones":
			all_phones = " ".join([word[0] for word in p.cmu_word_prons]).split(" ")
			return all_phones

		# The follow is designed for my conviencince only, 
		# outputting strings of wanted parts watching string for preindexing
		if mode == "stress":
			return ''.join([str(s) for s in ss])
		if mode == "vowels":
			return ''.join(vs)
		if mode == "cons":
			return ''.join(flatten(cs))
		if mode == "rime":
			return " ".join([vs[-1]," ".join(cs[-1])]).rstrip(" ")
		if mode == "rime_min":
			try:
				#try to take only the first in last cons clusters
				return " ".join([vs[-1]," ".join(cs[-1][0])]).rstrip(" ")
				# will except if out of list range, return the vowel joined empty cons cluster
			except:
				return " ".join([vs[-1]," ".join(cs[-1])]).rstrip(" ")

		if mode == "masculine":
			if ss[-1] == 1:
				return " ".join([vs[-1],str(ss[-1])," ".join(cs[-1])]).rstrip(" ")

		if mode == "feminine":
			try:
				if ss[-2] == 1 and ss[-1] == 0:
					return " ".join([vs[-2],str(ss[-2]),vs[-1],str(ss[-1])," ".join(cs[-1])]).rstrip(" ")
			except:
				pass
		if mode == "dactylic":
			try:
				if ss[-3] == 1 and ss[-2] == 0 and ss[-1] == 0:

					return " ".join([vs[-3],str(ss[-3]),vs[-2],str(ss[-2]),vs[-1],str(ss[-1])," ".join(cs[-1])]).rstrip(" ")
			except:
				pass

		if mode == "alliteration":
			try:
				return " ".join(cs[0]).rstrip(" ")
			except:
				pass

	######################## TO REMOVE ########################

	# change p to self
	def get_cons(p,mode='blocks',flat="no",removeDelims="no", placeholder="no"):
		"""
		Extract Consonants from a phonomial

		Note that we can either run this func live if we need it, or we can run in in a loop, adding it as an attr to each phonomial
		
		:param param1: phonomial
		:returns: int of block or syllable count
		:raises keyError: raises an exception
		"""
		# iterate across only vowel blocks in a phonomial
		if mode=='blocks':
			# move cons instansiation outside if
			cons = []
			for i, block in enumerate(p.blocks):
				if i % 2 != 1:
					cons.append(block)

		elif mode=='onsets-codas':
			onsets_codas = []
			for i, syll in enumerate(p.syllables):
				onsets_codas.append([syll[0],syll[2]])
			
			# Because Syllable struct is a 3 - lvl list, and preserving 3 lvls is not useful enough (too similar to original p.syllable form), 
			# we flatten once to get it to a 2 lvl list of lists for the remainder of processing in this function.
			# arised partially b/c of onsets_codas.append([syll[0],syll[2]]) 
			cons = flatten(onsets_codas)
		elif mode=='onsets':
			onsets = []
			for i, syll in enumerate(p.syllables):
				onsets.append(syll[0])
			cons = onsets
		elif mode=='codas':
			codas = []
			for i, syll in enumerate(p.syllables):
				codas.append(syll[2])
			cons = codas

		if placeholder == "yes":
			for i,sublist in enumerate(cons):
				if cons[i] == []:
					cons[i] = ["_"]


		# Option Flatten
		if flat=="no":
			pass
		elif flat=="yes":
			cons = flatten(cons)

		# Option Delim
		# abstract to fucn - pass arg: list of items to remove
		if removeDelims == "no":
			pass
		elif removeDelims == "yes":
			if flat == "no":
				temp = copy.deepcopy(cons)
				for i, cblock in enumerate(temp):
					for j, item in enumerate(temp[i]):
						# was hitting all sorts fow rods in cons streams so added this last check as tmep
						if item == "." or item == "-" or any(char for char in item if char.islower()):
							del temp[i][j]
				cons = temp
			elif flat == "yes":

				temp = copy.deepcopy(cons)
				for i, item in enumerate(temp):
					if item == "." or item == "-" or any(char for char in item if char.islower()):
						del temp[i]
				cons = temp

		return cons



	def get_vowels(p,mode='blocks',stress=False):
		"""
		Extract vowels from a phonomial
		
		iterate across only vowels blocks in a phonomial

		:param param1: phonomial
		:returns: int of block or syllable count
		:raises keyError: raises an exception
		"""

		vowels = []
		if mode == 'blocks':
			for i, block in enumerate(p.blocks):
				if i % 2 == 1:
					if stress==False:
						vowels.append(p.blocks[i][0])
					else:
						vowels.append(p.blocks[i][1])

		elif mode == 'syllables':
			for i, syllable in enumerate(p.syllables):
					if stress==False:
						vowels.append(p.syllables[i][1][0])
					else:
						vowels.append(p.syllables[i][1][1])

		return vowels






############################ SEARCH WIDGET ###########################

search_counter = 0






def create_expanded_button(description, button_style):
    return Button(description=description, button_style=button_style, layout=Layout(height='auto', width='auto'))


def create(description, button_style):
    return Button(description=description, button_style=button_style, layout=Layout(height='auto', width='auto'))

def create_CV_label(description, button_style):
    return Text(
        value='Hello',
        placeholder='Type something',
        description='String:',
        disabled=False
    )
    
def create_CV_HTML(description):
    return HTML(
        value="<b>"+description+"</b>",
    )
def create_CV_dropdowns(segment_array):
    return Dropdown(
        options=segment_array,
        value=segment_array[0],
        disabled=False,
        layout=Layout(height='auto', width='auto')
    )

def create_toggle_buttons():
    return ToggleButtons(
        options=['ON', 'OFF'],
        disabled=False,
        button_style='success', # 'success', 'info', 'warning', 'danger' or ''
        tooltips=['Description of slow', 'Description of regular'],
    #     icons=['check'] * 3
    )

def create_active_checkbox():

    return Checkbox(
        value=False,
        description='ACTIVE',
        disabled=False,
        indent=False,
        layout=Layout(height='auto', width='auto')

    )

def create_int_slider():
    return IntSlider(description='a',layout=Layout(height='auto', width='auto'))






def search2(pc, location='left'):
    global search_counter
    if search_counter == 0:
        print("Loading Dictionary...")
        words = get_cmu.cmuSyllDict.keys()
#         global phonomial_dict 
        global phonomial_dict # make sure its accessible if this if isn't met
        phonomial_dict = [phonomial.from_string(word.lower(), replace_missing=False) for word in tqdm(words)]
    search_counter+=1
    results = []
    for word in phonomial_dict: # Check condition on each word

        # how many conditions are there in pc?
        condition_count = len([y for y in pc.blocks if y != []])
        passed_condition_count = 0
        # Make sure dictionary word is at least as long as the phonomial condition
        block_gap = abs(len(pc.blocks)-len(word.blocks))

        if len(pc.blocks)<= len(word.blocks):
            if location == 'left':
                for i, element in enumerate(pc.blocks): # check every element - only works on perfect equality
                    if element != []:
                        if element[0] in word.blocks[i]: # matches starting from left
                            passed_condition_count +=1
#                             print("------LEFT PASSED")
            elif location == 'right':
                # the adjusted starting index for right slignment
                new_index = len(word.blocks) - len(pc.blocks) - block_gap
                for i, element in enumerate(pc.blocks): # check every element - only works on perfect equality
                    if element != []:
                        if element[0] in word.blocks[i+block_gap]: # matches starting from left
                            passed_condition_count +=1
#                             print("------RIGHT PASSED")
            elif location == 'anywhere':
                
                # the adjusted starting index for right slignment
                new_index = len(word.blocks) - len(pc.blocks)
                passed_condition_count_anywhere_list = []
                for moving_index in range(0,new_index+1,2):
                
                    passed_condition_count_anywhere = 0
                    for i, element in enumerate(pc.blocks): # check every element - only works on perfect equality
                        if element != []:
                            if element[0] in word.blocks[moving_index:][i]: # matches starting from left
                                passed_condition_count_anywhere +=1
                    passed_condition_count_anywhere_list.append(passed_condition_count_anywhere)
                if max(passed_condition_count_anywhere_list)>=condition_count:
                    results.append(word)
            if passed_condition_count>=condition_count:
                results.append(word)
        else:
            pass
    return results
def search_pattern(x):
    display(x)

def search_widget(syllables=1):
    """
    Returns an interactive widget that allow building X syllables sound patterns and finding word matches. Specify X with the syllables parameter. 

    :param syllables: int
    :returns: pywidget
    
    """


    cols = 2
    rows = syllables*2+1
    grid = GridspecLayout(cols, rows)

    
    def str_2_phonomial_pattern(test_str):
        test_str = test_str.split(" ")
        test_str = [block.strip('\"\'') for block in test_str] # clean user blocks
        cp = phonomial(int((len(test_str)-1)/2))
        v_pos = 1
        c_pos = 1
        for i, block in enumerate(test_str): 

            if i % 2 == 0: # consonant
                block_location = 'C'+str(c_pos)
                c_pos+=1
                if block == '*':
                    block = ''
                element = block

            else: # vowel

                element = block
                block_location = 'V'+str(v_pos)
                v_pos+=1
            cp = set_element(cp, block_location=block_location, element=element)
        return cp

    def get_final_results(ignore, location):

        display_strs = []
        for child in ui.children:
            display_strs.append(str(child)[73:].split("\"")[0].strip("'"))

        display_str = ' '.join(display_strs)
        pc = str_2_phonomial_pattern(ignore)
        pc = str_2_phonomial_pattern(display_str)
        
        matches = search2(pc,location=location)
        print(len(matches),'MATCHES')
        return ' '.join([match.ortho for match in matches])																																																																																																				
    
    
    for i in range(cols):
        for j in range(rows):
            if i == 0: # in the first row - CVC LABELS
                if j % 2 == 1:
                    grid[i, j] = create_CV_HTML('V')
                else:
                    grid[i, j] = create_CV_HTML('C')
            elif i == 1: # second row - SOUND
                if j % 2 == 1:
                    grid[i, j] = create_CV_dropdowns(['*']+vowels())
                else:
                    grid[i, j] = create_CV_dropdowns(['*']+consonants())

    # Show search interface
    display(grid)
    

    res = grid.children[5].value
    
    interactive_output_blocks = []
    blocks = []
    for block in range(rows): # really columns
        interactive_output_block = widgets.interactive_output(search_pattern, {'x': grid.children[rows+block]} )
        interactive_output_blocks.append(interactive_output_block)
        blocks.append(grid.children[rows+block].value)
    ui = widgets.HBox(interactive_output_blocks)
    display(ui)
    align_choice = widgets.RadioButtons(
        options=['left', 'right', 'anywhere'],
    #    value='pineapple', # Defaults to 'pineapple'
    #    layout={'width': 'max-content'}, # If the items' names are long
    description='Alignment:',
    disabled=False
    )
    # alt set button name, runtime setting not working
    widgets.interact_manual.opts['manual_name'] = 'Search'
    go = widgets.interact_manual(get_final_results,ignore='Select Alignment Below',location=align_choice,manual_name='Search')
    # go = interactive(get_final_results, {'manual': True}, ignore='Below put left, right, or anywhere',location='left',button_style='danger')


    








class data:
	"""

	"""	

	def __init__(self, syll_length):
		"""
		>>> type(phonomial(1))
		<type 'instance'>

		"""


	@staticmethod
	def get_sample(sample='none'):
		"""
		Flatten a 2D list into a 1D list - Use recursion on more than 2D lists if needed.
		
		:param param1: phonomial
		:returns: int of block or syllable count
		:raises keyError: raises an exception

		>>> phonomial.flatten([[1,2],[3,4]])
		[1, 2, 3, 4]

		"""
		

		sonnet = """Shall I compare thee to a summer’s day?
	Thou art more lovely and more temperate:
	Rough winds do shake the darling buds of May,
	And summer’s lease hath all too short a date;
	Sometime too hot the eye of heaven shines,
	And often is his gold complexion dimm'd;
	And every fair from fair sometime declines,
	By chance or nature’s changing course untrimm'd;
	But thy eternal summer shall not fade,
	Nor lose possession of that fair thou ow’st;
	Nor shall death brag thou wander’st in his shade,
	When in eternal lines to time thou grow’st:
	So long as men can breathe or eyes can see,
	So long lives this, and this gives life to thee."""
		rap_eminem = """His palms are sweaty, knees weak, arms are heavy
	There's vomit on his sweater already, mom's spaghetti
	He's nervous, but on the surface he looks calm and ready
	To drop bombs, but he keeps on forgettin'
	What he wrote down, the whole crowd goes so loud
	He opens his mouth, but the words won't come out
	He's chokin', how, everybody's jokin' now
	The clocks run out, times up, over, blaow
	Snap back to reality, ope there goes gravity
	Ope, there goes Rabbit, he choked
	He's so mad, but he won't give up that easy? No
	He won't have it, he knows his whole back's to these ropes
	It don't matter, he's dope, he knows that, but he's broke
	He's so stagnant, he knows, when he goes back to this mobile home, that's when it's
	Back to the lab again, yo, this whole rhapsody
	Better go capture this moment and hope it don't pass him"""
		limerick_1 = """A fly and a flea in the flue
	Were imprisoned, so what could they do?
	Said the fly, "Let us flee!"
	"Let us fly!" said the flea.
	So they flew through a flaw in the flue."""
		random_sentences = """
His thought process was on so many levels that he gave himself a phobia of heights.
Courage and stupidity were all he had.
I am never at home on Sundays.
He found rain fascinating yet unpleasant.
The ants enjoyed the barbecue more than the family.
The tour bus was packed with teenage girls heading toward their next adventure.
The tart lemonade quenched her thirst, but not her longing.
Twin 4-month-olds slept in the shade of the palm tree while the mother tanned in the sun.
He decided to live his life by the big beats manifesto.
Hit me with your pet shark!
She wasn't sure whether to be impressed or concerned that he folded underwear in neat little packages.
Abstraction is often one floor above you.
I think I will buy the red car, or I will lease the blue one.
He colored deep space a soft yellow.
The clock within this blog and the clock on my laptop are 1 hour different from each other.
She thought there'd be sufficient time if she hid her watch.
He knew it was going to be a bad day when he saw mountain lions roaming the streets.
The quick brown fox jumps over the lazy dog.
The complicated school homework left the parents trying to help their kids quite confused.
The sudden rainstorm washed crocodiles into the ocean.
		"""
		bar_pong_rhymes = """
Vivian Banks
really though, thanks
video tapes
hillary banks
live in the lake
gimmie a shank
amphibious face
fill in the blank
giving me breaks
permisquious skank
obsidian tank
lifted off dank
squishin' the grapes
sniffin' the paint
women get raped
into her taint
little bit late
little kid face
rippin' the drapes
hit it to drake
listens to drake
miniture gate
squids in the tank
nipples in clamps
gettin' him saved
spliff of some crank
shitty in spanks
get in the ranks
city is great
live in a maze
infinity days
literly' stank
lifting these weights
liftin' these plates
grippin' the cakes
hit wit a train
stepped on a rake
riggin' me thanks
give me a taste
shit was just gay
fish in the tank
		"""
		wiki_sentences = """
Outside North America, the word pants generally means underwear and not trousers.
Shorts are similar to trousers, but with legs that come down only to around the area of the knee, higher or lower depending on the style of the garment.
To distinguish them from shorts, trousers may be called 'long trousers' in certain contexts such as school uniform, where tailored shorts may be called 'short trousers', especially in the UK.
The oldest known trousers were found at the Yanghai cemetery in Turpan, Xinjiang, western China and dated to the period between the 10th and the 13th centuries BC.
Made of wool, the trousers had straight legs and wide crotches and were likely made for horseback riding.
		"""
		rap_biggy = """
It was all a dream, I used to read Word Up! magazine
Salt-n-Pepa and Heavy D up in the limousine
Hangin' pictures on my wall
Every Saturday Rap Attack, Mr. Magic, Marley Marl
I let my tape rock 'til my tape popped
Smokin' weed in Bambu, sippin' on Private Stock
Way back, when I had the red and black lumberjack
With the hat to match
Remember Rappin' Duke? Duh-ha, duh-ha
You never thought that hip-hop would take it this far
Now I'm in the limelight 'cause I rhyme tight
Time to get paid, blow up like the World Trade
Born sinner, the opposite of a winner
Remember when I used to eat sardines for dinner
Peace to Ron G, Brucie B, Kid Capri
Funkmaster Flex, Lovebug Starski
I'm blowin' up like you thought I would
Call the crib, same number, same hood
It's all good (It's all good)
And if you don't know, now you know, nigga
		"""
		battle_rap = """
I’m pissed off at people even thinking the match was even.
I mean I should Rerun my rounds cause this is a classic beating.
I’m like Popeye the sailor: I’m gripping a can and squeezing.
Talking out the side of your mouth gon get you the Bambi treatment.
I brought the Thumper in the club. Now it’s switching to rabbit season:
if he Bugs, I’ll get him smoked like Yosemite Sam was creeping.
Morris’ll catch a flashback like he’s sitting in class and dreaming,
while I’m at his bae side, high, like Tiffani-Amber Thiessen.
Mario Lopez is in the building. This is your chance to meet him.
The silencer is Andy Griff how it’s whistling back to greet him.
This is Aladdin dreaming, I’m wishing we had a reason.
I just had to get a couple bands off him before giving him back his freedom.
		"""
		rap_mfdoom = """
Living off borrowed time, the clock tick faster
That'd be the hour they knock the slick blaster
Dick Dastardly and Muttley with sick laughter
A gun fight and they come to cut the mixmaster
I C E cold, nice to be old
Y 2 G stee twice to threefold
He sold scrolls, lo and behold
Know who's the illest ever like the greatest story told
Keep your glory, gold and glitter
For half, half of his niggas'll take him out the picture
The other half is rich and it don't mean shit-ta
Villain: a mixture between both with a twist of liquor
Chase it with more beer, taste it like truth or dare
When he have the mic, it's like the place get like:'Aw yeah!''
It's like they know what's 'bout to happen
Just keep ya eye out, like 'Aye, aye captain'
Is he still a fly guy clapping if nobody ain't hear it
And can they testify from inner spirit
In living, the true gods
Giving y'all nothing but the lick like two broads
Got more lyrics than the church got 'Ooh Lords'
And he hold the mic and your attention like two swords
Either that or either one with two blades on it
Hey you, don't touch the mic like it's AIDS on it
It's like the end to the means
Fucked type of message that sends to the fiends
That's why he brings his own needles
And get more cheese than Doritos, Cheetos or Fritos
Slip like Freudian
Your first and last step to playing yourself like accordion
		"""
		limerick_2 = """
Hickory dickory dock,
the mouse ran up the clock;
the clock struck one
and down he run;
hickory dickory dock.
		"""
		
		hamilton_myshot = '''
A L E X A N D
E R, we are, meant to be…
A colony that runs independently
Meanwhile, Britain keeps shittin’ on us endlessly
Essentially, they tax us relentlessly
Then King George turns around, runs a spending spree
He ain’t ever gonna set his descendants free
So there will be a revolution in this century
Enter me!
(He says in parentheses)
Don’t be shocked when your hist’ry book mentions me
I will lay down my life if it sets us free
Eventually, you’ll see my ascendancy
		'''
		inaugural2009 = '''
My fellow citizens:

I stand here today humbled by the task before us, grateful for the trust you have bestowed, mindful of the sacrifices borne by our ancestors. I thank President Bush for his service to our nation, as well as the generosity and cooperation he has shown throughout this transition.

Forty-four Americans have now taken the presidential oath. The words have been spoken during rising tides of prosperity and the still waters of peace. Yet, every so often the oath is taken amidst gathering clouds and raging storms. At these moments, America has carried on not simply because of the skill or vision of those in high office, but because We the People have remained faithful to the ideals of our forbearers, and true to our founding documents.

So it has been. So it must be with this generation of Americans.

That we are in the midst of crisis is now well understood. Our nation is at war, against a far-reaching network of violence and hatred. Our economy is badly weakened, a consequence of greed and irresponsibility on the part of some, but also our collective failure to make hard choices and prepare the nation for a new age. Homes have been lost; jobs shed; businesses shuttered. Our health care is too costly; our schools fail too many; and each day brings further evidence that the ways we use energy strengthen our adversaries and threaten our planet.

These are the indicators of crisis, subject to data and statistics. Less measurable but no less profound is a sapping of confidence across our land -- a nagging fear that America's decline is inevitable, that the next generation must lower its sights.

Today I say to you that the challenges we face are real. They are serious and they are many. They will not be met easily or in a short span of time. But know this, America -- they will be met.

On this day, we gather because we have chosen hope over fear, unity of purpose over conflict and discord.

On this day, we come to proclaim an end to the petty grievances and false promises, the recriminations and worn-out dogmas that for far too long have strangled our politics.

We remain a young nation, but in the words of Scripture, the time has come to set aside childish things. The time has come to reaffirm our enduring spirit; to choose our better history; to carry forward that precious gift, that noble idea, passed on from generation to generation: the God-given promise that all are equal, all are free, and all deserve a chance to pursue their full measure of happiness.

In reaffirming the greatness of our nation, we understand that greatness is never a given. It must be earned. Our journey has never been one of shortcuts or settling for less. It has not been the path for the faint-hearted -- for those who prefer leisure over work, or seek only the pleasures of riches and fame. Rather, it has been the risk-takers, the doers, the makers of things'some celebrated but more often men and women obscure in their labor, who have carried us up the long, rugged path towards prosperity and freedom.

For us, they packed up their few worldly possessions and traveled across oceans in search of a new life.

For us, they toiled in sweatshops and settled the West; endured the lash of the whip and plowed the hard earth.

For us, they fought and died, in places like Concord and Gettysburg; Normandy and Khe Sahn.

Time and again these men and women struggled and sacrificed and worked till their hands were raw so that we might live a better life. They saw America as bigger than the sum of our individual ambitions; greater than all the differences of birth or wealth or faction.

This is the journey we continue today. We remain the most prosperous, powerful nation on Earth. Our workers are no less productive than when this crisis began. Our minds are no less inventive, our goods and services no less needed than they were last week or last month or last year. Our capacity remains undiminished. But our time of standing pat, of protecting narrow interests and putting off unpleasant decisions -- that time has surely passed. Starting today, we must pick ourselves up, dust ourselves off, and begin again the work of remaking America.

For everywhere we look, there is work to be done. The state of our economy calls for action, bold and swift, and we will act -- not only to create new jobs, but to lay a new foundation for growth. We will build the roads and bridges, the electric grids and digital lines that feed our commerce and bind us together. We will restore science to its rightful place, and wield technology's wonders to raise health care's quality and lower its cost. We will harness the sun and the winds and the soil to fuel our cars and run our factories. And we will transform our schools and colleges and universities to meet the demands of a new age. All this we can do. All this we will do.

Now, there are some who question the scale of our ambitions -- who suggest that our system cannot tolerate too many big plans. Their memories are short. For they have forgotten what this country has already done; what free men and women can achieve when imagination is joined to common purpose, and necessity to courage.

What the cynics fail to understand is that the ground has shifted beneath them -- that the stale political arguments that have consumed us for so long no longer apply. The question we ask today is not whether our government is too big or too small, but whether it works -- whether it helps families find jobs at a decent wage, care they can afford, a retirement that is dignified. Where the answer is yes, we intend to move forward. Where the answer is no, programs will end. And those of us who manage the public's dollars will be held to account -- to spend wisely, reform bad habits, and do our business in the light of day -- because only then can we restore the vital trust between a people and their government.

Nor is the question before us whether the market is a force for good or ill. Its power to generate wealth and expand freedom is unmatched, but this crisis has reminded us that without a watchful eye, the market can spin out of control -- the nation cannot prosper long when it favors only the prosperous. The success of our economy has always depended not just on the size of our Gross Domestic Product, but on the reach of our prosperity; on the ability to extend opportunity to every willing heart -- not out of charity, but because it is the surest route to our common good.

As for our common defense, we reject as false the choice between our safety and our ideals. Our Founding Fathers, faced with perils that we can scarcely imagine, drafted a charter to assure the rule of law and the rights of man, a charter expanded by the blood of generations. Those ideals still light the world, and we will not give them up for expedience's sake. And so to all the other peoples and governments who are watching today, from the grandest capitals to the small village where my father was born: know that America is a friend of each nation and every man, woman, and child who seeks a future of peace and dignity, and we are ready to lead once more.

Recall that earlier generations faced down fascism and communism not just with missiles and tanks, but with the sturdy alliances and enduring convictions. They understood that our power alone cannot protect us, nor does it entitle us to do as we please. Instead, they knew that our power grows through its prudent use; our security emanates from the justness of our cause, the force of our example, the tempering qualities of humility and restraint.

We are the keepers of this legacy. Guided by these principles once more, we can meet those new threats that demand even greater effort -- even greater cooperation and understanding between nations. We will begin to responsibly leave Iraq to its people, and forge a hard-earned peace in Afghanistan. With old friends and former foes, we will work tirelessly to lessen the nuclear threat, and roll back the specter of a warming planet. We will not apologize for our way of life, nor will we waver in its defense, and for those who seek to advance their aims by inducing terror and slaughtering innocents, we say to you now that our spirit is stronger and cannot be broken; you cannot outlast us, and we will defeat you.

For we know that our patchwork heritage is a strength, not a weakness. We are a nation of Christians and Muslims, Jews and Hindus -- and non-believers. We are shaped by every language and culture, drawn from every end of this Earth; and because we have tasted the bitter swill of civil war and segregation, and emerged from that dark chapter stronger and more united, we cannot help but believe that the old hatreds shall someday pass; that the lines of tribe shall soon dissolve; that as the world grows smaller, our common humanity shall reveal itself; and that America must play its role in ushering in a new era of peace.

To the Muslim world, we seek a new way forward, based on mutual interest and mutual respect. To those leaders around the globe who seek to sow conflict, or blame their society's ills on the West -- know that your people will judge you on what you can build, not what you destroy. To those who cling to power through corruption and deceit and the silencing of dissent, know that you are on the wrong side of history; but that we will extend a hand if you are willing to unclench your fist.

To the people of poor nations, we pledge to work alongside you to make your farms flourish and let clean waters flow; to nourish starved bodies and feed hungry minds. And to those nations like ours that enjoy relative plenty, we say we can no longer afford indifference to the suffering outside our borders; nor can we consume the world's resources without regard to effect. For the world has changed, and we must change with it.

As we consider the road that unfolds before us, we remember with humble gratitude those brave Americans who, at this very hour, patrol far-off deserts and distant mountains. They have something to tell us, just as the fallen heroes who lie in Arlington whisper through the ages. We honor them not only because they are the guardians of our liberty, but because they embody the spirit of service; a willingness to find meaning in something greater than themselves. And yet, at this moment -- a moment that will define a generation -- it is precisely this spirit that must inhabit us all.

For as much as government can do and must do, it is ultimately the faith and determination of the American people upon which this nation relies. It is the kindness to take in a stranger when the levees break, the selflessness of workers who would rather cut their hours than see a friend lose their job which sees us through our darkest hours. It is the firefighter's courage to storm a stairway filled with smoke, but also a parent's willingness to nurture a child, that finally decides our fate.

 Our challenges may be new. The instruments with which we meet them may be new. But those values upon which our success depends -- honesty and hard work, courage and fair play, tolerance and curiosity, loyalty and patriotism -- these things are old. These things are true. They have been the quiet force of progress throughout our history. What is demanded then is a return to these truths. What is required of us now is a new era of responsibility -- a recognition, on the part of every American, that we have duties to ourselves, our nation, and the world, duties that we do not grudgingly accept but rather seize gladly, firm in the knowledge that there is nothing so satisfying to the spirit, so defining of our character, than giving our all to a difficult task.

This is the price and the promise of citizenship.

This is the source of our confidence -- the knowledge that God calls on us to shape an uncertain destiny.

This is the meaning of our liberty and our creed -- why men and women and children of every race and every faith can join in celebration across this magnificent mall, and why a man whose father less than sixty years ago might not have been served at a local restaurant can now stand before you to take a most sacred oath.

So let us mark this day with remembrance, of who we are and how far we have traveled. In the year of America's birth, in the coldest of months, a small band of patriots huddled by dying campfires on the shores of an icy river. The capital was abandoned. The enemy was advancing. The snow was stained with blood. At a moment when the outcome of our revolution was most in doubt, the father of our nation ordered these words be read to the people:

"Let it be told to the future world ... that in the depth of winter, when nothing but hope and virtue could survive ... that the city and the country, alarmed at one common danger, came forth to meet ... it."

America! In the face of our common dangers, in this winter of our hardship, let us remember these timeless words. With hope and virtue, let us brave once more the icy currents, and endure what storms may come. Let it be said by our children's children that when we were tested we refused to let this journey end, that we did not turn back nor did we falter; and with eyes fixed on the horizon and God's grace upon us, we carried forth that great gift of freedom and delivered it safely to future generations.

Thank you. God bless you. And God bless the United States of America.  
		'''
			
		data = {"sonnet":sonnet,
				"rap_eminem":rap_eminem,
				"limerick_1":limerick_1,
				"limerick_2":limerick_2,
				"inagural2009":inaugural2009,
				"battle_rap":battle_rap,
				"random_sentences":random_sentences,
				"wiki_sentences":wiki_sentences,
				"rap_biggy":rap_biggy,
				"rap_mfdoom":rap_mfdoom,
				"bar_pong_rhymes":bar_pong_rhymes,

			}

		if sample == 'none':# display the whole list of possible sample data
			print("Pass one of the sample text names to data e.g. data('sonnet')")
			print("Sample Text names:")
			
			print(data.keys())
			return


		return data[sample]
		# sample_text_names = ['sonnet','rap_verse','limerick','inagural2008','random_sentences']




	@staticmethod
	def get_DCET():
		"""
		***NOT HERE YET***

		Provide a truncated 14k-words set.

		Offer optional download of the entire dataset


		"""







if __name__ == "__main__":


    import doctest
    doctest.testmod()




