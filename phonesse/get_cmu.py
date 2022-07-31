# ERROR IN PULLING DATA  - reorder ' and ( in sorted entries or fix this algo'
# READ  R EH1 D
# READ'S  R IY1 D Z
# READ(2)  R IY1 D
# 
# /Users/verbal/Desktop/verbal/projects/phonomials/data
import os 
# dir_path = os.path.dirname(os.path.realpath(__file__))
# print dir_path

# dirname, filename = os.path.split(os.path.abspath(__file__))

# print dirname, filename
# print os.getcwd(), "1"

# APPARENTLY pkg_resources CORRECT FOR PIP PACKAGES
# https://stackoverflow.com/questions/12201928/python-open-gives-ioerror-errno-2-no-such-file-or-directory/12201952#12201952
# import pkg_resources
# print pkg_resources.resource_filename("phonomials","cmudict.rep")


# Generate the path to the file relative to your python script:

from pathlib2 import Path


script_location = Path(__file__).absolute().parent
file_location = script_location / 'data/cmudict.rep'

file = file_location.open('r')

cmuSyll = file.readlines()
cmuSyllDict = {}


startIndex = 48
altIndex = [1,2,3,4,5,6,7,8]
altMark = [2,3,4,5,6,7,8,9]
altsFound = 0;

while startIndex < len(cmuSyll)-9:
	pronList = []
	word, firstPron = cmuSyll[startIndex].split(' ',1)
	pronList.append(firstPron.lstrip(' ').rstrip('\n'))
	for j, alt in enumerate(altIndex):
		altword, altPron = cmuSyll[startIndex+alt].split(' ',1)
		if altword == word + '(' + str(altMark[j]) +')':

			pronList.append(altPron.lstrip(' ').rstrip('\n'))
			altsFound += 1
	cmuSyllDict[word] = pronList
	if altsFound > 0:

		startIndex += altsFound + 1
	else:

		startIndex += 1
	altsFound = 0

