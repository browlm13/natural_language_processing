import numpy as np
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk import FreqDist
#from nltk.stem import *
from nltk.stem.porter import *

import math

from nltk.corpus import abc
from nltk.corpus import brown

from nltk.corpus import stopwords

from nltk.tag import pos_tag


"""
	Rapid Automatic Keyword Extraction
"""
import string
import operator
def isPunct(word):
  return len(word) == 1 and word in string.punctuation

def isNumeric(word):
	try:
		float(word) if '.' in word else int(word)
		return True
	except ValueError:
		return False

def generate_candidate_keywords(sentences):
	stop_words = nltk.corpus.stopwords.words()

	phrase_list = []
	for sentence in sentences:
	  words = map(lambda x: "|" if x in stop_words else x,
	    nltk.word_tokenize(sentence.lower()))
	  phrase = []
	  for word in words:
	    if word == "|" or isPunct(word):
	      if len(phrase) > 0:
	        phrase_list.append(phrase)
	        phrase = []
	    else:
	      phrase.append(word)

	return phrase_list

def calculate_phrase_scores(phrase_list, word_scores):
	phrase_scores = {}
	for phrase in phrase_list:
		phrase_score = 0
		for word in phrase:
			phrase_score += word_scores[word]
		phrase_scores[" ".join(phrase)] = phrase_score
	return phrase_scores

def calculate_word_scores(phrase_list):
	word_freq = nltk.FreqDist()
	word_degree = nltk.FreqDist()
	for phrase in phrase_list:
		degree = len(list(filter(lambda x: not isNumeric(x), phrase))) - 1
		for word in phrase:
			#word_freq.inc(word)
			word_freq[word] += 1
			#word_degree.inc(word, degree) # other words
			word_degree[word] += degree
	for word in word_freq.keys():
		word_degree[word] = word_degree[word] + word_freq[word] # itself
	# word score = deg(w) / freq(w)
	word_scores = {}
	for word in word_freq.keys():
		word_scores[word] = word_degree[word] / word_freq[word]
	return word_scores

def extract(text):
	sentences = nltk.sent_tokenize(text)

	phrase_list = generate_candidate_keywords(sentences)
	word_scores = calculate_word_scores(phrase_list)
	phrase_scores = calculate_phrase_scores(phrase_list, word_scores)

	sorted_phrase_scores = sorted(phrase_scores.items(), key=operator.itemgetter(1), reverse=True)
	n_phrases = len(sorted_phrase_scores)

	top_fraction = 1 #1/3
	return list(map(lambda x: x[0],sorted_phrase_scores[0:int(n_phrases/top_fraction)]))

def top_words(text):
	sentences = nltk.sent_tokenize(text)

	phrase_list = generate_candidate_keywords(sentences)
	word_scores = calculate_word_scores(phrase_list)
	sorted_word_scores = sorted(word_scores.items(), key=operator.itemgetter(1), reverse=True)
	
	top_picks = [ws[0] for ws in sorted_word_scores]
	return list(top_picks)

"""
	rank sentences based on rank top phrases
"""
def rank_sentences(text, n):
	tokenizer = RegexpTokenizer(r'\w+')
	sentences = nltk.sent_tokenize(text)
	top_phrases = extract(text)
	top_sentences = []

	for i in range(n):
		tokenized_phrase = tokenizer.tokenize(top_phrases[i])
		for s in sentences:
			if (set(tokenized_phrase) < set(tokenizer.tokenize(s))) and (s not in top_sentences):
				top_sentences.append(s)
	return top_sentences

#print(brown.words())
#print(brown.fileids())
#fileid = 'cr05'
#print(brown.words(fileid))
#print(brown.raw(fileid))

#print(abc.fileids())
#print(abc.raw('science.txt'))

stemmer = PorterStemmer()

plurals = ['caresses', 'flies', 'dies', 'mules', 'denied','died', 'agreed', 'owned', 'humbled', 'sized','meeting', 'stating', 'siezing', 'itemization','sensational', 'traditional', 'reference', 'colonizer','plotted']
singles = [stemmer.stem(plural) for plural in plurals]
#print(' '.join(singles))


# First, you're going to need to import wordnet:
from nltk.corpus import wordnet

# Then, we're going to use the term "program" to find synsets like so:
syns = wordnet.synsets("program")


"""
# An example of a synset:
print(syns[0].name())

# Just the word:
print(syns[0].lemmas()[0].name())

# Definition of that first synset:
print(syns[0].definition())

# Examples of the word in use in sentences:
print(syns[0].examples())


# Let's compare the verbs of "run" and "sprint:"
w1 = wordnet.synset('run.v.01') # v here denotes the tag verb
w2 = wordnet.synset('sprint.v.01')
print(w1.wup_similarity(w2))

# Let's compare the noun of "ship" and "boat:"
w1 = wordnet.synset('ship.n.01')
w2 = wordnet.synset('boat.n.01') # n denotes noun
print(w1.wup_similarity(w2))

# Let's compare the verbs of "sprint" and "sprint:"
w1 = wordnet.synset('sprint.v.01')
w2 = wordnet.synset('sprint.v.01')
print(w1.wup_similarity(w2))
"""

"""
prun corpus of non cadidate words

for each candidate word test against similarity scores for each 
other remaining word and if similarities are above a certain level
map to a category and replace occurences with that category

(original_word:string, original_location:location, 
	word_synset:string, similarity_groups:list)


"""

def syns_tag(term):
	""" returns word wordnet.synset """
	try:
		syns = wordnet.synsets(term)
		name = syns[0].name()
		return wordnet.synset(name)
	except:
		return None

def similarity_score(term, tokenized_document):
	""" Calculates similarity score for word against all other words in document """
	term = processes_and_tokenize(term)[0]	#make sure term is in correct form
	
	#minimum score
	minimum_score = .7

	# format words to wordnet.synset
	main_tagged_word = syns_tag(term)

	if main_tagged_word is not None:
		tokenized_tagged_document = []
		for t in tokenized_document:
			try:
				tokenized_tagged_document.append(syns_tag(t))
			except: pass
		#return tokenized_tagged_document

		# calculate similarity score
		similarity_scores = []
		for t in tokenized_tagged_document:
			try:
				ss = main_tagged_word.wup_similarity(t)
			except:
				ss = None
			if ss is not None and type(ss) :
				if ss > minimum_score:
					similarity_scores.append(ss)
		summed_similarity_scores = sum(similarity_scores)

		assert len(tokenized_tagged_document) > 0
		score = summed_similarity_scores / len(tokenized_tagged_document)
		return score

	else: return 0


"""
keyword criteria:
	-appearence frequency realitive to docuemnt size minimum

	-remove stop words (or seriously hinder their score)
	-remove punctuation
	-minumum size

1) Remove all stop words from the text( eg for, the, are, is , and etc.)
2) create an array of candidate keywords which are set of words separated by stop words
3) find the frequency of the words. (stemming might be used.)
4) find the degree of the each word. Degree of a word is the number of how many times a word is used by other candidate keywords
5) for every candidate keyword find the total frequency and degree by summing all word's scores.
6) finally degree/frequency gives the score for being keyword.

-Add phrases of up to certain length
"""

document_0 = "China has a strong economy that is growing at a rapid pace. However politically it differs greatly from the US Economy."
document_1 = "At last, China seems serious about confronting an endemic problem: domestic violence and corruption."
document_2 = "Japan's prime minister, Shinzo Abe, is working towards healing the economic turmoil in his own country for his view on the future of his people."
document_3 = "Vladimir Putin is working hard to fix the economy in Russia as the Ruble has tumbled."
document_4 = "What's the future of Abenomics? We asked Shinzo Abe for his views"
document_5 = "Obama has eased sanctions on Cuba while accelerating those against the Russian Economy, even as the Ruble's value falls almost daily."
document_6 = "Vladimir Putin is riding a horse while hunting deer. Vladimir Putin always seems so serious about things - even riding horses. Is he crazy?"

document_7 = """President Donald Trump misrepresented the crisis facing Puerto Rico Friday,
bragging that people “can’t believe how successful” the administration has been at saving lives following 
Hurricane Maria. “The loss of life, it’s always tragic. But it’s been incredible,” 
Trump told reporters Friday before taking off for his golf course in New Jersey. 
“The results that we’ve had with respect to loss of life. People can’t believe how 
successful that has been, relatively speaking.”"""

document_8 = """HONG KONG (Reuters) - Hundreds of protesters in Hong Kong on Thursday 
demanded full democracy outside government headquarters, speaking out against China's 
suppression of civil liberties on the third anniversary of a major pro-democracy movement.
Unfurling a mass of yellow umbrellas, a symbol of the 2014 movement that blocked major 
roads in the financial hub for close to three months, the demonstrators gathered at the 
same spot where police fired tear gas on the crowds three years ago."""

document_9 = """North Korea's military has been spotted moving missiles from a 
government rocket facility in Pyongyang to an undisclosed location amid reports 
the country may be gearing up for another missile test, according to a new report.
Reuters reported late on Friday that South Korea’s Korean Broadcasting System (KBS) 
spotted the weapons' movement late Friday, but didn't say where the missiles had been moved.
President Trump threatened to "totally destroy" North Korea during his first speech 
to the United Nations last week, warning the country that the U.S. would defend itself 
and its allies."""

document_10 = """A "quantum satellite" sounds at home in the James Bond franchise, 
but there really is a satellite named Micius with some truly quantum assignments. 
In this case, it helped the president of the Chinese Academy of Science make a video 
call. A quantum-safe video call."""

document_11 = """Puerto Rico is still literally powerless. Though 
Hurricane Maria made landfall as a category four hurricane over a week ago, 
the storm has left the island almost entirely without electrical power.
The island’s electrical grid was unable to resist the one-two punch of Hurricane 
Irma followed by Maria. 250-kilometre-per-hour winds and 76 centimetres of rain have 
left nearly 100 per cent of the island without power. Puerto Rico’s governor 
Ricardo Rosselló calls the situation a “humanitarian emergency”."""

document_12 = """People in England who commit the most serious crimes of 
animal cruelty could face up to five years in prison, the government has said.
The move - an increase on the current six-month maximum sentence - follows a 
number of cases where English courts wanted to hand down tougher sentences.
Environment Secretary Michael Gove said it would target "those who commit the 
most shocking cruelty towards animals".
The RSPCA said it would "deter people from abusing and neglecting animals"."""

all_documents = [document_0, document_1, document_2, document_3, document_4, document_5, document_6, document_7, document_8,document_9,document_10,document_11,document_12]

def processes_and_tokenize(raw_document):
	""" remove punctuation, convert to lower case, and return list of tokens """
	tokenizer = RegexpTokenizer(r'\w+')
	tokens = tokenizer.tokenize(raw_document.lower())		# tokens = nltk.word_tokenize(corpus.lower()) # without removing punctiation

	#remove stop words
	stop_words = set(stopwords.words('english'))
	filtered_tokens = [w for w in tokens if not w in stop_words]
	return filtered_tokens


def word_frequency_dict(tokens):
	""" returns a dictionary of word and their assosiated frequencies from token list """

	fdist = FreqDist(tokens) 						# fdist.keys() fdist.values()
	return dict(fdist)

def term_fequency(term,tokens):
	term = processes_and_tokenize(term)[0]	#make sure term is in correct form

	tf = tokens.count(term)
	return tf/len(tokens)

def augmented_term_fequency(term,tokens):
	""" returns term frequency in tokens over maximum term frequency of tokens """
	term = processes_and_tokenize(term)[0] #make sure term is in correct form

	max_count = max([tokens.count(t) for t in tokens])
	return tokens.count(term)/max_count

def inverse_document_frequency(term, tokenized_documents_list):
	""" IDF(t) = ln( Number Of Documents / Number Of Documents Containg Term )."""
	term = processes_and_tokenize(term)[0]	#make sure term is in correct form

	num_documents = len(tokenized_documents_list)
	num_documents_with_term = len([document for document in tokenized_documents_list if term in document])
	
	assert num_documents_with_term > 0
	return math.log(num_documents / num_documents_with_term)


def nolog_inverse_document_frequency(term, tokenized_documents_list):
	""" IDF(t) = ln( Number Of Documents / Number Of Documents Containg Term )."""
	term = processes_and_tokenize(term)[0]	#make sure term is in correct form

	num_documents = len(tokenized_documents_list)
	num_documents_with_term = len([document for document in tokenized_documents_list if term in document])
	
	assert num_documents_with_term > 0
	return num_documents / num_documents_with_term

def tf_idf(term, tokenized_document, tokenized_documents_list):
	""" Term Frequency - Inverse Document Frequency : returns tf * idf """
	#return term_fequency(term, tokenized_document) * inverse_document_frequency(term, tokenized_documents_list)
	#return augmented_term_fequency(term, tokenized_document) * inverse_document_frequency(term, tokenized_documents_list)
	return term_fequency(term, tokenized_document) * nolog_inverse_document_frequency(term, tokenized_documents_list)


def keyword_score(term, tokenized_document, tokenized_documents_list):
	tf_idf_scaler = 2
	term_tf_idf_score = tf_idf(term,tokenized_document,tokenized_documents_list)
	term_similarity_score = similarity_score(term, tokenized_document)
	return tf_idf_scaler*term_tf_idf_score + term_similarity_score


def keyword_scores_for_part_of_speech(pos, tokenized_document, tokenized_documents_list):
	tagged_tokenized_document = pos_tag(tokenized_document)
	filtered_tokens = [term for term, tag in tagged_tokenized_document if tag == pos]
	return list(filtered_tokens)

tokenized_documents_list = list(map(processes_and_tokenize, all_documents))

"""
for d in all_documents:
	d_tokens = processes_and_tokenize(d)
	for t in d_tokens:
		print ("term: %s,\t ss: %f" %(t, similarity_score(t,d_tokens)))
"""
"""
tf_idf_list = []
all_document_keyword_scores = []
for d in tokenized_documents_list:
	document_keyword_scores = []
	for t in d:
		#print ("word: %s,\t tf: %f,\t idf: %f,\t tf-idf: %f,\t similarity_score: %f" % (t, term_fequency(t,d), inverse_document_frequency(t,tokenized_documents_list), tf_idf(t,d,tokenized_documents_list), similarity_score(t,d)))
		tf_idf_list.append((t, tf_idf(t,d,tokenized_documents_list)))
		document_keyword_scores.append((t,keyword_score(t,d,tokenized_documents_list)))
	document_keyword_scores.sort(key=lambda x: x[1])
	all_document_keyword_scores.append(document_keyword_scores)
"""
#tf_idf_list.sort(key=lambda x: x[1])

"""
for i in tf_idf_list:
	print ("term: %s,\t\t tf-idf: %f" % (i[0], i[1]))
"""
"""
for d in all_document_keyword_scores:
	print ("\n\n\n")
	for i in d:
		print ("term: %s,\t\t keyword_score: %f" % (i[0], i[1]))
"""
#keyword_scores_for_part_of_speech('NN',tokenized_documents_list[11],tokenized_documents_list )
#sentences = nltk.sent_tokenize(all_documents[11])

for d in all_documents:
	#print(extract(d))
	print("\n\n\n")
	print("original:")
	print(d)
	print("\nreduced:")
	print(rank_sentences(d,2))
	print(extract(d)[0])
	print(extract(d)[1])
	print(top_words(d)[:3])
