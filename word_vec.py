# coding=utf-8
# author : liangchen
# decription :  class line_generator yield one line at a time
# 			 :  class word_vector to train all words in one directory path,generate word vectors


import gensim.models.word2vec as w2v
import logging
import numpy as np

import os
from multiprocessing import cpu_count
from datetime import datetime

import split_word


class line_generator():
	def __init__(self,srcdir):
		self.srcdir = srcdir

	def __iter__(self):
		for fname in os.listdir(self.srcdir):
			for line in open(os.path.join(self.srcdir, fname)):
				yield line.split()


class word_vector():
	def __init__(self,linesrc,modelpath=None,retrain=True):
		self.lines = linesrc

		#self.sentences = w2v.LineSentence(self.lines)
		#self.sentences = self.linesrc

		if modelpath:
			self.model = w2v.Word2Vec.load(modelpath)
		else:
			self.model = None
		self.retrain = retrain

	def train_model(self):
		if not self.retrain:
			return self.model

		first_flag = 1

		cpu_num = cpu_count()
		print u'time: %s ==> 模型训练开始，使用%d核\n' % (datetime.now(), cpu_num)
		vector_size = 50
		skip_gram = 0
		hierarch_sm = 1
		negative_sample = 0
		context_window = 5
		word_min_count = 5
		print u'time: %s ==> 模型参数： Vector size %d\tSkip-gram %d\tHierarchical softmax %d\tNeg Sampling %d\tWindows %d\tMin_count %d\n' % (datetime.now(), \
																																		 vector_size,\
																																		 skip_gram, \
																																		 hierarch_sm, \
																																		 negative_sample, \
																																		 context_window, \
																																		 word_min_count)
		for fname in os.listdir(self.lines):
			fullpath = os.path.join(self.lines, fname)
			if first_flag:
				self.model = w2v.Word2Vec(w2v.LineSentence(fullpath), size=50, sg=0, hs=1, negative=0, window=5,
									  min_count=5, workers=cpu_num)
				first_flag = 0
			else:
				old_corpus_count = self.model.corpus_count
				self.model.build_vocab(w2v.LineSentence(fullpath),update=True)
				new_example_count = self.model.corpus_count - old_corpus_count
				self.model.train(w2v.LineSentence(fullpath),total_examples=new_example_count,epochs=self.model.iter)


		print u'time: %s ==> 模型训练结束，使用%d核\n' % (datetime.now(), cpu_num)

		return self.model

	def update_model(self,newfile):
		if not self.model:
			return None

		if not os.path.isfile(newfile):
			return self.model

		with open(newfile,'r') as f:
			train_word_count = self.model.train(w2v.LineSentence(f))
			print u'time: %s ==> 文件%s更新模型结束，更新%d个词\n' % (datetime.now(), newfile, train_word_count)

		return self.model

	def get_vecs(self):
		if self.model:
			return self.model.wv

		return None

	def get_word_vec(self,word):
		if not self.model:
			return None

		return self.model[word]

	def save_model(self,path):
		if path:
			self.model.save(path)

		return True

	def model_test(self):
		print self.model.similarity('赵敏'.decode('utf-8'), '赵敏'.decode('utf-8'))
		print self.model.similarity('赵敏'.decode('utf-8'), '周芷若'.decode('utf-8'))
		print self.model.similarity('赵敏'.decode('utf-8'), '韦一笑'.decode('utf-8'))

		for k in self.model.similar_by_word('张三丰'.decode('utf-8')):
			print k[0], k[1]

		print self.model.wv['张三丰'.decode('utf-8')]
		return



#class_test
if __name__=='__main__':
	ds = split_word.doc_splitter('/home/lc/ht_work/ML/old_txt', '/home/lc/ht_work/ML/new_txt', '/home/lc/ht_work/xwparse/stopwords_merge.txt','/home/lc/ht_work/ML/xw_parse/userdict.txt', True)
	ds.split_all()

	#word_v = word_vector(line_generator('/home/lc/ht_work/ML/new_txt/'),None,True)
	word_v = word_vector('/home/lc/ht_work/ML/new_txt/', None, True)
	word_v.train_model()

	word_v.model_test()

