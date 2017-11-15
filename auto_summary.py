#coding:utf-8
# Simple Auto Summary
# based on word frequency
# 分词-去停用词-词频统计-文本句子切分-选择关键词首先出现的句子-按句子出现顺序构成summary
import re
import cchardet
import jieba.posseg as pseg

def readFromFile(filename):
	lines=[]
	with open(filename) as f:
		for line in f:
			# unicode

			# res=pseg.cut(line.strip())
			# split_line=' '.join([w.word for w in res])
			lines.append(line.replace(' ','').strip())

	return lines

# 计算词频
def count_word(lines,stop_words):
	words={}
	for line in lines:
		res=pseg.cut(line)
		for w in res:
			w=w.word.encode('utf-8')
			if w not in stop_words:
				words[w]=words.get(w,0)+1
	return words

# 加载停用词表
def load_stop_words(filename):
	stop_words=[]
	lines=open(filename).readlines()
	for line in lines:
		if line.strip()!='':
			stop_words.append(line.strip())
	return stop_words

# 切分句子
def cut_sentences(lines):
	sentences=[]
	for line in lines:
		pieces=re.split('，|。|？|！|；|、|,',line)
		# print pieces
		sentences+=pieces
	return sentences

# 生成摘要 k句
def get_summary(sentences,items,k):
	summary=[]
	count=0
	for item in items:
		word=item[0]
		for i,sen in enumerate(sentences):
			if (word in sen) and ([sen,i] not in summary):
				# 首次出现当前关键词的句子 sen
				summary.append([sen,i])
				# print sen.decode('utf-8').encode('GB18030'),i
				count+=1
				break

		if count==k:
			break
	# 按句子的出现顺序构成summary
	sens=sorted(summary,key=lambda x:x[1])

	summary=[s[0] for s in sens]

	return summary


def main():
	lines=readFromFile('test3.txt')
	# print lines[0].decode('utf-8').encode('GB18030')

	# 加载stop_words
	stop_words=load_stop_words('stop_words.txt')

	# print stop_words[0].decode('utf-8').encode('GB18030')
	# for sw in stop_words:
	# 	print sw.decode('utf-8').encode('GB18030')

	# 词频统计
	words=count_word(lines,stop_words)

	items=sorted(words.items(),key=lambda x:x[1],reverse=True)

	# for item in items:
	# 	print item[0].decode('utf-8').encode('GB18030'),item[1]

	# 切分句子
	sentences=cut_sentences(lines)
	# for sen in sentences:
	# 	print sen.decode('utf-8').encode('GB18030')

	k=10
	summary=get_summary(sentences,items,k)
	print len(summary)
	for s in summary:
		print s.decode('utf-8').encode('GB18030')

if __name__ == '__main__':
	main()