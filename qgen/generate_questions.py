"""
Generate templated questions from SPICE tuples
"""
import os
import sys
import nltk
import en
import profile
import simplejson as json
import cPickle as pickle
from nltk.tag import StanfordPOSTagger
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer

# Set constants and paths
# Max ID is the offset based on highest VQA question ID (including train and val set)
MAX_ID = 5819292
os.environ['STANFORD_MODELS'] = 'stanfordnlp/stanford-postagger'
path_to_model = 'stanfordnlp/stanford-postagger/models/english-bidirectional-distsim.tagger'
path_to_jar = 'stanfordnlp/stanford-postagger/stanford-postagger.jar'

# Load models
st = StanfordPOSTagger(path_to_model, path_to_jar)
wnl = WordNetLemmatizer()

# Load tags and keywords to check for special cases
with open('tags_and_words.pkl','r') as tfile:
    [verb_tags, adj_tags, spl_attributes, uncountables, human_synset, animates, verb_exp] = pickle.load(tfile)

# Functions for different question types
# Binary/Existential question for tuple type <subject/node>
def node_question(node, tag, qid):
    gen_q = {}
    try:
    	word = wn.synsets(unicode(node, "utf-8"))
    	hyperword = list(set([i for i in word.closure(lambda s:s.hypernyms())]))
    	flag = any(str(h) == "Synset('abstraction.n.06')" for h in hyperword)
    except Exception:
        flag = False

    if flag == True:
        pass
    else:
    	gen_q['question_id'] = qid
    	gen_q['a'] = 'yes'

	if tag == 'NNS':
        	gen_q['q'] = 'Are there ' + node + ' in the image?'
    	else:
        	if any(node == u for u in uncountables):
            		gen_q['q'] = 'Is there ' + node + ' in the image?'
        	else:
            		gen_q['q'] = 'Is there ' + en.noun.article(node) + ' in the image?'
    gen_qs = []
    gen_qs.append(gen_q)
    if any(node == a for a in animates):
	   gen_qs.append({'question_id':qid+1,'a':node,'q':'Which animal can you see in the image?'})

    return gen_qs

# Color question for tuple type <subject, attribute>
def color_question(node, color, qid):
    gen_q = {}
    gen_q['question_id'] = qid
    gen_q['a'] = color
    gen_q['q'] = 'What is the color of ' + node + ' in the image?'

    return gen_q

# 'at least one' object question
def loc_question(node, qid):
    gen_q = {}
    gen_q['question_id'] = qid
    gen_q['a'] = 'yes'
    gen_q['q'] ='Can you see ' + node + ' in the image?'

    return gen_q

# Questions for tuple type <node, verb>
def verb_question(node, tag, verb, qid):
    form = 'are' if (tag == 'NNS') else 'is'
    if (any(node == a for a in animates) or any(node == h for h in human_synset)):
        try:
            verb = en.verb.present_participle(en.verb.infinitive(verb))
        except Exception:
            pass
    gen_q = {}
    if all(en.verb.infinitive(verb) != v for v in verb_exp):
        gen_q['question_id'] = qid
        gen_q['a'] = verb
        gen_q['q'] ='What ' + form + ' the ' + node + ' doing?'

        return gen_q

# Question for tuple type <subject, 'be', attribute> or <subject, attribute>
def be_question(node, t, attr, qid):
    gen_q = {}
    try:
        word = wn.synsets(unicode(node, "utf-8"))
        hyperword = list(set([i for i in word.closure(lambda s:s.hypernyms())]))
        flag = any(str(h) == "Synset('abstraction.n.06')" for h in hyperword)
    except Exception:
        flag = False

    if flag == True:
        pass
    else:
    	gen_q['question_id'] = qid
    	gen_q['a'] = wnl.lemmatize(attr,'a') if all(t[1] != v for v in verb_tags) else attr
    	form = 'do' if (t[0] == 'NNS') else 'does'
   	gen_q['q'] = 'How ' + form + ' the ' + node + ' seem?'

    return gen_q

# Question for tuple type <subject, verb, object>
def verbasrel_question_g2(t, tag_0, qid):
    gen_q = []
    form = 'are' if (tag_0 == 'NNS') else 'is'
    form2 = 'do' if (tag_0 == 'NNS') else 'does'
    rel_parts = t[1].split(' ')
    try:
	   rel_parts[0] = en.verb.present_participle(en.verb.infinitive(rel_parts[0]))
    except Exception:
	   pass

    if all(en.verb.infinitive(rel_parts[0]) !=  v for v in verb_exp):
        gen_q.append({'question_id':qid,'a':t[2],'q':'What ' + form + ' the ' + t[0] + ' ' + rel_parts[0] + ' ' + ' '.join(rel_parts[1::]) + '?'})
    elif en.verb.infinitive(rel_parts[0]) == 'have':
        gen_q.append({'question_id':qid,'a':t[2],'q':'What ' + form2 + ' the ' + t[0] + ' ' + 'have' + ' ' + ' '.join(rel_parts[1::]) + '?'})
    elif en.verb.infinitive(rel_parts[0]) == 'be':
        gen_q.append({'question_id':qid,'a':t[2],'q':'What ' + form + ' the ' + t[0] + ' ' + ' '.join(rel_parts[1::]) + '?'})
    qid_i = qid
    if any(t[0] in h for h in human_synset):
        qid_i =  qid_i + 1
        gen_q.append({'question_id':qid_i,'a':t[0], 'q':'Who is ' + rel_parts[0] + ' ' + ' '.join(rel_parts[1::]) + ' the ' + t[2] + '?'})

    return gen_q

# Questions for tuple type <subject, verb preposition, object>
def verbasrel_question_l2(t, tag_0, qid):
    gen_q = []
    try:
        t[1] = en.verb.present_participle(en.verb.infinitive(t[1]))
    except Exception:
        pass

    form = 'are' if (tag_0 == 'NNS') else 'is'
    form2 = 'do' if (tag_0 =='NNS') else 'does'

    if all(en.verb.infinitive(t[1]) != v for v in verb_exp):
    	gen_q.append({'question_id':qid,'a':t[2],'q':'What ' + form +' the ' + t[0] + ' ' + t[1] + '?'})
    elif en.verb.infinitive(t[1]) == 'have':
        gen_q.append({'question_id':qid,'a':t[2],'q':'What ' + form2 +' the ' + t[0] + ' ' + 'have' + '?'})    
    qid_i = qid
    if any(t[0] == h for h in human_synset):
        qid_i = qid_i + 1
        gen_q.append( {'question_id':qid_i,'a':t[0],'q':'Who is ' + t[1] +' the ' + t[2] + '?'})

    return gen_q

# Questions for tuple type <subject, preposition, object>
def prep_question(t, tag_0, qid):
    gen_q = []
    form = 'are' if (tag_0 == 'NNS') else 'is'
    gen_q.append({'question_id': qid, 'a':t[2],'q':'What ' + form + ' the ' + t[0] + ' ' + t[1] + '?'})
    qid_i = qid
    if any(t[0] == h for h in human_synset):
        qid_i = qid_i + 1
        gen_q.append({'question_id': qid_i,'a':t[0],'q':'Who is' + ' ' + t[1] + ' the ' + t[2] + '?'})

    return gen_q

# where:spl preposition question
def where_question(t, tag_0, qid):
    gen_q = []
    form = 'are' if (tag_0 == 'NNS') else 'is'
    gen_q.append({'question_id': qid,'a':t[2],'q':'Where ' + form + ' ' + t[0] + '?'})
    qid_i  = qid
    if any(t[0] == h for h in human_synset):
	   qid_i = qid_i + 1
	   gen_q.append({'question_id': qid_i,'a':t[0],'q':'Who is' + ' '+ t[1] + ' the ' + t[2] + '?'})

    return gen_q

""" Master function to sort tuple type and generate questions """
def generate_questions(ip_filepath = 'vqa_oe_tuples_filtered.json', op_filepath = 'generated_questions/vqa_prem_questions.json', j = -1, k = -1):
    with open(ip_filepath, 'r') as ipfile:
        tuple_list = json.load(ipfile)
    all_tuples = []
    err_tuples = []
    j = 0 if (j < 0) else j
    k = min(k, len(tuple_list))
    print(j,k)
    # Loop over source VQA questions
    for i  in range(j,k):
        tuple_set = tuple_list[i]
        if i%500 == 0:
            print(i)
        # Set up objects and base ID
        q_object = {}
        q_object['image_id'] = tuple_set['image_id']
        q_object['question_id'] = tuple_set['question_id']
        q_object['question'] = tuple_set['question']
        q_object['tuples'] = []
        qid = MAX_ID + int(tuple_set['question_id']) * 20
        all_tags = st.tag([q_object['question'][:-1]])
        # Loop over tuples for each question
        for t in tuple_set['tuples']:
            question_object = {}
            question_object['tuple'] = t
            question_object['premise_questions'] = []
            ttype = len(t)
            words = []
            # Loop over parts of each tuple
            for ph in t:
                words.extend(ph.split(' '))
            # Find tags from sentence
            tags = []
            for w in words:
                i_flag = False
                for a in all_tags:
                    curr_a = (a[0] if(a[0][-2:] != '\'s') else a[0][:-2], a[1])
                    if w == curr_a[0]:
				        tags.append(curr_a)
				        i_flag = True
				        break
                    elif en.noun.singular(w) == en.noun.singular(curr_a[0]):
				        tags.append(curr_a)
				        i_flag = True
				        break
                if i_flag == False:
                    tags.append(st.tag([w])[0])
            # Check if first element is a noun
            if all(tags[0][1] != t_n for t_n in ['NN','NNS','NNP']):
                pass
            else:
            # Generate question based on type of tuple
                if ttype == 1:
                    question_object['premise_questions'].extend(node_question(t[0],tags[0][1],qid))
                    qid = qid + 1
                elif ttype == 2:
                    if any(t[1] == s for s in spl_attributes['color']):
                        question_object['premise_questions'].append(color_question(t[0],t[1],qid))
                        qid = qid + 1
                    elif any(t[1] == s for s in spl_attributes['location']):
                        question_object['premise_questions'].append(loc_question(t[0],qid))
                        qid =  qid + 1
                    elif any(tags[1][1] == v for v in verb_tags):
                        question_object['premise_questions'].append(verb_question(t[0],tags[0][1],t[1],qid))
                        qid = qid + 1
                    elif any(tags[0][1] == n for n in ['NN','NNS','NNP']) and any(tags[1][1] == n for n in ['NN','NNS','NNP']):
                        question_object['premise_questions'].extend(node_question(t[1]+' '+t[0],tags[0][1],qid))
                        qid = qid + 1
                    elif t[1] != 'same':
                        question_object['premise_questions'].append(be_question(t[0],[tags[0][1],tags[1][1]],t[1],qid))
                        qid = qid + 1
                elif ttype == 3:
                    num = len(t[1].split())
                    if (any(t[2] == s for s in spl_attributes['color']) or (t[2] == 'color')):
                        question_object['premise_questions'].append(color_question(t[0],t[2],qid))
                        qid = qid + 1
                    elif en.verb.infinitive(t[1]) == 'be':
                        question_object['premise_questions'].append(be_question(t[0],[tags[0][1],tags[2][1]],t[2],qid))
                        qid = qid + 1
                    elif (any(tags[1][1] == v for v in verb_tags) and (num == 1)):
                        question_object['premise_questions'].extend(verbasrel_question_l2(t,tags[0][1],qid))
                        qid = qid + 2
                    elif (any(tags[1][1] == v for v in verb_tags) and (num == 2)):
                        question_object['premise_questions'].extend(verbasrel_question_g2(t,tags[0][1],qid))
                        qid = qid + 2
                    elif (tags[1][1] == 'IN' and t[1] != 'of' and t[1] != 'in'):
                        question_object['premise_questions'].extend(prep_question(t,tags[0][1],qid))
                        qid = qid + 2
                    elif (t[1] == 'in'):
                        question_object['premise_questions'].extend(where_question(t,tags[0][1],qid))
                        qid = qid + 2
                    else:
                        err_tuples.append(t)
            q_object['tuples'].append(question_object)
        all_tuples.append(q_object)
    # Dump generated questions in JSON format
    with open(op_filepath, 'w') as opfile:
        json.dump(all_tuples, opfile)
    # Dump error tuples in error log
    #with open(op_filepath[:-5]+'_log.json','w') as op_log:
	#json.dump(err_tuples, op_log) 

    return all_tuples

# Main()
def main():
    args = sys.argv[1:]
    generate_questions(args[0], args[1], int(args[2]), int(args[3]))

if __name__ == '__main__':
   main()