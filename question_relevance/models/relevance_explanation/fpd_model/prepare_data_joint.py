from random import shuffle, seed
import sys
import os.path
import argparse
import h5py
from nltk.tokenize import word_tokenize
import json
import spacy.en
import numpy as np

def prepro_tuple(imgs, params):
  
    # preprocess all the question
    print 'example processed tokens:'
    for i,img in enumerate(imgs):
        s = img['tuple']
        if params['token_method'] == 'nltk':
            txt = word_tokenize(str(s).lower())
        elif params['token_method'] == 'spacy':
            txt = [token.norm_ for token in params['spacy'](s)]
        else:
            txt = tokenize(s)
        img['processed_tokens'] = txt
        if i < 10: print txt
        if i % 1000 == 0:
            sys.stdout.write("processing %d/%d (%.2f%% done)   \r" %  (i, len(imgs), i*100.0/len(imgs)) )
            sys.stdout.flush()   
    return imgs

def build_vocab_tuples(imgs, params):
    # build vocabulary for question and answers.

    count_thr = params['word_count_threshold']

    # count up the number of words
    counts = {}
    for img in imgs:
        for w in img['processed_tokens']:
            counts[w] = counts.get(w, 0) + 1
    cw = sorted([(count,w) for w,count in counts.iteritems()], reverse=True)
    print 'top words and their counts:'
    print '\n'.join(map(str,cw[:20]))

    # print some stats
    total_words = sum(counts.itervalues())
    print 'total words:', total_words
    bad_words = [w for w,n in counts.iteritems() if n <= count_thr]
    vocab = [w for w,n in counts.iteritems() if n > count_thr]
    bad_count = sum(counts[w] for w in bad_words)
    print 'number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words)*100.0/len(counts))
    print 'number of words in vocab would be %d' % (len(vocab), )
    print 'number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count*100.0/total_words)


    # lets now produce the final annotation
    # additional special UNK token we will use below to map infrequent words to
    print 'inserting the special UNK token'
    vocab.append('UNK')
  
    for img in imgs:
        txt = img['processed_tokens']
        question = [w if counts.get(w,0) > count_thr else 'UNK' for w in txt]
        img['final_tuple'] = question

    return imgs, vocab

def encode_tuples(imgs, params, wtoi):

    max_length = int(params['tuple_size'])
    N = len(imgs)
    vocab_size = len(wtoi.keys())
    label_arrays = np.zeros((N, max_length*vocab_size), dtype='uint32')
    for i,img in enumerate(imgs):
        tup_length = len(img['final_tuple'])
        for k in range(max_length):
            if k < tup_length:
                label_arrays[i][k*vocab_size+wtoi[img['final_tuple'][k]]] = 1
            #else:
            #    label_arrays[i][k*vocab_size+wtoi['UNK']] = 1
    
    return label_arrays

def split_pairs(json_data, tuple_size):
    N = len(json_data)
    all_tuples, labels, im_ids, ttype = [],[],[], []
    for json_obj in json_data:
        all_tuples.append({'tuple':json_obj['rel_tuple'].replace('_',' ')})
        all_tuples.append({'tuple':json_obj['rel_tuple'].replace('_',' ')})
        labels.append(1)
        labels.append(2)
        im_ids.append(str(json_obj['rel_imid']))
        im_ids.append(str(json_obj['irr_imid']))
    return all_tuples, labels, im_ids

def construct_splits(encoded_tuples, labels, im_ids):
    piv = int(0.75*len(labels))
    print("Total data points in train: "+str(len(encoded_tuples[0:piv]))+", Number of relevant tuples: "+str(labels[0:piv].count(1)))
    print("Total data points in val: "+str(len(encoded_tuples[piv:len(labels)]))+", Number of relevant tuples: "+str(labels[piv:len(labels)].count(1)))
    return encoded_tuples[0:piv], encoded_tuples[piv:len(labels)], \
    labels[0:piv], labels[piv:len(labels)], im_ids[0:piv], im_ids[piv:len(labels)]

def main(params):
    if params['token_method'] == 'spacy':
        print 'loading spaCy tokenizer for NLP'
        params['spacy'] = spacy.en.English(data_dir=params['spacy_data'])

    tuples_train, labels_train, imid_train = split_pairs(json.load(open(params['train_json'], 'r')),params['tuple_size'])
    ttype_train = [0]*len(labels_train)
    tuples_val, labels_val, imid_val = split_pairs(json.load(open(params['test_json'], 'r')),params['tuple_size'])
    ttype_val = [0]*len(labels_val)


    #shuffle train
    z = list(zip(tuples_train, labels_train, imid_train, ttype_train))
    shuffle(z)
    tuples_train, labels_train, imid_train, ttype_train = zip(*z)   
    #seed(123) # make reproducible
    #shuffle(all_tuples) # shuffle the order

    # tokenization and preprocessing training question
    tuples_train = prepro_tuple(tuples_train, params)
    tuples_val = prepro_tuple(tuples_val, params)
    all_tuples = []
    all_tuples.extend(tuples_train)
    all_tuples.extend(tuples_val)
    # create the vocab for question
    all_tuples, vocab = build_vocab_tuples(all_tuples, params)
    itow = {i+1:w for i,w in enumerate(vocab)} # a 1-indexed vocab translation table
    wtoi = {w:i+1 for i,w in enumerate(vocab)} # inverse table

    encoded_tuples = encode_tuples(all_tuples, params, wtoi)
    #tuples_train, tuples_val, labels_train, labels_val, imid_train, \
    #imid_val = construct_splits(encoded_tuples, labels, im_ids)
    num_train = len(labels_train)
    num_val = len(labels_val)
    tuples_train = encoded_tuples[0:num_train]
    tuples_val = encoded_tuples[num_train:num_train+num_val]
    # create output h5 file for training set.
    N = len(all_tuples)
    f = h5py.File(params['output_h5'], "w")
    f.create_dataset("tuples_train", dtype='uint32', data=tuples_train)
    f.create_dataset("labels_train", dtype='uint32', data=labels_train)
    f.create_dataset("tuples_val", dtype='uint32', data=tuples_val)
    f.create_dataset("labels_val", dtype='uint32', data=labels_val)
    f.create_dataset("ttype_train", dtype='uint32', data=ttype_train)
    f.create_dataset("ttype_val", dtype='uint32', data=ttype_val)
    f.close()
    print 'wrote ', params['output_h5']

    # create output json file
    out = {}
    out['ix_to_word'] = itow # encode the (1-indexed) vocab
    out['word_to_ix'] = wtoi
    out['imid_train'] = imid_train
    out['imid_val'] = imid_val
    json.dump(out, open(params['output_json'], 'w'))
    print 'wrote ', params['output_json']

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--train_json', required=True, help='input json file to process into hdf5')
    parser.add_argument('--test_json', required=True, help='input json file to process into hdf5')

    parser.add_argument('--tuple_size', default=2, help='Size of tuples being processed')
    parser.add_argument('--output_json', default='new2.json', help='output json file')
    parser.add_argument('--output_h5', default='new2.h5', help='output h5 file')
  
    # options
    #parser.add_argument('--max_length', default=26, type=int, help='max length of a caption, in number of words. captions longer than this get clipped.')
    parser.add_argument('--word_count_threshold', default=0, type=int, help='only words that occur more than this number of times will be put in vocab')
    #parser.add_argument('--num_test', default=0, type=int, help='number of test images (to withold until very very end)')
    parser.add_argument('--token_method', default='nltk', help='token method. set "spacy" for unigram paraphrasing')
    parser.add_argument('--spacy_data', default='spacy_data', help='location of spacy NLP model')

    parser.add_argument('--batch_size', default=10, type=int)

    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict
    print 'parsed input parameters:'
    print json.dumps(params, indent = 2)
    main(params)

