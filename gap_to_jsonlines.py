import sys
import json
import os
from bert import tokenization

def read_tsv(tsv_file):
    data = []
    first = True
    with  open(tsv_file) as f:
        for line in f:
            cols = line.split('\t')
            if first:
                fields = list(enumerate(cols))
                first = False
                continue
            data += [{col : (cols[index] if 'offset' not in col else int(cols[index])) for index, col in fields}]
    return data

def is_start(char_offset, char_to_word_offset, text):
    # print(char_offset, char_to_word_offset[char_offset-1: char_offset +1], text[char_offset-10:char_offset+10])
    return char_offset == 0 or char_to_word_offset[char_offset] != char_to_word_offset[char_offset - 1]

def tokenize(dataset, vocab_file):
    tokenizer = tokenization.FullTokenizer(
                vocab_file=vocab_file, do_lower_case=False) if vocab_file is not None else None
    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False
    def is_punctuation(c):
        if c == '.' or c == "," or c == "`" or c == '"' or c == "'" or c == '(' or c == ')' or c == '-' or c == '/' or c == '' or c == '*':
            return True
        return False
    for datum in dataset:
        paragraph_text = datum["Text"]
        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace, prev_is_punc = True, True
        for c in paragraph_text:
            if is_punctuation(c):
                prev_is_punc = True
                doc_tokens.append(c)
            elif is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace or prev_is_punc:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
                prev_is_punc = False
            char_to_word_offset.append(len(doc_tokens) - 1)
        tok_to_subtoken, para_subtokens, sentence_map = ([], ['[CLS]'], [0]) if tokenizer is not None else ([], [], [])
        current_sentence = 0
        clusters = []
        for tok_index, token in enumerate(doc_tokens):
            subtokens = [token] if tokenizer is None else tokenizer.tokenize(token)
            sentence_map += [current_sentence] * len(subtokens)
            tok_to_subtoken.append((len(para_subtokens), len(para_subtokens) + len(subtokens) - 1))
            para_subtokens += subtokens
            if token == '.':
                current_sentence += 1
        datum['speakers'] = ['[SPL]'] + ['Speaker#1'] * (len(para_subtokens)-1) + ['[SPL]'] if tokenizer is not None else ['Speaker#1'] * (len(para_subtokens)-1)
        datum['sentences'] = para_subtokens + ['[SEP]'] if tokenizer is not None else para_subtokens
        datum['sentence_map'] = sentence_map  + [sentence_map[-1]] if tokenizer is not None else sentence_map
        clusters = []
        a_start, a_end =  datum['A-offset'],  datum['A-offset'] + len(datum['A'])
        b_start, b_end =  datum['B-offset'],  datum['B-offset'] + len(datum['B'])
        pronoun_start, pronoun_end =  datum['Pronoun-offset'],  datum['Pronoun-offset'] + len(datum['Pronoun'])
        entity_start, entity_end = (a_start, a_end) if datum['A-coref'] == 'TRUE' else (b_start, b_end)
        
        datum['a_subtoken_span'] = tok_to_subtoken[char_to_word_offset[a_start]][0], tok_to_subtoken[char_to_word_offset[a_end]][1]
        datum['b_subtoken_span'] = tok_to_subtoken[char_to_word_offset[b_start]][0], tok_to_subtoken[char_to_word_offset[b_end]][1]
        datum['pronoun_subtoken_span'] = tok_to_subtoken[char_to_word_offset[pronoun_start]][0], tok_to_subtoken[char_to_word_offset[pronoun_end]][1]

        if datum['A-coref'] == 'TRUE' or datum['B-coref'] == 'TRUE':
            entity_span = datum['a_subtoken_span'] if datum['A-coref'] else datum['b_subtoken_span']
            clusters.append(entity_span)
            clusters.append(datum['pronoun_subtoken_span'])
        datum['clusters'] = [clusters]
    ext = 'tok.jsonlines' if tokenizer is None else 'jsonlines'
    with open(tsv_file.replace('tsv', ext), 'w') as f:
        for datum in dataset:
            json_datum = {'doc_key': datum['ID'], 'sentences': [datum['sentences']], 'speakers': [datum['speakers']],
                    'sentence_map': datum['sentence_map'], 'clusters': datum['clusters'],
                    'a_subtoken_span': datum['a_subtoken_span'], 'b_subtoken_span': datum['b_subtoken_span'], 'pronoun_subtoken_span': datum['pronoun_subtoken_span']}
            f.write(json.dumps(json_datum) + '\n')

def convert(tsv_file, vocab_file):
    dataset = read_tsv(tsv_file)
    tokenize(dataset, vocab_file)


if __name__ == '__main__':
    tsv_file = sys.argv[1]
    vocab_file = sys.argv[2] if len(sys.argv) == 3 else None
    convert(tsv_file, vocab_file)
