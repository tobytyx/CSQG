from nltk.tokenize import TweetTokenizer
import json


class GWTokenizer(object):
    """ """
    def __init__(self, dictionary_file):
        with open(dictionary_file, 'r') as f:
            self.word2i = json.load(f)['word2i']
        self.wpt = TweetTokenizer(preserve_case=False)

        if "<stop_dialogue>" not in self.word2i:
            self.word2i["<stop_dialogue>"] = len(self.word2i)

        for word in ["<q_object>", "<action>",  "<color>", "<size>"
                     "<texture>", "<shape>", "<location>", "<other>"]:
            if word in self.word2i:
                raise("word {} dump".format(word))
            self.word2i[word] = len(self.word2i)

        self.i2word = {}
        for (k, v) in self.word2i.items():
            self.i2word[v] = k

        # Retrieve key values
        self.c_other_token = self.word2i["<other>"]
        self.no_words = len(self.word2i)  # 所有单词的数目
        self.start_token = self.word2i["<start>"]
        self.stop_token = self.word2i["<stop>"]
        self.stop_dialogue = self.word2i["<stop_dialogue>"]
        self.padding_token = self.word2i["<padding>"]
        self.yes_token = self.word2i["<yes>"]
        self.no_token = self.word2i["<no>"]
        self.non_applicable_token = self.word2i["<n/a>"]

        self.answers = [self.yes_token, self.no_token, self.non_applicable_token]
        self.vocab_list = [k for k in self.word2i.keys()]

    """
    Input: String
    Output: List of tokens
    """
    def apply(self, question, is_answer=False):
        # 将输入的question转为数字
        tokens = []
        if is_answer:
            token = '<' + question.lower() + '>'
            tokens.append(self.word2i[token])
        else:
            for token in self.wpt.tokenize(question):
                if token not in self.word2i:
                    token = '<unk>'
                tokens.append(self.word2i[token])

        return tokens

    def decode(self, tokens):
        # 将数字转为英语单词
        return ' '.join([self.i2word[tok] for tok in tokens if tok != self.stop_token])

    def split_questions(self, dialogue_tokens):
        # 将整个句子的token拆分为多个qa
        qas = []
        qa = []
        for token in dialogue_tokens:
            assert token != self.padding_token, "Unexpected padding token"
            # check for end of dialogues
            if token == self.stop_dialogue:
                break
            if token == self.start_token:
                continue
            qa.append(token)
            # check for end of question
            if token in self.answers:
                qas.append(qa)
                qa = []
        return qas