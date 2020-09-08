import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import multiprocessing

#torch.manual_seed(1)

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 5
HIDDEN_DIM = 4


class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, word_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tag_to_ix[START_TAG] = len(tag_to_ix)
        self.tag_to_ix[STOP_TAG] = len(tag_to_ix)
        #print(self.tag_to_ix)
        self.ix_to_tag = {v: k for k, v in tag_to_ix.items()}
        self.tagset_size = len(tag_to_ix)
        ###############
        param1 = nn.Parameter(torch.FloatTensor(embedding_dim), True)
        param2 = nn.Parameter(torch.FloatTensor(hidden_dim), True)
        params = [param1, param1]
        self.parameters = nn.ParameterList(params)
        self.optimizer = optim.SGD(self.parameters(), lr=0.01, weight_decay=1e-4)
        self.word_to_ix = word_to_ix
        ##################
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        ###################################################################
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        '''print("Embeds: "+str(embeds))
        print("Word_embeds: "+str(self.word_embeds))
        print("Sentence: "+str(sentence))
        print("Word_embeds(sentence): " + str(self.word_embeds(sentence)))'''
        ###################################################################
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            try:
                score = score + \
                    self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
            except IndexError:
                '''print(tags)
                print(feat)
                print(feats)
                exit(0)'''
                continue
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

    def train_model(self, training_data, epoch_range):
        for epoch in range(epoch_range):
            print("Epoch: "+str(epoch))
            for characters, tags in training_data:
                # Step 1. Remember that Pytorch accumulates gradients.
                # We need to clear them out before each instance

                #self.model.zero_grad()
                self.zero_grad()

                # Step 2. Get our inputs ready for the network, that is,
                # turn them into Tensors of word indices.
                char_in = prepare_sequence(characters, self.word_to_ix)
                #print(char_in)
                targets = torch.tensor([self.tag_to_ix[t] for t in tags], dtype=torch.long)

                # Step 3. Run our forward pass.
                #loss = self.model.neg_log_likelihood(char_in, targets)
                loss = self.neg_log_likelihood(char_in, targets)

                # Step 4. Compute the loss, gradients, and update the parameters by
                # calling optimizer.step()
                loss.backward()
                self.optimizer.step()


    def predictions(self, test_Data):
        output = []
        #with torch.no_grad:
        for characters, tags in test_Data:
            formatted_word = prepare_sequence(characters, self.word_to_ix)
            #ans = self.model(formatted_word)
            #print(ans)
            ans = self(formatted_word)
            #print(ans)
            str = []
            for tag in ans[1]:
                str.append(self.ix_to_tag[tag])
            output.append(str)
        return output


# Make up some training data
'''
training_data = [(
    "the wall street journal reported today that apple corporation made money".split(),
    "B I I I O O O B I O O".split()
), (
    "georgia tech is a university in georgia".split(),
    "B I O O O O B".split(),
), (
    "georgia tech is a university in georgia".split(),
    "B I O O O O B".split()
)]

training_data = [(
    "n g e z i n k o n z o".split(),
    "B M E B M E B M M M E".split()
), (
    "w u k u t h o l a k a l a".split(),
    "B E B E B M M E B M M E S".split(), 
), (
    "k o m t h o m b o".split(),
    "B E S B M M M M E".split()
)]

word_to_ix = {}
for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
print(word_to_ix)

alphabet = 'abcdefghijklmnopqrstuvwxyz'
word_to_ix = {}
for letters in alphabet:
    word_to_ix[letters] = len(word_to_ix)
print(word_to_ix)



#tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}
tag_to_ix = {"S":0, "B": 1, "M": 2, "E": 3, START_TAG: 4, STOP_TAG: 5}
ix_to_tag = {v: k for k, v in tag_to_ix.items()}

model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

# Check predictions before training
with torch.no_grad():

    #precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    #precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
    precheck_sent = prepare_sequence("n g e n h l o s o".split(), word_to_ix)
    precheck_tags = torch.tensor([tag_to_ix[t] for t in "B M E S B M M M E".split()], dtype=torch.long)
    print(precheck_sent)
    print(precheck_tags)
    print(model(precheck_sent))

# Make sure prepare_sequence from earlier in the LSTM section is loaded
for epoch in range(
        300):  # again, normally you would NOT do 300 epochs, it is toy data
    for sentence, tags in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is,
        # turn them into Tensors of word indices.
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)

        # Step 3. Run our forward pass.
        loss = model.neg_log_likelihood(sentence_in, targets)

        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss.backward()
        optimizer.step()

# Check predictions after training
with torch.no_grad():
    #precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)

    precheck_sent = prepare_sequence("n g e n h l o s o".split(), word_to_ix)
    precheck_tags = torch.tensor([tag_to_ix[t] for t in "B M E S B M M M E".split()], dtype=torch.long)

    x = model(precheck_sent)
    print("precheck sent: "+str(precheck_sent))
    print(x)
    print(x[1])
    str = []
    for tag in x[1]:
        str.append(ix_to_tag[tag])
    print(str)

with torch.enable_grad():
    precheck_sent = prepare_sequence("n g e n h l o s o".split(), word_to_ix)
    precheck_tags = torch.tensor([tag_to_ix[t] for t in "B M E S B M M M E".split()], dtype=torch.long)

    x = model(precheck_sent)
    print(precheck_sent)
    print(x)
    print(x[1])
    str = []
    for tag in x[1]:
        str.append(ix_to_tag[tag])

    #print(model(precheck_sent))
# We got it!'''