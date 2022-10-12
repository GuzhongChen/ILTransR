import warnings
warnings.filterwarnings('ignore')

import argparse
import time
import random
import os
import io
import logging
import numpy as np
import mxnet as mx
from mxnet import gluon
import gluonnlp as nlp
from utils import nmt
from gluonnlp.model.transformer import ParallelTransformer, get_transformer_encoder_decoder
import math
nlp.utils.check_version('0.7.0')

np.random.seed(100)
random.seed(100)
mx.random.seed(10000)
ctx = mx.gpu(0)

# parameters for dataset
dataset = 'pubchem_100'
src_lang, tgt_lang = 'random_smiles', 'rdkit_canonical_smiles'
src_max_len, tgt_max_len = 100, 100

# parameters for model
num_units=128
hidden_size=1024
dropout=0.1
epsilon=0.1
num_layers=3
num_heads=4
scaled=True
share_embed=True
embed_size=128
tie_weights=True
embed_initializer=None

# parameters for training
batch_size, test_batch_size = 256, 64
num_buckets = 10
epochs = 10
clip = 5
lr = 0.001
lr_update_factor = 0.5
log_interval = 100
magnitude = 3.0
home_dir = ''
save_dir = os.path.join(home_dir,'pretraining')

#parameters for testing
beam_size = 4
lp_alpha = 0.6
lp_k = 5

nmt.utils.logging_config(folder=save_dir,name='il_transformer_128_1024')

def cache_dataset(dataset, prefix):
    """Cache the processed npy dataset  the dataset into an npz file

    Parameters
    ----------
    dataset : gluon.data.SimpleDataset
    file_path : str
    """
    if not os.path.exists(nmt._constants.CACHE_PATH):
        os.makedirs(nmt._constants.CACHE_PATH)
    src_data = np.concatenate([e[0] for e in dataset])
    tgt_data = np.concatenate([e[1] for e in dataset])
    src_cumlen = np.cumsum([0]+[len(e[0]) for e in dataset])
    tgt_cumlen = np.cumsum([0]+[len(e[1]) for e in dataset])
    np.savez(os.path.join(nmt._constants.CACHE_PATH, prefix + '.npz'),
             src_data=src_data, tgt_data=tgt_data,
             src_cumlen=src_cumlen, tgt_cumlen=tgt_cumlen)


def load_cached_dataset(prefix):
    cached_file_path = os.path.join(nmt._constants.CACHE_PATH, prefix + '.npz')
    if os.path.exists(cached_file_path):
        print('Load cached data from {}'.format(cached_file_path))
        npz_data = np.load(cached_file_path)
        src_data, tgt_data, src_cumlen, tgt_cumlen = [npz_data[n] for n in
                ['src_data', 'tgt_data', 'src_cumlen', 'tgt_cumlen']]
        src_data = np.array([src_data[low:high] for low, high in zip(src_cumlen[:-1], src_cumlen[1:])])
        tgt_data = np.array([tgt_data[low:high] for low, high in zip(tgt_cumlen[:-1], tgt_cumlen[1:])])
        return gluon.data.ArrayDataset(np.array(src_data), np.array(tgt_data))
    else:
        return None

class TrainValDataTransform(object):
    """Transform the machine translation dataset.

    Clip source and the target sentences to the maximum length. For the source sentence, append the
    EOS. For the target sentence, append BOS and EOS.

    Parameters
    ----------
    src_vocab : Vocab
    tgt_vocab : Vocab
    src_max_len : int
    tgt_max_len : int
    """

    def __init__(self, src_vocab, tgt_vocab, src_max_len, tgt_max_len):
        # On initialization of the class, we set the class variables
        self._src_vocab = src_vocab
        self._tgt_vocab = tgt_vocab
        self._src_max_len = src_max_len
        self._tgt_max_len = tgt_max_len

    def __call__(self, src, tgt):
        # On actual calling of the class, we perform the clipping then the appending of the EOS and BOS tokens.
        if self._src_max_len > 0:
            src_sentence = self._src_vocab[src.split()[:self._src_max_len]]
        else:
            src_sentence = self._src_vocab[src.split()]
        if self._tgt_max_len > 0:
            tgt_sentence = self._tgt_vocab[tgt.split()[:self._tgt_max_len]]
        else:
            tgt_sentence = self._tgt_vocab[tgt.split()]
        src_sentence.append(self._src_vocab[self._src_vocab.eos_token])
        tgt_sentence.insert(0, self._tgt_vocab[self._tgt_vocab.bos_token])
        tgt_sentence.append(self._tgt_vocab[self._tgt_vocab.eos_token])
        src_npy = np.array(src_sentence, dtype=np.int32)
        tgt_npy = np.array(tgt_sentence, dtype=np.int32)
        return src_npy, tgt_npy

def process_dataset(dataset, src_vocab, tgt_vocab, src_max_len=-1, tgt_max_len=-1):
    start = time.time()
    dataset_processed = dataset.transform(TrainValDataTransform(src_vocab, tgt_vocab,
                                                                src_max_len,
                                                                tgt_max_len), lazy=False)
    end = time.time()
    print('Processing time spent: {}'.format(end - start))
    return dataset_processed

def _get_pair_key(src_lang, tgt_lang):
    return '_'.join(sorted([src_lang, tgt_lang]))

from mxnet.gluon.data import ArrayDataset
class _TranslationDataset(ArrayDataset):
    def __init__(self, namespace, segment, src_lang, tgt_lang, root):
        self._segment = segment
        self._src_lang = src_lang
        self._tgt_lang = tgt_lang
        self._src_vocab = None
        self._tgt_vocab = None
        self._pair_key = _get_pair_key(src_lang, tgt_lang)
        root = os.path.expanduser(root)
        os.makedirs(root, exist_ok=True)
        self._root = root
        if isinstance(segment, str):
            segment = [segment]
        src_corpus = []
        tgt_corpus = []
        for ele_segment in segment:
            [src_corpus_path, tgt_corpus_path] = self._get_data(ele_segment)
            src_corpus.extend(TextLineDataset(src_corpus_path))
            tgt_corpus.extend(TextLineDataset(tgt_corpus_path))
        # Filter 0-length src/tgt sentences
        src_lines = []
        tgt_lines = []
        for src_line, tgt_line in zip(list(src_corpus), list(tgt_corpus)):
            if len(src_line) > 0 and len(tgt_line) > 0:
                src_lines.append(src_line)
                tgt_lines.append(tgt_line)
        super(_TranslationDataset, self).__init__(src_lines, tgt_lines)

    def _fetch_data_path(self,file_name):
        paths = []
        root = self._root
        for data_file_name in file_name:
            path = os.path.join(root, data_file_name)
            paths.append(path)
        return paths

    def _get_data(self, segment):
        src_corpus_file_name =\
            self._data_file[self._pair_key][segment + '_' + self._src_lang]
        tgt_corpus_file_name =\
            self._data_file[self._pair_key][segment + '_' + self._tgt_lang]
        return self._fetch_data_path([(src_corpus_file_name),
                                      (tgt_corpus_file_name)])

    @property
    def src_vocab(self):
        """Source Vocabulary of the Dataset.

        Returns
        -------
        src_vocab : Vocab
            Source vocabulary.
        """
        if self._src_vocab is None:
            src_vocab_file_name = \
                self._data_file[self._pair_key]['vocab' + '_' + self._src_lang]
            [src_vocab_path] = self._fetch_data_path([(src_vocab_file_name)])
            with io.open(src_vocab_path, 'r', encoding='utf-8') as in_file:
                self._src_vocab = nlp.Vocab.from_json(in_file.read())
        return self._src_vocab

    @property
    def tgt_vocab(self):
        """Target Vocabulary of the Dataset.

        Returns
        -------
        tgt_vocab : Vocab
            Target vocabulary.
        """
        if self._tgt_vocab is None:
            tgt_vocab_file_name = \
                self._data_file[self._pair_key]['vocab' + '_' + self._tgt_lang]
            [tgt_vocab_path] = self._fetch_data_path([(tgt_vocab_file_name)])
            with io.open(tgt_vocab_path, 'r', encoding='utf-8') as in_file:
                self._tgt_vocab = nlp.Vocab.from_json(in_file.read())
        return self._tgt_vocab

from gluonnlp.data.dataset import TextLineDataset
class PUBCHEM(_TranslationDataset):
    """
    Parameters
    ----------
    segment : str or list of str, default 'train'
        Dataset segment. Options are 'train', 'val', 'test' or their combinations.
    src_lang : str, default 'random_smiles'
        The source language. Option for source and target languages are 'random_smiles' <-> 'rdkit_canonical_smiles'
    tgt_lang : str, default 'rdkit_canonical_smiles'
        The target language. Option for source and target languages are 'random_smiles' <-> 'rdkit_canonical_smiles'

    """
    def __init__(self, segment='train', src_lang='random_smiles', tgt_lang='rdkit_canonical_smiles',
                 root=os.path.join(home_dir, 'datasets', 'pubchem')):
        self._supported_segments = ['train', 'val', 'test']
        self._data_file = {_get_pair_key('random_smiles', 'rdkit_canonical_smiles'):
                               {'train_random_smiles': ('train.random_smiles'),
                                'train_rdkit_canonical_smiles': ('train.rdkit_canonical_smiles'),
                                'val_random_smiles': ('val.random_smiles'),
                                'val_rdkit_canonical_smiles': ('val.rdkit_canonical_smiles'),
                                'test_random_smiles': ('test.random_smiles'),
                                'test_rdkit_canonical_smiles': ('test.rdkit_canonical_smiles'),
                                'vocab_random_smiles': ('vocab.random_smiles.json'),
                                'vocab_rdkit_canonical_smiles' : ('vocab.rdkit_canonical_smiles.json')
                        }}
        super(PUBCHEM, self).__init__('pubchem', segment=segment, src_lang=src_lang,
                                        tgt_lang=tgt_lang, root=root)

def load_translation_data(dataset, src_lang='random_smiles', tgt_lang='rdkit_canonical_smiles'):
    """Load translation dataset

    Parameters
    ----------
    dataset : str
    src_lang : str, default 'random_smiles'
    tgt_lang : str, default 'rdkit_canonical_smiles'

    Returns
    -------
    data_train_processed : Dataset
        The preprocessed training sentence pairs
    data_val_processed : Dataset
        The preprocessed validation sentence pairs
    data_test_processed : Dataset
        The preprocessed test sentence pairs
    val_tgt_sentences : list
        The target sentences in the validation set
    test_tgt_sentences : list
        The target sentences in the test set
    src_vocab : Vocab
        Vocabulary of the source language
    tgt_vocab : Vocab
        Vocabulary of the target language
    """
    common_prefix = 'pubchem_100'.format(src_lang, tgt_lang,
                                                   src_max_len, tgt_max_len)

    # Load the three datasets from files
    data_train = PUBCHEM('train', src_lang=src_lang, tgt_lang=tgt_lang)
    data_val = PUBCHEM('val', src_lang=src_lang, tgt_lang=tgt_lang)
    data_test = PUBCHEM('test', src_lang=src_lang, tgt_lang=tgt_lang)
    src_vocab, tgt_vocab = data_train.src_vocab, data_train.tgt_vocab
    data_train_processed = load_cached_dataset(common_prefix + '_train')

    # Check if each dataset has been processed or not, and if not, process and cache them.
    if not data_train_processed:
        data_train_processed = process_dataset(data_train, src_vocab, tgt_vocab,
                                               src_max_len, tgt_max_len)
        cache_dataset(data_train_processed, common_prefix + '_train')
    data_val_processed = load_cached_dataset(common_prefix + '_val')
    if not data_val_processed:
        data_val_processed = process_dataset(data_val, src_vocab, tgt_vocab)
        cache_dataset(data_val_processed, common_prefix + '_val')
    data_test_processed = load_cached_dataset(common_prefix + '_test')
    if not data_test_processed:
        data_test_processed = process_dataset(data_test, src_vocab, tgt_vocab)
        cache_dataset(data_test_processed, common_prefix + '_test')

    # Pull out the target sentences for both test and validation
    fetch_tgt_sentence = lambda src, tgt: tgt.split()
    val_tgt_sentences = list(data_val.transform(fetch_tgt_sentence))
    test_tgt_sentences = list(data_test.transform(fetch_tgt_sentence))

    # Return all of the necessary pieces we can extract from the data for training our model
    return data_train_processed, data_val_processed, data_test_processed, \
           val_tgt_sentences, test_tgt_sentences, src_vocab, tgt_vocab

def get_data_lengths(dataset):
    return list(dataset.transform(lambda srg, tgt: (len(srg), len(tgt))))

def evaluate(data_loader):
    """Evaluate given the data loader

    Parameters
    ----------
    data_loader : gluon.data.DataLoader

    Returns
    -------
    avg_loss : float
        Average loss
    real_translation_out : list of list of str
        The translation output
    """
    translation_out = []
    all_inst_ids = []
    avg_loss_denom = 0
    avg_loss = 0.0

    for _, (src_seq, tgt_seq, src_valid_length, tgt_valid_length, inst_ids) \
            in enumerate(data_loader):
        src_seq = src_seq.as_in_context(ctx)
        tgt_seq = tgt_seq.as_in_context(ctx)
        src_valid_length = src_valid_length.as_in_context(ctx)
        tgt_valid_length = tgt_valid_length.as_in_context(ctx)

        # Calculate Loss
        out, _ = model(src_seq, tgt_seq[:, :-1], src_valid_length, tgt_valid_length - 1)
        loss = loss_function(out, tgt_seq[:, 1:], tgt_valid_length - 1).mean().asscalar()
        all_inst_ids.extend(inst_ids.asnumpy().astype(np.int32).tolist())
        avg_loss += loss * (tgt_seq.shape[1] - 1)
        avg_loss_denom += (tgt_seq.shape[1] - 1)

        # Translate the sequences and score them
        samples, _, sample_valid_length =\
            translator.translate(src_seq=src_seq, src_valid_length=src_valid_length)
        max_score_sample = samples[:, 0, :].asnumpy()
        sample_valid_length = sample_valid_length[:, 0].asnumpy()

        # Iterate through the tokens and stitch the tokens together for the sentence
        for i in range(max_score_sample.shape[0]):
            translation_out.append(
                [tgt_vocab.idx_to_token[ele] for ele in
                 max_score_sample[i][1:(sample_valid_length[i] - 1)]])

    # Calculate the average loss and initialize a None-filled translation list
    avg_loss = avg_loss / avg_loss_denom
    real_translation_out = [None for _ in range(len(all_inst_ids))]

    # Combine all the words/tokens into a sentence for the final translation
    for ind, sentence in zip(all_inst_ids, translation_out):
        real_translation_out[ind] = sentence

    # Return the loss and the translation
    return avg_loss, real_translation_out


def write_sentences(sentences, file_path):
    with io.open(file_path, 'w', encoding='utf-8') as of:
        for sent in sentences:
            of.write(' '.join(sent) + '\n')

if __name__ == '__main__':

    data_train, data_val, data_test, val_tgt_sentences, test_tgt_sentences, src_vocab, tgt_vocab\
    = load_translation_data(dataset=dataset, src_lang=src_lang, tgt_lang=tgt_lang)
    data_train_lengths = get_data_lengths(data_train)
    data_val_lengths = get_data_lengths(data_val)
    data_test_lengths = get_data_lengths(data_test)

    with io.open(os.path.join(save_dir, 'val_smiles.txt'), 'w', encoding='utf-8') as of:
        for ele in val_tgt_sentences:
            of.write(' '.join(ele) + '\n')

    with io.open(os.path.join(save_dir, 'test_smiles.txt'), 'w', encoding='utf-8') as of:
        for ele in test_tgt_sentences:
            of.write(' '.join(ele) + '\n')


    data_train = data_train.transform(lambda src, tgt: (src, tgt, len(src), len(tgt)), lazy=False)
    data_val = gluon.data.SimpleDataset([(ele[0], ele[1], len(ele[0]), len(ele[1]), i)
                                     for i, ele in enumerate(data_val)])
    data_test = gluon.data.SimpleDataset([(ele[0], ele[1], len(ele[0]), len(ele[1]), i)
                                      for i, ele in enumerate(data_test)])
    train_batchify_fn = nlp.data.batchify.Tuple(nlp.data.batchify.Pad(pad_val=0),
                                            nlp.data.batchify.Pad(pad_val=0),
                                            nlp.data.batchify.Stack(dtype='float32'),
                                            nlp.data.batchify.Stack(dtype='float32'))
    test_batchify_fn = nlp.data.batchify.Tuple(nlp.data.batchify.Pad(pad_val=0),
                                           nlp.data.batchify.Pad(pad_val=0),
                                           nlp.data.batchify.Stack(dtype='float32'),
                                           nlp.data.batchify.Stack(dtype='float32'),
                                           nlp.data.batchify.Stack())
    bucket_scheme = nlp.data.ExpWidthBucket(bucket_len_step=1.2)
    train_batch_sampler = nlp.data.FixedBucketSampler(lengths=data_train_lengths,
                                                  batch_size=batch_size,
                                                  num_buckets=num_buckets,
                                                  shuffle=True,
                                                  bucket_scheme=bucket_scheme)
    logging.info('Train Batch Sampler:\n{}'.format(train_batch_sampler.stats()))
    val_batch_sampler = nlp.data.FixedBucketSampler(lengths=data_val_lengths,
                                                batch_size=test_batch_size,
                                                num_buckets=num_buckets,
                                                shuffle=False)
    logging.info('Valid Batch Sampler:\n{}'.format(val_batch_sampler.stats()))
    test_batch_sampler = nlp.data.FixedBucketSampler(lengths=data_test_lengths,
                                                 batch_size=test_batch_size,
                                                 num_buckets=num_buckets,
                                                 shuffle=False)
    logging.info('Test Batch Sampler:\n{}'.format(test_batch_sampler.stats()))

    train_data_loader = gluon.data.DataLoader(data_train,
                                          batch_sampler=train_batch_sampler,
                                          batchify_fn=train_batchify_fn,
                                          num_workers=0)
    val_data_loader = gluon.data.DataLoader(data_val,
                                        batch_sampler=val_batch_sampler,
                                        batchify_fn=test_batchify_fn,
                                        num_workers=0)
    test_data_loader = gluon.data.DataLoader(data_test,
                                         batch_sampler=test_batch_sampler,
                                         batchify_fn=test_batchify_fn,
                                         num_workers=0)

    encoder, decoder, one_step_ahead_decoder = get_transformer_encoder_decoder(
    units=num_units,
    hidden_size=hidden_size,
    dropout=dropout,
    num_layers=num_layers,
    num_heads=num_heads,
    max_src_length=src_max_len,
    max_tgt_length=tgt_max_len,
    scaled=scaled)
    model = nlp.model.translation.NMTModel(src_vocab=src_vocab,
                 tgt_vocab=tgt_vocab,
                 encoder=encoder,
                 decoder=decoder,
                 one_step_ahead_decoder=one_step_ahead_decoder,
                 embed_size=num_units,
                 embed_initializer=None,
                 prefix='transformer_')
    model.initialize(init=mx.init.Xavier(magnitude=magnitude), ctx=ctx)
    static_alloc = True
    model.hybridize(static_alloc=static_alloc)
    logging.info(model)

    # Due to the paddings, we need to mask out the losses corresponding to padding tokens.
    loss_function = nlp.loss.MaskedSoftmaxCELoss()
    loss_function.hybridize(static_alloc=static_alloc)

    translator = nmt.translation.BeamSearchTranslator(model=model, beam_size=beam_size,
                                                  scorer=nlp.model.BeamSearchScorer(alpha=lp_alpha,
                                                                                    K=lp_k),
                                                  max_length=tgt_max_len)
    logging.info('Use beam_size={}, alpha={}, K={}'.format(beam_size, lp_alpha, lp_k))

    trainer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': lr})

    best_valid_bleu = 0.0
    step_num = 0
    warmup_steps = 16000
    grad_interval = 5
    # Run through each epoch
    for epoch_id in range(epochs):
        log_avg_loss = 0
        log_avg_gnorm = 0
        log_wc = 0
        log_start_time = time.time()

        # Iterate through each batch
        for batch_id, (src_seq, tgt_seq, src_valid_length, tgt_valid_length)\
                in enumerate(train_data_loader):

            src_seq = src_seq.as_in_context(ctx)
            tgt_seq = tgt_seq.as_in_context(ctx)
            src_valid_length = src_valid_length.as_in_context(ctx)
            tgt_valid_length = tgt_valid_length.as_in_context(ctx)

            # Compute gradients and losses
            with mx.autograd.record():
                out, _ = model(src_seq, tgt_seq[:, :-1], src_valid_length, tgt_valid_length - 1)
                loss = loss_function(out, tgt_seq[:, 1:], tgt_valid_length - 1).mean()
                loss = loss * (tgt_seq.shape[1] - 1) / (tgt_valid_length - 1).mean()
                loss.backward()

            grads = [p.grad(ctx) for p in model.collect_params().values() if p.grad_req != 'null']
            gnorm = gluon.utils.clip_global_norm(grads, clip)
            trainer.step(1)
            src_wc = src_valid_length.sum().asscalar()
            tgt_wc = (tgt_valid_length - 1).sum().asscalar()
            step_loss = loss.asscalar()
            log_avg_loss += step_loss
            log_avg_gnorm += gnorm
            log_wc += src_wc + tgt_wc
            if (batch_id + 1) % log_interval == 0:
                wps = log_wc / (time.time() - log_start_time)
                logging.info('[Epoch {} Batch {}/{}] loss={:.4f}, ppl={:.4f}, gnorm={:.4f}, '
                            'throughput={:.2f}K wps, wc={:.2f}K'
                            .format(epoch_id, batch_id + 1, len(train_data_loader),
                                    log_avg_loss / log_interval,
                                    np.exp(log_avg_loss / log_interval),
                                    log_avg_gnorm / log_interval,
                                    wps / 1000, log_wc / 1000))
                log_start_time = time.time()
                log_avg_loss = 0
                log_avg_gnorm = 0
                log_wc = 0
            '''
            # Update the learning rate based on the number of batchs that have passed
            if batch_id % grad_interval == 0:
                step_num += 1
                new_lr = max(lr / math.sqrt(num_units) * min(1. / math.sqrt(step_num), step_num * warmup_steps ** (-1.5)),0.00000001)
                logging.info('Learning rate change to {}'.format(new_lr))
                trainer.set_learning_rate(new_lr)
            '''
        # Evaluate the losses on validation and test datasets and find the corresponding BLEU score and log it
        valid_loss, valid_translation_out = evaluate(val_data_loader)
        valid_bleu_score, _, _, _, _ = nmt.bleu.compute_bleu([val_tgt_sentences], valid_translation_out)
        logging.info('[Epoch {}] valid Loss={:.4f}, valid ppl={:.4f}, valid bleu={:.2f}'
                    .format(epoch_id, valid_loss, np.exp(valid_loss), valid_bleu_score * 100))
        test_loss, test_translation_out = evaluate(test_data_loader)
        test_bleu_score, _, _, _, _ = nmt.bleu.compute_bleu([test_tgt_sentences], test_translation_out)
        logging.info('[Epoch {}] test Loss={:.4f}, test ppl={:.4f}, test bleu={:.2f}'
                    .format(epoch_id, test_loss, np.exp(test_loss), test_bleu_score * 100))

        # Output the sentences we predicted on the validation and test datasets             
        write_sentences(valid_translation_out,
                        os.path.join(save_dir, 'epoch{:d}_valid_out.txt').format(epoch_id))
        write_sentences(test_translation_out,
                        os.path.join(save_dir, 'epoch{:d}_test_out.txt').format(epoch_id))

        # Save the model if the BLEU score is better than the previous best
        if valid_bleu_score > best_valid_bleu:
            best_valid_bleu = valid_bleu_score
            save_path = os.path.join(save_dir, 'valid_best.params')
            logging.info('Save best parameters to {}'.format(save_path))
            model.save_parameters(save_path)

        # Update the learning rate based on the number of epochs that have passed
        
        if epoch_id + 1 >= (epochs * 2) // 3:
            new_lr = trainer.learning_rate * lr_update_factor
            logging.info('Learning rate change to {}'.format(new_lr))
            trainer.set_learning_rate(new_lr)


