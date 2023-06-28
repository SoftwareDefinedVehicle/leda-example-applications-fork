# python3 -m pip install cantools
# python3 -m pip install vss-tools
# python3 -m pip install tensorflow --no-cache-dir
# python3 -m pip install tensorflow_hub
# python3 -m pip install tensorflow_text
# python3 -m pip install --upgrade gensim
# python3 -m pip install -U sentence-transformers
# python3 -m pip install tensorrt
# /usr/local/python/3.10.8/lib/python3.10/site-packages/vspec
# export PYTHONPATH=$PYTHONPATH:/usr/local/python/3.10.8/lib/python3.10/site-packages/
# export PYTHON_PATH=$PYTHON_PATH:/usr/local/python/3.10.8/lib/python3.10/site-packages/

import argparse
import logging
import sys
import numbers
import decimal
import itertools
import os
import re
import torch
logging.info("Importing modules...Tensorflow")

#import tensorflow as tf
import tensorflow_hub as hub
#import tensorflow_text as text
import numpy as np

logging.info("Importing modules...Rest")
#from gensim import corpora, models, similarities, downloader
from collections.abc import Iterable
from collections import defaultdict
from pprint import pprint
from dataclasses import dataclass, field
from functools import total_ordering
from anytree import PreOrderIter, Resolver
from vspec.model.vsstree import VSSNode
from vspec.model.constants import VSSTreeType
from vspec.loggingconfig import initLogging
from cantools.database import Database, Message, Signal

print("Hello00")
from sentence_transformers import SentenceTransformer, util
print("Hello01")

import vspec
import cantools
from absl import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


@dataclass(eq=False, unsafe_hash=True)
@total_ordering
class Similarity():
    """Collect weighted similarities"""

    can_message: Message
    can_signal: Signal
    vss_signal: VSSNode

    similarity_unit: float = 0.0
    similarity_signal_name: float = 0.0
    similarity_parent_name: float = 0.0
    similarity_signal_description: float = 0.0
    similarity_parent_description: float = 0.0
    similarity_minmax_values: float = 0.0

    weight_unit: float = 1.0
    weight_signal_name: float = 1.0
    weight_parent_name: float = 0.5
    weight_signal_description: float = 0.7
    weight_parent_description: float = 0.3
    weight_minmax_values: float = 1.0

    def __post_init__(self):
        self.similarity_unit = self.compare_units()

        self.similarity_signal_description = self.jaccard_similarity(self.can_signal.comment,
                                                                     self.vss_signal.description)

        if self.vss_signal.parent is not None:
            self.similarity_parent_description = self.jaccard_similarity(self.can_message.comment,
                                                                         self.vss_signal.parent.description)

        self.similarity_minmax_values = (self.jaccard_similarity(self.can_signal.minimum,
                                                                 self.vss_signal.min) +
                                         self.jaccard_similarity(self.can_signal.maximum,
                                                                 self.vss_signal.max))/2

        self.similarity_signal_name = self.jaccard_similarity(self.can_message.name,
                                                              self.vss_signal.qualified_name())

        if self.vss_signal.parent is not None:
            self.similarity_parent_name = self.jaccard_similarity(self.can_signal.name,
                                                                  self.vss_signal.parent.qualified_name())

    def total_similarity(self) -> float:
        """Calculated the total similarity of the DBC and the VSS element"""
        return ((self.similarity_unit * self.weight_unit)
                + (self.similarity_signal_description *
                   self.weight_signal_description)
                + (self.similarity_parent_description *
                   self.weight_parent_description)
                + (self.similarity_minmax_values * self.weight_minmax_values)
                + (self.similarity_signal_name * self.weight_signal_name)
                + (self.similarity_parent_name * self.weight_parent_name)) / 6

    def __eq__(self, other):
        return self.total_similarity() == other.total_similarity()

    def __lt__(self, other):
        return self.total_similarity() < other.total_similarity()

    def __gt__(self, other):
        return self.total_similarity() > other.total_similarity()

    def compare_units(self) -> float:
        """Compare units, such as km/h or m/s/s"""
        if self.can_signal.unit is None:
            return 0
        if self.vss_signal.unit is None:
            return 0
        if self.can_signal.unit is not None:
            if self.vss_signal.unit is not None:
                if (self.can_signal.unit == "m/s/s") & (self.vss_signal.unit.label == "m/s^2"):
                    return 1
        return self.can_signal.unit == self.vss_signal.unit

    @staticmethod
    def jaccard_similarity(first, second) -> float:
        """ returns the jaccard similarity between two lists """

        if first is None:
            return 0

        if second is None:
            return 0

        if first == second:
            return 1

        # Equality has already been checked.
        # If it's a number, it's not similar, so we return 0 as similarity.
        if isinstance(first, int) or isinstance(second, int):
            return 0
        if isinstance(first, float) or isinstance(second, float):
            return 0

        if isinstance(first, numbers.Number):
            if isinstance(second, numbers.Number):
                return abs(first.compare(second))

        if not isinstance(first, Iterable):
            raise ValueError(
                f"First element is not an iterable: {first} of type {type(first)}")

        if not isinstance(second, Iterable):
            raise ValueError(
                f"Second element is not an iterable: {second} of type {type(second)}")

        intersection_cardinality = len(
            set.intersection(*[set(first), set(second)]))
        union_cardinality = len(set.union(*[set(first), set(second)]))
        return intersection_cardinality/float(union_cardinality)


# module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
# model = hub.load(module_url)
# print("module %s loaded" % module_url)

# def embed(input):
#     return model(input)

def sentence_case(string):
    if string != '':
        result = re.sub('([A-Z])', r' \1', string)
        return result[:1].upper() + result[1:].lower()
    return

def semantic_similarity(data_base: Database, tree: VSSNode):
    # word = "Elephant"
    # sentence = "I am a sentence for which I would like to get its embedding."
    # paragraph = (
    #     "Universal Sentence Encoder embeddings also support short paragraphs. "
    #     "There is no hard limit on how long the paragraph is. Roughly, the longer "
    #     "the more 'diluted' the embedding will be.")
    # messages = [word, sentence, paragraph]
    # # Reduce logging output.
    # logging.set_verbosity(logging.ERROR)
    # message_embeddings = embed(messages)
    # pprint(message_embeddings)
    # for i, message_embedding in enumerate(np.array(message_embeddings).tolist()):
    #     print("Message: {}".format(messages[i]))
    #     print("Embedding size: {}".format(len(message_embedding)))
    #     message_embedding_snippet = ", ".join(
    #         (str(x) for x in message_embedding[:3]))
    #     print("Embedding: [{}, ...]\n".format(message_embedding_snippet))

    # corr = np.inner(np.squeeze(np.asarray(messages)), np.squeeze(np.asarray(message_embeddings)))

    # print(corr)
    
    
    # train_examples = [ 'Mike', 'Michael', 'Markus', 'Thorsten', 'Lydia' ]
    # train_labels = [ 0,0,0,1,2 ]

    # test_examples = [ 'Maik', 'Torben', 'Larissa' ]
    # test_labels = [ 0,1,2 ]
    
    # hub_layer = hub.KerasLayer("https://tfhub.dev/google/nnlm-en-dim50/2", input_shape=[], dtype=tf.string, trainable=True)
    # hub_layer(train_examples)
    # model = tf.keras.Sequential()
    # model.add(hub_layer)
    # model.add(tf.keras.layers.Dense(16, activation='relu'))
    # model.add(tf.keras.layers.Dense(1))
    # model.summary()
    # model.compile(optimizer='adam',
    #           loss=tf.losses.BinaryCrossentropy(from_logits=True),
    #           metrics=[tf.metrics.BinaryAccuracy(threshold=0.0, name='accuracy')])
    # x_val = train_examples[:4]
    # partial_x_train = train_examples[4:]
    # y_val = train_labels[:4]
    # partial_y_train = train_labels[4:]
    # history = model.fit(partial_x_train,
    #                 partial_y_train,
    #                 epochs=4,
    #                 batch_size=8,
    #                 validation_data=(x_val, y_val),
    #                 verbose=1)
    
    # results = model.evaluate(test_examples, test_labels)
    # print(results)
    
    # ACC1.AdaptiveCruiseCtrlSetSpeed [km/h] 0-250
    # Text: "Value of the desired (chosen) velocity of the adaptive cruise control system."
    #
    # VSS Branch: Vehicle.ADAS.CruiseControl.SpeedSet
    # Text: "Set cruise control speed in kilometers per hour."

    # dbc_doc = "Value of the desired (chosen) velocity of the adaptive cruise control system."
    # vss_doc = "Set cruise control speed in kilometers per hour."
    # corpus = tf.constant([dbc_doc, vss_doc])
    # hub_url = "https://tfhub.dev/google/sentence-t5/st5-base/1"
    # encoder = hub.KerasLayer(hub_url)
    # english_embeds = encoder(corpus)
    # print (english_embeds)

    # https://huggingface.co/sentence-transformers/sentence-t5-base
    # https://www.sbert.net/docs/quickstart.html#comparing-sentence-similarities
    # sentences = [
    #     "Set cruise control speed in kilometers per hour."
    #     "Value of the desired (chosen) velocity of the adaptive cruise control system.",
    #     "The quick brown fox jumps over the fense."
    # ]
    #model = SentenceTransformer('sentence-transformers/sentence-t5-base')
    
    # https://www.sbert.net/examples/applications/paraphrase-mining/README.html
    logging.info("Loading SentenceTransformer")
    print("Hallo")
    model = SentenceTransformer('all-MiniLM-L6-v2',
                                device='cpu',
                                )

    print("Hallo2")
    
    # Two lists of sentences - QUERY
    #sentences1 = 

    # sentences2 = ['The dog plays in the garden',
    #             'A woman watches TV',
    #             'The new movie is so great']
    corpus = []
    corpus_ids = []
    corpus_can_messages = []
    corpus_can_signals = []
    can_message = data_base.get_message_by_name('ACC1')
    for can_message in data_base.messages:
        for can_signal in can_message.signals:
            print("CAN-Signal: {} : {}".format(can_signal.name, can_signal.comment))
            signal_name_with_spaces=sentence_case(can_signal.name)
            corpus.append(str(signal_name_with_spaces) + " " + str(can_signal.comment) + " " + str(can_message.comment) + " in " + str(can_signal.unit))
            corpus_ids.append(can_message.name + "." + can_signal.name)
            corpus_can_messages.append(can_message)
            corpus_can_signals.append(can_signal)
    corpus_embeddings = model.encode(corpus, convert_to_tensor=True, show_progress_bar=True)

    #Compute embedding for both lists
    #embeddings1 = model.encode(sentences, convert_to_tensor=True, show_progress_bar=True)
    #query = ['Set cruise control speed in kilometers per hour.']
    
    top_k = min(5, len(corpus))
    for vss_node in PreOrderIter(tree):
        query = sentence_case(vss_node.name)
        query += " " + str(vss_node.description)
        if vss_node.parent:
            query += sentence_case(vss_node.parent.name)
            query += " " + str(vss_node.parent.description)
        query_embedding = model.encode(query, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
        
        good_results = torch.topk(cos_scores, k=top_k)
        print("======================")
        print("VSS Datapoint:", vss_node.qualified_name())
        print("Query Text:", query)
        print("\nTop 5 best matching CAN Signals:")
        for score, idx in zip(good_results[0], good_results[1]):
            print("\tMatch: {:.4f} {} - {}".format(score,corpus_ids[idx],corpus[idx]))
            # score,corpus[idx], corpus_ids[idx], ": ", corpus[idx], "(Score: {:.4f})".format(score))

    # cosine_scores = util.cos_sim(embeddings1, embeddings2)

    # #Find the pairs with the highest cosine similarity scores
    # pairs = []
    # for i in range(len(cosine_scores)-1):
    #     for j in range(i+1, len(cosine_scores)):
    #         pairs.append({'index': [i, j], 'score': cosine_scores[i][j]})

    # #Sort scores in decreasing order
    # pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)

    # for pair in pairs[0:10]:
    #     i, j = pair['index']
    #     score=pair['score']
    #     #print("{} \t\t {} \t\t Score: {:.4f}".format(sentences[i], sentences[j], pair['score']))
    #     print("==============================")
    #     print("i={} j={}".format(i,j))
    #     print("Sentence 1: {}".format(sentences1[i]))
    #     print("Sentence 2: {}".format(sentences2[j]))
    #     print("Score: {:.4f}".format(score))
    
    #Output the pairs with their score
    # for i in range(len(sentences1)):
    #     score=cosine_scores[i][i]
    #     print("==============================")
    #     print("i={}".format(i))
    #     print("Sentence 1: {}".format(sentences1[i]))
    #     print("Sentence 2: {}".format(sentences2[i]))
    #     print("Score: {:.4f}".format(score))
        #print("{} \t\t {} \t\t Score: {:.4f}".format(sentences1[i], sentences2[i], cosine_scores[i][i]))
    
    # paraphrases = util.paraphrase_mining(model,
    #                                      sentences,
    #                                      max_pairs=10,
    #                                      top_k=3,
    #                                      show_progress_bar=True)
    
    # print("\n\n\n")
    # for paraphrase in paraphrases[0:10]:
    #     score, i, j = paraphrase
    #     print("==============================")
    #     print("i={} j={}".format(i,j))
    #     print("Sentence i: {}".format(sentences[i]))
    #     print("Sentence j: {}".format(sentences[j]))
    #     print("Score: {:.4f}".format(score))
        #print("{} \t\t {} \t\t Score: {:.4f}".format(sentences[i], sentences[j], score))
    
    # embeddings = model.encode(sentences)
    # print("Embeddings:", embeddings)
    
    #Compute cosine similarity between all pairs
    # cos_sim = util.cos_sim(embeddings, embeddings)
    # print("Cosine-Similarity:", cos_sim)
    
    #Add all pairs to a list with their cosine similarity score
    # all_sentence_combinations = []
    # for i in range(len(cos_sim)-1):
    #     for j in range(i+1, len(cos_sim)):
    #         all_sentence_combinations.append([cos_sim[i][j], i, j])

    #Sort list by the highest cosine similarity score
#    all_sentence_combinations = sorted(all_sentence_combinations, key=lambda x: x[0], reverse=True)

    # print("Top-5 most similar pairs:")
    # for score, i, j in all_sentence_combinations[0:5]:
    #     print("{} \t {} \t {:.4f}".format(sentences[i], sentences[j], cos_sim[i][j]))


def correlate_vss_dbc(data_base: Database,
                      tree: VSSNode,
                      data_type_tree: VSSNode):
    """Iterate over complete DBC and VSS tree and check for correlation"""

    # A list of Similarity objects
    threshold = 0.3
    matches = defaultdict(set)
    for vss_signal in PreOrderIter(tree):
        for can_message in data_base.messages[:10]:
            for can_signal in can_message.signals[:10]:
                similar = Similarity(can_message, can_signal, vss_signal)
                if (similar.total_similarity() > threshold):
                    # logging.info("Found match for: %s %f", vss_signal.qualified_name(), similar.total_similarity())
                    matches[vss_signal].add(similar)
        okay_matches = sorted(
            matches[vss_signal], key=lambda x: x.total_similarity())[:1]
        if okay_matches:
            logging.info(
                "========================================================")
            logging.info("%s", vss_signal.qualified_name())
            for okay_match in okay_matches:
                logging.info(" Match: %s.%s %f",
                             okay_match.can_message.name,
                             okay_match.can_signal.name,
                             okay_match.total_similarity())
                dump_similarity(okay_match)

    # logging.info("Number of VSS signals with matches above threshold: %d", len(matches))
    # logging.info("Printing Top-10")
    # top10 = sorted(matches, reverse=True)[:10]
    # for element in top10:
    #     logging.info("%s = %f",
    #                  element.vss_signal.qualified_name(),
    #                  element.total_similarity())
    #     logging.info("%s", element.vss_signal.qualified_name())
    #     logging.info("  datatype: %s", element.vss_signal.datatype.value)
    #     logging.info("  unit: %s", element.vss_signal.unit)
    #     logging.info("  dbc:")
    #     logging.info("    message: %s", element.can_message.name)
    #     logging.info("    signal: %s", element.can_signal.name)
    #     logging.info("  dbc_meta:")
    #     logging.info("    vss_description: %s", element.vss_signal.description)
    #     logging.info("    can_message_comment: %s", element.can_message.comment)
    #     logging.info("    can_signal_comment: %s", element.can_signal.comment)
    #     logging.info("    similarity: %f", element.total_similarity())


def dump_similarity(similarity: Similarity):
    """Logs a comparison of the CAN and VSS items"""
    can_message = similarity.can_message
    can_signal = similarity.can_signal
    vss_signal = similarity.vss_signal

    logging.info("Comparing CAN %s.%s with VSS %s results in %f", can_message.name,
                 can_signal.name, vss_signal.qualified_name(), similarity.total_similarity())
    logging.info("%-10s: %-30s %-50s", 'Domain', 'DBC', 'VSS')

    logging.info("%-10s: %-30s %-50s %f", 'Parent',
                 can_message.name,
                 vss_signal.parent.qualified_name(),
                 similarity.similarity_parent_name)

    logging.info("%-10s: %-30s %-50s %f", 'Signal Name',
                 can_signal.name,
                 vss_signal.name,
                 similarity.similarity_signal_name)

    logging.info("%-10s: %-30s %-50s %f", 'Unit',
                 can_signal.unit,
                 vss_signal.unit,
                 similarity.similarity_unit)

    logging.info("%-10s: %-30s %-50s %f", 'Min/max',
                 str(can_signal.minimum) + '/' + str(can_signal.maximum),
                 str(vss_signal.min) + '/' + str(vss_signal.max),
                 similarity.similarity_minmax_values)

    logging.info("Comments / Descriptions:")
    logging.info(" DBC Comment: %s", can_signal.comment)
    logging.info(" VSS Description: %s", vss_signal.description)
    logging.info(" Similarity: %f", similarity.similarity_signal_description)
    logging.info(" Parent DBC: %s", can_message.comment)
    logging.info(" Parent VSS: %s", vss_signal.parent.description)
    logging.info(" Parent Similarity: %f",
                 similarity.similarity_parent_description)
    logging.info("Total Similarity: %f", similarity.total_similarity())


def print_dummy_dbc(data_base: Database,
                    message_name,
                    signal_name):
    """Prints info about a CAN signal"""

    # message_name = 'ACC1'
    can_message = data_base.get_message_by_name(message_name)
    logging.info("CAN Message Frame ID: %s (decimal)", can_message.frame_id)
    logging.info("CAN Message Name: %s", can_message.name)
    logging.info("CAN Message Comment: %s", can_message.comment)

    # signal_name = 'AdaptiveCruiseCtrlSetSpeed
    can_signal = can_message.get_signal_by_name(signal_name)
    logging.info("CAN Signal Name: %s", can_signal.name)
    logging.info("CAN Signal Comment: %s", can_signal.comment)
    logging.info("CAN Signal Unit: %s", can_signal.unit)
    logging.info("CAN Signal Minimum: %s", can_signal.minimum)
    logging.info("CAN Signal Maximum: %s", can_signal.maximum)
    logging.info("CAN Signal J1939 SPN: %s", can_signal.spn)

    pprint(vars(can_signal))

    # ACC1.AdaptiveCruiseCtrlSetSpeed [km/h] 0-250
    # Text: "Value of the desired (chosen) velocity of the adaptive cruise control system."
    #
    # VSS Branch: Vehicle.ADAS.CruiseControl.SpeedSet
    # Text: "Set cruise control speed in kilometers per hour."


def load_dbc(dbcfile) -> Database:
    """Load a CAN-Bus Database"""
    data_base = cantools.database.load_file(dbcfile)
    return data_base


def process_data_type_tree(args,
                           include_dirs,
                           abort_on_namestyle: bool) -> VSSNode:
    """Loading the data type tree from VSpec"""
    first_tree = True
    for type_file in args.vspec_types_file:
        logging.info(
            "Loading and processing struct/data type tree from %s", type_file)
        new_tree = vspec.load_tree(type_file, include_dirs,
                                   VSSTreeType.DATA_TYPE_TREE,
                                   break_on_name_style_violation=abort_on_namestyle,
                                   expand_inst=False)
        if first_tree:
            tree = new_tree
            first_tree = False
        else:
            vspec.merge_tree(tree, new_tree)
    vspec.check_type_usage(tree, VSSTreeType.DATA_TYPE_TREE)
    return tree


parser = argparse.ArgumentParser(description="Show vspec information.")


def main(arguments):
    """Main function"""
    initLogging()

    parser.add_argument('-I', '--include-dir',
                        action='append',
                        metavar='dir',
                        type=str,
                        default=[],
                        help='Add include directory to search for included vspec files.')

    parser.add_argument('-e', '--extended-attributes',
                        type=str,
                        default="",
                        help='Whitelisted extended attributes as comma separated list. ')

    parser.add_argument('-s', '--strict',
                        action='store_true',
                        help='Quit when not-recommended usage of VSS')

    parser.add_argument('--abort-on-unknown-attribute',
                        action='store_true',
                        help=" Terminate when an unknown attribute is found.")

    parser.add_argument('--abort-on-name-style',
                        action='store_true',
                        help=" Terminate naming style not follows recommendations.")

    parser.add_argument('-o', '--overlays',
                        action='append',
                        metavar='overlays',
                        type=str,
                        default=[],
                        help='Add overlays in order of appearance.')

    parser.add_argument('-u', '--unit-file',
                        action='append',
                        metavar='unit_file',
                        type=str,
                        default=[],
                        help='Unit file to be used. Argument -u may be used multiple times.')

    parser.add_argument('vspec_file',
                        metavar='<vspec_file>',
                        help='The vehicle specification file to convert.')

    type_group = parser.add_argument_group(
        'VSS Data Type Tree arguments',
        'Arguments related to struct/type support')

    type_group.add_argument('-vt', '--vspec-types-file',
                            action='append',
                            metavar='vspec_types_file',
                            type=str,
                            default=[],
                            help='Data types file in vspec format.')

    args = parser.parse_args(arguments)

    include_dirs = ["."]
    include_dirs.extend(args.include_dir)

    abort_on_unknown_attribute = False
    abort_on_namestyle = False

    if args.abort_on_unknown_attribute or args.strict:
        abort_on_unknown_attribute = True
    if args.abort_on_name_style or args.strict:
        abort_on_namestyle = True

    known_extended_attributes_list = args.extended_attributes.split(",")
    if len(known_extended_attributes_list) > 0:
        vspec.model.vsstree.VSSNode.whitelisted_extended_attributes = known_extended_attributes_list
        logging.info("Known extended attributes: %s",
                     ', '.join(known_extended_attributes_list))

    vspec.load_units(args.vspec_file, args.unit_file)

    data_type_tree = None
    if args.vspec_types_file:
        data_type_tree = process_data_type_tree(
            args, include_dirs, abort_on_namestyle)
        vspec.verify_mandatory_attributes(
            data_type_tree, abort_on_unknown_attribute)

    try:
        logging.info("Loading vspec from %s ...", args.vspec_file)
        tree = vspec.load_tree(
            args.vspec_file, include_dirs, VSSTreeType.SIGNAL_TREE,
            break_on_name_style_violation=abort_on_namestyle,
            expand_inst=False, data_type_tree=data_type_tree)

        for overlay in args.overlays:
            logging.info("Applying VSS overlay from %s ...", overlay)
            othertree = vspec.load_tree(overlay,
                                        include_dirs,
                                        VSSTreeType.SIGNAL_TREE,
                                        break_on_name_style_violation=abort_on_namestyle,
                                        expand_inst=False,
                                        data_type_tree=data_type_tree)
            vspec.merge_tree(tree, othertree)

        vspec.check_type_usage(tree, VSSTreeType.SIGNAL_TREE, data_type_tree)
        vspec.expand_tree_instances(tree)

        vspec.clean_metadata(tree)
        vspec.verify_mandatory_attributes(tree, abort_on_unknown_attribute)

        logging.info("Loading DBC...")
        dbc = load_dbc('j1939.dbc')

        #print_dummy_dbc(dbc, 'ACC1', 'AdaptiveCruiseCtrlSetSpeed')

        # correlate_vss_dbc(dbc, tree, data_type_tree)

        semantic_similarity(dbc, tree)

    except vspec.VSpecError as exception:
        logging.error("Error during processing of VSpec: %s", exception)
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
