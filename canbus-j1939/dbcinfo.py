#!/usr/bin/python3
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
import yaml
import pickle
from anytree import AnyNode
from anytree.exporter import DictExporter
import tensorflow_hub as hub
import numpy as np
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
from typing import Dict, Any
from sentence_transformers import SentenceTransformer, util

import vspec
import cantools
from absl import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def sentence_case(string):
    """Convert a camel case signal name into separate words"""
    if string != '':
        result = re.sub('([A-Z])', r' \1', string)
        return result[:1].upper() + result[1:].lower()
    return

def semantic_similarity(data_base: Database,
                        tree: VSSNode,
                        output_vspec_file: str):
    """ Use NLP model to find DBC matches for each VSS datapoint"""
    # https://www.sbert.net/examples/applications/paraphrase-mining/README.html
    logging.info("Loading SentenceTransformer")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # The corpus is the CAN Signals as describes
    corpus = []
    corpus_ids = []
    corpus_can_messages = []
    corpus_can_signals = []
    for can_message in data_base.messages:
        for can_signal in can_message.signals:
            logging.debug("CAN-Signal: {} : {}".format(can_signal.name, can_signal.comment))
            signal_name_with_spaces=sentence_case(can_signal.name)
            corpus.append(str(signal_name_with_spaces) + " " + str(can_signal.comment) + " " + str(can_message.comment) + " in " + str(can_signal.unit))
            corpus_ids.append(can_message.name + "." + can_signal.name)
            corpus_can_messages.append(can_message)
            corpus_can_signals.append(can_signal)

    pickle_file = 'doc_embedding.pickle'
    if os.path.isfile(pickle_file):
        logging.info(f"Loading embeddings from persisted cache file {pickle_file}")
        with open(pickle_file, 'rb') as pkl:
            corpus_embeddings = pickle.load(pkl)
    else:
        corpus_embeddings = model.encode(corpus, convert_to_tensor=True, show_progress_bar=True)
        logging.info(f"Storing embeddings to persistent cache file {pickle_file}")
        with open(pickle_file, 'wb') as pkl:
            pickle.dump(corpus_embeddings, pkl)

    #Compute embedding for both lists
    #embeddings1 = model.encode(sentences, convert_to_tensor=True, show_progress_bar=True)
    #query = ['Set cruise control speed in kilometers per hour.']

    # Only use the best match by setting top_k to 1
    # top_k = min(5, len(corpus))
    top_k=10
    
    # Annotate the VSSTree with additional "DBC" information
    vss_node : VSSNode
    for vss_node in PreOrderIter(tree):
        query = sentence_case(vss_node.name)
        query += " " + str(vss_node.description)
        if vss_node.parent:
            query += sentence_case(vss_node.parent.name)
            query += " " + str(vss_node.parent.description)
        query_embedding = model.encode(query, convert_to_tensor=True, show_progress_bar=False)
        cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
        
        good_results = torch.topk(cos_scores, k=top_k)
        for score, idx in zip(good_results[0], good_results[1]):
            
            # The text-based matching must now be validated with
            # hard facts, such as matching data types, units
            # and minimum/maximum values
            this_vss_node = vss_node
            this_can_message = corpus_can_messages[idx]
            this_can_signal = corpus_can_signals[idx]
            
            unit_match = units_match(this_vss_node, this_can_message, this_can_signal)
            if unit_match:
                vss_node.extended_attributes['dbc'] = {
                    "message": corpus_can_messages[idx].name,
                    "signal": corpus_can_signals[idx].name,
                    "metadata": {
                        "score": "{:.4f}".format(score),
                        "unit_match": unit_match
                    }
                }
                break

    # Write annotated VSS tree to yaml file
    yaml_content: Dict[str, Any] = {}
    export_node(yaml_content, tree)
    with open(output_vspec_file, "w", encoding="UTF-8") as output_file:
        yaml.dump(yaml_content, output_file, default_flow_style=False, Dumper=NoAliasDumper,
                  sort_keys=True, width=1024, indent=2, encoding='utf-8', allow_unicode=True)


def units_match(this_vss_node: VSSNode,
                this_can_message: Message,
                this_can_signal: Signal) -> str:
    """Validate a DBC-VSS match by their datatypes and units compatibility"""

    if (this_vss_node.unit != None and this_can_signal.unit != None) and (this_vss_node.unit == this_can_signal.unit):
        return "exact-match"

    # This is actually the same ... both means Acceleration
    if this_vss_node.unit == 'm/s^2' and this_can_signal.unit == 'm/s/s':
        return "alias-match"
    
    # Rotation rate / pitch / yaw. Could potentially be a match
    if this_vss_node.unit == 'degrees' and this_can_signal.unit == 'deg':
        return "alias-match"
    if this_vss_node.unit == 'degrees/s' and this_can_signal.unit == 'rad/s':
        return "alias-match"
    
    if not this_vss_node.unit and this_can_signal.unit:
        return None

    # Databroker Types: string, bool, int8, int16, int32, int64, uint8, uint16, uint32, uint64, float, double, timestamp
    vss_datatype = this_vss_node.data_type_str
    if not vss_datatype:
        if this_vss_node.has_datatype():
            vss_datatype = this_vss_node.get_datatype()

    if this_can_signal.is_float:
        if vss_datatype == 'float':
            return "float-match"
        if vss_datatype == 'double':
            return "double-match"

    if this_can_signal.is_signed:
        if vss_datatype.startswith('int'):
            return "int-match"

    if not this_can_signal.is_signed:
        if vss_datatype.startswith('uint'):
            return "uint-match"

    if vss_datatype == 'float' and not this_can_signal.is_float:
        return None
    

    if this_can_signal.choices:
        if this_vss_node.allowed:
            return None

    # Try matching the units first
    if not this_vss_node.unit and not this_can_signal.unit:
        return "no-units"
    
    # TODO: Trying to figure out enum mapping is too complicated for now
    # Hence, returning false.    
    # if this_can_signal.choices:
    #     if this_vss_node.has_unit:
    #         return False
    #     # Both could be ENUMs
    #     if this_vss_node.allowed:
    #         return True
    # this_vss_node.get_datatype()
    # this_can_signal.is_signed
    # this_can_signal.is_float
    # pprint(this_vss_node, depth=1)
    # print(repr(this_can_signal))
    # error_message = f"No datatype match found for combination: VSS:{vss_datatype} and DBC is_float:{this_can_signal.is_float} is_signed:{this_can_signal.is_signed} scale:{this_can_signal.scale}"
    return None

def export_node(yaml_dict, node: VSSNode):

    node_path = node.qualified_name()

    yaml_dict[node_path] = {}

    yaml_dict[node_path]["type"] = str(node.type.value)

    if node.is_signal() or node.is_property():
        yaml_dict[node_path]["datatype"] = node.get_datatype()

    # many optional attributes are initilized to "" in vsstree.py
    if node.min != "":
        yaml_dict[node_path]["min"] = node.min
    if node.max != "":
        yaml_dict[node_path]["max"] = node.max
    if node.allowed != "":
        yaml_dict[node_path]["allowed"] = node.allowed
    if node.default != "":
        yaml_dict[node_path]["default"] = node.default
    if node.deprecation != "":
        yaml_dict[node_path]["deprecation"] = node.deprecation

    # in case of unit or aggregate, the attribute will be missing
    try:
        yaml_dict[node_path]["unit"] = str(node.unit.value)
    except AttributeError:
        pass
    try:
        yaml_dict[node_path]["aggregate"] = node.aggregate
    except AttributeError:
        pass

    yaml_dict[node_path]["description"] = node.description

    if node.comment != "":
        yaml_dict[node_path]["comment"] = node.comment

    for k, v in node.extended_attributes.items():
        yaml_dict[node_path][k] = v

    for child in node.children:
        export_node(yaml_dict, child)

# create dumper to remove aliases from output and to add nice new line after each object for a better readability
class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True

    def write_line_break(self, data=None):
        super().write_line_break(data)
        if len(self.indents) == 1:
            super().write_line_break()

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

    parser.add_argument('-vf','--vspec-file',
                        required=True,
                        metavar='vspec_file',
                        help='The vehicle specification file to read.')

    parser.add_argument('-dbc','--dbc-file',
                        required=True,
                        metavar='dbc_file',
                        help='The CAN-Bus database (DBC) file to read.')

    parser.add_argument('-out','--output-vspec-file',
                        required=True,
                        metavar='output_vspec_file',
                        help='The Kuksa dbc2val mapping vspec file to write the matches to')

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
        dbc = load_dbc(args.dbc_file)

        semantic_similarity(dbc, tree, args.output_vspec_file)

    except vspec.VSpecError as exception:
        logging.error("Error during processing of VSpec: %s", exception)
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
