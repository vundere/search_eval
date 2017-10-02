import csv
import math
from statistics import median, mode
from collections import Counter
from enum import Enum


class AttrType(Enum):
    cat = 0  # categorical (qualitative) attribute
    num = 1  # numerical (quantitative) attribute
    target = 2  # target label


class NodeType(Enum):
    root = 0
    internal = 1
    leaf = 2


class SplitType(Enum):
    bin = 0  # binary split
    multi = 1  # multi-way split


class Attribute(object):
    def __init__(self, label, type):
        assert type in AttrType
        self.label = label
        self.type = type
        self.stat = None  # holds mean for numerical and mode for categorical attributes


class Splitting(object):
    def __init__(self, attr, infogain, split_type, cond, splits):
        self.attr = attr  # attribute ID (index in ATTR)
        self.infogain = infogain  # information gain if splitting is done on this attribute
        self.split_type = split_type  # one of SplitType
        self.cond = cond  # splitting condition, i.e., values on outgoing edges
        self.splits = splits  # list of training records (IDs) for each slitting condition


class Node(object):
    def __init__(self, id, type, parent_id, children=None, edge_value=None, val=None, split_type=None, split_cond=None,
                 infogain=None):
        self.id = id  # ID (same as the index in DT.model list)
        self.type = type  # one of NodeType
        self.parent_id = parent_id  # ID of parent node (None if root)
        self.children = children  # list of IDs of child nodes
        self.edge_value = edge_value  # the value of the incoming edge (only if not root node)
        self.val = val  # if root or internal node: the attribute that is compared at that node; if leaf node: the target value
        self.split_type = split_type  # one of SplitType
        self.split_cond = split_cond  # splitting condition (median value for binary splits on numerical values; otherwise a list of categorical values (corresponding to child nodes))
        self.infogain = infogain

    def append_child(self, node_id):
        self.children.append(node_id)


INFILE = "data/basketball.train.csv"
TESTFILE = "data/basketball.test.csv"
OUTPUT_FILE = "data/dt_basketball.pred.csv"


def load_test_data():
        with open(TESTFILE) as csvfile:
            data = []
            csvreader = csv.reader(csvfile, delimiter=',')
            for row in csvreader:
                rec = []
                for i in range(len(ATTR) - 1):
                    val = row[i].strip()
                    # convert numerical attributes
                    if ATTR[i].type == AttrType.num:  # Note that this will break for "?" (missing attribute)
                        val = float(val)
                    rec.append(val)
                data.append(rec)
        return data


ATTR = [Attribute("LOCATION", AttrType.cat), Attribute("W", AttrType.cat),
        Attribute("FINAL_MARGIN", AttrType.num), Attribute("SHOT_NUMBER", AttrType.num), 
        Attribute("PERIOD", AttrType.cat), Attribute("GAME_CLOCK", AttrType.num), 
        Attribute("SHOT_CLOCK", AttrType.num), Attribute("DRIBBLES", AttrType.num), 
        Attribute("TOUCH_TIME", AttrType.num), Attribute("SHOT_DIST", AttrType.num),
        Attribute("PTS_TYPE", AttrType.cat), Attribute("CLOSE_DEF_DIST", AttrType.num),
        Attribute("Target", AttrType.target)]

IDX_TARGET = len(ATTR) - 1 

TARGET_ATTR_SUCCESS = 'made'
TARGET_ATTR_FAILURE = 'missed'


class DT(object):
    def __init__(self):
        self.data = None  # training data set (loaded into memory)
        self.model = None  # decision tree model
        self.default_class = None  # default target class

    def __load_data(self):
        with open(INFILE) as csvfile:
            self.data = []
            csvreader = csv.reader(csvfile, delimiter=',')
            for row in csvreader:
                rec = []
                for i in range(len(ATTR)):
                    val = row[i].strip()
                    # convert numerical attributes
                    if ATTR[i].type == AttrType.num:  # Note that this will break for "?" (missing attribute)
                        val = float(val)
                    rec.append(val)
                self.data.append(rec)
                # self.data.append([element.strip() for element in row])  # strip spaces



    def __median(self, attrNum):
        attrList = []
        for i in range(len(self.data)):
            attrList.append(self.data[i][attrNum])
        return median(attrList)
        

    def __entropy(self, records):
        """
        Calculates entropy for a selection of records.

        :param records: Data records (given by indices)
        """
        yes = 0
        no = 0
        target_idx = len(self.data[0]) - 1

        for r in records:
            if self.data[r][target_idx] == TARGET_ATTR_SUCCESS:
                yes += 1
            if self.data[r][target_idx] == TARGET_ATTR_FAILURE:
                no += 1

        entropy = -no/(no+yes)*math.log2(no/(no+yes)) - yes/(no+yes)*math.log2(yes/(no+yes))

        return entropy

    
    def __find_best_attr(self, attrs, records):
        """
        Finds the attribute with the largest gain.

        :param attrs: Set of attributes
        :param records: Training set (list of record ids)
        :return:
        """
        entropy_p = self.__entropy(records)  # parent's entropy
        splittings = []  # holds the splitting information for each attribute

        for a in attrs:
            assert ATTR[a].type in AttrType
            splits = {}  # record IDs corresponding to each split
            # split_mode = None
            # split_cond = None
            # splitting condition depends on the attribute type
            if ATTR[a].type == AttrType.target:  # skip target attribute
                continue
            elif ATTR[a].type == AttrType.cat:  # categorical attribute
                # multi-way split on each possible value
                split_mode = SplitType.multi
                # each possible attr value corresponds to a split (indexed with categorical labels)
                # Note: it's important to consider attr values from the entire training set
                split_cond = set([self.data[idx][a] for idx in range(len(self.data))])
                
                # `splits[val]` holds a list of records for a given split,
                # where `val` is an element of `split_cond`
                for sc in split_cond:
                    conds = []

                    for r in records:
                        if self.data[r][a] == sc:
                            conds.append(r)

                    splits[sc] = conds
                
            elif ATTR[a].type == AttrType.num:  # numerical attribute => binary split on median value
                split_mode = SplitType.bin
                split_cond = self.__median(a)  # (i.e., if less or equal than this value)
                        
                upper = []
                lower = []

                for r in records:
                    if self.data[r][a] <= split_cond:
                        lower.append(r)
                    elif self.data[r][a] > split_cond:
                        upper.append(r)

                splits['>{}'.format(split_cond)] = upper

                splits['<={}'.format(split_cond)] = lower

            # compute gain for attribute a
            infogain = entropy_p

            for s in splits:
                yes = 0
                no = 0
                target_idx = len(self.data[0]) - 1

                # entropy of each split
                for r in splits[s]:
                    if self.data[r][target_idx] == TARGET_ATTR_SUCCESS:
                        yes += 1
                    if self.data[r][target_idx] == TARGET_ATTR_FAILURE:
                        no += 1        
                
                if no == 0 and yes == 0:
                    entropy = 1
                elif no == 0 or yes == 0:
                    entropy = 0
                else:
                    entropy = -no / (no + yes) * math.log2(no / (no + yes)) - yes / (no + yes) * math.log2(yes / (no + yes))
                infogain -= (yes + no) / len(records) * entropy
                
            splitting = Splitting(a, infogain, split_mode, split_cond, splits)
            splittings.append(splitting)

        # find best splitting
        best_splitting = sorted(splittings, key=lambda x: x.infogain, reverse=True)[0]
        return best_splitting

    def __add_node(self, parent_id, node_type=NodeType.internal, edge_value=None, val=None, split_type=None,
                   split_cond=None):

        node_id = len(self.model)  # id of the newly assigned node
        if not self.model:  # the tree is empty
            node_type = NodeType.root

        node = Node(node_id, node_type, parent_id, children=[], edge_value=edge_value, val=val, split_type=split_type,
                    split_cond=split_cond)
        self.model.append(node)

        # also add it as a child of the parent node
        if parent_id is not None:
            self.model[parent_id].append_child(node_id)

        return node_id

    def __id3(self, attrs, records, parent_id=None, value=None):
        # empty training set or empty set of attributes => create leaf node with default class
        if not records or not attrs:
            self.__add_node(parent_id, node_type=NodeType.leaf, edge_value=value, val=self.default_class)
            return

        # if all records have the same target value => create leaf node with that target value
        same = all(self.data[idx][IDX_TARGET] == self.data[records[0]][IDX_TARGET] for idx in records)
        if same:
            target = self.data[records[0]][IDX_TARGET]
            self.__add_node(parent_id, node_type=NodeType.leaf, edge_value=value, val=target)
            return

        # find the attribute with the largest gain
        splitting = self.__find_best_attr(attrs, records)
        # add node
        node_id = self.__add_node(parent_id, edge_value=value, val=splitting.attr, split_type=splitting.split_type,
                                  split_cond=splitting.cond)
        
        # TODO call tree construction recursively for each split
        print('Working...')
        attrs_copy = attrs.copy()
        attrs_copy.remove(splitting.attr)

        if splitting.split_type == SplitType.bin:

            upper = []
            lower = []
            split_cond = splitting.cond

            for r in records:
                if self.data[r][splitting.attr] > split_cond:
                    upper.append(r)
                elif self.data[r][splitting.attr] <= split_cond:
                    lower.append(r)

            try:
                '>{}'.format(split_cond)
                self.__id3(attrs_copy, upper, parent_id=node_id, value=False)
            except Exception as e:
                print('ERROR: {}'.format(e))

            try:
                '<={}'.format(split_cond)
                self.__id3(attrs_copy, lower, parent_id=node_id, value=True)
            except Exception as e:
                print('ERROR: {}'.format(e))

        if splitting.split_type == SplitType.multi:

            for sc in splitting.cond:
                subset = []

                for r in records:
                    if self.data[r][splitting.attr] == sc:
                        subset.append(r)

                try:
                    self.__id3(attrs_copy, subset, parent_id=node_id, value=sc)
                except Exception as e:
                    print('ERROR: {}'.format(e))


    def print_model(self, node_id=0, level=0):
        node = self.model[node_id]
        indent = "  " * level
        if node.type == NodeType.leaf:
            print(indent + str(node.edge_value) + " [Leaf node] class=" + node.val)
        else:
            cond = " <= " + str(node.split_cond) if ATTR[node.val].type == AttrType.num else " == ? "
            if node.type == NodeType.root:
                print("[Root node] '" + ATTR[node.val].label + "'" + cond)
            else:
                print(indent + str(node.edge_value) + " [Internal node] '" + ATTR[node.val].label + "'" + cond)
            # print tree for child notes recursively
            for n_id in node.children:
                self.print_model(n_id, level + 1)

    def build_model(self):
        self.__load_data()
        self.model = []  # holds the decision tree model, represented as a list of nodes
        self.default_class = Counter([x[IDX_TARGET] for x in self.data]).most_common(1)[0][0]
        self.__id3(set(range(len(ATTR) - 1)), list(range(len(self.data))))

    def apply_model(self, record):
        node = self.model[0]
        while node.type != NodeType.leaf:
            attr_val = record[node.val]

            # check which edge to follow based on attr value
            if ATTR[node.val].type == AttrType.num:
                if record[node.val] > node.split_cond:
                    edge = False  # ">" + str(node.split_cond) # TODO set it <=median or >median
                elif record[node.val] <= node.split_cond:
                    edge = True  # "<=" + str(node.split_cond)
            else:
                edge = attr_val

            for child_id in node.children:
                child_edge_value = self.model[child_id].edge_value

                if self.model[child_id].edge_value == edge:
                    node = self.model[child_id]
                    
        return node.val

def main():
    dt = DT()
    print("Build model:")
    dt.build_model()
    dt.print_model()
    
    test = load_test_data()

    print("\nApply model:")
    with open(OUTPUT_FILE, 'w') as output:
        output.write("Id,Target\n")
        for id, data in enumerate(test):
            res = dt.apply_model(data)
            # write results to file
            output.write("%s,%s\n" % (id + 1, res))
    print("Results written to {0}".format(OUTPUT_FILE))


if __name__ == "__main__":
    main()



