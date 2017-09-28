import json

# Change these variables as needed
NB_FILE = 'assign_2/2_Decision_tree_example.ipynb'
OUT_FILE = 'converted.py'


def convert(input_file, output_file):
    with open(input_file, 'r') as f, open(output_file, 'w') as g:
        data = json.load(f)
        for obj in data['cells']:
            if obj['cell_type'] == 'code':
                for line in obj['source']:
                    g.write(line)
                g.write('\n\n')


if __name__ == '__main__':
    convert(NB_FILE, OUT_FILE)
