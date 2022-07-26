# Source code: https://github.com/tiffsea/python-code/blob/master/demos/convert-json.py
import json
import pypandoc


def get_var_comment(var_name, var_comment):
    str_comment = '* ' + '**' + var_name + '**: '  # make name of variable bold and each variable a new bullet point
    if len(var_comment) > 1:
        # make first line italics if more than 1 as first line indicates data type
        str_comment = str_comment + '*' + var_comment[0] + '*. '
        str_comment = str_comment + ' '.join(var_comment[1:])
    else:
        str_comment = str_comment + var_comment[0]
    return str_comment + '\n\n'


class Convert_Json():

    def __init__(self, json_fp, h1):
        self.fp = json_fp
        self.h1 = h1
        self.jdata = self.get_json()
        self.mddata = self.format_json_to_md()

    def get_json(self):
        with open(self.fp) as f:
            res = json.load(f)
        return res

    def convert_json_to_txt(self, output_fn):
        with open(output_fn, 'w') as f:
            json.dump(self.jdata, f)
        print('Json file successfully converted to txt')

    def format_json_to_md(self):
        text = f'# {self.h1}\n'
        dct = self.jdata
        for page_title, page_variables in dct.items():
            text += f'## {page_title}\n'
            text += ' '.join(page_variables['DESCRIPTION']) + '\n\n'
            for var_name, var_comment in page_variables.items():
                if var_name == 'DESCRIPTION':
                    continue
                text += get_var_comment(var_name, var_comment)
        return text

    def convert_dict_to_md(self, output_file):
        with open(output_file, 'w') as writer:
            writer.writelines(self.mddata)
        print('json file successfully converted to md')


def convert_json_to_md(json_file, out_file, title):
    converter = Convert_Json(json_file, title)
    converter.convert_dict_to_md(output_file=out_file) # uncomment for markdown output


if __name__ == '__main__':
    convert_json_to_md('iss/setup/notebook_comments.json', 'docs/notebook_comments.md', 'Notebook Comments')
