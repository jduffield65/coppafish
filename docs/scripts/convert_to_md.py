# Source code: https://github.com/tiffsea/python-code/blob/master/demos/convert-json.py
import json
import numpy as np
from typing import Tuple, List
from coppafish.setup.config import _options
import os


def get_var_comment(var_name: str, var_comment: List[str]) -> str:
    """
    Constructs a large string which starts a new bullet point, indicates var_name in bold and then adds var_comment
    after a colon.

    Args:
        var_name: Name of variable to get string for.
        var_comment: List of strings giving a comment for var_name.

    Returns:
        Large string giving a comment for var_name.

    """
    str_comment = '* ' + '**' + var_name + '**: '  # make name of variable bold and each variable a new bullet point
    if len(var_comment) > 1:
        # Add new line to comment wherever there is an empty line
        for j in range(len(var_comment)):
            if len(var_comment[j]) == 0:
                var_comment[j] = '\n\n\t'
        # make first line italics if more than 1 as first line indicates data type. Also start new line after it.
        str_comment = str_comment + '*' + var_comment[0] + '*.\n\n\t'
        str_comment = str_comment + ' '.join(var_comment[1:])
    else:
        str_comment = str_comment + var_comment[0]
    return str_comment + '\n\n'


def extract_headings(ini_txt: str) -> Tuple[dict, dict]:
    """
    From the .ini string, the headers are extracted and a comment describing each is added.

    Args:
        ini_txt: text read in from .ini file including comments with a `';'` or `'; '` prefix.

    Returns:
        json_dict - Contains a DESCRIPTION key for each header (know by []) in the .ini file but no other keys.
            e.g. `json_dict['file_names']['DESCRIPTION']`
        variable_dict - For each header, all lines in ini file are kept.
            e.g. `json_dict['file_names']` is a list of all lines in .ini file in the file_names section.
            Each line is a different element in the list.

    """
    txt_list = ini_txt.split('\n')
    # Find position of headers
    section_heading_inds = [ind for ind in np.arange(len(txt_list)) if len(txt_list[ind]) > 0 and
                            txt_list[ind][0] == '[']  # remove empty lines
    # Find position of all empty lines
    empty_line_ind = np.asarray([ind for ind in np.arange(len(txt_list)) if len(txt_list[ind]) == 0])

    # Finds comment lines describing each section as those between header line and first empty line after it.
    section_comment_inds = []
    for i in range(len(section_heading_inds)):
        end_comment_ind = np.asarray(empty_line_ind)[np.asarray(empty_line_ind) > section_heading_inds[i]].min()
        section_comment_inds = section_comment_inds + [np.arange(section_heading_inds[i] + 1, end_comment_ind)]

    header_names = [txt_list[i][1:-1] for i in section_heading_inds]
    # remove comment indicator from comments
    for ind in np.concatenate(section_comment_inds):
        txt_list[ind] = txt_list[ind].replace('; ', '')
        txt_list[ind] = txt_list[ind].replace(';', '')  # for case with blank comment
    txt_list = np.asarray(txt_list)
    json_dict = {header_names[i]: {'DESCRIPTION': txt_list[section_comment_inds[i]].tolist()} for i in
                 range(len(header_names))}
    section_heading_inds = section_heading_inds + [len(txt_list)]  # add last line of txt_list+1 for next line
    variable_dict = {header_names[i]: txt_list[np.arange(section_comment_inds[i][-1] + 1,
                                                         section_heading_inds[i+1])].tolist() for
                     i in range(len(header_names))}
    return json_dict, variable_dict


def add_variable_info(json_dict: dict, variable_dict: dict):
    """
    Updates json_dict by adding a key for each variable in the relevant section which includes a comment describing it.
    e.g. json_dict['file_names']['input_dir'] is added.

    Args:
        json_dict: Contains a DESCRIPTION key for each header (know by []) in the .ini file but no other keys.
            e.g. `json_dict['file_names']['DESCRIPTION']`
        variable_dict: For each header, all lines in ini file are kept.
            e.g. `json_dict['file_names']` is a list of all lines in .ini file in the file_names section.
            Each line is a different element in the list.

    """
    for key in variable_dict.keys():
        txt_list = [val for val in variable_dict[key] if len(val) > 0]  # remove blank lines
        comment_start_ind = 0
        for i in range(len(txt_list)):
            if txt_list[i][0] == ';':
                continue
            var_name, default_value = txt_list[i].split(' =')
            if default_value == '' and 'maybe' in _options[key][var_name]:
                default_value = 'None'
            elif default_value == '':
                default_value = 'MUST BE SPECIFIED'
            elif default_value[0] == ' ':
                default_value = default_value[1:]
            # add as first line of comment, the _option specified in setup.config (will be set to italics in markdown).
            # remove ';' or ' ;'prefix from comments.
            var_comments = [_options[key][var_name]] + \
                           [var.replace('; ', '').replace(';', '') for var in txt_list[comment_start_ind:i]]
            var_comments = var_comments + [f'\n\n\tDefault: `{default_value}`']
            json_dict[key][var_name] = var_comments
            comment_start_ind = i + 1


class ConvertToMD:
    def __init__(self, file_name: str, out_file: str, title: str):
        """
        This converts the notebook_comments.json and the settings.default.ini files to markdown.

        Args:
            file_name: The path to the .json or .ini file.
            out_file: The path where the markdown (.md) file should be saved.
            title: Title added to the markdown file.
        """
        self.file_name = file_name
        self.title = title
        if self.file_name.endswith('.json'):
            self.jdata = self.get_json()
        elif self.file_name.endswith('.ini'):
            self.jdata = self.ini_to_json()
        else:
            raise ValueError(f"file_name must refer to a .json or .ini file but file_name = {file_name}.")
        self.mddata = self.format_json_to_md()
        self.convert_dict_to_md(out_file)

    def get_json(self):
        with open(self.file_name) as f:
            res = json.load(f)
        return res

    def ini_to_json(self):
        txt = open(self.file_name, "r").read()
        json_dict, variable_dict = extract_headings(txt)
        add_variable_info(json_dict, variable_dict)
        return json_dict

    def format_json_to_md(self):
        text = f'# {self.title}\n'
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
        print(f'{self.file_name} file successfully converted to {output_file}')


if __name__ == '__main__':
    ConvertToMD(os.path.abspath(__file__ + "/../../../coppafish/setup/settings.default.ini"),
                os.path.abspath(__file__ + "/../../config.md"), 'Default Config Settings')
    ConvertToMD(os.path.abspath(__file__ + "/../../../coppafish/setup/notebook_comments.json"),
                os.path.abspath(__file__ + "/../../notebook_comments.md"), 'Notebook Comments')
