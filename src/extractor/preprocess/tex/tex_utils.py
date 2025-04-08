import re


def get_tex_file_names(folder_path):
    """
    Get the file names from a folder (the top level folder of a paper)
    """
    # Get the file names
    file_names = []
    for file in folder_path.iterdir():
        if file.is_file() and file.suffix == '.tex':
            file_names.append(file)
    return file_names


# Remove potential duplicate tables as \begin{center} tables are included in \begin{figure} tables
def remove_duplicate_tables(table_latex_org):
    table_latex = []
    table_latex_string = ''

    # Sort the table latex by the length of the table latex
    table_latex_org_sorted = sorted(table_latex_org, key=len, reverse=True)

    for item in table_latex_org_sorted:
        if item not in table_latex_string:
            table_latex.append(item)
            table_latex_string += item

    return [item for item in table_latex_org if item in table_latex]


def extract_tabular_code(table_code):
    """
    Extract the tabular code from the table code.
    """
    table_latex_code_tabular = []
    
    begin_list = [[m.start(), '{'] for m in re.finditer(r'\\begin\s*?{\s*?tabular.*?}', table_code)]
    end_list = [[m.end(), '}'] for m in re.finditer(r'\\end\s*?{\s*?tabular.*?}', table_code)]
    begin_end_list = begin_list + end_list
    begin_end_list.sort(key=lambda x: x[0])

    # Find the matching begin and end
    if len(begin_end_list) == 0:
        return table_latex_code_tabular

    try: 
        assert len(begin_end_list) % 2 == 0
        assert begin_end_list[0][1] == '{'
    except AssertionError:
        return table_latex_code_tabular

    begin_end_stack = []
    begin_end_pairs = []

    begin_end_stack.append(begin_end_list[0])
    for i in range(1, len(begin_end_list)):
        if begin_end_list[i][1] == '{':
            begin_end_stack.append(begin_end_list[i])
        else:
            begin_postion = begin_end_stack.pop()
            assert begin_postion[1] == '{'
            begin_end_pairs.append([begin_postion[0], begin_end_list[i][0]])

    assert len(begin_end_stack) == 0
    begin_end_pairs.sort(key=lambda x: x[0])

    # Extract the tabular code
    end_index = -1
    for begin, end in begin_end_pairs:

        if begin < end_index:
            continue
        else:
            table_latex_code_tabular.append(table_code[begin:end])
            end_index = end

    return table_latex_code_tabular


def extract_table_data(latex_code):
    """Extract the table data from the latex code.
    
    Args:
        latex (str): The latex code of the table.

    Returns:
        table_latex: The table data.
    """
    table_latex = []
    
    # Extract tables using regex
    table_latex += re.findall(r'\\begin\s*?{\s*?table.*?\\end\s*?{\s*?table.*?}', latex_code, flags=re.DOTALL)

    # Extract wrapped tables
    table_latex_wrapped = [item for item in re.findall(r'\\begin\s*?{\s*?wraptable.*?wraptable.*?}', latex_code, flags=re.DOTALL) if '\\begin{tabular' in item and '\\end{tabular' in item]
    table_latex += table_latex_wrapped

    # Extract rotating tables
    table_latex_rotating = [item for item in re.findall(r'\\begin\s*?{\s*?sidewaystable.*?sidewaystable.*?}', latex_code, flags=re.DOTALL) if '\\begin{tabular' in item and '\\end{tabular' in item]
    table_latex += table_latex_rotating

    # Extract tables located in figures
    table_latex_figures = [item for item in re.findall(r'\\begin\s*?{figure.*?\\end\s*?{figure.*?}', latex_code, flags=re.DOTALL) if '\\begin{tabular' in item and '\\end{tabular' in item]
    table_latex += table_latex_figures

    # Extract tables located in center
    table_latex_center = [item for item in re.findall(r'\\begin\s*?{center.*?\\end\s*?{center.*?}', latex_code, flags=re.DOTALL) if '\\begin{tabular' in item and '\\end{tabular' in item]
    table_latex += table_latex_center

    # Extract tables located in minipage
    table_latex_minipage = [item for item in re.findall(r'\\begin\s*?{minipage.*?\\end\s*?{minipage.*?}', latex_code, flags=re.DOTALL) if '\\begin{tabular' in item and '\\end{tabular' in item]
    table_latex += table_latex_minipage

    # Extract tables located in subfigures
    table_latex_subfigures = [item for item in re.findall(r'\\begin\s*?subfigure.*?\\end\s*?subfigure.*?}', latex_code, flags=re.DOTALL) if '\\begin{tabular' in item and '\\end{tabular' in item]
    table_latex += table_latex_subfigures

    table_latex = remove_duplicate_tables(table_latex)

    # Remove the comments at the end of each line
    table_latex_new = []
    for item in table_latex:

        try:
            # Remove the comments at the end of each line
            item_new = remove_comments_at_the_end(item) 

            # Check if the table is valid (i.e., the number of '{' and '}' are the same)
            # if item_new.count('{') != item_new.count('}'):
            if not check_curly_brackets(item_new):
                item_new_index = latex_code.index(item_new)
                count = 0
                for i in range(item_new_index, item_new_index+len(item_new)):
                    if latex_code[i] == '{':
                        count += 1
                    elif latex_code[i] == '}':
                        count -= 1

                assert count > 0
                for i in range(item_new_index+len(item_new), len(latex_code)):
                    if latex_code[i] == '{':
                        count += 1
                    elif latex_code[i] == '}':
                        count -= 1
                    if count == 0 and latex_code[item_new_index:i+1].count('begin{') == latex_code[item_new_index:i+1].count('end{'):
                        item_new = latex_code[item_new_index:i+1]
                        break
            
            table_latex_new.append(item_new)
        except (ValueError, AssertionError):
            continue

    table_latex = table_latex_new

    table_latex = remove_duplicate_tables(table_latex)

    return table_latex


def check_curly_brackets(input_string):
    return input_string.count('{') == input_string.count('}') or input_string.count('{') - input_string.count('\{') == input_string.count('}') - input_string.count('\}')


def find_snippets(input_string, command):
    command_indeces = re.finditer(pattern=r'\\\s*'+command, string=input_string, flags=re.DOTALL)
    command_indeces = [item.start() for item in command_indeces]
    assert len(command_indeces) > 0
    command_snippets = []
    for index in command_indeces:
        count = 0
        for i in range(index+1, len(input_string)):
            if input_string[i] == '{':
                count += 1
            elif input_string[i] == '}':
                count -= 1
            if count == 0 and 'end{' in input_string[index:i+1]:
                command_snippets.append(input_string[index:i+1])
                break
    return command_snippets


def remove_comments_at_the_end(table_latex):
    # Remove comments at the end of each line (after the symbol %)
    table_lines = table_latex.split('\n')
    table_lines_new = []
    for line in table_lines:
        if '%' in line and ('\\%' not in line or line.index('\\%') > line.index('%') or '\\\\%' in line):
            line = line[:line.index('%')]
        table_lines_new.append(line)
    table_latex = '\n'.join(table_lines_new)
    return table_latex


def find_complete_tabular_code_label(paper_latex, table_latex):
    # Get all tables in the paper
    table_all = extract_table_data(paper_latex)
    assert sum([table_latex in item for item in table_all])

    for item in table_all:
        if not check_curly_brackets(item):
            break
    # assert sum([item.count('{') == item.count('}') for item in table_all]) == len(table_all)
    assert sum([check_curly_brackets(item) for item in table_all]) == len(table_all)

    # Find the source table of the input table
    table_source = [item for item in table_all if table_latex in item]

    # assert len(table_source) == 1
    table_source = table_source[0]

    # Find the label of the table_latex
    table_label = re.findall(r'\\label\s*{.*?}', table_latex, re.S)

    if len(table_label) == 0:

        # Use the label of the source table if it only has one label
        if len(re.findall(r'\\label\s*{.*?}', table_source, re.S)) == 1:
            table_label = re.findall(r'\\label\s*{.*?}', table_source, re.S)
            table_label = table_label[0]
        
        # Check if there are multiple labels in the source table
        elif len(re.findall(r'\\label\s*{.*?}', table_source, re.S)) > 1:

            # Check whether '\subfloat' is in the table_source
            if '\\subfloat' in table_source:

                # Find the '\\subfloats' snippets in the table_source based on the brackets
                subfloat_snippets = find_snippets(table_source, 'subfloat')
                subfloat_snippets = [item for item in subfloat_snippets if table_latex in item]
                assert len(subfloat_snippets) == 1

                subfloat_snippet = subfloat_snippets[0]

                # Find the label of the subfloat snippet
                subfloat_label = re.findall(r'\\label\s*{.*?}', subfloat_snippet, re.S)
                if len(subfloat_label) == 1:
                    subfloat_label = subfloat_label[0]
                    table_label = subfloat_label
                    table_latex = subfloat_snippet
                else:
                    subfloat_label_freq = [paper_latex.count(item) for item in subfloat_label]
                    subfloat_label = subfloat_label[subfloat_label_freq.index(max(subfloat_label_freq))]
                    table_label = subfloat_label
                    table_latex = subfloat_snippet
            elif '\\subcaptionbox' in table_source:
                # Find the '\\subcaptionbox' snippets in the table_source based on the brackets
                subcaptionbox_snippets = find_snippets(table_source, 'subcaptionbox')
                subcaptionbox_snippets = [item for item in subcaptionbox_snippets if table_latex in item]
                assert len(subcaptionbox_snippets) == 1

                subcaptionbox_snippet = subcaptionbox_snippets[0]

                # Find the label of the subcaptionbox snippet
                subcaptionbox_label = re.findall(r'\\label\s*{.*?}', subcaptionbox_snippet, re.S)
                if len(subcaptionbox_label) == 1:
                    subcaptionbox_label = subcaptionbox_label[0]
                    table_label = subcaptionbox_label
                    table_latex = subcaptionbox_snippet
                else:
                    subcaptionbox_label_freq = [paper_latex.count(item) for item in subcaptionbox_label]
                    subcaptionbox_label = subcaptionbox_label[subcaptionbox_label_freq.index(max(subcaptionbox_label_freq))]
                    table_label = subcaptionbox_label
                    table_latex = subcaptionbox_snippet
            elif '\\parbox' in table_source:
                # Find the '\\parbox' snippets in the table_source based on the brackets
                parbox_snippets = find_snippets(table_source, 'parbox')
                parbox_snippets = [item for item in parbox_snippets if table_latex in item]
                assert len(parbox_snippets) == 1

                parbox_snippet = parbox_snippets[0]
                
                # Find the label of the parbox snippet
                parbox_label = re.findall(r'\\label\s*{.*?}', parbox_snippet, re.S)
                assert len(parbox_label) == 1
                parbox_label = parbox_label[0]
                table_label = parbox_label
                table_latex = parbox_snippet
            elif 'resizebox' in table_source and len(re.findall(r'\\label\s*{.*?}', table_source, re.S)) == len(find_snippets(table_source, 'resizebox')):
                # Find the '\\resizebox' snippets in the table_source based on the brackets
                resizebox_snippets = find_snippets(table_source, 'resizebox')
                resizebox_snippets = [item for item in resizebox_snippets if table_latex in item]
                assert len(resizebox_snippets) == 1

                resizebox_snippet = resizebox_snippets[0]
                resizebox_snippet_index = find_snippets(table_source, 'resizebox').index(resizebox_snippet)

                # Find the label of the resizebox snippet
                assert len(re.findall(r'\\label\s*{.*?}', table_source, re.S)) == len(find_snippets(table_source, 'resizebox'))
                resizebox_label = re.findall(r'\\label\s*{.*?}', table_source, re.S)[resizebox_snippet_index]
                table_label = resizebox_label
                table_latex = resizebox_snippet
            elif '\\begin{subtable}' in table_source:
                # Extract subtables
                subtable_snippets = [item for item in re.findall(r'\\begin\s*?{\s*?subtable.*?subtable.*?}', table_source, flags=re.DOTALL) if '\\begin{tabular' in item and '\\end{tabular' in item]
                subtable_snippets = [item for item in subtable_snippets if table_latex in item]
                assert len(subtable_snippets) == 1

                subtable_snippet = subtable_snippets[0]
                
                # Find the label of the subtable snippet
                subtable_label = re.findall(r'\\label\s*{.*?}', subtable_snippet, re.S)
                if len(subtable_label) == 1:
                    subtable_label = subtable_label[0]
                    table_label = subtable_label
                    table_latex = subtable_snippet
                else:
                    subtable_label_freq = [paper_latex.count(item) for item in subtable_label]
                    subtable_label = subtable_label[subtable_label_freq.index(max(subtable_label_freq))]
                    table_label = subtable_label
                    table_latex = subtable_snippet
            elif '\\begin{minipage}' in table_source:
                # Extract minipages
                minipage_snippets = [item for item in re.findall(r'\\begin\s*?{\s*?minipage.*?minipage.*?}', table_source, flags=re.DOTALL) if '\\begin{tabular' in item and '\\end{tabular' in item]
                minipage_snippets = [item for item in minipage_snippets if table_latex in item]
                assert len(minipage_snippets) == 1

                minipage_snippet = minipage_snippets[0]
                
                # Find the label of the minipage snippet
                minipage_label = re.findall(r'\\label\s*{.*?}', minipage_snippet, re.S)
                if len(minipage_label) == 1 and len(extract_tabular_code(minipage_snippet)) == 1:
                    minipage_label = minipage_label[0]
                    table_label = minipage_label
                    table_latex = minipage_snippet
                else:
                    if len(minipage_label) == len(extract_tabular_code(minipage_snippet)):
                        table_latex_index = extract_tabular_code(minipage_snippet).index(table_latex)
                        table_label = minipage_label[table_latex_index]
                    else:
                        table_label = None
            elif len(extract_tabular_code(table_source)) >= len(re.findall(r'\\label\s*{.*?}', table_source, re.S)):
                # Find the closest label after the table
                table_latex_index_start, table_latex_index_end = table_source.index(table_latex), table_source.index(table_latex) + len(table_latex)
                label_w_indexes = [(table_source.index(item), item) for item in re.findall(r'\\label\s*{.*?}', table_source, re.S)]
                label_w_indexes_selected = [item for item in label_w_indexes if item[0] >= table_latex_index_end]
                if label_w_indexes_selected:
                    label_w_index_closest = min(label_w_indexes_selected, key=lambda x: x[0])
                    if len(re.findall(r'\\begin\s*?{\s*?tabular.*?}', table_source[table_latex_index_end:label_w_index_closest[0]])) == 0:
                        table_label = label_w_index_closest[1]
                        table_latex = table_source[table_latex_index_start:label_w_index_closest[0]+len(label_w_index_closest[1])]
                    else:
                        table_label = None
                else:
                    table_label = None
            else:
                raise ValueError('Multiple labels found in the table but no \\subfloat found.')
        else:
            pass

    elif len(table_label) == 1: 
        if table_latex.count('{') != table_latex.count('}'):
            if '\\subfloat' in table_latex:
                subfloat_snippets = find_snippets(table_source, 'subfloat')
                subfloat_snippets = [item for item in subfloat_snippets if table_latex in item]
                assert len(subfloat_snippets) == 1

                subfloat_snippet = subfloat_snippets[0]

                # Find the label of the subfloat snippet
                subfloat_label = re.findall(r'\\label\s*{.*?}', subfloat_snippet, re.S)                
                if len(subfloat_label) == 1:
                    subfloat_label = subfloat_label[0]
                    table_label = subfloat_label
                    table_latex = subfloat_snippet
                else:
                    subfloat_label_freq = [paper_latex.count(item) for item in subfloat_label]
                    subfloat_label = subfloat_label[subfloat_label_freq.index(max(subfloat_label_freq))]
                    table_label = subfloat_label
                    table_latex = subfloat_snippet
            elif '\\subcaptionbox' in table_source:
                # Find the '\\subcaptionbox' snippets in the table_source based on the brackets
                subcaptionbox_snippets = find_snippets(table_source, 'subcaptionbox')
                subcaptionbox_snippets = [item for item in subcaptionbox_snippets if table_latex in item]
                assert len(subcaptionbox_snippets) == 1

                subcaptionbox_snippet = subcaptionbox_snippets[0]
                
                # Find the label of the subcaptionbox snippet
                subcaptionbox_label = re.findall(r'\\label\s*{.*?}', subcaptionbox_snippet, re.S)
                if len(subcaptionbox_label) == 1:
                    subcaptionbox_label = subcaptionbox_label[0]
                    table_label = subcaptionbox_label
                    table_latex = subcaptionbox_snippet
                else:
                    subcaptionbox_label_freq = [paper_latex.count(item) for item in subcaptionbox_label]
                    subcaptionbox_label = subcaptionbox_label[subcaptionbox_label_freq.index(max(subcaptionbox_label_freq))]
                    table_label = subcaptionbox_label
                    table_latex = subcaptionbox_snippet
            elif '\\parbox' in table_latex:
                # Find the '\\parbox' snippets in the table_source based on the brackets
                parbox_snippets = find_snippets(table_source, 'parbox')
                parbox_snippets = [item for item in parbox_snippets if table_latex in item]
                assert len(parbox_snippets) == 1

                parbox_snippet = parbox_snippets[0]
                
                # Find the label of the parbox snippet
                parbox_label = re.findall(r'\\label\s*{.*?}', parbox_snippet, re.S)
                assert len(parbox_label) == 1
                parbox_label = parbox_label[0]
                table_label = parbox_label
                table_latex = parbox_snippet
            elif 'resizebox' in table_latex and len(re.findall(r'\\label\s*{.*?}', table_source, re.S)) == len(find_snippets(table_source, 'resizebox')):
                # Find the '\\resizebox' snippets in the table_source based on the brackets
                resizebox_snippets = find_snippets(table_source, 'resizebox')
                resizebox_snippets = [item for item in resizebox_snippets if table_latex in item]
                assert len(resizebox_snippets) == 1

                resizebox_snippet = resizebox_snippets[0]
                resizebox_snippet_index = find_snippets(table_source, 'resizebox').index(resizebox_snippet)
                
                # Find the label of the resizebox snippet
                assert len(re.findall(r'\\label\s*{.*?}', table_source, re.S)) == len(find_snippets(table_source, 'resizebox'))
                resizebox_label = re.findall(r'\\label\s*{.*?}', table_source, re.S)[resizebox_snippet_index]
                table_label = resizebox_label
                table_latex = resizebox_snippet
            elif '\\begin{subtable}' in table_latex:
                # Extract subtables
                subtable_snippets = [item for item in re.findall(r'\\begin\s*?{\s*?subtable.*?subtable.*?}', table_source, flags=re.DOTALL) if '\\begin{tabular' in item and '\\end{tabular' in item]
                subtable_snippets = [item for item in subtable_snippets if table_latex in item]
                assert len(subtable_snippets) == 1

                subtable_snippet = subtable_snippets[0]
                
                # Find the label of the subtable snippet
                subtable_label = re.findall(r'\\label\s*{.*?}', subtable_snippet, re.S)
                if len(subtable_label) == 1:
                    subtable_label = subtable_label[0]
                    table_label = subtable_label
                    table_latex = subtable_snippet
                else:
                    subtable_label_freq = [paper_latex.count(item) for item in subtable_label]
                    subtable_label = subtable_label[subtable_label_freq.index(max(subtable_label_freq))]
                    table_label = subtable_label
                    table_latex = subtable_snippet
            elif '\\begin{minipage}' in table_source:
                # Find the '\\minipage' snippets in the table_source based on the brackets
                minipage_snippets = find_snippets(table_source, 'minipage')
                minipage_snippets = [item for item in minipage_snippets if table_latex in item]
                assert len(minipage_snippets) == 1

                minipage_snippet = minipage_snippets[0]
                
                # Find the label of the minipage snippet
                minipage_label = re.findall(r'\\label\s*{.*?}', minipage_snippet, re.S)
                assert len(minipage_label) == 1
                if len(minipage_label) == 1 and len(extract_tabular_code(minipage_snippet)) == 1:
                    minipage_label = minipage_label[0]
                    table_label = minipage_label
                    table_latex = minipage_snippet
                else:
                    if len(minipage_label) == len(extract_tabular_code(minipage_snippet)):
                        table_latex_index = extract_tabular_code(minipage_snippet).index(table_latex)
                        table_label = minipage_label[table_latex_index]
                    else:
                        table_label = None
            elif len(extract_tabular_code(table_source)) >= len(re.findall(r'\\label\s*{.*?}', table_source, re.S)):
                # Find the closest label after the table
                table_latex_index_start, table_latex_index_end = table_source.index(table_latex), table_source.index(table_latex) + len(table_latex)
                label_w_indexes = [(table_source.index(item), item) for item in re.findall(r'\\label\s*{.*?}', table_source, re.S)]
                label_w_indexes_selected = [item for item in label_w_indexes if item[0] >= table_latex_index_end]
                if label_w_indexes_selected:
                    label_w_index_closest = min(label_w_indexes_selected, key=lambda x: x[0])
                    if len(re.findall(r'\\begin\s*?{\s*?tabular.*?}', table_source[table_latex_index_end:label_w_index_closest[0]])) == 0:
                        table_label = label_w_index_closest[1]
                        table_latex = table_source[table_latex_index_start:label_w_index_closest[0]+len(label_w_index_closest[1])]
                    else:
                        table_label = None
                else:
                    table_label = None
            else:
                raise ValueError('Unmatched brackets in the table and no \subfloat in table_latex.')
        else:
            table_label = table_label[0]

    else:

        # More than one label found in the table_latex
        # Select the label exists the most in the paper_latex
        table_label = sorted(table_label, key=lambda x: paper_latex.count(x.replace('label', 'ref'))-(''.join(table_all)).count(x.replace('label', 'ref')), reverse=True)[0]
    
    return table_latex, table_label, table_source


# # Get the tables from the latex code
def get_tables(latex_code, select_numeric_table=True):

    table_latex = extract_table_data(latex_code)

    # Select tables with numeric values
    if select_numeric_table:

        tabular_latex = []
        tabular_label = []

        for each_table in table_latex:
            each_table_label = re.findall(r'\\label\s*{.*?}', each_table, re.S)
            table_latex_code_tabular = extract_tabular_code(each_table)

            if len(table_latex_code_tabular) == 1:
                tabular_latex.append(each_table)

                if len(each_table_label) == 1:
                    tabular_label.append(each_table_label[0])
                elif len(each_table_label) > 1:

                    # If there are more than one labels, select the first one that only happens once in all the tables
                    label_flag = False
                    for each_label in each_table_label:
                        if ' '.join(table_latex).count(each_label) == 1:
                            tabular_label.append(each_label)
                            label_flag = True
                            break

                    if not label_flag:
                        tabular_label.append(None)
                else:
                    tabular_label.append(None)
            elif len(table_latex_code_tabular) < 1: # NOTE: checked
                pass
            else:
                if len(table_latex_code_tabular) == each_table.count('\subfloat'):
                    table_latex_code_subfloat = []
                    subfloat_begin_list = [m.start() for m in re.finditer(r'\\subfloat', each_table)]
                    tabular_begin_end_list = [[each_table.find(each_tabular), each_table.find(each_tabular)+len(each_tabular)] for each_tabular in table_latex_code_tabular]
                    assert len(subfloat_begin_list) == len(tabular_begin_end_list)
                    for subfloat_begin, tabular_begin_end in zip(subfloat_begin_list, tabular_begin_end_list):
                        table_latex_code_subfloat.append(each_table[subfloat_begin:tabular_begin_end[1]])

                    table_latex_tabular_new = table_latex_code_subfloat

                else:
                    table_latex_tabular_new = table_latex_code_tabular

                for each_tabular in table_latex_tabular_new:
                    tabular_latex.append(each_tabular)

                    each_tabular_label = re.findall(r'\\label\s*{(.*?)}', each_tabular, re.S)

                    if len(each_tabular_label) == 1:
                        tabular_label.append(each_tabular_label[0])
                    elif len(each_tabular_label) == 0 and len(each_table_label) == 1:
                        tabular_label.append(each_table_label[0])
                    else:
                        tabular_label.append(None)

        table_latex_selected = []
        table_latex_label_selected = []

        # Select tables with numeric values
        for each_tabular, each_tabular_label in zip(tabular_latex, tabular_label):
            tabular_code = extract_tabular_code(each_tabular)
            
            if len(tabular_code) != 1:
                continue
            # assert len(tabular_code) == 1, "ERROR: More than one tabular found in {}".format(each_tabular)
            
            tabular_code = tabular_code[0]
            tabular_content = re.findall(r'\\begin\s*?{tabular.*?}.*?\n*?(.*)\n*?\\end\s*?{\s*?tabular.*?}', tabular_code, re.S)
            if len(tabular_content) != 1:
                continue
            # assert len(tabular_content) == 1, f"ERROR: {len(tabular_content)} tabular content found in {tabular_code}"
            tabular_content = tabular_content[0]

            tabular_content = tabular_content.replace('\\hline', '').replace('\\bottomrule', '').replace('\\toprule', '').replace('\\midrule', '')
            tabular_content = re.sub(r'\\cline{.*?}', '', tabular_content)
            tabular_content = ' '.join(tabular_content.split())
            
            # Split the table into rows
            tabular_content_rows = [each_row for each_row in tabular_content.split('\\\\') if each_row.strip()]

            tabular_content_cells = [[each_cell.strip() for each_cell in each_row.split('&')] for each_row in tabular_content_rows if each_row.strip()]

            def find_numeric_cells(tabular_data):
                numeric_cells = []
                for row_index, row in enumerate(tabular_data):
                    for col_index, cell in enumerate(row):
                        
                        # Check if the cell is numeric
                        def is_numeric(s):

                            # Extract the number from latex commands
                            if len(re.findall(r'\\text[a-zA-Z\s]*{(.*?)}', s)) == 1:
                                s = re.findall(r'\\text[a-zA-Z\s]*{(.*?)}', s)[0]

                            try:
                                float(s.strip().strip('\%').lower().strip('m').strip('b').strip('k'))
                                return s.strip('\%').strip()
                            except ValueError:
                                return False

                        if is_numeric(cell.strip()):
                            numeric_cells.append(is_numeric(cell.strip()))
                return numeric_cells

            numeric_cells = find_numeric_cells(tabular_content_cells)

            if len(numeric_cells) == 0:
                table_latex_selected.append(each_tabular)
                table_latex_label_selected.append(each_tabular_label)
        
        tables_list = list(zip(table_latex_selected, table_latex_label_selected))
    else:
        tables_list = list(zip(table_latex, [None] * len(table_latex)))

    return tables_list


def check_only_one_command(s):
    front_bracket_index_all = [(m.start(), '{') for m in re.finditer(r'{', s)]
    back_bracket_index_all = [(m.start(), '}') for m in re.finditer(r'}', s)]
    assert len(front_bracket_index_all) == len(back_bracket_index_all)
    
    front_back_bracket_index_all = sorted(front_bracket_index_all + back_bracket_index_all, key=lambda x: x[0])

    # Pair the front and back brackets using stack
    front_back_bracket_pair_dict = {}
    stack = []
    for i in range(len(front_back_bracket_index_all)):
        if front_back_bracket_index_all[i][1] == '{':
            stack.append(front_back_bracket_index_all[i][0])
        else:
            if front_back_bracket_index_all[i][0] > stack[-1]:
                front_back_bracket_pair_dict[stack.pop()] = front_back_bracket_index_all[i][0]
            else:
                raise ValueError('The brackets are not paired correctly.')

    if not s.startswith('\\multi') and front_back_bracket_pair_dict[front_bracket_index_all[0][0]] == len(s) - 1:
        return True
    elif s.startswith('\\multi') and front_back_bracket_pair_dict[front_bracket_index_all[min(len(front_back_bracket_pair_dict)-1, 2)][0]] == len(s) - 1:
        return True
    else:
        return False
    

# Check if the cell is numeric
def is_numeric(s):

    s = s.strip().replace('\%', '')

    if s.count('{') != s.count('}'):
        return False

    # Check if the string starts with { and ends with }
    if s.startswith('{') and s.endswith('}'):
        s = s[1:-1].strip()
        s = is_numeric(s)
        if not s:
            return False

    # Check if s is a command
    if s.startswith('\\') and ('{' in s and check_only_one_command(s)):
        if s.startswith('\\text') or s.startswith('\\underline') or s.startswith('\\bf'):
            s = re.findall(r'\\[a-zA-Z\s]*{(.*)}', s)[0]
            s = is_numeric(s)
            if not s:
                return False
        elif s.startswith('\\multicolumn') or s.startswith('\\multirow'):
            s = s.split('{')[-1].replace('}', '').strip()
            s = is_numeric(s)
            if not s:
                return False

    # Remove \cellcolor{*} from the cell
    if '\\cellcolor' in s:
        s = re.sub(r'\\cellcolor{.*}', '', s).strip()
        s = is_numeric(s)
        if not s:
            return False
        
    # Check if there is \pm in the cell
    if '\\pm' in s:
        s = s.strip()
        first_space_index = s.find(' ')
        if first_space_index != -1 and first_space_index < s.find('\\pm'):
            s = s[:first_space_index].strip()
        else:
            s = s.split('\\pm')[0].strip().strip('$')
            
        s = is_numeric(s)
        if not s:
            return False

    if '(' in s and ')' in s:
        s = s.split('(')[0].strip()
        s = is_numeric(s)
        if not s:
            return False

    if s.startswith('$') and s.endswith('$'):
        s = s[1:-1]
    elif s.count('$') % 2 != 0:
        s = s.replace('$', '')
    s = s.strip()
            
    # Check if the cell is numeric cell like '1,059,117'
    if ',' in s and False not in [is_numeric(item.strip()) and len(item.strip()) == 3 for item in s.split(',')[1:]]:
        return s

    try:
        # float(s.strip().strip('\%').lower().strip('m').strip('b').strip('k').strip('g').replace('^', '').replace('$', ''))
        float(s.strip().strip('\%').lower().strip('m').strip('b').strip('k').strip('g').replace('$', ''))
        return s.strip('\%').strip()
    except ValueError:
        if 'x' in s and len(s.split('x')) == 2 and False not in [is_numeric(item.strip()) for item in s.split('x')]:
            return s
        else:
            return False
        

def find_numeric_cells(tabular_data):
    numeric_cells = []
    for row_index, row in enumerate(tabular_data):
        for col_index, cell in enumerate(row):

            if is_numeric(cell.strip()):
                numeric_cells.append([is_numeric(cell.strip()), cell.strip()])

    return numeric_cells


def extract_numeric_cells(tabular_code):
    tabular_content = re.findall(r'\\begin\s*?{tabular.*?}.*?\n+?(.*)\n*?\\end\s*?{\s*?tabular.*?}', tabular_code, re.S)

    assert len(tabular_content) == 1, "ERROR: More than one tabular content found in {}".format(tabular_code)
    tabular_content_org = tabular_content[0]
    tabular_content = tabular_content_org

    # Substitute the \hline, \cline, \bottomrule commands
    tabular_content = tabular_content.replace('\\hline', '').replace('\\bottomrule', '').replace('\\toprule', '').replace('\\midrule', '')
    tabular_content = re.sub(r'\\cline{.*?}', '', tabular_content)
    # tabular_content = ' '.join(tabular_content.split())
    
    # Split the table into rows
    tabular_content_rows = [each_row for each_row in tabular_content.split('\\\\') if each_row.strip()]
    
    # Split the rows into columns
    tabular_content_cells = [[each_cell.strip() for each_cell in each_row.split('&')] for each_row in tabular_content_rows if each_row.strip()]
    
    extracted_numeric_cells_org = find_numeric_cells(tabular_content_cells)
    extracted_numeric_cells = [each_cell[0] for each_cell in extracted_numeric_cells_org]
    extracted_numeric_cells_raw = [each_cell[1] for each_cell in extracted_numeric_cells_org]

    if len(extracted_numeric_cells) != len([item for item in extracted_numeric_cells if item in tabular_content_org]):
        raise Exception("ERROR: Number of extracted numeric cells is not equal to number of cells in the tabular code")
    assert len(extracted_numeric_cells) == len([item for item in extracted_numeric_cells if item in tabular_content_org]), "ERROR: Number of extracted numeric cells is not equal to number of cells in the tabular code"

    tabular_content_processed = tabular_content_org.replace('\\\\', '&&').replace('\n', '&')
    assert len(tabular_content_processed) == len(tabular_content_org), "ERROR: Length of processed tabular content is not equal to original tabular content"

    # Find the position of all & symbols
    and_positions = [m.start() for m in re.finditer('&', tabular_content_processed)]

    cell_spans = [[tabular_content_processed[and_positions[i]+1:and_positions[i+1]], and_positions[i]+1, and_positions[i+1]] for i in range(0, len(and_positions)-1)]
    cell_spans_filtered = [each_cell for each_cell in cell_spans if each_cell[0].strip() in extracted_numeric_cells_raw]
  
    if len(cell_spans_filtered) != len(extracted_numeric_cells_raw):
        raise Exception("ERROR: Number of numeric cells and cell spans do not match")

    extracted_numeric_cells_org_idx = []
    for cell_idx in range(len(cell_spans_filtered)):
        cell_raw_w_space, cell_raw_w_space_start, cell_raw_w_space_end = cell_spans_filtered[cell_idx]
        cell_value, cell_raw = extracted_numeric_cells_org[cell_idx]
        assert len(cell_raw.strip()) == len(cell_raw)
        assert cell_raw_w_space.strip() == cell_raw.strip(), "ERROR: Cell span and extracted numeric cell do not match"
        assert cell_raw_w_space.count(cell_raw) == 1, "ERROR: Cell span contains more than one instance of the extracted numeric cell"
        assert tabular_content_org[cell_raw_w_space_start:cell_raw_w_space_end] == cell_raw_w_space, "ERROR: Cell span and extracted numeric cell do not match"

        cell_raw_start = cell_raw_w_space_start + cell_raw_w_space.find(cell_raw) + tabular_code.find(tabular_content_org)
        cell_raw_end = cell_raw_start + len(cell_raw)
        assert tabular_code[cell_raw_start:cell_raw_end] == cell_raw, "ERROR: Cell span and extracted numeric cell do not match"

        if cell_raw.count(cell_value) == 1:
            cell_value_start = cell_raw_start + cell_raw.find(cell_value)
            cell_value_end = cell_value_start + len(cell_value)
            assert tabular_code[cell_value_start:cell_value_end] == cell_value, "ERROR: Cell span and extracted numeric cell do not match"
            extracted_numeric_cells_org_idx.append([cell_value, cell_raw, cell_value_start, cell_value_end, cell_raw_start, cell_raw_end])
        else:
            # Find the last occurence of the cell value in the cell raw
            cell_value_start = cell_raw_start + (cell_raw.rfind(cell_value) if '\\multi' in cell_raw else cell_raw.find(cell_value))
            cell_value_end = cell_value_start + len(cell_value)
            assert tabular_code[cell_value_start:cell_value_end] == cell_value, "ERROR: Cell span and extracted numeric cell do not match"
            extracted_numeric_cells_org_idx.append([cell_value, cell_raw, cell_value_start, cell_value_end, cell_raw_start, cell_raw_end])

    return extracted_numeric_cells_org_idx


def check_only_one_command_axcell(s):
    front_bracket_index_all = [(m.start(), '{') for m in re.finditer(r'{', s)]
    back_bracket_index_all = [(m.start(), '}') for m in re.finditer(r'}', s)]
    assert len(front_bracket_index_all) == len(back_bracket_index_all)
    
    front_back_bracket_index_all = sorted(front_bracket_index_all + back_bracket_index_all, key=lambda x: x[0])
    
    # TODO: Re-write the command parsing logic
    if len(front_back_bracket_index_all) % 2 != 0 or front_back_bracket_index_all[0][1] != '{' or front_back_bracket_index_all[-1][1] != '}':
        # raise ValueError('The brackets are not paired correctly.')
        return False
    
    count = 0
    for i in range(len(front_back_bracket_index_all)):
        if front_back_bracket_index_all[i][1] == '{':
            count += 1
        else:
            count -= 1
        if count < 0:
            return False

    # Pair the front and back brackets using stack
    front_back_bracket_pair_dict = {}
    stack = []
    for i in range(len(front_back_bracket_index_all)):
        if front_back_bracket_index_all[i][1] == '{':
            stack.append(front_back_bracket_index_all[i][0])
        else:
            if front_back_bracket_index_all[i][0] > stack[-1]:
                front_back_bracket_pair_dict[stack.pop()] = front_back_bracket_index_all[i][0]
            else:
                raise ValueError('The brackets are not paired correctly.')

    # Check if there is only one command
    if not (s.startswith('\\multi') or s.startswith('\\tabincell')) and front_back_bracket_pair_dict[front_bracket_index_all[0][0]] == len(s) - 1:
        return True
    elif (s.startswith('\\multi') or s.startswith('\\tabincell')) and front_back_bracket_pair_dict[front_bracket_index_all[min(len(front_back_bracket_pair_dict)-1, 2)][0]] == len(s) - 1:
        return True
    else:
        return False


# Check if the cell is numeric
def is_numeric_axcell(s):

    if 'begin{' in s and 'end{' in s:
        s = re.findall(r'\\begin{.*?}(.*)\\end{.*}', s)[0]
        s = s.strip()
        
    if s.startswith('\\tabincell') and s.count('{') == s.count('}')+1:
        s = s + '}'

    # s = s.strip().replace('\%', '')
    s = s.strip().rstrip('\%').strip('~').strip()
    s = re.sub(r'\\\s+', '', s)
    
    s = s.replace('\!\!', '')

    if s.count('{') != s.count('}'):
        return False

    # Check if the string starts with { and ends with }
    if s.startswith('{') and s.endswith('}'):
        s = s[1:-1].strip()
        if s.startswith('\\'):
            s_find = re.findall(r'\\[a-zA-Z]+\s*\{*\}*(.*)\s*', s)
            if len(s_find) != 0:
                s = s_find[0]
            s = is_numeric_axcell(s)
            if not s:
                return False
        else:
            s = is_numeric_axcell(s)
            if not s:
                return False
    elif s.endswith('}') and s[0].isdigit():
        first_index_list = [s.find('{'), s.find(' '), s.find('\\')]
        first_index_list = [i for i in first_index_list if i != -1]
        if first_index_list:
            first_index = min(first_index_list)
            s = s[:first_index].strip()
            s = is_numeric_axcell(s)
            if not s:
                return False
        
    # Check if the string starts with $ and ends with $
    if s.startswith('$'):
        if s.endswith('$'):
            s = s[1:-1].strip()
            s = is_numeric_axcell(s)
            if not s:
                return False
        else:
            second_dollar_index = s.find('$', 1)
            if second_dollar_index != -1:
                s = s[1:second_dollar_index].strip()
                s = is_numeric_axcell(s)
                if not s:
                    return False
    elif s.endswith('$'):
        first_dollar_index = s.find('$')
        if first_dollar_index != -1:
            s = s[:first_dollar_index].strip()
            s = is_numeric_axcell(s)
            if not s:
                return False

    # Check if s is a command
    # if len(re.findall(r'\\[a-zA-Z\s]*{(.*)}', s)) == 1 and len(re.findall(r'\\[a-zA-Z\s]*{.*}', s)[0]) == len(s):
    if s.startswith('\\'):
        if '{' in s and check_only_one_command_axcell(s):
            if s.startswith('\\text') or s.startswith('\\math') or s.startswith('\\underline') or s.startswith('\\bf') or s.startswith('\\bm') or s.startswith('\\t{') or s.startswith('\\b{') or s.startswith('\\small{') or s.startswith('\\bleu{'):
                s = re.findall(r'\\[a-zA-Z\s]*{(.*)}', s)[0]
                s = is_numeric_axcell(s)
                if not s:
                    return False
            elif s.startswith('\\multicolumn') or s.startswith('\\multirow') or s.startswith('\\tabincell'):
                s = s.split('{')[-1].replace('}', '').strip()
                s = is_numeric_axcell(s)
                if not s:
                    return False
        elif '{' not in s:
            if re.findall(r'\\[a-zA-Z]+\s*(.*)\s*', s) and (s.startswith('\\bf') or s.startswith('\\small') or s.startswith('\\scriptsize')):
                s = re.findall(r'\\[a-zA-Z]+\s*(.*)\s*', s)[0]
                s = is_numeric_axcell(s)
                if not s:
                    return False
            else:
                return False

    # Remove \cellcolor{*} from the cell
    if '\\cellcolor' in s:
        s = re.sub(r'\\cellcolor\[*.*\]*{.*}', '', s).strip()
        s = is_numeric_axcell(s)
        if not s:
            return False
        
    # Remove \textcolor{*} from the cell
    if s.startswith('\\textcolor'):
        s = re.findall(r'\\textcolor\s*\[*.*\]*\s*{.*}{(.*)}', s)[0].strip()
        s = is_numeric_axcell(s)
        if not s:
            return False
        
    # Check if there is \pm in the cell
    if '\\pm' in s or '\\textpm' in s:
        pm_str = '\\pm' if '\\pm' in s else '\\textpm'
        s = s.strip()
        first_space_index = s.find(' ')
        if first_space_index != -1 and first_space_index < s.find(pm_str):
            s = s[:first_space_index].strip()
        else:
            s = s.split(pm_str)[0].strip().strip('$')
            
        s = is_numeric_axcell(s)
        if not s:
            return False

    if '(' in s and ')' in s:
        s = s.split('(')[0].strip()
        s = is_numeric_axcell(s)
        if not s:
            return False
        
    if '/' in s:
        s = s.split('/')[0].strip()
        s = is_numeric_axcell(s)
        if not s:
            return False

    if s.startswith('$') and s.endswith('$'):
        s = s[1:-1]
    elif s.count('$') % 2 != 0:
        s = s.replace('$', '')
    s = s.strip()
    
    if '$\\downarrow$' in s:
        s = s.split('$\\downarrow$')[0].strip()
        s = is_numeric_axcell(s)
        if not s:
            return False
    elif '$\\uparrow$' in s:
        s = s.split('$\\uparrow$')[0].strip()
        s = is_numeric_axcell(s)
        if not s:
            return False
        
    # if s ends with m, b, k, g, then it is not a numeric cell
    if s and s.strip().lower()[-1] in ['m', 'b', 'k', 'g']:
        return False
    
    # Check if the cell is numeric cell like '1,059,117'
    if ',' in s and len((s if '.' not in s else s.split('.')[0]).split(',')[1:]) and len((s if '.' not in s else s.split('.')[0]).split(',')[1:]) == sum([is_numeric_axcell(item.strip()) and len(item.strip()) == 3 and ' ' not in item for item in (s if '.' not in s else s.split('.')[0]).split(',')[1:]]):
        return s

    try:
        float(s.strip().strip('\%').lower().replace('$', ''))
        return s.strip().strip('\%').strip().replace('$', '')
    except ValueError:
        return False


def find_numeric_cells_axcell(tabular_data):
    numeric_cells = []
    for row_index, row in enumerate(tabular_data):
        for col_index, cell in enumerate(row):

            if is_numeric_axcell(cell.strip()):
                numeric_cells.append([is_numeric_axcell(cell.strip()), cell.strip()])

    return numeric_cells


def extract_numeric_cells_axcell(tabular_code):

    tabular_code_org = tabular_code

    if 'tabularnewline' in tabular_code:
        tabular_code_new = tabular_code.replace('\\tabularnewline', '\\\\'+len('abularnewline')*' ')
        assert len(tabular_code_new) == len(tabular_code)
        tabular_code = tabular_code_new

    tabular_content = re.findall(r'\\begin\s*?{tabular.*?}.*?\n+?(.*)\n*?\\end\s*?{\s*?tabular.*?}', tabular_code, re.S)
    if len(tabular_content) != 1:
        return []
    assert len(tabular_content) == 1, "ERROR: More than one tabular content found in {}".format(tabular_code)
    tabular_content_org = tabular_content[0]

    tabular_content = tabular_content_org

    # Substitute the \hline, \cline, \bottomrule commands
    tabular_content = tabular_content.replace('\\hline', '').replace('\\bottomrule', '').replace('\\toprule', '').replace('\\midrule', '')
    tabular_content = re.sub(r'\\cline{.*?}', '', tabular_content)
    
    # Split the table into rows
    tabular_content_rows = [each_row for each_row in tabular_content.split('\\\\') if each_row.strip()]
    
    # Split the rows into columns
    tabular_content_cells = [[each_cell.strip() if '\n' not in each_cell.strip() else each_cell.split('\n')[-1].strip() for each_cell in each_row.split('&')] for each_row in tabular_content_rows if each_row.strip()]
    
    extracted_numeric_cells_org = find_numeric_cells_axcell(tabular_content_cells)
    extracted_numeric_cells = [each_cell[0] for each_cell in extracted_numeric_cells_org]
    extracted_numeric_cells_raw = [each_cell[1] for each_cell in extracted_numeric_cells_org]
    
    if len(extracted_numeric_cells) != len([item for item in extracted_numeric_cells if item in tabular_content_org]):
        return []
    assert len(extracted_numeric_cells) == len([item for item in extracted_numeric_cells if item in tabular_content_org]), "ERROR: Number of extracted numeric cells is not equal to number of cells in the tabular code"

    tabular_content_processed = tabular_content_org.replace('\\\\', '&&').replace('\n', '&')
    assert len(tabular_content_processed) == len(tabular_content_org), "ERROR: Length of processed tabular content is not equal to original tabular content"

    # Find the position of all & symbols
    and_positions = [m.start() for m in re.finditer('&', tabular_content_processed)]
    cell_spans = [[tabular_content_processed[and_positions[i]+1:and_positions[i+1]], and_positions[i]+1, and_positions[i+1]] for i in range(0, len(and_positions)-1)]
    cell_spans_filtered = [each_cell for each_cell in cell_spans if each_cell[0].strip() in extracted_numeric_cells_raw]
    
    if len(cell_spans_filtered) != len(extracted_numeric_cells_raw):
        return []

    extracted_numeric_cells_org_idx = []
    for cell_idx in range(len(cell_spans_filtered)):
        cell_raw_w_space, cell_raw_w_space_start, cell_raw_w_space_end = cell_spans_filtered[cell_idx]
        cell_value, cell_raw = extracted_numeric_cells_org[cell_idx]
        assert len(cell_raw.strip()) == len(cell_raw)
        assert cell_raw_w_space.strip() == cell_raw.strip(), "ERROR: Cell span and extracted numeric cell do not match"
        assert cell_raw_w_space.count(cell_raw.strip()) == 1, "ERROR: Cell span contains more than one instance of the extracted numeric cell"
        assert tabular_content_org[cell_raw_w_space_start:cell_raw_w_space_end] == cell_raw_w_space, "ERROR: Cell span and extracted numeric cell do not match"

        cell_raw_start = cell_raw_w_space_start + cell_raw_w_space.find(cell_raw) + tabular_code.find(tabular_content_org)
        cell_raw_end = cell_raw_start + len(cell_raw)
        assert tabular_code_org[cell_raw_start:cell_raw_end] == cell_raw, "ERROR: Cell span and extracted numeric cell do not match"

        if cell_raw.count(cell_value) == 1:
            cell_value_start = cell_raw_start + cell_raw.find(cell_value)
            cell_value_end = cell_value_start + len(cell_value)
            assert tabular_code_org[cell_value_start:cell_value_end] == cell_value, "ERROR: Cell span and extracted numeric cell do not match"
            extracted_numeric_cells_org_idx.append([cell_value, cell_raw, cell_value_start, cell_value_end, cell_raw_start, cell_raw_end])
        else:
            # Find the last occurence of the cell value in the cell raw
            cell_value_start = cell_raw_start + (cell_raw.rfind(cell_value) if '\\multi' in cell_raw else cell_raw.find(cell_value))
            cell_value_end = cell_value_start + len(cell_value)
            assert tabular_code_org[cell_value_start:cell_value_end] == cell_value, "ERROR: Cell span and extracted numeric cell do not match"
            extracted_numeric_cells_org_idx.append([cell_value, cell_raw, cell_value_start, cell_value_end, cell_raw_start, cell_raw_end])

    return extracted_numeric_cells_org_idx
