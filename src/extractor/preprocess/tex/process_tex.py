import re

from pathlib import Path

from .tex_utils import get_tex_file_names, get_tables


def preprocess_tex_src(paper_source_dir: str, remove_appendix: bool = True):

    paper_path_cleaned = paper_source_dir
    
    # Check if the paper path exists or paper folder is empty, if so, use the original paper
    if not Path(paper_path_cleaned).exists() or len(get_tex_file_names(Path(paper_path_cleaned))) == 0:
        return None

    file_names = get_tex_file_names(Path(paper_path_cleaned))

    # Collect the tex file which contains \begin{document}
    tex_files = []
    for file in file_names:
        with open(file, 'r', errors='ignore') as f:
            content = f.read()
            if '\\begin{document}' in content:
                tex_files.append(file)

    if len(tex_files) == 0:
        return None
    
    latex_code_list = [] # Load the latex code in the main tex file

    for main_tex_file in tex_files:
        with open(main_tex_file, 'r', errors='ignore') as f:
            latex_code = f.read()

            def merge_latex_file(latex_code_org):

                latex_code_merged = latex_code_org

                # Replace \input{...} with the content of the file
                for match in re.finditer(r'\\input{(.*)}', latex_code_merged):

                    if match.group(1).strip().endswith('bbl'):
                        latex_code_merged = latex_code_merged.replace(match.group(0), '')
                        continue

                    file_name = match.group(1).strip() if match.group(1).strip().endswith('.tex') or match.group(1).strip().endswith('_tex') else f'{match.group(1).strip()}.tex'
                    if len(file_name) > 100:
                        continue
                    file_path = main_tex_file.parent / file_name

                    # Check if the file exists
                    if file_path.exists():
                        with open(file_path, 'r', errors='ignore') as f:
                            content = f.read()
                        latex_code_merged = latex_code_merged.replace(match.group(0), '\n'+content+'\n')
                    else:
                        file_name_tail = file_name.split('/')[-1] # HACK: some relative paths are not existent in the path
                        if len(file_name_tail) > 100:
                            continue
                        file_path_tail = main_tex_file.parent / file_name_tail                       
                    
                        if file_path_tail.exists():
                            with open(file_path_tail, 'r', errors='ignore') as f:
                                content = f.read()
                            latex_code_merged = latex_code_merged.replace(match.group(0), '\n'+content+'\n')
                        else:
                            pass

                # Replace \include{...} with the content of the file
                for match in re.finditer(r'\\include{(.*)}', latex_code_merged):

                    if match.group(1).strip().endswith('bbl'):
                        latex_code_merged = latex_code_merged.replace(match.group(0), '')
                        continue

                    file_name = match.group(1).strip() if match.group(1).strip().endswith('.tex') else f'{match.group(1).strip()}.tex'
                    if len(file_name) > 100:
                        continue                    
                    file_path = main_tex_file.parent / file_name

                    # Check if the file exists
                    if file_path.exists():
                        with open(file_path, 'r', errors='ignore') as f:
                            content = f.read()
                        latex_code_merged = latex_code_merged.replace(match.group(0), '\n'+content+'\n')
                    else:
                        file_name_tail = file_name.split('/')[-1] # HACK: some relative paths are not existent in the path
                        if len(file_name_tail) > 100:
                            continue
                        file_path_tail = main_tex_file.parent / file_name_tail
                                                
                        if file_path_tail.exists():
                            with open(file_path_tail, 'r', errors='ignore') as f:
                                content = f.read()
                            latex_code_merged = latex_code_merged.replace(match.group(0), '\n'+content+'\n')
                        else:
                            pass                    
                return latex_code_merged

            latex_code = merge_latex_file(latex_code)

            def remove_redundant_empty_lines(latex_code_org):

                latex_code = latex_code_org
                # Remove redundant empty lines
                for match in re.finditer(r'\n{2,}', latex_code):
                    latex_code = latex_code.replace(match.group(0), '\n\n')

                return latex_code

            latex_code = remove_redundant_empty_lines(latex_code)
            
            # Remove the % at the end of the line
            latex_code = re.sub(r'%\s*$', '', latex_code, flags=re.MULTILINE)

            # Get commands in the latex code (starting with \newcommand)
            commands = [item for item in latex_code.split('\n') if item.strip().startswith('\\newcommand')]
            defs = [item for item in latex_code.split('\n') if item.strip().startswith('\\def\\') or item.startswith('\\def{')]
            defs = [('\def'+item1).strip() for item in defs for item1 in item.split('\def') if item1] # There might be multiple \def in one line

            def parse_self_defined_command(command, command_name):

                # With arguments
                p1 = re.compile(command_name+r'\s*\{(?P<command_name>\\.*)\}\s*\[(?P<arg_num>\d)\]\s*\{(?P<command_content>.*)\}')
                p2 = re.compile(command_name+r'\s*(?P<command_name>\\.*)\s*\[(?P<arg_num>\d)\]\s*\{(?P<command_content>.*)\}')
                m1 = p1.match(command)
                m2 = p2.match(command)
                if m1:
                    return m1.group('command_name'), m1.group('command_content'), int(m1.group('arg_num'))
                elif m2:
                    return m2.group('command_name'), m2.group('command_content'), int(m2.group('arg_num'))
                else:
                    # Without arguments
                    p1 = re.compile(command_name+r'\s*\{(?P<command_name>\\.*?)\}\s*\{(?P<command_content>.*)\}')
                    p2 = re.compile(command_name+r'\s*(?P<command_name>\\.*?)\s*\{(?P<command_content>.*)\}')
                    m1 = p1.match(command)
                    m2 = p2.match(command)
                    if m1:
                        return m1.group('command_name'), m1.group('command_content'), 0
                    elif m2:
                        return m2.group('command_name'), m2.group('command_content'), 0
                    else:
                        # If the command is not in the correct format, print the command
                        if command_name == r'\\newcommand' and command.startswith(r'\\newcommand\*'):
                            return None, None, None
                        else:
                            return None, None, None

            # Replace the commands with the content of the commands
            commands_dict = {}
            for command in commands:
                
                command = command.strip()

                if command.startswith('\\newcommand*'):
                    command_name, command_content, arg_num = parse_self_defined_command(command, r'\\newcommand\*')
                elif command.startswith('\\newcommand '):
                    command_name, command_content, arg_num = parse_self_defined_command(command, r'\\newcommand\s+')
                elif command.startswith('\\newcommand'):
                    command_name, command_content, arg_num = parse_self_defined_command(command, r'\\newcommand')
                else:
                    continue
                
                # Select the commands that are used in the paper and have no arguments
                # if command_name and command_name in latex_code and arg_num == 0 and command_content.count('\\') == 0:
                if command_name and command_name in latex_code and arg_num == 0 and command_content.count('\\') <= 3:
                    commands_dict[command_name.strip()] = command_content.strip()[1:-1] if command_content.strip().startswith('{') and command_content.strip().endswith('}') else command_content.strip()
            
            for command in defs:
                command_name, command_content, arg_num = parse_self_defined_command(command, r'\\def')
                
                # Select the commands that are used in the paper and have no arguments
                # if command_name and command_name in latex_code and arg_num == 0 and command_content.count('\\') == 0:
                if command_name and command_name in latex_code and arg_num == 0 and command_content.count('\\') <= 3:
                    commands_dict[command_name.strip()] = command_content.strip()[1:-1] if command_content.strip().startswith('{') and command_content.strip().endswith('}') else command_content.strip()
            
            # Replace the commands in latex_code with the content of the commands
            for command_name, command_content in commands_dict.items():
                if command_name+'{}' in latex_code:
                    latex_code = latex_code.replace(command_name+'{}', command_content)
                elif command_name in latex_code:
                    latex_code = latex_code.replace(command_name, command_content)

            # Extract content between \begin{document} and \end{document}
            try:
                latex_code = re.search(r'\\begin\s*?{\s*?document\s*?}(.*)\\end\s*?{\s*?document\s*?}', latex_code, flags=re.DOTALL).group(1)
            except AttributeError:
                pass

            # Remove the empty lines after the (sub-)section title
            for match in re.finditer(r'\\.*?section\s*?{.*?}\n{2,}', latex_code):
                try:
                    latex_code = latex_code.replace(match.group(0), match.group(0).strip()+'\n')
                except AttributeError:
                    pass

            # Remove the empty lines after the \begin command
            for match in re.finditer(r'\\begin\s*?{.*?}\n{2,}', latex_code):
                try:
                    latex_code = latex_code.replace(match.group(0), match.group(0).rstrip()+'\n')
                except AttributeError:
                    pass

            # Remove the empty lines before the \end command
            for match in re.finditer(r'\n{2,}\\end\s*?{.*?}', latex_code):
                try:
                    latex_code = latex_code.replace(match.group(0), '\n'+match.group(0).lstrip())
                except AttributeError:
                    pass

            latex_code_list.append(latex_code)

    # Find the longest latex code as the main latex code
    latex_code = max(latex_code_list, key=len)

    if remove_appendix:
        prev_latex_code = len(latex_code)
        appendix_pattern = re.compile(r'\\appendix.*?(?=\\end\{document\})', re.DOTALL)
        latex_code = re.sub(appendix_pattern, '', latex_code)
        if len(latex_code) == prev_latex_code: # HACK
            clearpage_pattern = re.compile(r'\\clearpage.*?(?=\\end\{document\})', re.DOTALL)
            latex_code = re.sub(clearpage_pattern, '', latex_code)

    return latex_code


def extract(paper_source_dir: str):
    latex_code = preprocess_tex_src(paper_source_dir, remove_appendix=True)

    if latex_code is None:
        return None

    tables_list = get_tables(latex_code, select_numeric_table=False)
    tables_index = [j for j in range(len(tables_list))]

    if len(tables_list) == 0:
        return None
    
    result = {
                'full_paper_latex_code': latex_code,
                'tables_list': tables_list,
                'tables_index': tables_index,
            }
        
    return result
