import os
import re
import xml.etree.ElementTree as ET

from bs4 import BeautifulSoup
from create_so_data import get_lines, get_first, parse_args
from so_twokenize import tokenize

code_file_number = 1


# Figure out which qids are relevant
def get_qids_of_interest(labeled_data_files):
    qids = set()
    for f in labeled_data_files:
        lines = get_lines(f)
        for l in lines:
            if get_first(l).lower() in ['question_id', 'answer_to_question_id']:
                next(lines)
                qid = next(lines).strip().split("\t")[0]
                qids.add(qid)
    return qids


def extract_text_from_xml(body):
    """Copied and minimally edited from
    https://github.com/jeniyat/StackOverflowNER/blob/master/code/DataReader/read_so_post_info.py"""
    body_str = body.encode("utf-8").strip()
    extracted_text = ""

    soup = BeautifulSoup(body_str, "lxml")
    all_tags = soup.find_all(True)
    for para in soup.body:
        text_for_current_block = str(para)

        #
        temp_soup = BeautifulSoup(text_for_current_block, "lxml")
        list_of_tags = [tag.name for tag in temp_soup.find_all()]
        tag_len = len(list_of_tags)

        if set(list_of_tags) == {'html', 'body', 'pre', 'code'}:
            code_string = temp_soup.pre.string
            temp_soup.pre.string = "CODE_BLOCK: Q_" + str(code_file_number) + " (code omitted for annotation)\n"
        elif "code" in list_of_tags:
            all_inline_codes = temp_soup.find_all("code")

            for inline_code in all_inline_codes:
                inline_code_string_raw = str(inline_code)

                temp_code_soup = BeautifulSoup(inline_code_string_raw, "lxml")
                inline_code_string_list_of_text = temp_code_soup.findAll(text=True)
                inline_code_string_text = "".join(inline_code_string_list_of_text).strip()

                try:
                    if "\n" in inline_code_string_text:
                        code_string = inline_code_string_text

                        inline_code.string = "CODE_BLOCK: Q_" + str(
                            code_file_number) + " (code omitted for annotation)\n"


                    elif inline_code_string_text.count('.') >= 1:
                        inline_code.string = "--INLINE_CODE_BEGIN---" + inline_code_string_text.replace(".",
                                                                                                        "..").replace(
                            '\r', '').replace('\n', '') + "--INLINE_CODE_END---"

                    elif inline_code_string_text.count('?') >= 1:
                        inline_code.string = "--INLINE_CODE_BEGIN---" + inline_code_string_text.replace("?",
                                                                                                        "<?-?>").replace(
                            '\r', '').replace('\n', '') + "--INLINE_CODE_END---"
                    else:
                        inline_code.string = "--INLINE_CODE_BEGIN---" + inline_code_string_text.replace('\r',
                                                                                                        '').replace(
                            '\n', '') + "--INLINE_CODE_END---"
                except Exception as e:

                    print("DEBUG----- inisde except for inline code -------- error", e)

                    continue

        if "blockquote" in list_of_tags:
            op_string = temp_soup.blockquote.string
            temp_soup.blockquote.string = "OP_BLOCK: (output omitted for annotation)\n"

        if "kbd" in list_of_tags:
            all_keyboard_ip = temp_soup.find_all("kbd")
            # print("DEBUG-keyboard-input", all_keyboard_ip)
            # print("DEBUG---inputxml: ",body_str)
            for keyboard_ip in all_keyboard_ip:
                print("DEBUG-keyboard-input", keyboard_ip.string)
                keyboard_ip.string = "--KEYBOARD_IP_BEGIN---" + keyboard_ip.string + "--KEYBOARD_IP_END---"

            # print(keyboard_ip.string)

        list_of_texts = temp_soup.findAll(text=True)
        text = "".join(list_of_texts)

        extracted_text += text
        extracted_text += "\n\n"

    # print("DEBUG--extracted-text-for-xml: \n",extracted_text)

    return extracted_text


def find_string_indices(input_string, string_to_search):
    """Copied from
        https://github.com/jeniyat/StackOverflowNER/blob/master/code/DataReader/read_so_post_info.py"""
    string_indices=[]
    location=-1
    while True:
        location = input_string.find(string_to_search, location + 1)
        if location == -1: break
        string_indices.append(location)
    return string_indices


def tokenize_and_annotate_post_body(xml_filtered_string, post_id):
    """Copied and minimally edited from
    https://github.com/jeniyat/StackOverflowNER/blob/master/code/DataReader/read_so_post_info.py"""
    tokenized_body = tokenize(xml_filtered_string)

    tokenized_body_extra_qmark_and_stop_removed = []
    for sentence in tokenized_body:
        if "--INLINE_CODE_BEGIN---" in sentence:
            sentence = sentence.replace("..", ".")
            sentence = sentence.replace("<?-?>", "?")
        tokenized_body_extra_qmark_and_stop_removed.append(sentence)

    tokenized_body_new_line_removed = [re.sub(r"\n+", "\n", sentence) for sentence in
                                       tokenized_body_extra_qmark_and_stop_removed]

    tokenized_body_str = "\n".join(tokenized_body_new_line_removed)

    # -----------------postions for inline codes------------------------------

    inline_code_begining_positions = find_string_indices(tokenized_body_str, "--INLINE_CODE_BEGIN")
    inline_code_ending_positions_ = find_string_indices(tokenized_body_str, "INLINE_CODE_END---")
    len_inline_code_end_tag = len("INLINE_CODE_END---")
    inline_code_ending_positions = [x + len_inline_code_end_tag for x in inline_code_ending_positions_]

    # -----------------postions for keyboard inputs------------------------------

    keyboard_ip_begining_positions = find_string_indices(tokenized_body_str, "--KEYBOARD_IP_BEGIN")
    keyboard_ip_ending_positions_ = find_string_indices(tokenized_body_str, "KEYBOARD_IP_END---")
    len_keyboard_ip_end_tag = len("KEYBOARD_IP_END---")
    keyboard_ip_ending_positions = [x + len_keyboard_ip_end_tag for x in keyboard_ip_ending_positions_]

    # -----------------postions for code blocks-----------------------------

    code_block_begining_positions = find_string_indices(tokenized_body_str, "CODE_BLOCK:")
    code_block_ending_positions_ = find_string_indices(tokenized_body_str, "(code omitted for annotation)")
    len_code_block_end_tag = len("(code omitted for annotation)")
    code_block_ending_positions = [x + len_code_block_end_tag for x in code_block_ending_positions_]

    # -----------------postions for output blocks------------------------------

    op_block_begining_positions = find_string_indices(tokenized_body_str, "OP_BLOCK:")
    op_block_ending_positions_ = find_string_indices(tokenized_body_str, "(output omitted for annotation)")
    len_op_block_end_tag = len("(output omitted for annotation)")
    op_block_ending_positions = [x + len_op_block_end_tag for x in op_block_ending_positions_]

    question_url_text = "Question_URL: " + "https://stackoverflow.com/questions/" + str(post_id) + "/"
    intro_text = "Question_ID: " + str(post_id) + "\n" + question_url_text + "\n\n"

    op_string = tokenized_body_str.replace("--INLINE_CODE_BEGIN---", "").replace("--INLINE_CODE_END---", "").replace(
        "--KEYBOARD_IP_BEGIN---", "").replace("--KEYBOARD_IP_END---", "")

    return intro_text + "\n" + op_string + "\n"


def write_to_file(post_id, create_date, body):
    with open("so_data/posts_texts/%s.txt" % post_id, 'w') as f:
        f.write("%s\n%s\n%s" % (post_id, create_date, extract_text_from_xml(body)))


# Use etree to go through XML. If qid is relevant, write a file that has the create date of the post and other things
def write_out_files_with_create_dates(posts_file, qids_of_interest):
    with open(posts_file, 'r') as f:
        ctr = 0
        for l in f:
            try:
                tree = ET.fromstring(l)
                attrib = tree.attrib
                if attrib["PostTypeId"] == '1' and attrib["Id"] in qids_of_interest:
                    write_to_file(attrib["Id"], attrib["CreationDate"], attrib["Body"])
                elif attrib["PostTypeId"] == '2' and attrib["ParentId"] in qids_of_interest:
                    post_id = "%s_%s" % (attrib["ParentId"], attrib["Id"])
                    write_to_file(post_id, attrib["CreationDate"], attrib["Body"])
                ctr += 1
                if ctr % 100000 == 0:
                    print(attrib["Id"])
            except ET.ParseError:
                pass


if __name__ == "__main__":
    args = parse_args()

    labeled_data_files = ["train.txt", "dev.txt", "test.txt"]
    labeled_data_files = [os.path.join(args.labeled_data_dir, "StackOverflowNER/resources/annotated_ner_data/StackOverflow",
                                       f_name) for f_name in labeled_data_files]
    qids = get_qids_of_interest(labeled_data_files)

    write_out_files_with_create_dates(os.path.join("so_data", "Posts.xml"), qids)
