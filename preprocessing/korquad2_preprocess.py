import bs4
from bs4 import BeautifulSoup


def get_wiki_context(wiki_html, is_eval=False):
    soup = BeautifulSoup(wiki_html, 'lxml')
    structured_context = get_structure_context(soup, wiki_html, is_eval)
    return structured_context


def get_structure_context(soup, html, is_eval=False):
    output = []
    last_position = 0
    a_tag = soup.find('a', text='검색하러 가기')
    last_position = get_tag_position(a_tag, html, last_position, is_eval)
    div_soup = a_tag.find_next('div').find('div')
    last_position = get_tag_position(div_soup, html, last_position, is_eval)
    last_position = get_tag_position(div_soup, html, last_position, is_eval)
    for child in div_soup.children:
        last_position, temp_output = get_structure_tag_info(
            child, html, last_position, is_eval)
        if temp_output is not None:
            output.extend(temp_output)

    return output


def get_structure_tag_info(soup, html, last_position, is_eval=False):
    if type(soup) is bs4.element.NavigableString or soup is None or type(soup) is bs4.element.Comment:
        return last_position, None

    if soup.name == 'div':
        last_position = get_tag_position(soup, html, last_position, is_eval)
        tag_html = str(soup)
        last_position = last_position + len(tag_html)
        return last_position, None

    tag_name = soup.name
    target_list = soup.find_all(
        ['table', 'ul', 'dl', 'p', 'h2', 'h3', 'dt', 'dd', 'li', 'td', 'th', 'tr']
        )
    if target_list is not None and len(target_list) > 0:
        if tag_name in ['table', 'ul', 'dl', 'tr']:
            tag_html = str(soup)
            last_position = get_tag_position(
                soup, html, last_position, is_eval)
            output = {"text": [], 'tag': tag_name, 'start': last_position,
                      'end': last_position + len(tag_html)}
            for child in soup.children:
                last_position, child_output = get_structure_tag_info(
                    child, html, last_position, is_eval)
                if child_output is not None:
                    output['text'].extend(child_output)

            if len(output['text']) == 0:
                return last_position, None

            return last_position, [output]
        else:
            output = []
            for child in soup.children:
                last_position, child_output = get_structure_tag_info(
                    child, html, last_position, is_eval)
                if child_output is not None:
                    output.extend(child_output)
            return last_position, output
    else:
        if soup.name in ['p', 'h2', 'h3', 'dt', 'dd', 'li', 'td', 'th']:

            text = soup.text.replace('\xa0', '').strip()

            noises = ['각주', '외부 링크', '같이 보기', '위키미디어', '참고 문헌', '참고문헌', '관련 항목',
                      '출처가 분명하지 않습니다', '출처를 표기해 주세요', '삭제될 수도 있습니다',
                      '이 문단은 비어있습니다']

            if text == '' or text is None or any(noise in text for noise in noises):
                return last_position, None

#             if text == '' or text is None:
#                 return last_position, None

            last_position, position_ids = get_text_positions(
                text, html, last_position, is_eval)
            output = {'text': text, 'text_pos': position_ids,
                      'start': last_position, 'tag': tag_name}
            return last_position, [output]
        else:
            return last_position, None


def get_text_positions(text, html, last_position, is_eval=False):
    if is_eval:
        return 0, []

    position_ids = []
    for substr in text:
        last_position = get_str_position(substr, html, last_position, is_eval)
        position_ids.append(last_position)
    return last_position, position_ids


def get_str_position(substr, html, last_position, is_eval=False):
    if is_eval:
        return 0

    new_position = html.find(substr, last_position + 1)
    if new_position == -1:
        return last_position
    return new_position


def get_tag_position(soup, html, last_position, is_eval=False):
    if is_eval:
        return 0

    tag_name = soup.name
    tag_html = str(soup)

    count = html.count(tag_html)
    if count == 1:
        return html.index(tag_html)

    count = html.count(tag_html[:20])
    if count == 1:
        return html.index(tag_html[:20])

    count = html.count(tag_html[:10])
    if count == 1:
        return html.index(tag_html[:10])

    last_position = get_str_position(
        "<" + tag_name, html, last_position, is_eval)
    return last_position