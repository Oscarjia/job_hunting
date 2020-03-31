import requests
from lxml import etree
import re
from urllib import parse
from .editDistance import edit_distance


class WebSpider:
    def __init__(self):
        self.bdbk_headers = {
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7,la;q=0.6',
            'Host': 'baike.baidu.com',
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36'
        }
        self.bdzd_headers = {
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7,la;q=0.6',
            'Host': 'zhidao.baidu.com',
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36'
        }

    @staticmethod
    def get_page(url, headers, encode):
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                response.encoding = encode
                return response.text
            else:
                print('Request failed', response.status_code)
        except requests.exceptions.ConnectionError as e:
            print('Error', e.args)

    @staticmethod
    def bdbk_extract(html):
        patterns = [r'[\n\xa0]', r'<div.*?>', r'<i>', r'<a.*?>', r'<sup.*?/sup>', r'</.*?>']
        for pattern in patterns:
            html = re.sub(pattern, '', html)
        return html

    @staticmethod
    def bdbk_parser(html):
        selector = etree.HTML(html)
        description = selector.xpath('//div[@class="lemma-summary"]/div')
        description = [etree.tostring(d, encoding='utf-8', method='html').decode('utf-8') for d in description]
        return description

    def search_bdbk(self, inputQ):
        html = self.get_page('https://baike.baidu.com/item/' + inputQ, headers=self.bdbk_headers, encode='utf-8')
        description = self.bdbk_parser(html)
        description = [self.bdbk_extract(d) for d in description]
        return description[0] if description else []

    @staticmethod
    def make_url(inputQ):
        base_url = 'https://zhidao.baidu.com/search?lm=0&rn=10&pn=0&fr=search&ie=gbk&word='
        inputQ_convert = parse.quote(inputQ, encoding='gb2312')
        return base_url + inputQ_convert

    @staticmethod
    def bdzd_parser(html):
        selector = etree.HTML(html)
        questions = selector.xpath('//div[@class="list-inner"]/div/dl/dt/a')
        links = selector.xpath('//div[@class="list-inner"]/div/dl/dt/a/@href')
        questions = [(q.xpath('string(.)'), link) for q, link in zip(questions, links)]
        return questions

    @staticmethod
    def inner_parser(html):
        selector = etree.HTML(html)
        best_answer = selector.xpath('//div[@class="best-text mb-10"]')
        best_answer = [a.xpath('string(.)') for a in best_answer]
        best_answer = ''.join(best_answer).strip()
        best_answer = re.sub(r'[\n\xa0]', '', best_answer)
        best_answer = re.sub(r'展开全部', '', best_answer)
        return best_answer

    def search_bdzd(self, inputQ):
        html = self.get_page(self.make_url(inputQ), headers=self.bdzd_headers, encode='gb2312')
        questions = self.bdzd_parser(html)
        if questions:
            best_answer = sorted([(q, link, edit_distance(inputQ, q)[0]) for (q, link) in questions], key=lambda x: x[2])[0]
            if best_answer[2] <= 2 and len(inputQ) >= best_answer[2] * 2:
                best_answer_html = self.get_page(best_answer[1], headers=self.bdzd_headers, encode='gb2312')
                best_answer = self.inner_parser(best_answer_html)
                return best_answer
        return []

    def searchAnswer(self, inputQ):
        bdbk_result = self.search_bdbk(inputQ)
        return (bdbk_result, 'From bdbk') if bdbk_result else (self.search_bdzd(inputQ), 'From bdzd')