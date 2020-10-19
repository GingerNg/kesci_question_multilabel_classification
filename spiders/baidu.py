from pyquery import PyQuery as pq
import requests
from common.mongo_utils import MongoDao
from common.hash_utils import get_md5
import datetime
import time
import random
from common.proxy_utils import get_proxy
from spiders.setting import Tasks

root_url = "http://www.baidu.com"

# Tasks = {}

Tasks["db_name"] = 'nlp_db'
Tasks["coll_name"] = 'baidu_html' + "_style"
# Tasks["labels"] = Styles
# Tasks["coll_name"] = 'baidu_html' + "_scene"
# Tasks["labels"] = Scenes

mongodao = MongoDao(url='mongodb://192.168.235.223:27017/',
                        db_name=Tasks["db_name"],
                        coll_name=Tasks["coll_name"])


# class Spider(object):
#     def __init__(self):
#         pass

#     def get_next_url():
#         pass

#     def fetch_first


def fetch_html(url, keyword):
    """[fetch and save]

    Args:
        url ([type]): [description]
        keyword ([type]): [description]
    """
    proxy = get_proxy().get("proxy")
    print(url)
    res = mongodao.search_one(filter={"url": url})
    if res and res["fetch_status"] == 1:  # 相同链接不重复爬取
        return
    # time.sleep(random.randint(1, 10))
    headers = {
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.90 Safari/537.36',
        'Cookie': 'BIDUPSID=68BB799DC03E5ABF5A9987DC33390F10; PSTM=1596071180; BAIDUID=68BB799DC03E5ABFB283D59D10702583:FG=1; BD_UPN=12314753; BDUSS=dzNE5aNjZmdVE3Sjd4LS02dlNYLWlST21ON2ZCbUJJaDdqTFd-cW85cWRPa3RmSVFBQUFBJCQAAAAAAAAAAAEAAADBLy4OamluamllNjAzODA5AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJ2tI1-drSNfYT; ispeed_lsm=2; BDORZ=B490B5EBF6F3CD402E515D22BCDA1598; MCITY=-%3A; BDSFRCVID=pbIOJeC62wIBsrQr-n8pKUhN4opIK4vTH6aoG9qcIWeTjazl1wQaEG0PHU8g0KubpgWlogKK3mOTH4-F_2uxOjjg8UtVJeC6EG0Ptf8g0M5; H_BDCLCKID_SF=tJ-J_IPKJD-3j5ruM-rV-JD0-fTBa4oXHD7yWCvMKqTcOR5Jj65byfDyKRQHe4CO-g3PBpu2a-JzOIbC3MA--t41eqQ2Blb-QIQKXJ4hbMjHsq0x0M7le-bQypoabPbUaIOMahkbal7xOK-zQlPK5JkgMx6MqpQJQeQ-5KQN3KJmfbL9bT3YjjISKx-_J5tOtn7P; BDRCVFR[feWj1Vr5u3D]=I67x6TjHwwYf0; delPer=0; BD_CK_SAM=1; PSINO=6; BDUSS_BFESS=dzNE5aNjZmdVE3Sjd4LS02dlNYLWlST21ON2ZCbUJJaDdqTFd-cW85cWRPa3RmSVFBQUFBJCQAAAAAAAAAAAEAAADBLy4OamluamllNjAzODA5AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJ2tI1-drSNfYT; H_PS_PSSID=1456_32534_32356_32327_31660_32352_32046_32398_32405_32115_31321_26350_32525_32481; sug=3; sugstore=0; ORIGIN=0; bdime=0; H_PS_645EC=e9062UvAlNBHvw%2FWtg5%2BglHqZldHyCmRxSY%2FLpKeNt61v0EAVLZAPok3a1w; WWW_ST=1597214743145'
    }
    resp = None
    try:
        resp = requests.get(url=url,
                            proxies={"http": "http://{}".format(proxy)},
                            headers=headers,
                            timeout=30
                            )
    except Exception as e:
        print(e)

    # resp = requests.get(url, headers)
    fetch_status = 0
    if resp:
        if "百度安全验证" in resp.content.decode("utf-8"):
            print("error")
            fetch_status = 0
        else:
            fetch_status = 1
        # time.sleep(10)
        # resp = requests.get(url)
    data = {
        "html": resp.content.decode("utf-8") if resp else "",
        "timestamp": str(datetime.datetime.now()),
        "url": url,
        "keyword": keyword,
        "_id": get_md5(url + "||"+str(datetime.date.today())),
        "fetch_status": fetch_status,
        "parse_status": 0
    }
    # open("res.html", "wb").write(resp.content)
    res = mongodao.search_one(filter={"_id": data["_id"]})
    if not res:
        mongodao.insert_one(data=data)
    else:
        if res["fetch_status"] == 0:
            mongodao.update_one(filter={"_id": data["_id"]}, data=data)
    return data


def get_next_url(html_doc):
    doc = pq(html_doc)
    next_url = root_url + \
        doc("div#wrapper_wrapper div#page div.page-inner a.n").attr('href')
    return next_url


if __name__ == "__main__":
    # keyword = "金融投资"
    keywords = Tasks["labels"]
    # url = "http://www.baidu.com/s?ie=utf-8&newi=1&mod=1&isbd=1&isid=a1aa77f300044e57&wd="+keyword+"&rsv_spt=1&rsv_iqid=0xd06bf0090003f3d2&issp=1&f=8&rsv_bp=1&rsv_idx=2&ie=utf-8&rqlang=cn&tn=baiduhome_pg&rsv_enter=0&rsv_dl=tb&rsv_t=498dag0E%2BL5k35XJ4ijilScFHLf08YPYU2IgFo%2F5jRtWABizwy9QfX1Wm8ei%2FiKt4Piq&oq=%25E9%2587%2591%25E8%259E%258D%25E6%258A%2595%25E8%25B5%2584&rsv_btype=t&rsv_pq=a1aa77f300044e57&bs=%E9%87%91%E8%9E%8D%E6%8A%95%E8%B5%84&rsv_sid=32292_1456_31669_32379_32356_32327_31660_32352_32046_32398_32405_32115_31321_26350_32525&_ss=1&clist=&hsug=&f4s=1&csor=4&_cr1=41229"
    for keyword in keywords:
        print(keyword)
        for p in range(250):
            url = "http://www.baidu.com/s?wd={}&pn={}&cl=3".format(keyword, p)
            data = fetch_html(url=url, keyword=keyword)
        # next_url = get_next_url(data["html"])
        # url = next_url
