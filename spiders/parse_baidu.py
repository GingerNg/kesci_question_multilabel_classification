from common.mongo_utils import MongoDao
from pyquery import PyQuery as pq
from common.hash_utils import get_md5
import datetime
from spiders.setting import Tasks


Tasks["db_name"] = 'nlp_db'
Tasks["coll_name"] = 'baidu_html' + "_scene"

Tasks["parsed_coll_name"] = 'baidu_text' + "_scene"

html_mongodao = MongoDao(url='mongodb://localhost:27017/',
                         db_name=Tasks["db_name"],
                         coll_name=Tasks["coll_name"])
text_mongodao = MongoDao(url='mongodb://localhost:27017/',
                         db_name=Tasks["db_name"],
                         coll_name=Tasks["parsed_coll_name"])


def parse(page):
    html_doc = page["html"]
    # keyword = page["keyword"]
    try:
        doc = pq(html_doc)
        for item in doc("div#content_left div.c-container").items():
            title = item("h3").text()
            # abstract = item("div.c-abstract").text()
            content = item("div.c-abstract").text()
            print(title)
            # print(content)
            data = {
                "parent_id": page["_id"],
                "content": content,
                "title": title,
                "_id": get_md5(title + "||" + page["timestamp"]),
                "keyword": page["keyword"],
                "timestamp": str(datetime.datetime.now())
            }
            res = text_mongodao.search_one(filter={"_id": data["_id"]})
            if not res:
                text_mongodao.insert_one(data=data)
        html_mongodao.update_one(filter={"_id": page["_id"]},
                                 data={"parse_status": 1})
    except Exception as e:
        print(e)
    # print(item)
    # print("-------")


if __name__ == "__main__":
    pages = html_mongodao.search(filter={})
    for page in pages:
        if page["parse_status"] == 0:
            parse(page)
