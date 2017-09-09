from elasticsearch import Elasticsearch

INDEX_NAME = "aquaint"
DOC_TYPE = "doc"
FIELDS = ["title", "content"]


def main():
    # FIND AVG TITLE & CONTENT LENGTH
    # Initialize the scroll
    es = Elasticsearch()
    title_lengths = []
    content_lengths = []
    page = es.search(
        index=INDEX_NAME,
        doc_type=DOC_TYPE,
        scroll='2m',
        size=10000,
        body={
            "query": {
                "match_all": {}
            }
        }
    )
    sid = page['_scroll_id']
    scroll_size = page['hits']['total']
    # Start scrolling
    while scroll_size > 0:
        res = page['hits']['hits']
        for hit in res:
            title_lengths.append(len(hit['_source']['title']))
            content_lengths.append(len(hit['_source']['content']))
        print("Scrolling...")
        page = es.scroll(scroll_id=sid, scroll='2m')
        # Update the scroll ID
        sid = page['_scroll_id']
        # Get the number of results that we returned in the last scroll
        scroll_size = len(page['hits']['hits'])
        # Scroll 10000 at a time (max)
        print("scroll size: " + str(scroll_size))
    avg_title_len = sum(title_len for title_len in title_lengths) / len(title_lengths)
    avg_content_len = sum(content_len for content_len in content_lengths) / len(content_lengths)
    print("Average title length: {0:05.5f}.\nAverage content length: {1:05.5f}\n"
          "Number of titles: {2}\nNumber of contents: {3}".format(avg_title_len, avg_content_len, len(title_lengths), len(content_lengths)))

if __name__ == "__main__":
    main()
