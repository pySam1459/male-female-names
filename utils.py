import requests as r
from bs4 import BeautifulSoup


def load_data() -> tuple[list[str], list[str]]:
    mURL = "https://www.pampers.co.uk/pregnancy/baby-names/article/top-baby-names-for-boys"
    html = r.get(mURL)
    soup = BeautifulSoup(html.content, 'html.parser')
    names_htmls = soup.find_all("ol")
    boy_names_html = []
    for names_html in names_htmls:
        boy_names_html += names_html.find_all("p", {"class": "rich-text-text"})
    boy_names = [tag.text.strip("\t").lower() for tag in boy_names_html]

    fURL = "https://www.goodhousekeeping.com/life/parenting/a37668901/top-baby-girl-names/"
    html = r.get(fURL)
    soup = BeautifulSoup(html.content, 'html.parser')
    names_html = soup.find("ol", {"class": "css-1rk79nl et3p2gv0", "data-node-id": "34"})
    girl_names_html = names_html.find_all("li")
    girl_names = [tag.text.lower() for tag in girl_names_html]
    return boy_names, girl_names