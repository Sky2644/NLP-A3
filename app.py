import base64
import difflib
import html
import os
import re
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx
from gensim import downloader as api
from gensim.models import FastText, KeyedVectors, Word2Vec
from nltk import pos_tag, sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.data import find
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


st.set_page_config(page_title="文本表示与词向量模型实验系统", layout="wide")
if get_script_run_ctx() is None:
    raise SystemExit(
        "This is a Streamlit app. Run:\n"
        "/Users/bytedance/Desktop/A3/.venv/bin/streamlit run /Users/bytedance/Desktop/A3/app.py"
    )
st.session_state.setdefault("processed_text", "")
st.session_state.setdefault("processed_ready", False)
gensim_data_dir = Path(__file__).resolve().parent / ".gensim-data"
gensim_data_dir.mkdir(parents=True, exist_ok=True)
api.BASE_DIR = str(gensim_data_dir)


def ensure_nltk_resource(resource_path, download_name):
    try:
        find(resource_path)
        return True
    except LookupError:
        import nltk

        try:
            nltk.download(download_name, quiet=True)
            find(resource_path)
            return True
        except Exception:
            return False


HAVE_PUNKT = ensure_nltk_resource("tokenizers/punkt", "punkt")
HAVE_STOPWORDS = ensure_nltk_resource("corpora/stopwords", "stopwords")
HAVE_WORDNET = ensure_nltk_resource("corpora/wordnet", "wordnet")
HAVE_TAGGER = ensure_nltk_resource("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger")


SAMPLE_TEXT = (
    "In a coastal city the harbor welcomes ships every morning. The captain checks the map, "
    "the crew loads cargo, and the harbor master records arrivals. A scientist visits the "
    "city to study tidal energy. The scientist studies currents and designs turbines. In the "
    "laboratory the scientist measures power and writes reports. The city council reviews "
    "the reports and plans new projects. Meanwhile a journalist interviews the captain and "
    "the scientist, connecting their stories in a daily report. The story repeats when the "
    "ship returns at night. In another part of the world a mountain village watches a river "
    "cut through stone. The river feeds farms, and the farmers plant wheat and barley. The "
    "farmer watches clouds, predicts rain, and prepares the soil. The teacher in the village "
    "explains how water supports life. The children draw maps of the river and label fields, "
    "bridges, and forests. Each season the village celebrates the harvest. A traveler reads "
    "about this village and writes a letter describing the farms and mountains. The letter "
    "travels to the coastal city, where the journalist includes it in a report. The reader "
    "in the city imagines the mountains and the river. In a university the professor teaches "
    "about planets, stars, and galaxies. The student studies physics and observes the sky. "
    "The telescope captures light, and the researcher analyzes signals from distant stars. "
    "The astronomer explains that a planet orbits a star, and a moon orbits a planet. The "
    "researcher writes a paper, and the editor reviews the paper. The library stores the "
    "paper next to books on history and language. A linguist in the same university studies "
    "words, meanings, and context. The linguist notes that a word co-occurs with another "
    "word in similar contexts. The linguist compares synonyms like river and stream, city "
    "and town, professor and teacher, scientist and researcher. The linguist prepares a "
    "lesson and teaches students how word order changes meaning. On a quiet evening the "
    "student reads a novel about a captain, a scientist, and a journalist. The novel describes "
    "the harbor, the city council, and the mountain village. The student notices repeated "
    "phrases such as scientist studies, captain commands, farmer plants, and teacher explains. "
    "These phrases appear in different chapters but remain connected in the reader's mind. "
    "The story demonstrates how people, places, and actions co-occur in narrative text. It "
    "also shows how related ideas appear even when words do not directly co-occur. The city "
    "needs energy, the village needs water, and the university needs knowledge. These needs "
    "shape the lives of the captain, scientist, farmer, teacher, and student. Together their "
    "stories form a rich example for exploring statistical text models and semantic spaces."
)


def normalize_text(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z\s]", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
    return cleaned


def safe_sent_tokenize(text: str) -> List[str]:
    if HAVE_PUNKT:
        return sent_tokenize(text)
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def safe_word_tokenize(text: str) -> List[str]:
    if HAVE_PUNKT:
        return word_tokenize(text)
    return re.findall(r"[A-Za-z]+", text)


def preprocess_documents(
    text: str,
    remove_stopwords: bool = True,
    normalize_method: str = "无",
) -> Tuple[List[str], List[str], List[List[str]]]:
    sentences = safe_sent_tokenize(text)
    stop_words = set(stopwords.words("english")) if remove_stopwords and HAVE_STOPWORDS else set()
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer() if HAVE_WORDNET else None
    processed_docs = []
    tokenized_docs = []
    for sentence in sentences:
        cleaned = normalize_text(sentence)
        tokens = [t for t in safe_word_tokenize(cleaned) if t.isalpha()]
        if remove_stopwords:
            tokens = [t for t in tokens if t not in stop_words]
        if normalize_method == "词干提取":
            tokens = [stemmer.stem(t) for t in tokens]
        elif normalize_method == "词形还原" and lemmatizer:
            tokens = [lemmatizer.lemmatize(t) for t in tokens]
        tokenized_docs.append(tokens)
        processed_docs.append(" ".join(tokens))
    return sentences, processed_docs, tokenized_docs


def tokenize_for_models(text: str, remove_stopwords: bool = True) -> List[List[str]]:
    _, _, tokenized_docs = preprocess_documents(
        text, remove_stopwords=remove_stopwords, normalize_method="词形还原"
    )
    return [tokens for tokens in tokenized_docs if tokens]


def build_tfidf_dataframe(tfidf_matrix, terms):
    coo = tfidf_matrix.tocoo()
    data = pd.DataFrame(
        {"doc_id": coo.row, "term": [terms[i] for i in coo.col], "weight": coo.data}
    )
    return data.sort_values(["doc_id", "weight"], ascending=[True, False])


def extract_top_keywords(tfidf_matrix, terms, top_k=5):
    top_keywords = {}
    top_values = {}
    for doc_id in range(tfidf_matrix.shape[0]):
        row = tfidf_matrix.getrow(doc_id).toarray().ravel()
        if row.sum() == 0:
            top_keywords[doc_id] = []
            top_values[doc_id] = []
            continue
        top_indices = row.argsort()[-top_k:][::-1]
        keywords = [terms[i] for i in top_indices if row[i] > 0]
        values = [float(row[i]) for i in top_indices if row[i] > 0]
        top_keywords[doc_id] = keywords
        top_values[doc_id] = values
    return top_keywords, top_values


def highlight_keywords(sentences, keywords_by_doc, colors):
    highlighted = []
    for doc_id, sentence in enumerate(sentences):
        safe_text = html.escape(sentence)
        keywords = keywords_by_doc.get(doc_id, [])
        for kw in sorted(set(keywords), key=len, reverse=True):
            color = colors[doc_id % len(colors)]
            pattern = re.compile(rf"\b{re.escape(kw)}\b", flags=re.IGNORECASE)
            safe_text = pattern.sub(
                f'<span style="background-color: {color}; padding: 2px 4px; '
                f'border-radius: 4px;">{kw}</span>',
                safe_text,
            )
        highlighted.append(safe_text)
    return "<br/>".join(highlighted)


def build_wordcloud(freqs):
    try:
        from wordcloud import WordCloud

        wc = WordCloud(width=800, height=400, background_color="white")
        image = wc.generate_from_frequencies(freqs).to_image()
        return image
    except Exception:
        return None


def compute_term_groups_by_pos(terms):
    if not HAVE_TAGGER:
        return {term: "Other" for term in terms}
    tagged = pos_tag(terms)
    mapping = {}
    for term, tag in tagged:
        if tag.startswith("NN"):
            mapping[term] = "Noun"
        elif tag.startswith("VB"):
            mapping[term] = "Verb"
        elif tag.startswith("JJ"):
            mapping[term] = "Adj"
        elif tag.startswith("RB"):
            mapping[term] = "Adv"
        else:
            mapping[term] = "Other"
    return mapping


def compute_term_groups_by_doc(matrix, terms):
    dense = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
    doc_ids = dense.argmax(axis=0)
    return {term: f"Doc {doc_ids[i] + 1}" for i, term in enumerate(terms)}


def compute_cooccurrence_pairs(binary_matrix, terms, coords, top_n=20):
    cooc = binary_matrix.T @ binary_matrix
    np.fill_diagonal(cooc, 0)
    pairs = []
    for i in range(cooc.shape[0]):
        for j in range(i + 1, cooc.shape[1]):
            pairs.append((i, j, cooc[i, j]))
    pairs = sorted(pairs, key=lambda x: x[2], reverse=True)
    results = []
    for i, j, score in pairs[:top_n]:
        dist = np.linalg.norm(coords[i] - coords[j])
        results.append(
            {
                "term_a": terms[i],
                "term_b": terms[j],
                "cooccurrence": int(score),
                "distance": float(dist),
            }
        )
    return pd.DataFrame(results)


def compute_noncooccur_pairs(binary_matrix, terms, coords, top_n=20):
    cooc = binary_matrix.T @ binary_matrix
    np.fill_diagonal(cooc, 0)
    results = []
    for i in range(cooc.shape[0]):
        for j in range(i + 1, cooc.shape[1]):
            if cooc[i, j] == 0:
                dist = np.linalg.norm(coords[i] - coords[j])
                results.append((i, j, dist))
    results = sorted(results, key=lambda x: x[2])
    rows = []
    for i, j, dist in results[:top_n]:
        rows.append(
            {"term_a": terms[i], "term_b": terms[j], "cooccurrence": 0, "distance": dist}
        )
    return pd.DataFrame(rows)


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)) or 1.0
    return float(np.dot(vec_a, vec_b) / denom)


def preprocess_query(text: str, remove_stopwords: bool, normalize_method: str) -> str:
    cleaned = normalize_text(text)
    tokens = [t for t in safe_word_tokenize(cleaned) if t.isalpha()]
    if remove_stopwords:
        stop_words = set(stopwords.words("english")) if HAVE_STOPWORDS else set()
        tokens = [t for t in tokens if t not in stop_words]
    if normalize_method == "词干提取":
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens]
    elif normalize_method == "词形还原" and HAVE_WORDNET:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)


def compute_sparse_cosine(tfidf_matrix, query_vector):
    doc_norms = np.sqrt(tfidf_matrix.multiply(tfidf_matrix).sum(axis=1)).A1
    query_norm = float(np.sqrt(query_vector.multiply(query_vector).sum()))
    if query_norm == 0.0:
        return np.zeros(tfidf_matrix.shape[0])
    scores = (tfidf_matrix @ query_vector.T).toarray().ravel()
    return scores / (doc_norms * query_norm + 1e-12)


def image_to_data_url(file_bytes: bytes, mime_type: str) -> str:
    b64 = base64.b64encode(file_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


@st.cache_resource(show_spinner=False)
def train_word2vec(
    corpus: Tuple[Tuple[str, ...], ...],
    vector_size: int,
    window: int,
    min_count: int,
    sg: int,
    epochs: int,
):
    model = Word2Vec(
        sentences=list(corpus),
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        workers=1,
        seed=42,
        epochs=epochs,
    )
    return model


@st.cache_resource(show_spinner=False)
def train_fasttext(
    corpus: Tuple[Tuple[str, ...], ...],
    vector_size: int,
    window: int,
    min_count: int,
    min_n: int,
    max_n: int,
    epochs: int,
):
    model = FastText(
        sentences=list(corpus),
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        min_n=min_n,
        max_n=max_n,
        workers=1,
        seed=42,
        epochs=epochs,
    )
    return model


@st.cache_resource(show_spinner=False)
def build_offline_glove_25():
    words = [
        "man",
        "woman",
        "king",
        "queen",
        "boy",
        "girl",
        "prince",
        "princess",
        "paris",
        "france",
        "china",
        "beijing",
        "rome",
        "italy",
        "london",
        "england",
        "city",
        "town",
        "river",
        "stream",
        "scientist",
        "researcher",
    ]
    vecs = np.zeros((len(words), 25), dtype=np.float32)
    idx = {w: i for i, w in enumerate(words)}

    def set_dims(word, royalty=0.0, gender=0.0, geo=0.0, role=0.0):
        vecs[idx[word], 0] = royalty
        vecs[idx[word], 1] = gender
        vecs[idx[word], 2] = geo
        vecs[idx[word], 3] = role

    set_dims("man", gender=-1.0)
    set_dims("woman", gender=1.0)
    set_dims("boy", gender=-0.8)
    set_dims("girl", gender=0.8)
    set_dims("king", royalty=1.0, gender=-1.0)
    set_dims("queen", royalty=1.0, gender=1.0)
    set_dims("prince", royalty=0.8, gender=-1.0)
    set_dims("princess", royalty=0.8, gender=1.0)

    set_dims("france", geo=1.0)
    set_dims("paris", geo=1.0, role=0.3)
    set_dims("china", geo=2.0)
    set_dims("beijing", geo=2.0, role=0.3)
    set_dims("italy", geo=3.0)
    set_dims("rome", geo=3.0, role=0.3)
    set_dims("england", geo=4.0)
    set_dims("london", geo=4.0, role=0.3)

    set_dims("city", role=0.2)
    set_dims("town", role=0.18)
    set_dims("river", role=0.1)
    set_dims("stream", role=0.09)
    set_dims("scientist", role=0.6)
    set_dims("researcher", role=0.58)

    kv = KeyedVectors(vector_size=25)
    kv.add_vectors(words, vecs)
    kv.fill_norms()
    return kv


def load_keyedvectors_from_glove_text(file_bytes: bytes, encoding: str, limit: int):
    lines = file_bytes.decode(encoding, errors="ignore").splitlines()
    words = []
    vectors = []
    vector_size = None
    for line in lines:
        if not line.strip():
            continue
        parts = line.rstrip().split()
        if len(parts) < 3:
            continue
        word = parts[0]
        try:
            vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
        except ValueError:
            continue
        if vector_size is None:
            vector_size = vec.shape[0]
        if vec.shape[0] != vector_size:
            continue
        words.append(word)
        vectors.append(vec)
        if limit and len(words) >= limit:
            break
    if not words:
        raise ValueError("empty vectors")
    kv = KeyedVectors(vector_size=vector_size)
    kv.add_vectors(words, np.vstack(vectors))
    kv.fill_norms()
    return kv


@st.cache_resource(show_spinner=False)
def load_glove_model():
    return api.load("glove-twitter-25")


st.title("文本表示与词向量模型实验系统")
st.markdown("包含传统统计模型、Word2Vec、GloVe 与 FastText/Sent2Vec 四个模块。")

tabs = st.tabs(["传统统计模型", "Word2Vec 对比", "预训练 GloVe", "FastText 与 Sent2Vec"])


with tabs[0]:
    st.subheader("用户输入界面")
    col_left, col_right = st.columns([2, 1], gap="large")
    with col_left:
        text_input = st.text_area(
            "输入或粘贴英文文本",
            height=240,
            value=st.session_state.get("input_text", ""),
            placeholder="建议输入 500-1000 词的英文文本",
        )
        uploaded = st.file_uploader("上传 .txt 文件", type=["txt"], key="module1_upload")
        action_col1, action_col2 = st.columns(2)
        with action_col1:
            if st.button("加载示例文本"):
                st.session_state["input_text"] = SAMPLE_TEXT
                text_input = SAMPLE_TEXT
        with action_col2:
            if uploaded is not None:
                file_text = uploaded.read().decode("utf-8", errors="ignore")
                st.session_state["input_text"] = file_text
                text_input = file_text
        process_clicked = st.button("处理文本")
    with col_right:
        word_count = len(re.findall(r"\b\w+\b", text_input))
        st.metric("当前字数", word_count)
        st.info("点击处理文本后执行分句、向量化与可视化。")
        st.markdown("示例文本包含明显共现关系：scientist studies, captain commands, farmer plants。")

    if not text_input.strip():
        st.warning("请输入英文文本或上传 .txt 文件后再进行分析。")
        st.stop()

    st.subheader("文本预处理配置")
    pre_col1, pre_col2, pre_col3 = st.columns(3)
    with pre_col1:
        remove_stop = st.checkbox("移除停用词", value=True)
    with pre_col2:
        normalize_method = st.selectbox("规范化方法", ["无", "词干提取", "词形还原"])
    with pre_col3:
        min_df = st.slider("最小文档频率(min_df)", 1, 3, 1)

    if process_clicked:
        st.session_state["processed_text"] = text_input
        st.session_state["processed_ready"] = True

    if not st.session_state.get("processed_ready"):
        st.info("等待处理文本按钮触发计算。")
        st.stop()

    start_time = time.time()
    sentences, processed_docs, tokenized_docs = preprocess_documents(
        st.session_state["processed_text"],
        remove_stopwords=remove_stop,
        normalize_method=normalize_method,
    )
    st.caption(f"切分为 {len(sentences)} 个文档，预处理耗时 {time.time() - start_time:.2f} 秒。")

    if len(processed_docs) < 2:
        st.warning("文本过短，至少需要两个句子进行分析。")
        st.stop()

    st.subheader("TF-IDF 计算与展示")
    tfidf_vectorizer = TfidfVectorizer(
        tokenizer=str.split,
        preprocessor=None,
        lowercase=False,
        min_df=min_df,
    )
    try:
        tfidf_matrix = tfidf_vectorizer.fit_transform(processed_docs)
        terms = tfidf_vectorizer.get_feature_names_out()
    except ValueError:
        st.error("词表为空，请关闭停用词移除或增加文本长度。")
        st.stop()
    if tfidf_matrix.shape[1] == 0:
        st.warning("没有可用词语，请降低 min_df 或增加文本长度。")
        st.stop()
    tfidf_df = build_tfidf_dataframe(tfidf_matrix, terms)

    st.dataframe(
        tfidf_df.head(200).rename(
            columns={"doc_id": "文档编号", "term": "词语", "weight": "权重"}
        ),
        use_container_width=True,
        height=320,
    )

    st.subheader("文本相似度查询")
    query_col1, query_col2 = st.columns([3, 1])
    with query_col1:
        query_text = st.text_area(
            "输入查询文本",
            height=100,
            value="The scientist studies tidal energy.",
        )
    with query_col2:
        top_k = st.slider("返回结果数", 3, 10, 5)
        query_clicked = st.button("查询相似度")

    if query_clicked:
        processed_query = preprocess_query(query_text, remove_stop, normalize_method)
        if not processed_query.strip():
            st.error("查询文本无有效词语，请调整输入。")
        else:
            query_vec = tfidf_vectorizer.transform([processed_query])
            scores = compute_sparse_cosine(tfidf_matrix, query_vec)
            ranked = np.argsort(scores)[::-1][:top_k]
            result_rows = []
            for idx in ranked:
                result_rows.append(
                    {
                        "文档编号": f"Doc {idx + 1}",
                        "相似度": float(scores[idx]),
                        "原句": sentences[idx],
                    }
                )
            st.dataframe(pd.DataFrame(result_rows), use_container_width=True)

    top_keywords, top_values = extract_top_keywords(tfidf_matrix, terms, top_k=5)
    keyword_rows = []
    for doc_id, keywords in top_keywords.items():
        for idx, kw in enumerate(keywords):
            value = top_values[doc_id][idx] if idx < len(top_values[doc_id]) else 0.0
            keyword_rows.append(
                {"文档编号": f"Doc {doc_id + 1}", "关键词": kw, "TF-IDF": value}
            )
    st.dataframe(pd.DataFrame(keyword_rows), use_container_width=True)

    st.subheader("关键词高亮与词云")
    colors = [
        "#ffd166",
        "#a0c4ff",
        "#bdb2ff",
        "#caffbf",
        "#ffadad",
        "#fdffb6",
        "#9bf6ff",
        "#ffc6ff",
    ]
    highlighted_html = highlight_keywords(sentences, top_keywords, colors)
    st.markdown(
        f"<div style='line-height: 1.7; font-size: 16px;'>{highlighted_html}</div>",
        unsafe_allow_html=True,
    )

    freqs = {}
    for doc_id, keywords in top_keywords.items():
        row = tfidf_matrix.getrow(doc_id).toarray().ravel()
        for kw in keywords:
            idx = np.where(terms == kw)[0]
            if idx.size > 0:
                freqs[kw] = freqs.get(kw, 0) + float(row[idx[0]])

    wordcloud_image = build_wordcloud(freqs)
    if wordcloud_image is not None:
        st.image(wordcloud_image, caption="关键词词云", use_container_width=True)
    elif freqs:
        freq_df = (
            pd.DataFrame(list(freqs.items()), columns=["term", "weight"])
            .sort_values("weight", ascending=False)
            .head(30)
        )
        st.plotly_chart(
            px.bar(freq_df, x="term", y="weight", title="关键词权重（替代词云）"),
            use_container_width=True,
        )
    else:
        st.info("当前文本未提取到关键词。")

    st.subheader("LSA 降维与可视化")
    lsa_col1, lsa_col2, lsa_col3 = st.columns(3)
    with lsa_col1:
        matrix_type = st.selectbox("输入矩阵类型", ["TF-IDF", "词袋模型"])
    with lsa_col2:
        n_components = st.slider(
            "LSA 组件数量",
            min_value=2,
            max_value=min(50, max(2, len(terms) - 1)),
            value=4,
        )
    with lsa_col3:
        group_mode = st.selectbox("词语分组方式", ["词性", "文档来源"])

    if matrix_type == "词袋模型":
        count_vectorizer = CountVectorizer(
            tokenizer=str.split,
            preprocessor=None,
            lowercase=False,
            min_df=min_df,
        )
        try:
            matrix = count_vectorizer.fit_transform(processed_docs)
            lsa_terms = count_vectorizer.get_feature_names_out()
        except ValueError:
            st.error("词袋矩阵为空，请降低 min_df 或增加文本长度。")
            st.stop()
    else:
        matrix = tfidf_matrix
        lsa_terms = terms

    if matrix.shape[1] < 2:
        st.warning("词汇数量不足以进行 LSA 分析。")
        st.stop()

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    svd.fit(matrix)
    term_vectors = svd.components_.T
    coords = term_vectors[:, :2]
    variance_2d = float(svd.explained_variance_ratio_[:2].sum())
    variance_total = float(svd.explained_variance_ratio_.sum())

    st.caption(f"2D 解释方差 {variance_2d:.2f} | 总解释方差 {variance_total:.2f}")
    if variance_2d < 0.6:
        st.warning("2D 解释方差低于 0.60，可提高 min_df 或增加文本长度。")

    if group_mode == "词性":
        group_map = compute_term_groups_by_pos(list(lsa_terms))
    else:
        group_map = compute_term_groups_by_doc(matrix, list(lsa_terms))

    lsa_df = pd.DataFrame(
        {
            "term": lsa_terms,
            "x": coords[:, 0],
            "y": coords[:, 1],
            "group": [group_map[t] for t in lsa_terms],
        }
    )

    fig = px.scatter(
        lsa_df,
        x="x",
        y="y",
        color="group",
        hover_data=["term"],
        title="LSA 2D 语义空间",
    )
    fig.update_layout(
        xaxis_title=f"Component 1 ({svd.explained_variance_ratio_[0]:.2f})",
        yaxis_title=f"Component 2 ({svd.explained_variance_ratio_[1]:.2f})",
        height=520,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("观察任务支持")
    st.markdown("观察共现频率高的词语对是否靠近，以及语义相关但不直接共现的词对。")
    top_terms_limit = min(120, len(lsa_terms))
    binary_matrix = (matrix[:, :top_terms_limit] > 0).astype(int).toarray()
    coords_top = coords[:top_terms_limit]
    cooc_df = compute_cooccurrence_pairs(
        binary_matrix, lsa_terms[:top_terms_limit], coords_top
    )
    noncooc_df = compute_noncooccur_pairs(
        binary_matrix, lsa_terms[:top_terms_limit], coords_top
    )
    obs_col1, obs_col2 = st.columns(2)
    with obs_col1:
        st.markdown("**高同现词语对**")
        st.dataframe(cooc_df, use_container_width=True, height=260)
    with obs_col2:
        st.markdown("**零共现但距离接近的词语对**")
        st.dataframe(noncooc_df, use_container_width=True, height=260)


with tabs[1]:
    st.subheader("Word2Vec 训练与对比")
    data_source = st.radio("数据来源", ["使用模块1语料", "上传新语料"], horizontal=True)
    if data_source == "上传新语料":
        uploaded_w2v = st.file_uploader("上传 .txt 文件", type=["txt"], key="w2v_upload")
        w2v_text = uploaded_w2v.read().decode("utf-8", errors="ignore") if uploaded_w2v else ""
    else:
        w2v_text = st.session_state.get(
            "processed_text", st.session_state.get("input_text", "")
        )

    if not w2v_text.strip():
        st.warning("请提供语料后再训练 Word2Vec 模型。")
        st.stop()

    w2v_tokens = tokenize_for_models(w2v_text, remove_stopwords=True)
    if len(w2v_tokens) < 2:
        st.warning("语料过短，无法训练 Word2Vec。")
        st.stop()

    w2v_col1, w2v_col2, w2v_col3, w2v_col4 = st.columns(4)
    with w2v_col1:
        window = st.slider("window", 2, 10, 5)
    with w2v_col2:
        vector_size = st.slider("vector_size", 50, 300, 100, step=50)
    with w2v_col3:
        min_count = st.slider("min_count", 1, 5, 1)
    with w2v_col4:
        epochs = st.slider("epochs", 5, 20, 10)

    arch_mode = st.radio("架构选择", ["CBOW", "Skip-Gram"], horizontal=True)
    sg_value = 0 if arch_mode == "CBOW" else 1

    train_clicked = st.button("训练模型")
    if train_clicked:
        progress = st.progress(0)
        start = time.time()
        corpus_tuple = tuple(tuple(tokens) for tokens in w2v_tokens)
        with st.spinner("训练中，请稍候..."):
            model = train_word2vec(
                corpus_tuple, vector_size, window, min_count, sg_value, epochs
            )
        progress.progress(100)
        st.session_state["word2vec_model"] = model
        st.success(
            f"训练完成 | 词汇表大小 {len(model.wv.index_to_key)} | 用时 {time.time() - start:.2f} 秒"
        )

    model = st.session_state.get("word2vec_model")
    if model:
        st.markdown("**相似词查询**")
        vocab = list(model.wv.index_to_key)
        query_mode = st.radio(
            "输入方式",
            ["从词表选择", "手动输入"],
            horizontal=True,
            key="w2v_query_mode",
        )
        query_col1, query_col2 = st.columns([2, 1])
        with query_col1:
            if query_mode == "从词表选择":
                default_word = "scientist" if "scientist" in model.wv else vocab[0]
                default_index = vocab.index(default_word) if default_word in vocab else 0
                query_word = st.selectbox("选择单词", options=vocab, index=default_index)
            else:
                query_word = st.text_input("输入单词", value="scientist")
        with query_col2:
            query_clicked = st.button("查询相似词", key="w2v_query_btn")
        if query_clicked:
            if query_word in model.wv:
                similar = model.wv.most_similar(query_word, topn=5)
                st.dataframe(pd.DataFrame(similar, columns=["word", "similarity"]))
            else:
                st.error("该词不在词汇表中。Word2Vec 无法为未登录词(OOV)计算相似度。")
                suggestions = difflib.get_close_matches(query_word, vocab, n=5, cutoff=0.6)
                if suggestions:
                    st.info(f"可尝试：{', '.join(suggestions)}")

    st.markdown("**CBOW vs Skip-Gram 对比**")
    preset_words = ["scientist", "captain", "farmer", "teacher", "city"]
    target_word = st.selectbox("选择目标词", preset_words)
    compare_clicked = st.button("训练并对比")
    if compare_clicked:
        progress = st.progress(0)
        corpus_tuple = tuple(tuple(tokens) for tokens in w2v_tokens)
        with st.spinner("训练 CBOW 模型"):
            cbow_model = train_word2vec(
                corpus_tuple, vector_size, window, min_count, 0, epochs
            )
        progress.progress(50)
        with st.spinner("训练 Skip-Gram 模型"):
            sg_model = train_word2vec(
                corpus_tuple, vector_size, window, min_count, 1, epochs
            )
        progress.progress(100)
        if target_word in cbow_model.wv and target_word in sg_model.wv:
            cbow_sim = cbow_model.wv.most_similar(target_word, topn=5)
            sg_sim = sg_model.wv.most_similar(target_word, topn=5)
            compare_df = pd.DataFrame(
                {
                    "CBOW": [f"{w} ({s:.2f})" for w, s in cbow_sim],
                    "Skip-Gram": [f"{w} ({s:.2f})" for w, s in sg_sim],
                }
            )
            st.dataframe(compare_df, use_container_width=True)
        else:
            st.error("目标词未进入词汇表，请调整语料或参数。")


with tabs[2]:
    st.subheader("预训练 GloVe 模型")
    load_mode = st.radio(
        "加载方式",
        ["在线加载 glove-twitter-25", "离线示例向量(无需网络)", "上传 GloVe 文本向量(.txt/.vec)"],
        horizontal=True,
    )
    if load_mode == "在线加载 glove-twitter-25":
        load_clicked = st.button("加载 glove-twitter-25 模型", key="glove_load_online")
        if load_clicked:
            try:
                with st.spinner("加载模型中，请稍候..."):
                    glove_model = load_glove_model()
                st.session_state["glove_model"] = glove_model
            except Exception as exc:
                st.error(f"加载失败: {exc}")
                st.info(
                    f"当前 gensim 缓存目录: {gensim_data_dir}\n"
                    "若无网络可改用离线示例向量或上传本地 GloVe 文本向量。"
                )
    elif load_mode == "离线示例向量(无需网络)":
        offline_clicked = st.button("启用离线示例向量", key="glove_load_offline")
        if offline_clicked:
            st.session_state["glove_model"] = build_offline_glove_25()
            st.warning("已启用离线示例向量（小词表，用于演示类比与相似度，非 glove-twitter-25 原模型）。")
    else:
        uploaded_vec = st.file_uploader(
            "上传 GloVe 文本向量文件（每行：word val1 val2 ...）",
            type=["txt", "vec"],
            key="glove_upload_text",
        )
        up_col1, up_col2, up_col3 = st.columns(3)
        with up_col1:
            encoding = st.selectbox("编码", ["utf-8", "latin-1"], index=0)
        with up_col2:
            limit = st.number_input("最多加载词数", min_value=1000, max_value=200000, value=50000, step=1000)
        with up_col3:
            parse_clicked = st.button("解析并加载", key="glove_parse_text")
        if parse_clicked:
            if uploaded_vec is None:
                st.error("请先上传向量文件。")
            else:
                try:
                    with st.spinner("解析中，请稍候..."):
                        kv = load_keyedvectors_from_glove_text(
                            uploaded_vec.getvalue(), encoding=encoding, limit=int(limit)
                        )
                    st.session_state["glove_model"] = kv
                    st.success(
                        f"加载完成 | 词汇量 {len(kv.key_to_index)} | 向量维度 {kv.vector_size}"
                    )
                except Exception as exc:
                    st.error(f"解析失败: {exc}")

    glove_model = st.session_state.get("glove_model")
    if glove_model:
        st.success(
            f"模型已加载 | 词汇量 {len(glove_model.key_to_index)} | 向量维度 {glove_model.vector_size}"
        )
        if load_mode == "在线加载 glove-twitter-25":
            st.caption("语料简介：Twitter 语料，适合短文本语义分析。")
        elif load_mode == "离线示例向量(无需网络)":
            st.caption("离线示例向量：小词表，仅用于演示类比与相似度交互。")
        else:
            st.caption("自定义上传向量：请确保为英文词向量文本格式。")

        st.markdown("**词类比计算器**")
        if "analogy_inputs" not in st.session_state:
            st.session_state["analogy_inputs"] = {"a": "king", "b": "man", "c": "woman"}
        input_col1, input_col2, input_col3, input_col4 = st.columns(4)
        with input_col1:
            word_a = st.text_input("A", st.session_state["analogy_inputs"]["a"])
        with input_col2:
            word_b = st.text_input("B", st.session_state["analogy_inputs"]["b"])
        with input_col3:
            word_c = st.text_input("C", st.session_state["analogy_inputs"]["c"])
        with input_col4:
            swap_clicked = st.button("交换 A/B")
            clear_clicked = st.button("清除输入")
        if swap_clicked:
            word_a, word_b = word_b, word_a
        if clear_clicked:
            word_a, word_b, word_c = "", "", ""
        st.session_state["analogy_inputs"] = {"a": word_a, "b": word_b, "c": word_c}

        analogy_clicked = st.button("计算类比")
        if analogy_clicked:
            if all([word_a, word_b, word_c]):
                missing = [w for w in [word_a, word_b, word_c] if w not in glove_model]
                if missing:
                    st.error(f"未登录词: {', '.join(missing)}")
                else:
                    results = glove_model.most_similar(
                        positive=[word_a, word_c], negative=[word_b], topn=5
                    )
                    st.dataframe(
                        pd.DataFrame(results, columns=["word", "similarity"]),
                        use_container_width=True,
                    )
                    if st.button("保存类比结果"):
                        st.session_state.setdefault("analogy_records", []).append(
                            {
                                "A": word_a,
                                "B": word_b,
                                "C": word_c,
                                "Top1": results[0][0],
                            }
                        )
            else:
                st.error("请填写 A、B、C 三个单词。")

        st.markdown("**经典示例**")
        example_col1, example_col2 = st.columns(2)
        with example_col1:
            if st.button("king - man + woman"):
                st.session_state["analogy_inputs"] = {
                    "a": "king",
                    "b": "man",
                    "c": "woman",
                }
        with example_col2:
            if st.button("paris - france + china"):
                st.session_state["analogy_inputs"] = {
                    "a": "paris",
                    "b": "france",
                    "c": "china",
                }

        if st.session_state.get("analogy_records"):
            st.markdown("**保存的类比结果**")
            st.dataframe(pd.DataFrame(st.session_state["analogy_records"]))

        st.markdown("**词义相似度计算**")
        sim_col1, sim_col2, sim_col3 = st.columns([2, 2, 1])
        with sim_col1:
            sim_word_a = st.text_input("词语 1", value="city")
        with sim_col2:
            sim_word_b = st.text_input("词语 2", value="town")
        with sim_col3:
            sim_clicked = st.button("计算相似度")
        if sim_clicked:
            if sim_word_a in glove_model and sim_word_b in glove_model:
                score = glove_model.similarity(sim_word_a, sim_word_b)
                st.metric("余弦相似度", f"{score:.3f}")
            else:
                missing = [w for w in [sim_word_a, sim_word_b] if w not in glove_model]
                st.error(f"未登录词: {', '.join(missing)}")


with tabs[3]:
    st.subheader("FastText 与 Sent2Vec")
    source_option = st.radio(
        "语料来源", ["使用模块1语料", "上传新语料"], horizontal=True, key="ft_source"
    )
    if source_option == "上传新语料":
        uploaded_ft = st.file_uploader("上传 .txt 文件", type=["txt"], key="ft_upload")
        fasttext_text = (
            uploaded_ft.read().decode("utf-8", errors="ignore") if uploaded_ft else ""
        )
    else:
        fasttext_text = st.session_state.get(
            "processed_text", st.session_state.get("input_text", "")
        )

    if not fasttext_text.strip():
        st.warning("请提供语料后再训练 FastText。")
        st.stop()

    ft_tokens = tokenize_for_models(fasttext_text, remove_stopwords=True)
    if len(ft_tokens) < 2:
        st.warning("语料过短，无法训练 FastText。")
        st.stop()

    ft_col1, ft_col2, ft_col3, ft_col4, ft_col5 = st.columns(5)
    with ft_col1:
        ft_vector_size = st.slider("vector_size", 50, 300, 100, step=50, key="ft_vec")
    with ft_col2:
        ft_window = st.slider("window", 2, 10, 5, key="ft_window")
    with ft_col3:
        ft_min_count = st.slider("min_count", 1, 5, 1, key="ft_min_count")
    with ft_col4:
        ft_min_n = st.slider("min_n", 2, 5, 3, key="ft_min_n")
    with ft_col5:
        ft_max_n = st.slider("max_n", 3, 6, 5, key="ft_max_n")
    ft_epochs = st.slider("epochs", 5, 20, 10, key="ft_epochs")

    ft_train_clicked = st.button("训练 FastText")
    if ft_train_clicked:
        progress = st.progress(0)
        start = time.time()
        corpus_tuple = tuple(tuple(tokens) for tokens in ft_tokens)
        with st.spinner("训练中，请稍候..."):
            ft_model = train_fasttext(
                corpus_tuple,
                ft_vector_size,
                ft_window,
                ft_min_count,
                ft_min_n,
                ft_max_n,
                ft_epochs,
            )
        progress.progress(100)
        st.session_state["fasttext_model"] = ft_model
        st.success(
            f"训练完成 | 词汇表大小 {len(ft_model.wv.index_to_key)} | 用时 {time.time() - start:.2f} 秒"
        )

    ft_model = st.session_state.get("fasttext_model")
    if ft_model:
        st.markdown("**OOV 测试**")
        oov_col1, oov_col2 = st.columns([2, 1])
        with oov_col1:
            oov_word = st.text_input("输入 OOV 单词", value="computeer")
        with oov_col2:
            oov_clicked = st.button("测试 OOV")
        if oov_clicked:
            word2vec_model = st.session_state.get("word2vec_model")
            if word2vec_model:
                try:
                    _ = word2vec_model.wv[oov_word]
                    st.info("Word2Vec 找到该词。")
                except KeyError:
                    st.warning(
                        "Word2Vec 未登录词(OOV)：该词未在训练词表中，这是预期现象。FastText 会利用子词信息生成向量。"
                    )
            else:
                st.info("尚未训练 Word2Vec，无法比较。")
            ft_similar = ft_model.wv.most_similar(oov_word, topn=5)
            st.dataframe(pd.DataFrame(ft_similar, columns=["word", "similarity"]))

        st.markdown("**Sent2Vec 平均池化**")
        example_pairs = {
            "相似语义": (
                "The scientist studies tidal energy.",
                "The researcher analyzes power from the sea.",
            ),
            "不相似语义": (
                "The farmer plants wheat in spring.",
                "The telescope captures distant stars.",
            ),
        }
        selected_pair = st.selectbox("选择示例句对", list(example_pairs.keys()))
        if st.button("加载示例句对"):
            sent_a_default, sent_b_default = example_pairs[selected_pair]
            st.session_state["sent_a"] = sent_a_default
            st.session_state["sent_b"] = sent_b_default

        sent_col1, sent_col2 = st.columns(2)
        with sent_col1:
            sent_a = st.text_area(
                "句子 A",
                height=120,
                value=st.session_state.get("sent_a", "The scientist studies tidal energy."),
            )
        with sent_col2:
            sent_b = st.text_area(
                "句子 B",
                height=120,
                value=st.session_state.get("sent_b", "The researcher analyzes power from the sea."),
            )
        if st.button("计算句向量相似度"):
            tokens_a = tokenize_for_models(sent_a, remove_stopwords=True)
            tokens_b = tokenize_for_models(sent_b, remove_stopwords=True)
            flat_a = [t for sub in tokens_a for t in sub]
            flat_b = [t for sub in tokens_b for t in sub]
            if not flat_a or not flat_b:
                st.error("句子过短或无有效词语。")
            else:
                vec_a = np.mean([ft_model.wv[t] for t in flat_a], axis=0)
                vec_b = np.mean([ft_model.wv[t] for t in flat_b], axis=0)
                score = cosine_similarity(vec_a, vec_b)
                st.metric("句向量余弦相似度", f"{score:.3f}")
                st.caption(f"句子 A 词数 {len(flat_a)} | 句子 B 词数 {len(flat_b)}")


st.markdown("---")
st.header("导出页面与报告生成")
st.markdown("上传截图并填写实验 prompt 与总结后生成报告与展示页面。")

prompt_text = st.text_area("实验 prompt", height=140, value=st.session_state.get("prompt_text", ""))
summary_text = st.text_area("实验总结与分析", height=160, value=st.session_state.get("summary_text", ""))
st.session_state["prompt_text"] = prompt_text
st.session_state["summary_text"] = summary_text

st.subheader("运行截图上传")
shot_col1, shot_col2 = st.columns(2)
with shot_col1:
    shot_tfidf = st.file_uploader(
        "TF-IDF/LSA 文本相似度查询界面截图", type=["png", "jpg", "jpeg"], key="shot_tfidf"
    )
    shot_w2v = st.file_uploader(
        "Word2Vec 相似度查询界面截图", type=["png", "jpg", "jpeg"], key="shot_w2v"
    )
with shot_col2:
    shot_glove = st.file_uploader(
        "GloVe 词类比界面截图", type=["png", "jpg", "jpeg"], key="shot_glove"
    )
    shot_fasttext = st.file_uploader(
        "FastText OOV 界面截图", type=["png", "jpg", "jpeg"], key="shot_fasttext"
    )


def render_image_block(uploaded, title):
    if uploaded is None:
        return f"<h3>{title}</h3><p>未上传截图</p>"
    data_url = image_to_data_url(uploaded.getvalue(), uploaded.type or "image/png")
    return f"<h3>{title}</h3><img src='{data_url}' style='max-width:100%;'/>"


report_html = f"""
<html>
<head>
<meta charset="utf-8"/>
<title>实验报告</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; line-height: 1.6; }}
h1 {{ margin-bottom: 8px; }}
section {{ margin-bottom: 24px; }}
</style>
</head>
<body>
<h1>文本表示与词向量模型实验报告</h1>
<section>
<h2>实验 Prompt</h2>
<p>{html.escape(prompt_text).replace(chr(10), "<br/>")}</p>
</section>
<section>
<h2>系统运行截图</h2>
{render_image_block(shot_tfidf, "TF-IDF/LSA 文本相似度查询")}
{render_image_block(shot_w2v, "Word2Vec 相似度查询")}
{render_image_block(shot_glove, "GloVe 词类比")}
{render_image_block(shot_fasttext, "FastText OOV")}
</section>
<section>
<h2>实验总结与分析</h2>
<p>{html.escape(summary_text).replace(chr(10), "<br/>")}</p>
</section>
</body>
</html>
"""

showcase_html = f"""
<html>
<head>
<meta charset="utf-8"/>
<title>功能展示页面</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; line-height: 1.6; }}
section {{ margin-bottom: 24px; }}
</style>
</head>
<body>
<h1>功能展示页面</h1>
<section>{render_image_block(shot_tfidf, "TF-IDF/LSA 文本相似度查询")}</section>
<section>{render_image_block(shot_w2v, "Word2Vec 相似度查询")}</section>
<section>{render_image_block(shot_glove, "GloVe 词类比")}</section>
<section>{render_image_block(shot_fasttext, "FastText OOV")}</section>
</body>
</html>
"""

export_col1, export_col2 = st.columns(2)
with export_col1:
    st.download_button(
        "下载实验报告 HTML",
        data=report_html.encode("utf-8"),
        file_name="experiment_report.html",
        mime="text/html",
    )
with export_col2:
    st.download_button(
        "下载功能展示页面 HTML",
        data=showcase_html.encode("utf-8"),
        file_name="showcase.html",
        mime="text/html",
    )

st.info("PDF 导出流程：下载实验报告 HTML 后，用浏览器打开并选择“打印/导出为 PDF”。")
