import streamlit as st
import numpy as np
import pandas as pd
import joblib
import re
import nltk
import requests
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURASI HALAMAN
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Financial Distress Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ─────────────────────────────────────────────────────────────────────────────
# COLORBLIND-SAFE PALETTE
# Palette yang aman untuk Deuteranopia, Protanopia, Tritanopia
# Berdasarkan ColorBrewer dan Okabe-Ito palette
# ─────────────────────────────────────────────────────────────────────────────
COLORBLIND_PALETTE = {
    "positive": "#0173B2",      # Biru (aman semua)
    "negative": "#DE8F05",      # Oranye (aman semua)
    "neutral": "#CC78BC",       # Ungu muda (aman semua)
    "accent": "#029E73",        # Hijau (aman Deuteranopia)
    "dark": "#CA9161",          # Coklat (aman semua)
    "light_gray": "#ECE2F0",    # Abu-abu terang
    "dark_gray": "#56B4E9",     # Biru terang
}


CSS_COLORBLIND = f"""
<style>
    .main-header {{
        background: linear-gradient(135deg, {COLORBLIND_PALETTE['positive']} 0%, {COLORBLIND_PALETTE['dark']} 100%);
        padding: 1.5rem 2rem; border-radius: 12px;
        margin-bottom: 1.5rem; color: white;
    }}
    .main-header h1 {{ margin: 0; font-size: 1.8rem; font-weight: 700; }}
    .main-header p  {{ margin: 0.3rem 0 0; font-size: 0.95rem; opacity: 0.85; }}


    .section-card {{
        background: #ffffff; border: 1px solid #e0e0e0;
        border-radius: 10px; padding: 1.2rem 1.5rem;
        margin-bottom: 1rem; box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    }}
    .section-title {{
        font-size: 1rem; font-weight: 700; color: {COLORBLIND_PALETTE['positive']};
        border-left: 4px solid {COLORBLIND_PALETTE['positive']}; padding-left: 0.6rem;
        margin-bottom: 0.9rem;
    }}


    .badge-ok   {{ background: #E8F5E9; color: #2e7d32; border: 1px solid #a5d6a7;
                  border-radius: 6px; padding: 0.35rem 0.75rem; font-size: 0.82rem;
                  display: inline-block; margin-bottom: 0.8rem; }}
    .badge-warn {{ background: #fff8e1; color: #f57f17; border: 1px solid #ffe082;
                  border-radius: 6px; padding: 0.35rem 0.75rem; font-size: 0.82rem;
                  display: inline-block; margin-bottom: 0.8rem; }}
    .badge-info {{ background: #e3f2fd; color: #1565c0; border: 1px solid #90caf9;
                  border-radius: 6px; padding: 0.35rem 0.75rem; font-size: 0.82rem;
                  display: inline-block; margin-bottom: 0.8rem; }}


    .result-box {{ border-radius: 10px; padding: 1rem 1.2rem; margin: 0.4rem 0; font-size: 0.94rem; }}
    .r-good    {{ background: #e8f5e9; border-left: 5px solid #43a047; color: #2e7d32; }}
    .r-bad     {{ background: #fff3e0; border-left: 5px solid #f57c00; color: #e65100; }}
    .r-invest  {{ background: #e3f2fd; border-left: 5px solid {COLORBLIND_PALETTE['positive']}; color: #0d47a1; }}
    .r-risk    {{ background: #fff9c4; border-left: 5px solid #f9a825; color: #f57f17; }}
    .r-no      {{ background: #ffebee; border-left: 5px solid #e53935; color: #b71c1c; }}
    .r-neutral {{ background: #f3e5f5; border-left: 5px solid {COLORBLIND_PALETTE['neutral']}; color: #6a1b9a; }}


    .metric-mini {{ background: #f5f7fa; border-radius: 8px; padding: 0.6rem 0.8rem;
                   text-align: center; border: 1px solid #e0e0e0; border-left: 4px solid {COLORBLIND_PALETTE['positive']}; }}
    .metric-mini .val {{ font-size: 1.05rem; font-weight: 700; color: {COLORBLIND_PALETTE['positive']}; }}
    .metric-mini .lbl {{ font-size: 0.72rem; color: #666; margin-top: 2px; }}


    .news-card {{
        background: #ffffff; border: 1px solid #e0e0e0; border-left: 4px solid {COLORBLIND_PALETTE['positive']};
        border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 0.7rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }}
    .news-meta {{ font-size: 0.76rem; color: #888; margin-bottom: 0.35rem; }}
    .news-title {{ font-size: 0.95rem; font-weight: 700; color: #1a1a2e; margin-bottom: 0.4rem; line-height: 1.4; }}
    .news-body {{ font-size: 0.85rem; color: #444; line-height: 1.55; margin-bottom: 0.6rem; }}
    .news-link {{
        display: inline-flex; align-items: center; gap: 0.3rem;
        background: {COLORBLIND_PALETTE['positive']}; color: white;
        padding: 0.35rem 0.9rem; border-radius: 6px; text-decoration: none;
        font-size: 0.82rem; font-weight: 600;
    }}


    .dataframe {{ font-size: 12px; }}
    footer {{ visibility: hidden; }}
</style>
"""


st.markdown(CSS_COLORBLIND, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# KONSTANTA & DATA
# ─────────────────────────────────────────────────────────────────────────────
YEAR_MAPPING = {2019: 0, 2020: 1, 2021: 2, 2022: 3, 2023: 4, 2024: 5}


NAME_LIST = [
    "Abadi Nusantara Hijau Investam", "Ace Oldfields Tbk.", "Adhi Kartiko Pratama Tbk.",
    "Agro Bahari Nusantara Tbk.", "Agro Yasa Lestari Tbk.", "Agung Menjangan Mas Tbk.",
    "Akasha Wira International Tbk.", "Alakasa Industrindo Tbk", "Alkindo Naratama Tbk.",
    "Alumindo Light Metal Industry", "Aman Agrindo Tbk.", "Andira Agro Tbk.",
    "Aneka Tambang Tbk.", "Argha Karya Prima Industry Tbk", "Arita Prima Indonesia Tbk.",
    "Arkha Jayanti Persada Tbk.", "Arwana Citramulia Tbk.", "Asahimas Flat Glass Tbk.",
    "Asia Sejahtera Mina Tbk.", "Asiaplast Industries Tbk.", "Astra Agro Lestari Tbk.",
    "Astra Graphia Tbk.", "Astra International Tbk.", "Ateliers Mecaniques D Indonesi",
    "Avia Avian Tbk.", "BISI International Tbk.", "Bakrie & Brothers Tbk",
    "Bakrie Sumatera Plantations Tb", "Benteng Api Technic Tbk.", "Berkah Beton Sadaya Tbk.",
    "Berkah Prima Perkasa Tbk.", "Berlina Tbk.", "Betonjaya Manunggal Tbk.",
    "Bintang Mitra Semestaraya Tbk", "Budi Starch & Sweetener Tbk.", "Bumi Teknokultura Unggul Tbk",
    "Buyung Poetra Sembada Tbk.", "Cahayaputra Asa Keramik Tbk.", "Campina Ice Cream Industry Tbk",
    "Cemindo Gemilang Tbk.", "Central Omega Resources Tbk.", "Central Proteina Prima Tbk.",
    "Cerestar Indonesia Tbk.", "Champion Pacific Indonesia Tbk", "Charoen Pokphand Indonesia Tbk",
    "Chemstar Indonesia Tbk.", "Cilacap Samudera Fishing Indus", "Cisadane Sawit Raya Tbk.",
    "Cisarua Mountain Dairy Tbk.", "Cita Mineral Investindo Tbk.", "Citatah Tbk.",
    "Citra Borneo Utama Tbk.", "Colorpak Indonesia Tbk.", "Communication Cable Systems In",
    "Cottonindo Arista Tbk.", "DFI Retail Nusantara Tbk.", "Darmi Bersaudara Tbk.",
    "Delta Djakarta Tbk.", "Dewi Shri Farmindo Tbk.", "Dharma Samudera Fishing Indust",
    "Dharma Satya Nusantara Tbk.", "Diamond Food Indonesia Tbk.", "Dosni Roha Indonesia Tbk.",
    "Dua Putra Utama Makmur Tbk.", "Duta Intidaya Tbk.", "Duta Pertiwi Nusantara Tbk.",
    "Dyandra Media International Tb", "Eagle High Plantations Tbk.", "Ecocare Indo Pasifik Tbk.",
    "Ekadharma International Tbk.", "Emdeki Utama Tbk.", "Enseval Putera Megatrading Tbk",
    "Era Mandiri Cemerlang Tbk.", "Esta Indonesia Tbk.", "Estee Gold Feet Tbk.",
    "Estika Tata Tiara Tbk.", "Eterindo Wahanatama Tbk", "FAP Agri Tbk.",
    "FKS Food Sejahtera Tbk.", "Fajar Surya Wisesa Tbk.", "Falmaco Nonwoven Industri Tbk.",
    "Formosa Ingredient Factory Tbk", "Garudafood Putra Putri Jaya Tb", "Geoprima Solusi Tbk.",
    "Golden Plantation Tbk.", "Gozco Plantations Tbk.", "Graha Prima Mentari Tbk.",
    "Green Power Group Tbk.", "Gudang Garam Tbk.", "Gunanusa Eramandiri Tbk.",
    "Gunawan Dianjaya Steel Tbk.", "H.M. Sampoerna Tbk.", "HK Metals Utama Tbk.",
    "Harapan Duta Pertiwi Tbk.", "Hassana Boga Sejahtera Tbk.", "Hatten Bali Tbk.",
    "Hoffmen Cleanindo Tbk.", "Ifishdeco Tbk.", "Impack Pratama Industri Tbk.",
    "Indal Aluminium Industry Tbk.", "Indo Acidatama Tbk", "Indo American Seafoods Tbk.",
    "Indo Boga Sukses Tbk.", "Indo Komoditi Korpora Tbk.", "Indo Oil Perkasa Tbk.",
    "Indo Pureco Pratama Tbk.", "Indocement Tunggal Prakarsa Tb", "Indofood CBP Sukses Makmur Tbk",
    "Indofood Sukses Makmur Tbk.", "Indonesia Fibreboard Industry", "Indonesian Tobacco Tbk.",
    "Intan Baru Prana Tbk.", "Intanwijaya Internasional Tbk", "Inter Delta Tbk",
    "Intikeramik Alamasri Industri", "Intraco Penta Tbk.", "Island Concepts Indonesia Tbk.",
    "Jakarta Kyoei Steel Works Tbk.", "Janu Putra Sejahtera Tbk.", "Japfa Comfeed Indonesia Tbk.",
    "Jasuindo Tiga Perkasa Tbk.", "Jaya Agra Wattie Tbk.", "Jaya Swarasa Agung Tbk.",
    "Jembo Cable Company Tbk.", "Jhonlin Agro Raya Tbk.", "Jobubu Jarum Minahasa Tbk.",
    "KMI Wire & Cable Tbk.", "Kabelindo Murni Tbk.", "Kapuas Prima Coal Tbk.",
    "Kedawung Setia Industrial Tbk.", "Keramika Indonesia Assosiasi T", "Kino Indonesia Tbk.",
    "Kirana Megatara Tbk.", "Kobexindo Tractors Tbk.", "Kokoh Inti Arebama Tbk",
    "Kurniamitra Duta Sentosa Tbk.", "Kusuma Kemindo Sentosa Tbk.", "Lautan Luas Tbk.",
    "Leyand International Tbk.", "Lion Metal Works Tbk.", "Lionmesh Prima Tbk.",
    "Lovina Beach Brewery Tbk.", "MNC Asia Holding Tbk.", "Madusari Murni Indah Tbk.",
    "Mahkota Group Tbk.", "Malindo Feedmill Tbk.", "Mandom Indonesia Tbk.",
    "Mark Dynamics Indonesia Tbk.", "Martina Berto Tbk.", "Master Print Tbk.",
    "Matahari Putra Prima Tbk.", "Maxindo Karya Anugerah Tbk.", "Mayora Indah Tbk.",
    "Megalestari Epack Sentosaraya", "Menthobi Karyatama Raya Tbk.", "Midi Utama Indonesia Tbk.",
    "Millennium Pharmacon Internati", "Mitra Pack Tbk.", "Mitra Tirta Buwana Tbk.",
    "Modern Internasional Tbk.", "Morenzo Abadi Perkasa Tbk.", "Mulia Boga Raya Tbk.",
    "Mulia Industrindo Tbk", "Multi Agro Gemilang Plantation", "Multi Bintang Indonesia Tbk.",
    "Multi Hanna Kreasindo Tbk.", "Multi Makmur Lemindo Tbk.", "Multifiling Mitra Indonesia Tb",
    "Multipolar Tbk.", "Mustika Ratu Tbk.", "Mutuagung Lestari Tbk.",
    "Nanotech Indonesia Global Tbk.", "Nippon Indosari Corpindo Tbk.", "Nusa Palapa Gemilang Tbk.",
    "Nusantara Sawit Sejahtera Tbk.", "Nusatama Berkah Tbk.", "OBM Drilchem Tbk.",
    "Optima Prima Metal Sinergi Tbk", "PAM Mineral Tbk.", "PP London Sumatra Indonesia Tb",
    "Palma Serasih Tbk.", "Panca Budi Idaman Tbk.", "Paperrocks Indonesia Tbk.",
    "Pelangi Indah Canindo Tbk", "Perdana Bangun Pusaka Tbk", "Perma Plasindo Tbk.",
    "Personel Alih Daya Tbk.", "Pinago Utama Tbk.", "Platinum Wahab Nusantara Tbk.",
    "Pradiksi Gunatama Tbk.", "Prasidha Aneka Niaga Tbk", "Prima Cakrawala Abadi Tbk.",
    "Primadaya Plastisindo Tbk.", "Pulau Subur Tbk.", "Samator Indo Gas Tbk.",
    "Sampoerna Agro Tbk.", "Sarana Mitra Luas Tbk.", "Saranacentral Bajatama Tbk.",
    "Saraswanti Anugerah Makmur Tbk", "Sariguna Primatirta Tbk.", "Satu Visi Putra Tbk.",
    "Satyamitra Kemas Lestari Tbk.", "Sawit Sumbermas Sarana Tbk.", "Segar Kumala Indonesia Tbk.",
    "Sekar Bumi Tbk.", "Sekar Laut Tbk.", "Semen Baturaja Tbk.",
    "Semen Indonesia (Persero) Tbk.", "Sentra Food Indonesia Tbk.", "Shield On Service Tbk.",
    "Siantar Top Tbk.", "Sinergi Inti Plastindo Tbk.", "Sinergi Multi Lestarindo Tbk.",
    "Singaraja Putra Tbk.", "Smart Tbk.", "Solusi Bangun Indonesia Tbk.",
    "Sreeya Sewu Indonesia Tbk.", "Sriwahana Adityakarta Tbk.", "Steadfast Marine Tbk.",
    "Steel Pipe Industry of Indones", "Sumber Alfaria Trijaya Tbk.", "Sumber Mineral Global Abadi Tb",
    "Sumber Tani Agung Resources Tb", "Suparma Tbk.", "Superior Prima Sukses Tbk.",
    "Superkrane Mitra Utama Tbk.", "Supra Boga Lestari Tbk.", "Supreme Cable Manufacturing &",
    "Surya Biru Murni Acetylene Tbk", "Surya Pertiwi Tbk.", "Surya Toto Indonesia Tbk.",
    "Tanah Laut Tbk", "Teladan Prima Agro Tbk.", "Tigaraksa Satria Tbk.",
    "Timah Tbk.", "Tira Austenite Tbk", "Tirta Mahakam Resources Tbk",
    "Tri Banyan Tirta Tbk.", "Trias Sentosa Tbk.", "Trimegah Bangun Persada Tbk.",
    "Trinitan Metals and Minerals T", "Triputra Agro Persada Tbk.", "Triwira Insanlestari Tbk.",
    "Tunas Alfin Tbk.", "Tunas Baru Lampung Tbk.", "Ultrajaya Milk Industry & Trad",
    "Uni-Charm Indonesia Tbk.", "Unilever Indonesia Tbk.", "United Tractors Tbk.",
    "Victoria Care Indonesia Tbk.", "Voksel Electric Tbk.", "Wahana Interfood Nusantara Tbk",
    "Wahana Inti Makmur Tbk.", "Wahana Pronatural Tbk.", "Waskita Beton Precast Tbk.",
    "Wicaksana Overseas Internation", "Widiant Jaya Krenindo Tbk.", "Widodo Makmur Perkasa Tbk.",
    "Widodo Makmur Unggas Tbk.", "Wijaya Cahaya Timber Tbk.", "Wijaya Karya Beton Tbk.",
    "Wilmar Cahaya Indonesia Tbk.", "Wilton Makmur Indonesia Tbk.", "Wismilak Inti Makmur Tbk.",
    "Xolare RCR Energy Tbk.", "Yanaprima Hastapersada Tbk",
]


NAME_MAPPING = {name: i for i, name in enumerate(NAME_LIST)}


FITUR = [
    "Total Pendapatan", "Total Pendapatan T - 1", "Laba Bersih Tahun Berjalan",
    "Aset", "Aset IDR", "Ekuitas", "Ekuitas IDR", "Liabilitas",
    "Aset Lancar", "Liabilitas Jangka Pendek", "Saham Beredar",
    "Market Value", "Close Price", "Adjusted Close Price",
    "KOMISARIS INDEPENDEN", "DEWAN KOMISARIS", "DEWAN DIREKSI",
    "ROE", "Current Ratio", "Ukuran Perusahaan",
    "Pertumbuhan Penjualan", "DER", "PBV", "Persentase Dewan Komisaris",
]


FITUR_RUPIAH = {
    "Total Pendapatan", "Total Pendapatan T - 1", "Laba Bersih Tahun Berjalan",
    "Aset", "Aset IDR", "Ekuitas", "Ekuitas IDR", "Liabilitas",
    "Aset Lancar", "Liabilitas Jangka Pendek", "Saham Beredar",
    "Market Value", "Close Price", "Adjusted Close Price",
}


KODE_ADA_SENTIMEN = {
    "AVIA", "BHIT", "BISI", "BNBR", "BRMS", "BRPT", "BWPT", "CAKK",
    "CLEO", "CMNT", "CMRY", "CSRA", "DKFT", "DLTA", "DSFI", "DYAN",
    "FASW", "GGRM", "GGRP", "GZCO",
}


NAMA_KE_KODE = {
    "Avia Avian Tbk."               : "AVIA",
    "MNC Asia Holding Tbk."         : "BHIT",
    "BISI International Tbk."       : "BISI",
    "Bakrie & Brothers Tbk"         : "BNBR",
    "Bumi Teknokultura Unggul Tbk"  : "BRMS",
    "Eagle High Plantations Tbk."   : "BWPT",
    "Cahayaputra Asa Keramik Tbk."  : "CAKK",
    "Sariguna Primatirta Tbk."      : "CLEO",
    "Cemindo Gemilang Tbk."         : "CMNT",
    "Cisarua Mountain Dairy Tbk."   : "CMRY",
    "Cisadane Sawit Raya Tbk."      : "CSRA",
    "Central Omega Resources Tbk."  : "DKFT",
    "Delta Djakarta Tbk."           : "DLTA",
    "Dharma Samudera Fishing Indust": "DSFI",
    "Dyandra Media International Tb": "DYAN",
    "Fajar Surya Wisesa Tbk."       : "FASW",
    "Gudang Garam Tbk."             : "GGRM",
    "Gozco Plantations Tbk."        : "GZCO",
}

# Peta kode -> nama untuk membangun query pencarian berita yang lebih spesifik
KODE_KE_KEYWORD = {
    "AVIA": "Avia Avian cat",
    "BHIT": "MNC Asia Holding BHIT",
    "BISI": "BISI International benih",
    "BNBR": "Bakrie Brothers",
    "BRMS": "Bumi Teknokultura Unggul BRMS",
    "BWPT": "Eagle High Plantations sawit",
    "CAKK": "Cahayaputra Asa Keramik",
    "CLEO": "CLEO Sariguna Primatirta air mineral",
    "CMNT": "Cemindo Gemilang semen merah putih",
    "CMRY": "Cisarua Mountain Dairy Cimory",
    "CSRA": "Cisadane Sawit Raya",
    "DKFT": "Central Omega Resources nikel",
    "DLTA": "Delta Djakarta",
    "DSFI": "Dharma Samudera Fishing Industries",
    "DYAN": "Dyandra Media International",
    "FASW": "Fajar Surya Wisesa kertas",
    "GGRM": "Gudang Garam",
    "GZCO": "Gozco Plantations sawit",
}


SEPARATOR_LABEL = "──────── 248 PERUSAHAAN TANPA DATA SENTIMEN ────────"


def build_company_options():
    with_sentimen    = sorted(NAMA_KE_KODE.keys())
    without_sentimen = sorted(n for n in NAME_LIST if n not in NAMA_KE_KODE)
    return with_sentimen + [SEPARATOR_LABEL] + without_sentimen


COMPANY_OPTIONS = build_company_options()


# Kunci session_state untuk kolom Judul & Isi Berita
KEY_JUDUL = "input_judul_berita"
KEY_ISI   = "input_isi_berita"


# ─────────────────────────────────────────────────────────────────────────────
# PEMUATAN RESOURCE (MODEL, NLTK, STEMMER) — DI-CACHE
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def setup_nltk():
    """Unduh resource NLTK yang dibutuhkan (sekali saja)."""
    for paket in ["punkt", "punkt_tab", "stopwords"]:
        try:
            nltk.download(paket, quiet=True)
        except Exception:
            pass


@st.cache_resource(show_spinner=False)
def load_stemmer():
    """Inisialisasi stemmer Sastrawi (Bahasa Indonesia)."""
    return StemmerFactory().create_stemmer()


@st.cache_resource(show_spinner=True)
def load_models():
    """
    Memuat model & artefak yang tersimpan.
    Kembalikan dict; nilai None jika file tidak ditemukan.
    """
    artefak = {
        "model_terstruktur": "model_ml_terstruktur.pkl",
        "model_tak_terstruktur": "ensemble_model_tak_terstruktur.pkl",
        "tfidf": "tfidf_model.pkl",
        "scaler": "minmax_scaler.pkl",
    }
    hasil = {}
    for nama, path in artefak.items():
        try:
            hasil[nama] = joblib.load(path)
        except Exception:
            hasil[nama] = None
    return hasil


@st.cache_data(show_spinner=False)
def load_stopwords_custom():
    """
    Muat daftar pengecualian stopword dari pengecualian_stopword.xlsx
    (kolom 'stopwords'). Jika gagal, pakai stopword bawaan NLTK saja.
    """
    base = set()
    try:
        base = set(stopwords.words("indonesian"))
    except Exception:
        base = set()
    try:
        df = pd.read_excel("pengecualian_stopword.xlsx")
        kolom = "stopwords" if "stopwords" in df.columns else df.columns[0]
        kata_dikecualikan = set(
            str(w).strip().lower() for w in df[kolom].dropna().tolist()
        )
        # Kata yang dikecualikan TIDAK dihapus -> buang dari daftar stopword
        base = base - kata_dikecualikan
    except Exception:
        pass
    return base


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING TEKS
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_teks(teks: str, stemmer, stop_set: set) -> str:
    """Case folding, cleaning, tokenisasi, stopword removal, stemming."""
    if not teks:
        return ""
    teks = teks.lower()
    teks = re.sub(r"http\S+|www\.\S+", " ", teks)         # hapus URL
    teks = re.sub(r"[^a-z\s]", " ", teks)                  # sisakan huruf
    teks = re.sub(r"\s+", " ", teks).strip()
    try:
        tokens = word_tokenize(teks)
    except Exception:
        tokens = teks.split()
    tokens = [t for t in tokens if t not in stop_set and len(t) > 2]
    teks_bersih = " ".join(tokens)
    try:
        teks_bersih = stemmer.stem(teks_bersih)
    except Exception:
        pass
    return teks_bersih


# ─────────────────────────────────────────────────────────────────────────────
# PENCARIAN BERITA (Google News RSS)
# ─────────────────────────────────────────────────────────────────────────────
def _bersih_html(t: str) -> str:
    t = re.sub(r"<[^>]+>", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _domain(url: str) -> str:
    try:
        return urllib.parse.urlparse(url).netloc.replace("www.", "") or "Media Online"
    except Exception:
        return "Media Online"


def _format_tanggal(pub: str) -> str:
    for fmt in ("%a, %d %b %Y %H:%M:%S %Z", "%a, %d %b %Y %H:%M:%S %z"):
        try:
            return datetime.strptime(pub, fmt).strftime("%d %b %Y, %H:%M WIB")
        except ValueError:
            continue
    return pub or "Tanggal tidak tersedia"


@st.cache_data(ttl=300, show_spinner=False)
def cari_berita(kode_saham: str, kata_tambahan: str = "", max_hasil: int = 8) -> list:
    """
    Cari berita dari Google News RSS berdasarkan kode saham perusahaan.
    Mengembalikan list of dict: {judul, isi, sumber, link, tanggal}.
    """
    query = KODE_KE_KEYWORD.get(kode_saham, kode_saham)
    if kata_tambahan.strip():
        query += f" {kata_tambahan.strip()}"

    url = (
        "https://news.google.com/rss/search?q="
        + urllib.parse.quote(query)
        + "&hl=id&gl=ID&ceid=ID:id"
    )
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "id-ID,id;q=0.9",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=12)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        channel = root.find("channel")
        if channel is None:
            return []

        hasil = []
        for item in channel.findall("item")[:max_hasil]:
            judul    = _bersih_html(item.findtext("title", ""))
            link     = item.findtext("link", "").strip()
            desc     = _bersih_html(item.findtext("description", ""))
            pub_date = item.findtext("pubDate", "").strip()

            src_el = item.find("source")
            sumber = src_el.text.strip() if (src_el is not None and src_el.text) else _domain(link)

            if judul:
                hasil.append({
                    "judul"  : judul,
                    "isi"    : desc if desc else "(Ringkasan tidak tersedia — buka tautan untuk artikel lengkap)",
                    "sumber" : sumber,
                    "link"   : link,
                    "tanggal": _format_tanggal(pub_date),
                })
        return hasil
    except requests.exceptions.ConnectionError:
        return [{"error": "Tidak dapat terhubung ke internet. Periksa koneksi Anda."}]
    except requests.exceptions.Timeout:
        return [{"error": "Permintaan timeout. Coba lagi beberapa saat."}]
    except Exception as e:
        return [{"error": f"Terjadi kesalahan: {e}"}]


# ─────────────────────────────────────────────────────────────────────────────
# KOMPONEN UI
# ─────────────────────────────────────────────────────────────────────────────
def render_header():
    st.markdown(
        """
        <div class="main-header">
            <h1>📊 Financial Distress Analysis</h1>
            <p>Analisis kesehatan keuangan perusahaan berbasis data terstruktur & sentimen berita</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kotak_hasil(label: str, kelas: str, ikon: str = ""):
    st.markdown(
        f'<div class="result-box {kelas}">{ikon} {label}</div>',
        unsafe_allow_html=True,
    )


# ── TAB 1: ANALISIS TERSTRUKTUR ──────────────────────────────────────────────
def tab_terstruktur(models):
    st.markdown('<div class="section-title">🏢 Pilih Perusahaan & Tahun</div>',
                unsafe_allow_html=True)

    c1, c2 = st.columns([3, 1])
    with c1:
        nama = st.selectbox(
            "Perusahaan",
            options=[o for o in COMPANY_OPTIONS if o != SEPARATOR_LABEL],
            key="sel_terstruktur",
        )
    with c2:
        tahun = st.selectbox("Tahun", options=list(YEAR_MAPPING.keys()),
                             index=len(YEAR_MAPPING) - 1, key="thn_terstruktur")

    st.markdown('<div class="section-title">💰 Input Fitur Keuangan</div>',
                unsafe_allow_html=True)

    nilai = {}
    kolom = st.columns(3)
    for i, fitur in enumerate(FITUR):
        with kolom[i % 3]:
            fmt = "%.4f" if fitur not in FITUR_RUPIAH else "%.0f"
            nilai[fitur] = st.number_input(
                fitur, value=0.0, format=fmt, key=f"f_{i}"
            )

    if st.button("🔮 Prediksi Financial Distress", type="primary",
                 key="btn_prediksi_terstruktur"):
        model = models.get("model_terstruktur")
        scaler = models.get("scaler")
        if model is None:
            st.error("❌ Model terstruktur (model_ml_terstruktur.pkl) tidak ditemukan.")
            return

        fitur_vektor = np.array([[nilai[f] for f in FITUR]], dtype=float)
        try:
            if scaler is not None:
                fitur_vektor = scaler.transform(fitur_vektor)
            pred = model.predict(fitur_vektor)[0]
            label = str(pred).upper()

            st.markdown('<div class="section-title">📋 Hasil Prediksi</div>',
                        unsafe_allow_html=True)
            if "TIDAK" in label or label in ("1", "DISTRESS"):
                kotak_hasil(f"Status: <strong>TIDAK BAIK / Financial Distress</strong>",
                            "r-bad", "⚠️")
            else:
                kotak_hasil(f"Status: <strong>BAIK / Sehat</strong>", "r-good", "✅")

            # Probabilitas jika tersedia
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(fitur_vektor)[0]
                kelas = list(getattr(model, "classes_", range(len(proba))))
                fig = go.Figure(go.Bar(
                    x=[str(k) for k in kelas], y=proba,
                    marker_color=COLORBLIND_PALETTE["positive"],
                ))
                fig.update_layout(title="Probabilitas Prediksi", height=300,
                                  yaxis_title="Probabilitas")
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Gagal melakukan prediksi: {e}")


# ── SEKSI CARI BERITA (mengisi otomatis kolom Judul & Isi) ───────────────────
def seksi_cari_berita():
    st.markdown('<div class="section-title">🔍 Cari Berita Otomatis</div>',
                unsafe_allow_html=True)
    st.caption("Cari berita terbaru untuk 18 perusahaan berdata sentimen, "
               "lalu klik *Gunakan* untuk mengisi kolom di bawah secara otomatis.")

    c1, c2, c3 = st.columns([3, 2, 1])
    with c1:
        nama = st.selectbox("Perusahaan (18 berdata sentimen)",
                            options=list(NAMA_KE_KODE.keys()),
                            key="sel_cari_berita")
    kode = NAMA_KE_KODE[nama]
    with c2:
        kw = st.text_input("Kata kunci tambahan (opsional)",
                           placeholder="mis: laba, dividen, akuisisi",
                           key="kw_cari_berita")
    with c3:
        st.write("")
        st.write("")
        cari = st.button("🔍 Cari Berita", type="primary", use_container_width=True)

    st.markdown(
        f'<span class="badge-info">Kode Saham: <strong>{kode}</strong></span>'
        f'&nbsp;<span class="badge-ok">Sumber: Google News (ID)</span>',
        unsafe_allow_html=True,
    )

    if cari:
        with st.spinner(f"Mencari berita untuk {nama}..."):
            st.session_state["_hasil_berita"] = cari_berita(kode, kw)
            st.session_state["_kode_aktif"] = kode

    hasil = st.session_state.get("_hasil_berita", [])
    if hasil and isinstance(hasil[0], dict) and "error" in hasil[0]:
        st.error(f"❌ {hasil[0]['error']}")
        return
    if not hasil:
        return

    st.success(f"✅ Ditemukan {len(hasil)} berita. Pilih salah satu:")
    for idx, b in enumerate(hasil):
        st.markdown(
            f"""
            <div class="news-card">
                <div class="news-meta">🕐 {b['tanggal']} &nbsp;•&nbsp; 🏷️ {b['sumber']}</div>
                <div class="news-title">{b['judul']}</div>
                <div class="news-body">{b['isi'][:300] + ('...' if len(b['isi']) > 300 else '')}</div>
                <a class="news-link" href="{b['link']}" target="_blank">🔗 Buka sumber asli</a>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("📥 Gunakan Berita Ini", key=f"gunakan_{idx}"):
            st.session_state[KEY_JUDUL] = b["judul"]
            st.session_state[KEY_ISI] = (
                f"{b['isi']}\n\nSumber: {b['sumber']}\nLink: {b['link']}"
            )
            st.success(f"✅ Kolom terisi: '{b['judul'][:60]}...'")
            st.rerun()


# ── TAB 2: ANALISIS TAK TERSTRUKTUR (SENTIMEN BERITA) ────────────────────────
def tab_tak_terstruktur(models, stemmer, stop_set):
    # 1) Tombol cari berita otomatis
    seksi_cari_berita()

    st.markdown("---")

    # 2) Seksi Data Teks Berita (sesuai screenshot)
    st.markdown('<div class="section-title">📰 Data Teks Berita</div>',
                unsafe_allow_html=True)

    col_judul, col_isi = st.columns(2)
    with col_judul:
        judul_berita = st.text_area(
            "Judul Berita",
            value=st.session_state.get(KEY_JUDUL, ""),
            key=KEY_JUDUL,
            placeholder="Contoh: Laba bersih meningkat 25% pada 2024",
            height=120,
        )
    with col_isi:
        isi_berita = st.text_area(
            "Isi Berita",
            value=st.session_state.get(KEY_ISI, ""),
            key=KEY_ISI,
            placeholder="Tulis atau tempel isi berita terkait perusahaan di sini...",
            height=120,
        )

    col_btn, col_clear = st.columns([1, 1])
    with col_btn:
        analisis = st.button("🧠 Analisis Sentimen Berita", type="primary",
                             key="btn_prediksi_sentimen", use_container_width=True)
    with col_clear:
        if st.button("🗑️ Bersihkan Kolom", key="btn_bersih", use_container_width=True):
            st.session_state[KEY_JUDUL] = ""
            st.session_state[KEY_ISI] = ""
            st.rerun()

    if analisis:
        teks_gabung = f"{judul_berita} {isi_berita}".strip()
        if not teks_gabung:
            st.warning("⚠️ Mohon isi Judul atau Isi Berita terlebih dahulu, "
                       "atau gunakan fitur Cari Berita di atas.")
            return

        model = models.get("model_tak_terstruktur")
        tfidf = models.get("tfidf")
        if model is None or tfidf is None:
            st.error("❌ Model sentimen atau TF-IDF tidak ditemukan "
                     "(ensemble_model_tak_terstruktur.pkl / tfidf_model.pkl).")
            return

        with st.spinner("Memproses teks & memprediksi sentimen..."):
            teks_bersih = preprocess_teks(teks_gabung, stemmer, stop_set)
            try:
                vektor = tfidf.transform([teks_bersih])
                pred = model.predict(vektor)[0]
                label = str(pred).upper()

                st.markdown('<div class="section-title">📋 Hasil Analisis Sentimen</div>',
                            unsafe_allow_html=True)
                if label in ("POSITIF", "POSITIVE", "1", "BAIK"):
                    kotak_hasil("Sentimen: <strong>POSITIF</strong>", "r-good", "😊")
                elif label in ("NEGATIF", "NEGATIVE", "0", "TIDAK BAIK"):
                    kotak_hasil("Sentimen: <strong>NEGATIF</strong>", "r-bad", "😟")
                else:
                    kotak_hasil(f"Sentimen: <strong>{label}</strong>", "r-neutral", "😐")

                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(vektor)[0]
                    kelas = list(getattr(model, "classes_", range(len(proba))))
                    fig = go.Figure(go.Bar(
                        x=[str(k) for k in kelas], y=proba,
                        marker_color=COLORBLIND_PALETTE["accent"],
                    ))
                    fig.update_layout(title="Probabilitas Sentimen", height=300,
                                      yaxis_title="Probabilitas")
                    st.plotly_chart(fig, use_container_width=True)

                with st.expander("🔎 Lihat teks setelah preprocessing"):
                    st.write(teks_bersih if teks_bersih else "(kosong)")
            except Exception as e:
                st.error(f"❌ Gagal menganalisis sentimen: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    setup_nltk()
    stemmer = load_stemmer()
    stop_set = load_stopwords_custom()
    models = load_models()

    render_header()

    tab1, tab2 = st.tabs([
        "📊 Analisis Terstruktur (Keuangan)",
        "📰 Analisis Tak Terstruktur (Sentimen Berita)",
    ])
    with tab1:
        tab_terstruktur(models)
    with tab2:
        tab_tak_terstruktur(models, stemmer, stop_set)


if __name__ == "__main__":
    main()
