import streamlit as st
import numpy as np
import pandas as pd
import joblib
import re
import nltk
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import requests
from bs4 import BeautifulSoup
 
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
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
        padding: 1.5rem 2rem; border-radius: 12px;
        margin-bottom: 1.5rem; color: white;
    }
    .main-header h1 { margin: 0; font-size: 1.8rem; font-weight: 700; }
    .main-header p  { margin: 0.3rem 0 0; font-size: 0.95rem; opacity: 0.85; }
    .section-card {
        background: #ffffff; border: 1px solid #e0e0e0;
        border-radius: 10px; padding: 1.2rem 1.5rem;
        margin-bottom: 1rem; box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    }
    .section-title {
        font-size: 1rem; font-weight: 700; color: #1e3a5f;
        border-left: 4px solid #2d6a9f; padding-left: 0.6rem;
        margin-bottom: 0.9rem;
    }
    .badge-ok   { background:#e8f5e9; color:#2e7d32; border:1px solid #a5d6a7;
                  border-radius:6px; padding:0.35rem 0.75rem; font-size:0.82rem;
                  display:inline-block; margin-bottom:0.8rem; }
    .badge-warn { background:#fff8e1; color:#f57f17; border:1px solid #ffe082;
                  border-radius:6px; padding:0.35rem 0.75rem; font-size:0.82rem;
                  display:inline-block; margin-bottom:0.8rem; }
    .badge-info { background:#e3f2fd; color:#1565c0; border:1px solid #90caf9;
                  border-radius:6px; padding:0.35rem 0.75rem; font-size:0.82rem;
                  display:inline-block; margin-bottom:0.8rem; }
    .result-box { border-radius:10px; padding:1rem 1.2rem; margin:0.4rem 0; font-size:0.94rem; }
    .r-good    { background:#e8f5e9; border-left:5px solid #43a047; color:#2e7d32; }
    .r-bad     { background:#fff3e0; border-left:5px solid #f57c00; color:#e65100; }
    .r-invest  { background:#e3f2fd; border-left:5px solid #1976d2; color:#0d47a1; }
    .r-risk    { background:#fff9c4; border-left:5px solid #f9a825; color:#f57f17; }
    .r-no      { background:#ffebee; border-left:5px solid #e53935; color:#b71c1c; }
    .r-neutral { background:#f3e5f5; border-left:5px solid #8e24aa; color:#6a1b9a; }
    .metric-mini { background:#f5f7fa; border-radius:8px; padding:0.6rem 0.8rem;
                   text-align:center; border:1px solid #e0e0e0; }
    .metric-mini .val { font-size:1.05rem; font-weight:700; color:#1e3a5f; }
    .metric-mini .lbl { font-size:0.72rem; color:#666; margin-top:2px; }
    .news-card {
        border:1px solid #e0e0e0; border-radius:8px; padding:0.9rem 1rem;
        margin:0.4rem 0; background:#f9fafb;
        border-left: 4px solid #2d6a9f;
    }
    .news-title { font-weight:600; color:#1e3a5f; font-size:0.95rem; margin-bottom:0.25rem; }
    .news-meta  { font-size:0.8rem; color:#666; margin-bottom:0.4rem; }
    .news-valid { color:#2e7d32; font-weight:500; }
    .news-invalid { color:#f57f17; font-weight:500; }
    footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)
 
# ─────────────────────────────────────────────────────────────────────────────
# KONSTANTA
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
 
# Query pencarian yang dioptimasi untuk setiap perusahaan
NAMA_QUERY = {
    "Avia Avian Tbk."               : "Avia Avian AVIA saham",
    "MNC Asia Holding Tbk."         : "MNC Asia BHIT saham",
    "BISI International Tbk."       : "BISI International saham",
    "Bakrie & Brothers Tbk"         : "Bakrie Brothers BNBR saham",
    "Bumi Teknokultura Unggul Tbk"  : "Bumi Teknokultura BRMS saham",
    "Eagle High Plantations Tbk."   : "Eagle High Plantations BWPT saham",
    "Cahayaputra Asa Keramik Tbk."  : "Cahayaputra Keramik CAKK saham",
    "Sariguna Primatirta Tbk."      : "CLEO Sariguna saham",
    "Cemindo Gemilang Tbk."         : "Cemindo Gemilang CMNT saham",
    "Cisarua Mountain Dairy Tbk."   : "CMRY Cisarua Mountain Dairy saham",
    "Cisadane Sawit Raya Tbk."      : "Cisadane Sawit CSRA saham",
    "Central Omega Resources Tbk."  : "Central Omega DKFT saham",
    "Delta Djakarta Tbk."           : "Delta Djakarta DLTA bir saham",
    "Dharma Samudera Fishing Indust": "Dharma Samudera DSFI saham",
    "Dyandra Media International Tb": "Dyandra Media DYAN saham",
    "Fajar Surya Wisesa Tbk."       : "Fajar Surya Wisesa FASW saham",
    "Gudang Garam Tbk."             : "Gudang Garam GGRM rokok saham",
    "Gozco Plantations Tbk."        : "Gozco Plantations GZCO saham",
}
 
# Media resmi Indonesia untuk validasi sumber berita
SUMBER_RESMI = {
    'idx.co.id', 'reuters.com', 'kompas.com', 'cnnindonesia.com',
    'bloomberg.com', 'idxchannel.com', 'investor.id', 'mediaindonesia.com',
    'kontan.co.id', 'bisnis.com', 'ojk.go.id', 'cnbc.com',
    'tempo.co', 'okezone.com', 'suara.com', 'kumparan.com',
}
 
SEPARATOR_LABEL = "──────── 248 PERUSAHAAN TANPA DATA SENTIMEN ────────"
 
def build_company_options():
    with_sentimen    = sorted(NAMA_KE_KODE.keys())
    without_sentimen = sorted(n for n in NAME_LIST if n not in NAMA_KE_KODE)
    return with_sentimen + [SEPARATOR_LABEL] + without_sentimen
 
COMPANY_OPTIONS = build_company_options()
 
# ─────────────────────────────────────────────────────────────────────────────
# FUNGSI PENCARIAN BERITA
# ─────────────────────────────────────────────────────────────────────────────
def validate_source(source_url: str) -> tuple:
    """Cek apakah URL berasal dari 16 sumber resmi. Return: (is_valid, domain)"""
    try:
        source_lower = source_url.lower()
        for media in SUMBER_RESMI:
            if media in source_lower:
                return True, media
        return False, "-"
    except:
        return False, "-"
 
 
def fetch_article_summary(url: str, max_words: int = 50) -> str:
    """
    Fetch halaman artikel dan ambil max_words kata pertama dari konten utama.
    Hanya dipanggil untuk URL dari 16 sumber resmi yang terverifikasi.
    Menggunakan html.parser (built-in Python — tidak perlu install lxml).
    """
    try:
        headers = {
            'User-Agent'      : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept-Language' : 'id-ID,id;q=0.9,en;q=0.8',
            'Accept'          : 'text/html,application/xhtml+xml',
        }
        resp = requests.get(url, headers=headers, timeout=7, allow_redirects=True)
        if resp.status_code != 200:
            return ""
 
        soup = BeautifulSoup(resp.content, features="html.parser")
 
        # Hapus elemen tidak relevan
        for tag in soup(["script", "style", "nav", "header",
                         "footer", "aside", "figure", "figcaption",
                         "form", "button", "iframe"]):
            tag.decompose()
 
        # Selector konten artikel — urutan dari yang paling spesifik
        konten = ""
        SELECTORS = [
            "article",
            ".article-content", ".article-body", ".article__content",
            ".detail-text", ".detail__body",
            ".read-page--article-body",
            ".post-content", ".post-body",
            ".entry-content", ".entry-body",
            ".content-body", ".content-detail",
            "div.detail", "div.artikel",
            ".story-body", ".story-content",
            ".news-content", ".news-body",
            "[itemprop='articleBody']",
        ]
        for selector in SELECTORS:
            tags = soup.select(selector)
            if tags:
                teks = " ".join(t.get_text(separator=" ", strip=True) for t in tags[:3])
                if len(teks.split()) >= 15:
                    konten = teks
                    break
 
        if not konten:
            # Fallback: ambil paragraf panjang (>20 kata)
            paragraphs = soup.find_all("p")
            parts = [p.get_text(strip=True) for p in paragraphs
                     if len(p.get_text(strip=True).split()) > 8]
            konten = " ".join(parts[:12])
 
        # Bersihkan whitespace berlebih
        konten = " ".join(konten.split())
        if not konten:
            return ""
 
        # Ambil max_words kata pertama
        words = konten.split()
        if len(words) <= max_words:
            return konten
        return " ".join(words[:max_words]) + "..."
 
    except Exception:
        return ""
 
 
def build_verified_rss_query(keyword: str) -> str:
    """
    Buat query Google News RSS yang hanya mengembalikan hasil
    dari 16 sumber resmi yang telah ditentukan.
    Dibagi 2 batch karena query terlalu panjang jika semua digabung.
    """
    sumber_list = list(SUMBER_RESMI)
    site_filters = " OR ".join(f"site:{s}" for s in sumber_list)
    return f"{keyword} ({site_filters})"
 
 
def search_news_verified(keyword: str, num_results: int = 5) -> list:
    """
    Cari berita HANYA dari 16 sumber resmi menggunakan Google News RSS
    dengan site-filter query. Setiap hasil yang terverifikasi akan di-fetch
    untuk mendapatkan summary 50 kata dari isi artikel.
    """
    import urllib.parse
 
    # Query dengan site filter — hanya tampilkan dari 16 sumber resmi
    site_filters = " OR ".join(f"site:{s}" for s in list(SUMBER_RESMI))
    full_query   = f"{keyword} ({site_filters})"
    encoded      = urllib.parse.quote(full_query)
 
    rss_url = (
        f"https://news.google.com/rss/search"
        f"?q={encoded}&hl=id&gl=ID&ceid=ID:id"
    )
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
 
    try:
        resp = requests.get(rss_url, headers=headers, timeout=10)
        if resp.status_code != 200:
            return [{"error": f"HTTP {resp.status_code}"}]
 
        soup  = BeautifulSoup(resp.content, features="html.parser")
        items = soup.find_all("item")
 
        results = []
        for item in items:
            if len(results) >= num_results:
                break
 
            title      = item.find("title")
            link       = item.find("link")
            pubdate    = item.find("pubdate")
            source_tag = item.find("source")
 
            title_text  = title.get_text(strip=True)      if title      else "Tanpa Judul"
            link_text   = link.get_text(strip=True)       if link       else "#"
            date_text   = pubdate.get_text(strip=True)    if pubdate    else ""
            source_name = source_tag.get_text(strip=True) if source_tag else ""
 
            # Verifikasi: hanya masukkan jika dari 16 sumber resmi
            is_valid, domain = validate_source(link_text + " " + source_name)
            if not is_valid:
                # Skip — tidak dari sumber resmi yang ditentukan
                continue
 
            # Fetch summary dari artikel (max 50 kata)
            summary = ""
            if link_text and link_text != "#":
                summary = fetch_article_summary(link_text, max_words=50)
 
            # Format tanggal lebih ringkas jika ada
            try:
                from email.utils import parsedate_to_datetime
                dt = parsedate_to_datetime(date_text)
                date_fmt = dt.strftime("%d %b %Y")
            except Exception:
                date_fmt = date_text[:16] if date_text else ""
 
            results.append({
                "title"   : title_text,
                "link"    : link_text,
                "date"    : date_fmt,
                "source"  : source_name if source_name else domain,
                "is_valid": True,   # dijamin valid karena sudah difilter
                "summary" : summary,
            })
 
        return results if results else [{"error": "Tidak ada berita dari sumber terverifikasi ditemukan."}]
 
    except Exception as e:
        return [{"error": str(e)}]
 
 
# ─────────────────────────────────────────────────────────────────────────────
# LOAD RESOURCES
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def download_nltk():
    for pkg in ["punkt", "stopwords", "punkt_tab"]:
        nltk.download(pkg, quiet=True)
 
download_nltk()
 
 
@st.cache_resource
def load_models():
    return (
        joblib.load("minmax_scaler.pkl"),
        joblib.load("model_ml_terstruktur.pkl"),
        joblib.load("tfidf_model.pkl"),
        joblib.load("ensemble_model_tak_terstruktur.pkl"),
    )
 
 
@st.cache_resource
def load_nlp():
    data_stop    = pd.read_excel("pengecualian_stopword.xlsx")
    stopwords_ex = set(data_stop["stopwords"])
    sw           = set(stopwords.words("indonesian")) - stopwords_ex
    stemmer      = StemmerFactory().create_stemmer()
    return sw, stemmer
 
 
@st.cache_data
def load_lookup() -> dict:
    df     = pd.read_excel("dc_dataCustom.xlsx")
    yr_rev = {0: 2019, 1: 2020, 2: 2021, 3: 2022, 4: 2023, 5: 2024}
    nm_rev = {i: n for i, n in enumerate(NAME_LIST)}
    lookup = {}
    for _, row in df.iterrows():
        tahun = yr_rev.get(int(row["Year"]))
        nama  = nm_rev.get(int(row["NAME"]))
        if tahun is None or nama is None:
            continue
        lookup[(nama, tahun)] = {
            "Total Pendapatan"          : float(row["Total Pendapatan"]),
            "Total Pendapatan T - 1"    : float(row["Total Pendapatan T - 1"]),
            "Laba Bersih Tahun Berjalan": float(row["Laba Bersih Tahun Berjalan"]),
            "Aset"                      : float(row["Aset"]),
            "Aset IDR"                  : float(row["Aset IDR"]),
            "Ekuitas"                   : float(row["Ekuitas"]),
            "Ekuitas IDR"               : float(row["Ekuitas IDR"]),
            "Liabilitas"                : float(row["Liabilitas"]),
            "Aset Lancar"               : float(row["Aset Lancar"]),
            "Liabilitas Jangka Pendek"  : float(row["Liabilitas Jangka Pendek"]),
            "Saham Beredar"             : float(row["Saham Beredar"]),
            "Market Value"              : float(row["Market Value"]),
            "Close Price"               : float(row["Close Price"]),
            "Adjusted Close Price"      : float(row["Ajusted Close Price"]),
            "KOMISARIS INDEPENDEN"      : float(row["KOMISARIS INDEPENDEN"]),
            "DEWAN KOMISARIS"           : float(row["DEWAN KOMISARIS"]),
            "DEWAN DIREKSI"             : float(row["DEWAN DIREKSI"]),
            "ROE"                       : float(row["ROE"]),
            "Current Ratio"             : float(row["Current Ratio"]),
            "Ukuran Perusahaan"         : float(row["Ukuran Perusahaan"]),
            "Pertumbuhan Penjualan"     : float(row["Pertumbuhan Penjualan"]),
            "DER"                       : float(row["DER"]),
            "PBV"                       : float(row["PBV"]),
            "Persentase Dewan Komisaris": float(row["Persentase Dewan Komisaris"]),
        }
    return lookup
 
 
@st.cache_data
def load_tren(nama: str) -> pd.DataFrame:
    lookup = load_lookup()
    rows   = []
    for tahun in range(2019, 2025):
        key = (nama, tahun)
        if key in lookup:
            r = {"Tahun": tahun}
            r.update(lookup[key])
            rows.append(r)
    return pd.DataFrame(rows).set_index("Tahun") if rows else pd.DataFrame()
 
 
# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
for f in FITUR:
    if f not in st.session_state:
        st.session_state[f] = 0.0
if "autofilled"    not in st.session_state: st.session_state["autofilled"]    = False
if "show_hasil"    not in st.session_state: st.session_state["show_hasil"]    = False
if "title_input"   not in st.session_state: st.session_state["title_input"]   = ""
if "content_input" not in st.session_state: st.session_state["content_input"] = ""
if "news_results"  not in st.session_state: st.session_state["news_results"]  = []
if "news_searched" not in st.session_state: st.session_state["news_searched"] = False
 
 
# ─────────────────────────────────────────────────────────────────────────────
# CALLBACK AUTOFILL
# ─────────────────────────────────────────────────────────────────────────────
def do_autofill():
    nama_dipilih = st.session_state["sel_nama"]
    if nama_dipilih == SEPARATOR_LABEL:
        for f in FITUR:
            st.session_state[f] = 0.0
        st.session_state["autofilled"]    = False
        st.session_state["show_hasil"]    = False
        st.session_state["news_results"]  = []
        st.session_state["news_searched"] = False
        return
    lookup = load_lookup()
    key    = (nama_dipilih, st.session_state["sel_tahun"])
    found  = key in lookup
    for f in FITUR:
        st.session_state[f] = lookup[key][f] if found else 0.0
    st.session_state["autofilled"]    = found
    st.session_state["show_hasil"]    = False
    st.session_state["news_results"]  = []
    st.session_state["news_searched"] = False
 
 
# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING TEKS
# ─────────────────────────────────────────────────────────────────────────────
def datacleaning(text: str) -> str:
    text = re.sub(r"@[A-Za-z0-9]+", "", text)
    text = re.sub(r"#[A-Za-z0-9]+", "", text)
    text = re.sub(r"RT[\s]", "", text)
    text = re.sub(r'[?|$|.|@#%^/&*=!_()\"+,-]', "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[0-9]+", "", text)
    return text.replace("\n", " ").strip()
 
 
def preprocess_text(raw: str, sw, stemmer) -> str:
    text     = datacleaning(raw).lower()
    tokens   = word_tokenize(text)
    filtered = " ".join(w for w in tokens if w not in sw)
    return stemmer.stem(filtered)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# GRAFIK TREN RASIO KEUANGAN
# ─────────────────────────────────────────────────────────────────────────────
def buat_grafik_tren(nama: str):
    df = load_tren(nama)
    if df.empty or len(df) < 2:
        return None
    tahun      = df.index.tolist()
    warna_line = "#2d6a9f"
    warna_pos  = "#43a047"
    warna_neg  = "#e53935"
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            "ROE — Return on Equity",       "Current Ratio (Likuiditas)",
            "DER — Debt-to-Equity Ratio",   "PBV — Price-to-Book Value",
            "Pertumbuhan Penjualan (%)",     "Ukuran Perusahaan (ln Aset)",
        ],
        vertical_spacing=0.15, horizontal_spacing=0.10,
    )
 
    def add_line(col_name, row, col_idx, fmt=".4f"):
        if col_name not in df.columns:
            return
        fig.add_trace(go.Scatter(
            x=tahun, y=df[col_name].tolist(), mode="lines+markers",
            line=dict(color=warna_line, width=2.5),
            marker=dict(size=7, color=warna_line),
            hovertemplate=f"<b>%{{x}}</b><br>{col_name}: %{{y:{fmt}}}<extra></extra>",
            showlegend=False,
        ), row=row, col=col_idx)
 
    add_line("ROE", 1, 1);            add_line("Current Ratio", 1, 2)
    add_line("DER", 2, 1);            add_line("PBV", 2, 2)
    add_line("Ukuran Perusahaan", 3, 2, fmt=".3f")
 
    if "Pertumbuhan Penjualan" in df.columns:
        vals   = df["Pertumbuhan Penjualan"].tolist()
        colors = [warna_pos if v >= 0 else warna_neg for v in vals]
        fig.add_trace(go.Bar(
            x=tahun, y=[v * 100 for v in vals], marker_color=colors,
            hovertemplate="<b>%{x}</b><br>Pertumbuhan: %{y:.2f}%<extra></extra>",
            showlegend=False,
        ), row=3, col=1)
        fig.add_hline(y=0, line_dash="dot", line_color="#aaa", line_width=1, row=3, col=1)
 
    fig.update_layout(
        title=dict(
            text=f"<b>Tren Rasio Keuangan 2019–2024 — {nama}</b>",
            font=dict(size=14, color="#1e3a5f"), x=0.01,
        ),
        height=680, paper_bgcolor="white", plot_bgcolor="#f9fafb",
        margin=dict(t=80, b=40, l=50, r=30), font=dict(size=11, color="#333"),
    )
    tick_txt = [str(t) for t in tahun]
    for i in range(1, 4):
        for j in range(1, 3):
            fig.update_xaxes(showgrid=True, gridcolor="#e8e8e8",
                             tickvals=tahun, ticktext=tick_txt, row=i, col=j)
            fig.update_yaxes(showgrid=True, gridcolor="#e8e8e8", row=i, col=j)
    return fig
 
 
# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📊 Financial Distress Analysis</h1>
    <p>Prediksi kesehatan keuangan perusahaan berbasis data laporan keuangan &amp; sentimen berita</p>
</div>
""", unsafe_allow_html=True)
 
with st.spinner("Memuat model & data..."):
    scaler, model_numerik, tfidf_vectorizer, model_teks = load_models()
    stop_words, stemmer = load_nlp()
 
# ─────────────────────────────────────────────────────────────────────────────
# SEKSI 1 — PILIH PERUSAHAAN & TAHUN
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🏢 Pilih Perusahaan & Tahun</div>', unsafe_allow_html=True)
 
cy, cn = st.columns([1, 3])
with cy:
    year = st.selectbox("Tahun", options=list(YEAR_MAPPING.keys()),
                        key="sel_tahun", on_change=do_autofill)
with cn:
    name = st.selectbox(
        "Nama Perusahaan", options=COMPANY_OPTIONS,
        key="sel_nama", on_change=do_autofill,
        help="18 perusahaan teratas memiliki data sentimen historis. "
             "248 perusahaan di bawah separator tidak punya data sentimen.",
    )
 
if name == SEPARATOR_LABEL:
    st.warning("⚠️ Baris pembatas tidak bisa dipilih. Silakan pilih nama perusahaan yang valid.")
    st.stop()
 
if st.session_state["autofilled"]:
    st.markdown('<span class="badge-ok">✅ Data numerik terisi otomatis. Dapat diedit manual.</span>',
                unsafe_allow_html=True)
else:
    st.markdown('<span class="badge-warn">⚠️ Data numerik tidak tersedia untuk kombinasi ini — isi manual.</span>',
                unsafe_allow_html=True)
 
kode         = NAMA_KE_KODE.get(name)
ada_sentimen = kode is not None and kode in KODE_ADA_SENTIMEN
 
if ada_sentimen:
    st.markdown(f'<span class="badge-info">📰 Data sentimen tersedia (kode: <b>{kode}</b>).</span>',
                unsafe_allow_html=True)
else:
    kode_disp = kode if kode else "tidak diketahui"
    st.markdown(
        f'<span class="badge-warn">⚠️ <b>{name}</b> (kode: <b>{kode_disp}</b>) tidak ada di '
        f'dataset sentimen — 18 dari 266 perusahaan tercakup. Input teks berita manual.</span>',
        unsafe_allow_html=True)
 
st.markdown("</div>", unsafe_allow_html=True)
 
# ─────────────────────────────────────────────────────────────────────────────
# SEKSI 2 — LAPORAN KEUANGAN
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📈 Laporan Keuangan</div>', unsafe_allow_html=True)
 
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f'<div class="metric-mini"><div class="val">Rp {st.session_state["Total Pendapatan"]:,.0f}</div><div class="lbl">Total Pendapatan</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown(f'<div class="metric-mini"><div class="val">{st.session_state["ROE"]:.4f}</div><div class="lbl">ROE</div></div>', unsafe_allow_html=True)
with m3:
    st.markdown(f'<div class="metric-mini"><div class="val">{st.session_state["DER"]:.4f}</div><div class="lbl">DER</div></div>', unsafe_allow_html=True)
with m4:
    st.markdown(f'<div class="metric-mini"><div class="val">{st.session_state["Current Ratio"]:.4f}</div><div class="lbl">Current Ratio</div></div>', unsafe_allow_html=True)
 
m5, m6, m7 = st.columns(3)
with m5:
    st.markdown(f'<div class="metric-mini"><div class="val">{st.session_state["Ukuran Perusahaan"]:.4f}</div><div class="lbl">Ukuran Perusahaan</div></div>', unsafe_allow_html=True)
with m6:
    st.markdown(f'<div class="metric-mini"><div class="val">{st.session_state["Pertumbuhan Penjualan"]*100:.2f}%</div><div class="lbl">Pertumbuhan Penjualan</div></div>', unsafe_allow_html=True)
with m7:
    st.markdown(f'<div class="metric-mini"><div class="val">{st.session_state["PBV"]:.4f}</div><div class="lbl">PBV</div></div>', unsafe_allow_html=True)
 
m8, m9, m10 = st.columns(3)
with m8:
    st.markdown(f'<div class="metric-mini"><div class="val">{st.session_state["Saham Beredar"]:,.0f}</div><div class="lbl">Saham Beredar</div></div>', unsafe_allow_html=True)
with m9:
    st.markdown(f'<div class="metric-mini"><div class="val">Rp {st.session_state["Market Value"]:,.0f}</div><div class="lbl">Market Value</div></div>', unsafe_allow_html=True)
with m10:
    st.markdown(f'<div class="metric-mini"><div class="val">Rp {st.session_state["Close Price"]:,.0f}</div><div class="lbl">Close Price</div></div>', unsafe_allow_html=True)
 
st.divider()
 
with st.expander("🔧 Detail Input 24 Fitur — klik untuk membuka/edit manual", expanded=False):
    st.caption("Semua fitur sudah terisi otomatis dari dataset. "
               "Buka panel ini hanya jika ingin mengubah nilai secara manual.")
    cols3  = st.columns(3)
    inputs = []
    for i, label in enumerate(FITUR):
        fmt = "%.2f" if label in FITUR_RUPIAH else "%.6f"
        with cols3[i % 3]:
            v = st.number_input(label,
                                value=float(st.session_state.get(label, 0.0)),
                                format=fmt, key=f"ni_{label}")
            inputs.append(v)
 
st.markdown("</div>", unsafe_allow_html=True)
 
# ─────────────────────────────────────────────────────────────────────────────
# SEKSI 3 — CARI BERITA ONLINE  ← TEPAT DI ATAS "Data Teks Berita"
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🔍 Cari Berita Online</div>', unsafe_allow_html=True)
 
nama_cari_list = sorted(NAMA_KE_KODE.keys())
col_cari1, col_cari2 = st.columns([3, 1])
 
with col_cari1:
    idx_default  = nama_cari_list.index(name) if name in nama_cari_list else 0
    pilihan_cari = st.selectbox(
        "Pilih perusahaan untuk dicari beritanya",
        options=nama_cari_list,
        index=idx_default,
        key="pilihan_cari_berita",
    )
 
with col_cari2:
    st.markdown("<br>", unsafe_allow_html=True)
    cari_btn = st.button("🔍 Cari Berita", use_container_width=True, type="primary")
 
if cari_btn:
    query = NAMA_QUERY.get(pilihan_cari, pilihan_cari + " saham Indonesia")
    with st.spinner(f"Mencari berita dari 16 sumber terverifikasi untuk **{pilihan_cari}**... (~10 detik)"):
        results = search_news_verified(query, num_results=5)
        st.session_state["news_results"]  = results
        st.session_state["news_searched"] = True
 
# Tampilkan hasil pencarian
if st.session_state["news_searched"]:
    results = st.session_state["news_results"]
 
    if not results:
        st.info("📭 Tidak ada berita dari sumber terverifikasi. Coba pilih perusahaan lain atau input berita manual.")
    elif "error" in results[0]:
        st.info(f"📭 {results[0]['error']} Coba pilih perusahaan lain atau input berita manual di bawah.")
    else:
        st.success(f"✅ Ditemukan **{len(results)}** berita dari sumber terverifikasi")
        st.caption("💡 Baca ringkasan di bawah, klik link untuk artikel lengkap, lalu copy isi ke kolom **Data Teks Berita**.")
 
        for i, article in enumerate(results, 1):
            source  = article.get("source", "")
            date    = article.get("date", "")
            link    = article.get("link", "#")
            summary = article.get("summary", "")
 
            # Meta line
            meta_parts = [p for p in [source, date] if p and p.strip() and p != "-"]
            meta_line  = " &nbsp;|&nbsp; ".join(meta_parts)
            if meta_line:
                meta_line += ' &nbsp;|&nbsp; <span class="news-valid">✅ Sumber Terverifikasi</span>'
            else:
                meta_line = '<span class="news-valid">✅ Sumber Terverifikasi</span>'
 
            # Summary block
            if summary:
                summary_html = (
                    f'<div style="font-size:0.88rem;color:#333;line-height:1.65;'
                    f'margin-top:0.55rem;padding-top:0.5rem;'
                    f'border-top:1px solid #dde;">'
                    f'<span style="font-weight:600;color:#1e3a5f;">Ringkasan:</span> {summary}'
                    f'</div>'
                )
            else:
                summary_html = (
                    '<div style="font-size:0.82rem;color:#aaa;margin-top:0.4rem;">'
                    'Ringkasan tidak tersedia — buka artikel untuk membaca isi lengkap.'
                    '</div>'
                )
 
            # Label link informatif
            link_label = f"🔗 Buka di {source}" if (source and source != "-") else "🔗 Buka Artikel Lengkap"
 
            st.markdown(
                f'<div class="news-card">'
                f'<div class="news-title">{i}. {article["title"]}</div>'
                f'<div class="news-meta">{meta_line}</div>'
                f'{summary_html}'
                f'</div>',
                unsafe_allow_html=True,
            )
            if link and link != "#":
                st.markdown(f"&nbsp;&nbsp;&nbsp;[{link_label}]({link})")
 
st.markdown("</div>", unsafe_allow_html=True)
 
# ─────────────────────────────────────────────────────────────────────────────
# SEKSI 4 — DATA TEKS BERITA
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📰 Data Teks Berita</div>', unsafe_allow_html=True)
 
if not ada_sentimen:
    st.markdown(
        '<div class="result-box r-neutral" style="margin-bottom:0.8rem;">'
        "ℹ️ <b>Perusahaan ini tidak ada dalam dataset sentimen historis.</b><br>"
        "Masukkan berita terkini secara manual. "
        "Model sentimen tetap memproses teks yang Anda tulis."
        "</div>", unsafe_allow_html=True)
 
tc, cc = st.columns([1, 2])
with tc:
    title_input = st.text_area(
        "Judul Berita", height=100,
        placeholder="Contoh: Laba bersih meningkat 25% pada 2024",
        key="title_input",
    )
with cc:
    content_input = st.text_area(
        "Isi Berita", height=100,
        placeholder="Tulis atau tempel isi berita terkait perusahaan di sini...",
        key="content_input",
    )
 
st.markdown("</div>", unsafe_allow_html=True)
 
# ─────────────────────────────────────────────────────────────────────────────
# TOMBOL PREDIKSI
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
col_btn = st.columns([1, 2, 1])
with col_btn[1]:
    prediksi_btn = st.button(
        "🔍 Jalankan Prediksi", use_container_width=True, type="primary"
    )
 
if prediksi_btn:
    st.session_state["show_hasil"] = True
 
# ─────────────────────────────────────────────────────────────────────────────
# HASIL PREDIKSI
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state["show_hasil"]:
 
    enc_year   = YEAR_MAPPING[year]
    enc_name   = NAME_MAPPING[name]
    arr_scaled = scaler.transform(np.array([[enc_year, enc_name] + inputs]))
    pred_num   = model_numerik.predict(arr_scaled)[0]
    label_num  = "Baik" if str(pred_num).upper() == "BAIK" else "Tidak Baik"
 
    combined    = (
        st.session_state["title_input"] + " " +
        st.session_state["content_input"]
    ).strip()
    teks_kosong = len(combined) < 5
 
    if teks_kosong:
        label_teks    = None
        mode_sentimen = "kosong"
    else:
        clean         = preprocess_text(combined, stop_words, stemmer)
        tfidf_vec     = tfidf_vectorizer.transform([clean]).toarray()
        pred_teks     = model_teks.predict(tfidf_vec)[0]
        label_teks    = "Positif" if str(pred_teks).lower() == "positif" else "Negatif"
        mode_sentimen = "historis" if ada_sentimen else "manual"
 
    st.markdown("---")
    st.markdown("### 📋 Hasil Prediksi")
 
    r1, r2 = st.columns(2)
 
    with r1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📊 Analisis Laporan Keuangan</div>', unsafe_allow_html=True)
        css_n  = "r-good" if label_num == "Baik" else "r-bad"
        icon_n = "✅" if label_num == "Baik" else "⚠️"
        st.markdown(
            f'<div class="result-box {css_n}"><b>{icon_n} Prediksi Keuangan: {label_num}</b><br>'
            f'Perusahaan <b>{name}</b> tahun <b>{year}</b> memiliki kondisi keuangan yang '
            f'<b>{"baik" if label_num == "Baik" else "kurang baik"}</b>.</div>',
            unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
 
    with r2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📰 Analisis Sentimen Berita</div>', unsafe_allow_html=True)
 
        if mode_sentimen == "kosong":
            st.markdown(
                '<div class="result-box r-neutral">⏭️ <b>Prediksi Sentimen Dilewati</b><br>'
                'Tidak ada teks berita yang diinput. '
                'Isi kolom Judul & Isi Berita untuk analisis sentimen.</div>',
                unsafe_allow_html=True)
        elif mode_sentimen == "manual":
            css_t  = "r-good" if label_teks == "Positif" else "r-bad"
            icon_t = "✅" if label_teks == "Positif" else "⚠️"
            st.markdown(
                f'<div class="result-box {css_t}"><b>{icon_t} Sentimen: {label_teks}</b> '
                f'<span style="font-size:0.8rem;opacity:0.75;">(dari teks input manual)</span><br>'
                f'Hasil berdasarkan berita yang Anda masukkan.</div>',
                unsafe_allow_html=True)
            st.markdown(
                '<div class="result-box r-neutral" style="margin-top:0.35rem;">'
                '💡 Dataset sentimen mencakup <b>18 dari 266</b> perusahaan. '
                'Tambahkan berita ke <code>hasil_sentimen_saham.xlsx</code> dan latih ulang '
                'model untuk akurasi lebih tinggi.</div>',
                unsafe_allow_html=True)
        else:
            css_t  = "r-good" if label_teks == "Positif" else "r-bad"
            icon_t = "✅" if label_teks == "Positif" else "⚠️"
            st.markdown(
                f'<div class="result-box {css_t}"><b>{icon_t} Sentimen: {label_teks}</b> '
                f'<span style="font-size:0.8rem;opacity:0.75;">(kode: {kode})</span><br>'
                f'Dipandang <b>{"baik" if label_teks == "Positif" else "kurang baik"}</b> '
                f'berdasarkan analisis berita.</div>',
                unsafe_allow_html=True)
 
        st.markdown("</div>", unsafe_allow_html=True)
 
    st.markdown("---")
    st.markdown("### 🏦 Kesimpulan Rekomendasi Investasi")
 
    if mode_sentimen == "kosong":
        if label_num == "Baik":
            st.markdown(
                '<div class="result-box r-risk">⚠️ <b>Kondisi keuangan baik, namun sentimen belum dianalisis.</b><br>'
                'Masukkan data berita untuk rekomendasi yang lebih lengkap.</div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="result-box r-no">❌ <b>Kondisi keuangan kurang baik. Sentimen belum dianalisis.</b><br>'
                'Tidak direkomendasikan untuk berinvestasi berdasarkan data yang tersedia.</div>',
                unsafe_allow_html=True)
    elif label_num == "Baik" and label_teks == "Positif":
        catatan = " <i>(sentimen berdasarkan input manual)</i>" if mode_sentimen == "manual" else ""
        st.markdown(
            f'<div class="result-box r-invest">✅ <b>Cocok untuk berinvestasi pada perusahaan ini.</b><br>'
            f'Kondisi keuangan sehat dan sentimen pasar positif.{catatan}</div>',
            unsafe_allow_html=True)
    elif label_num == "Baik" or label_teks == "Positif":
        catatan = " Sentimen berdasarkan input manual." if mode_sentimen == "manual" else ""
        st.markdown(
            f'<div class="result-box r-risk">⚠️ <b>Terdapat risiko dalam berinvestasi.</b><br>'
            f'Salah satu indikator menunjukkan sinyal negatif — pertimbangkan dengan cermat.{catatan}</div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="result-box r-no">❌ <b>Tidak direkomendasikan untuk berinvestasi.</b><br>'
            'Kondisi keuangan dan sentimen berita keduanya menunjukkan sinyal negatif.</div>',
            unsafe_allow_html=True)
 
    st.markdown("---")
    st.markdown("### 📉 Tren Rasio Keuangan (2019–2024)")
 
    fig_tren = buat_grafik_tren(name)
    if fig_tren is not None:
        st.plotly_chart(fig_tren, use_container_width=True)
 
        df_tren    = load_tren(name)
        kolom_show = ["ROE", "Current Ratio", "DER", "PBV",
                      "Pertumbuhan Penjualan", "Ukuran Perusahaan"]
        kolom_ada  = [c for c in kolom_show if c in df_tren.columns]
        df_tampil  = df_tren[kolom_ada].copy()
 
        if "Pertumbuhan Penjualan" in df_tampil.columns:
            df_tampil["Pertumbuhan Penjualan"] = df_tampil["Pertumbuhan Penjualan"].map(
                lambda x: f"{x*100:.2f}%"
            )
        for c in ["ROE", "Current Ratio", "DER", "PBV", "Ukuran Perusahaan"]:
            if c in df_tampil.columns:
                df_tampil[c] = df_tampil[c].map(lambda x: f"{x:.4f}")
 
        st.caption("Tabel ringkas rasio keuangan per tahun")
        st.dataframe(df_tampil, use_container_width=True)
    else:
        st.info(
            f"📭 Data tren tidak tersedia untuk **{name}**. "
            "Perusahaan ini tidak ditemukan dalam `dc_dataCustom.xlsx` periode 2019–2024."
        )
