"""
Sistem Rekomendasi Buku - Item-Based Collaborative Filtering + GUI Tkinter
==========================================================================
Hanya memakai library standar Python (tkinter, math, random), tanpa instalasi tambahan.

Cara kerja singkat (item-based CF):
1. Bangun matriks rating pengguna x buku.
2. Hitung kemiripan antar BUKU dengan Adjusted Cosine Similarity
   (rating dikurangi rata-rata rating tiap pengguna, supaya bias
   "pengguna pelit/murah hati memberi nilai" hilang).
3. Prediksi rating buku B untuk pengguna U = rata-rata terbobot dari
   rating U pada buku-buku lain yang paling mirip dengan B.
4. Rekomendasikan buku yang belum dibaca dengan prediksi tertinggi.

Jalankan:  python rekomendasi_buku.py
"""

import math
import random
import tkinter as tk
from tkinter import ttk, messagebox

# ----------------------------------------------------------------------------
# 1. DATA CONTOH
# ----------------------------------------------------------------------------
BUKU = {
    # Fiksi Indonesia
    1: ("Laskar Pelangi", "Fiksi Indonesia"),
    2: ("Bumi Manusia", "Fiksi Indonesia"),
    3: ("Negeri 5 Menara", "Fiksi Indonesia"),
    4: ("Ayat-Ayat Cinta", "Fiksi Indonesia"),
    5: ("Cantik Itu Luka", "Fiksi Indonesia"),
    6: ("Pulang", "Fiksi Indonesia"),
    # Fantasi
    7: ("Harry Potter and the Philosopher's Stone", "Fantasi"),
    8: ("The Hobbit", "Fantasi"),
    9: ("Percy Jackson: The Lightning Thief", "Fantasi"),
    10: ("Eragon", "Fantasi"),
    11: ("The Lion, the Witch and the Wardrobe", "Fantasi"),
    12: ("A Game of Thrones", "Fantasi"),
    # Pengembangan diri & bisnis
    13: ("Atomic Habits", "Pengembangan Diri"),
    14: ("Filosofi Teras", "Pengembangan Diri"),
    15: ("Rich Dad Poor Dad", "Pengembangan Diri"),
    16: ("The 7 Habits of Highly Effective People", "Pengembangan Diri"),
    17: ("Mindset", "Pengembangan Diri"),
    18: ("Start with Why", "Pengembangan Diri"),
    # Sains & sejarah
    19: ("Sapiens", "Sains & Sejarah"),
    20: ("A Brief History of Time", "Sains & Sejarah"),
    21: ("Cosmos", "Sains & Sejarah"),
    22: ("Guns, Germs, and Steel", "Sains & Sejarah"),
    23: ("Homo Deus", "Sains & Sejarah"),
    24: ("The Selfish Gene", "Sains & Sejarah"),
}


def buat_data_rating(jumlah_user=30, seed=42):
    """Membuat data rating sintetis (1-5): tiap pengguna punya 1-2 genre favorit."""
    rnd = random.Random(seed)
    genre_list = sorted({g for _, g in BUKU.values()})
    ratings = {}
    for u in range(1, jumlah_user + 1):
        favorit = set(rnd.sample(genre_list, rnd.choice([1, 2])))
        r = {}
        for bid, (_, genre) in BUKU.items():
            suka = genre in favorit
            if rnd.random() < (0.75 if suka else 0.2):
                r[bid] = rnd.randint(4, 5) if suka else rnd.randint(1, 3)
        if len(r) >= 3:
            ratings[f"User {u:02d}"] = r
    return ratings


# ----------------------------------------------------------------------------
# 2. MODEL: ITEM-BASED COLLABORATIVE FILTERING
# ----------------------------------------------------------------------------
class ItemBasedCF:
    def __init__(self, ratings, min_common=2):
        self.ratings = ratings            # {user: {item: rating}}
        self.min_common = min_common      # minimal pengguna yang menilai kedua buku
        self.user_mean = {u: sum(r.values()) / len(r) for u, r in ratings.items()}

        # Indeks terbalik: item -> {user: rating}
        self.item_users = {}
        for u, r in ratings.items():
            for i, v in r.items():
                self.item_users.setdefault(i, {})[u] = v

        self.sim = {}                     # {item: {item_lain: kemiripan}}
        self._hitung_similarity()

    def _hitung_similarity(self):
        items = list(self.item_users)
        for a in range(len(items)):
            for b in range(a + 1, len(items)):
                i, j = items[a], items[b]
                common = self.item_users[i].keys() & self.item_users[j].keys()
                if len(common) < self.min_common:
                    continue
                num = den_i = den_j = 0.0
                for u in common:
                    di = self.item_users[i][u] - self.user_mean[u]
                    dj = self.item_users[j][u] - self.user_mean[u]
                    num += di * dj
                    den_i += di * di
                    den_j += dj * dj
                if den_i == 0 or den_j == 0:
                    continue
                s = num / (math.sqrt(den_i) * math.sqrt(den_j))
                # Shrinkage: kurangi kepercayaan bila sedikit data bersama
                s *= len(common) / (len(common) + 5)
                self.sim.setdefault(i, {})[j] = s
                self.sim.setdefault(j, {})[i] = s

    def buku_mirip(self, item, n=5):
        """Daftar n buku paling mirip dengan `item`: [(item_id, skor)]."""
        hasil = sorted(self.sim.get(item, {}).items(), key=lambda x: -x[1])
        return hasil[:n]

    def prediksi(self, rating_user, item, k=10):
        """Prediksi rating `item` dari rating_user = {item: rating}. None bila tak bisa."""
        tetangga = [
            (self.sim[item][j], r)
            for j, r in rating_user.items()
            if j in self.sim.get(item, {}) and self.sim[item][j] > 0
        ]
        if not tetangga:
            return None
        tetangga.sort(key=lambda x: -x[0])
        tetangga = tetangga[:k]
        pembagi = sum(s for s, _ in tetangga)
        return sum(s * r for s, r in tetangga) / pembagi

    def rekomendasi(self, rating_user, n=5, k=10):
        """Top-n buku yang belum dinilai: [(item_id, prediksi)]."""
        kandidat = []
        for item in BUKU:
            if item in rating_user:
                continue
            p = self.prediksi(rating_user, item, k)
            if p is not None:
                kandidat.append((item, p))
        kandidat.sort(key=lambda x: -x[1])
        return kandidat[:n]


# ----------------------------------------------------------------------------
# 3. GUI TKINTER
# ----------------------------------------------------------------------------
class AplikasiRekomendasi(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Sistem Rekomendasi Buku - Item-Based Collaborative Filtering")
        self.geometry("820x520")
        self.minsize(720, 460)

        self.ratings = buat_data_rating()
        self.model = ItemBasedCF(self.ratings)
        self.rating_baru = {}             # untuk tab pengguna baru

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=8, pady=8)
        self._tab_user(nb)
        self._tab_mirip(nb)
        self._tab_baru(nb)

        ttk.Label(
            self, foreground="gray",
            text=f"Data contoh: {len(self.ratings)} pengguna, {len(BUKU)} buku, "
                 f"{sum(len(r) for r in self.ratings.values())} rating (sintetis)."
        ).pack(pady=(0, 6))

    # ---------- helper ----------
    @staticmethod
    def _tabel(parent, kolom, lebar):
        frame = ttk.Frame(parent)
        tv = ttk.Treeview(frame, columns=[c[0] for c in kolom], show="headings", height=12)
        for (cid, judul), w in zip(kolom, lebar):
            tv.heading(cid, text=judul)
            tv.column(cid, width=w, anchor="w" if cid in ("judul", "genre") else "center")
        sb = ttk.Scrollbar(frame, orient="vertical", command=tv.yview)
        tv.configure(yscrollcommand=sb.set)
        tv.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        return frame, tv

    @staticmethod
    def _isi(tv, baris):
        tv.delete(*tv.get_children())
        for b in baris:
            tv.insert("", "end", values=b)

    # ---------- Tab 1: rekomendasi untuk pengguna yang sudah ada ----------
    def _tab_user(self, nb):
        tab = ttk.Frame(nb, padding=10)
        nb.add(tab, text="Rekomendasi Pengguna")

        bar = ttk.Frame(tab)
        bar.pack(fill="x")
        ttk.Label(bar, text="Pengguna:").pack(side="left")
        self.cb_user = ttk.Combobox(bar, values=sorted(self.ratings), state="readonly", width=12)
        self.cb_user.current(0)
        self.cb_user.pack(side="left", padx=6)
        ttk.Label(bar, text="Jumlah rekomendasi:").pack(side="left", padx=(10, 0))
        self.sp_n1 = ttk.Spinbox(bar, from_=1, to=15, width=4)
        self.sp_n1.set(5)
        self.sp_n1.pack(side="left", padx=6)
        ttk.Button(bar, text="Tampilkan", command=self._aksi_user).pack(side="left", padx=10)

        isi = ttk.Frame(tab)
        isi.pack(fill="both", expand=True, pady=10)
        kiri = ttk.LabelFrame(isi, text="Buku yang sudah dinilai", padding=6)
        kanan = ttk.LabelFrame(isi, text="Rekomendasi untuk pengguna ini", padding=6)
        kiri.pack(side="left", fill="both", expand=True, padx=(0, 5))
        kanan.pack(side="left", fill="both", expand=True, padx=(5, 0))

        f1, self.tv_dinilai = self._tabel(
            kiri, [("judul", "Judul"), ("rating", "Rating")], [200, 55])
        f1.pack(fill="both", expand=True)
        f2, self.tv_rek1 = self._tabel(
            kanan, [("judul", "Judul"), ("genre", "Genre"), ("skor", "Prediksi")], [170, 100, 60])
        f2.pack(fill="both", expand=True)
        self._aksi_user()

    def _aksi_user(self):
        user = self.cb_user.get()
        n = int(self.sp_n1.get())
        r = self.ratings[user]
        self._isi(self.tv_dinilai,
                  [(BUKU[i][0], v) for i, v in sorted(r.items(), key=lambda x: -x[1])])
        rek = self.model.rekomendasi(r, n)
        self._isi(self.tv_rek1, [(BUKU[i][0], BUKU[i][1], f"{p:.2f}") for i, p in rek])

    # ---------- Tab 2: buku mirip ----------
    def _tab_mirip(self, nb):
        tab = ttk.Frame(nb, padding=10)
        nb.add(tab, text="Buku Mirip")

        bar = ttk.Frame(tab)
        bar.pack(fill="x")
        ttk.Label(bar, text="Pilih buku:").pack(side="left")
        self.judul_ke_id = {f"{j}": i for i, (j, _) in BUKU.items()}
        self.cb_buku = ttk.Combobox(bar, values=list(self.judul_ke_id), state="readonly", width=42)
        self.cb_buku.current(0)
        self.cb_buku.pack(side="left", padx=6)
        ttk.Label(bar, text="Jumlah:").pack(side="left", padx=(10, 0))
        self.sp_n2 = ttk.Spinbox(bar, from_=1, to=15, width=4)
        self.sp_n2.set(5)
        self.sp_n2.pack(side="left", padx=6)
        ttk.Button(bar, text="Cari", command=self._aksi_mirip).pack(side="left", padx=10)

        f, self.tv_mirip = self._tabel(
            tab, [("judul", "Judul"), ("genre", "Genre"), ("sim", "Kemiripan")], [350, 150, 90])
        f.pack(fill="both", expand=True, pady=10)
        self._aksi_mirip()

    def _aksi_mirip(self):
        item = self.judul_ke_id[self.cb_buku.get()]
        hasil = self.model.buku_mirip(item, int(self.sp_n2.get()))
        self._isi(self.tv_mirip, [(BUKU[i][0], BUKU[i][1], f"{s:.3f}") for i, s in hasil])

    # ---------- Tab 3: pengguna baru (cold start ringan) ----------
    def _tab_baru(self, nb):
        tab = ttk.Frame(nb, padding=10)
        nb.add(tab, text="Pengguna Baru")

        bar = ttk.Frame(tab)
        bar.pack(fill="x")
        ttk.Label(bar, text="Buku:").pack(side="left")
        self.cb_buku3 = ttk.Combobox(bar, values=list(self.judul_ke_id), state="readonly", width=38)
        self.cb_buku3.current(0)
        self.cb_buku3.pack(side="left", padx=6)
        ttk.Label(bar, text="Rating:").pack(side="left")
        self.cb_rating = ttk.Combobox(bar, values=[1, 2, 3, 4, 5], state="readonly", width=3)
        self.cb_rating.set(5)
        self.cb_rating.pack(side="left", padx=6)
        ttk.Button(bar, text="Tambah", command=self._tambah_rating).pack(side="left", padx=4)
        ttk.Button(bar, text="Reset", command=self._reset_rating).pack(side="left", padx=4)

        isi = ttk.Frame(tab)
        isi.pack(fill="both", expand=True, pady=10)
        kiri = ttk.LabelFrame(isi, text="Rating Anda", padding=6)
        kanan = ttk.LabelFrame(isi, text="Rekomendasi", padding=6)
        kiri.pack(side="left", fill="both", expand=True, padx=(0, 5))
        kanan.pack(side="left", fill="both", expand=True, padx=(5, 0))

        f1, self.tv_baru = self._tabel(
            kiri, [("judul", "Judul"), ("rating", "Rating")], [200, 55])
        f1.pack(fill="both", expand=True)
        f2, self.tv_rek3 = self._tabel(
            kanan, [("judul", "Judul"), ("genre", "Genre"), ("skor", "Prediksi")], [170, 100, 60])
        f2.pack(fill="both", expand=True)
        ttk.Button(kanan, text="Rekomendasikan", command=self._aksi_baru).pack(pady=(6, 0))

    def _tambah_rating(self):
        self.rating_baru[self.judul_ke_id[self.cb_buku3.get()]] = int(self.cb_rating.get())
        self._isi(self.tv_baru, [(BUKU[i][0], v) for i, v in self.rating_baru.items()])

    def _reset_rating(self):
        self.rating_baru.clear()
        self._isi(self.tv_baru, [])
        self._isi(self.tv_rek3, [])

    def _aksi_baru(self):
        if len(self.rating_baru) < 2:
            messagebox.showinfo("Info", "Beri rating minimal 2 buku terlebih dahulu.")
            return
        rek = self.model.rekomendasi(self.rating_baru, 8)
        if not rek:
            messagebox.showinfo("Info", "Belum ada rekomendasi. Coba beri rating pada buku lain.")
        self._isi(self.tv_rek3, [(BUKU[i][0], BUKU[i][1], f"{p:.2f}") for i, p in rek])


if __name__ == "__main__":
    AplikasiRekomendasi().mainloop()
