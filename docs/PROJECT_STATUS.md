# Proje Durumu – Türkçe TTS PoC

**Son güncelleme:** 2026-10-09 · **Son commit:** bu dosyayı içeren commit · **Repo:** https://github.com/cansuyildirimmm/turkish-tts-poc

Bu dosya, çalışmaya yeni bir oturumda kaldığı yerden devam edebilmek için
tutulur. Her faz sonunda güncellenmelidir.

## Çalışma kuralları (özet)

- Fazlar sırayla ilerler; **her faz sonunda durulur ve onay beklenir**.
- Training, **parametreler gösterilip onay alınmadan başlatılmaz**.
- Harici TTS API'si kullanılmaz; inference tamamen lokal/offline.
- ERP projesine dokunulmaz; model/dataset lisansları doğrulanır.
- Büyük dosyalar (model, ses, dataset, checkpoint) Git'e girmez.
- Raporlar Türkçe; commit/push kullanıcı onayından sonra yapılır.

## Faz durumu

| Faz | Durum | Not |
|---|---|---|
| 0 Sistem analizi | ✅ | Win 11, Ryzen 5 5600H, 16 GB RAM, **NVIDIA GPU yok** (AMD RX 6500M 4 GB, CUDA yok). CPU inference mümkün; fine-tuning lokalde mantıklı değil |
| 1 Model araştırması | ✅ | Seçilen: **FreyaTTS-small** (Apache-2.0, 183M, Türkçe native, CPU'da çalışır). Yedek: Chatterbox Multilingual (MIT). XTTS-v2, F5, MMS, Piper TR sesleri ticari değil |
| 2 Base inference | ✅ | Offline, sabit revizyonlu indirme, SHA-256 doğrulama |
| 3 ERP evaluation set | ✅ | `evaluation/sentences.txt` (32 cümle, unseen) |
| 4 Text normalization | ✅ | `src/text_normalizer.py`, ek uyumu dahil |
| 5 Dataset araştırması | ✅ | Ticari kullanıma uygun tek konuşmacılı hazır Türkçe dataset yok. Pilot kayıt kararı 2026-10-01'de **iptal** → sentetik veri. `docs/dataset_report.md` |
| 6 Dataset validation | 🟡 Kod hazır | Sentetik veri üretimi: `src/generate_synthetic.py`, 20 cümlelik deneme başarılı (2026-10-08) |
| 7 Train/val/test split | 🟡 Kod hazır | Sentetik veriye uyarlanacak |
| 8 Fine-tuning | 🟡 Kod hazır, **training başlatılmadı** | Sentetik veri + CPU parametreleri + onay bekleniyor |
| 9 Base vs fine-tuned | 🟡 Script hazır | Base ile test edildi (`comparison/`) |
| 10 FastAPI | ⏳ | Model kalitesi onaylanınca |
| 11 Docker | ⏳ | |
| 12 Azure/ERP mimarisi | ⏳ | Sadece öneri; Azure servis tipi henüz bilinmiyor |

## 2026-10-01 kararları (YÖN DEĞİŞİKLİĞİ)

- **Pilot kayıt YAPILMAYACAK.** Kullanıcı ses kaydetmek istemiyor; kayıt
  ortamı da yok. Gerçek insan sesi olmadığı için KVKK/rıza belgeleri gerekmez.
- **Şirket şartları:** (1) ücretli hiçbir şey yok, (2) yeni yabancı
  kod/model/üreticiye bağımlılık yok.
- Bu nedenle elenenler: Chatterbox (ek model), profesyonel seslendirme
  (ücretli), Azure/kiralık GPU (ücretli).
- **Seçilen yol (C): sentetik veriyle voice-lock fine-tuning.** FreyaTTS-small
  kendi çıktılarıyla eğitilir; kayma kontrolü (FAZ 4) ile sesi sabit kalan
  örnekler otomatik seçilir. Sadece mevcut yerel FreyaTTS kopyası kullanılır.
- **Eğitim yeri: bu bilgisayarın CPU'su** (ücretsiz). Süre bilinmiyor
  (kaba: birkaç saat – bir gece); başlamadan önce kısa hız testi yapılacak.
- Beklenti: kalite base'i geçmez; hedef sadece cümle sonu ses kaymasını
  gidermek. Başarı FAZ 9 karşılaştırmasıyla ölçülecek.
- Kısaltma okunuşları onaylandı: ERP "e re pe", CRM "si ar em", KPI "ke pe i",
  PDF "pe de ef" (CRM `outputs/base/27.wav` ile teyit edildi).

## 2026-10-08 oturumu (sentetik veri, Adım 1)

- Tasarım onaylandı: kaynak = 420 pilot cümle + **400 yeni uzun ERP cümlesi**
  (`recording/synthetic_sentences.txt` → `synthetic_script.tsv`, eval çakışması
  yok), tek ses (`LEYLA_SEED`=9), her parça ayrı üretilir, süre ölçekleri
  1.0/0.9/1.1/1.2 sırayla denenir. Kod: `src/generate_synthetic.py` (+12 test).
- **Deneme 1** (20 cümle, ölçüt "açılışa göre düşüş < 3 yarım ton"): 1/20 kabul.
  Reddedilenlerde düşüş 4,5–10 yarım ton.
- **Dinleme testi (kullanıcı):** 9–10 yarım ton düşen klipler (en düşük ~186 Hz)
  **aynı kadın sesi, sorun yok** → "açılışa göre düşüş" doğal cümle sonu
  tonlamasını kayma sanıyor. Önceki "22/32 kayma" bulgusu da bu ölçüte
  dayanıyordu, yani abartılıydı.
- Yeni ölçüt: **hiçbir 0,5 sn pencere < 180 Hz olmasın.** Base eval çıktılarında
  en düşük pencere çoğunlukla 170–210 Hz; belirgin uç değerler 96–130 Hz
  (uzun, parçalanan cümleler 31 ve 32).
- **Deneme 2** (aynı 20 cümle, 180 Hz): **20/20 kabul**, 1,34 dk ses, 2,3 sn
  işlem / 1 sn ses. Tam üretim tahmini ~50–55 dk ses, ~2–2,5 saat CPU.
  Çıktı `datasets/synthetic/`, ilk deneme `datasets/synthetic_trial_v1/`.
- Not: filtre artık neredeyse hiçbir şey elemiyor → fine-tuning kazancı sınırlı
  olabilir. Drift-guard da aynı hatalı düşüş ölçütünü kullanıyor (bu yüzden ~4x
  maliyet).

## 2026-10-09 oturumu (drift-guard ölçütü, B seçeneği)

- Drift-guard kabul ölçütü "düşüş < 4 yarım ton" yerine **"en düşük 0,5 sn
  pencere ≥ 180 Hz"** (`config.DRIFT_MIN_PITCH_HZ`, `voice_check.take_floor`).
  Sentetik üretim de aynı sabiti kullanıyor. 136 test geçiyor.
- **Eval ölçümü (32 cümle, CPU):**

  | | Guard kapalı | Guard (180 Hz) |
  |---|---|---|
  | < 180 Hz pencere | 12/32 | 4/32 (03, 26, 31, 32) |
  | Süre | 256 sn (RTF 1,74) | 492 sn (RTF 3,36, ~1,9x) |

  Çıktılar: `outputs/base_2026-10-09/`, `outputs/drift_guard_180/`.
- 32 (96 Hz) düzelmedi: 2. parçanın 4 denemesi de < 200 Hz açılıyor
  (kayma parçanın başında), ölçek değiştirmek işe yaramıyor. 26'nın son parçası da aynı.
- **Dinleme (kullanıcı):** base 31 sonda erkek sesine dönüyor; guard 31'de
  2. parça (ölçek 1.2) **başka bir kadın sesi**. Perde ölçütü konuşmacı
  değişimini göremez → **guard çözüm değil, varsayılan kapalı kalır.**
- **Dinleme (kullanıcı):** deneme 2'de 0.9/1.1 ölçekle kabul edilen tek
  parçalı klipler (g028, x182, x280, x321) **aynı ses** → sentetik üretim
  olduğu gibi kullanılabilir. 1.2 ölçekle kabul edilen klipler tam üretimden
  sonra ayrıca dinletilecek.
- Karar: **fine-tuning yolu.**

## Buradan devam (sıradaki adım)

1. Tam sentetik üretim: `python src/generate_synthetic.py` (~2–2,5 saat,
   kaldığı yerden devam eder). Başlatıldı: 2026-10-09.
2. 1.2 ölçekli kabul edilen klipleri kullanıcıya dinlet (konuşmacı değişimi?).
3. Doğrulama + split, CPU eğitim parametreleri (`finetune.py --plan`) →
   **ONAY** → eğitim → `compare_models.py`.

Eski pilot-kayıt akışı (referans için, artık kullanılmıyor):
`validate_dataset.py` → `split_dataset.py` → `prepare_latents.py` →
`finetune.py --config configs/finetune_pilot.json --plan` → `compare_models.py`.
`configs/finetune_pilot.json` GPU/gerçek kayıt varsayımıyla yazıldı; CPU ve
sentetik veri için yeniden ayarlanacak.

## Önemli teknik bulgular

- **Ses kayması:** FreyaTTS-small'da konuşmacı koşullaması yok (ses = seed).
  Cümle sonlarında perde ~1 oktav düşüp erkek sesine kayıyor (base: 22/32
  cümle). Kısa parça, adım sayısı, seed değişimi çözmüyor. `--drift-guard`
  (opsiyonel, varsayılan kapalı) 17/32'ye indiriyor, maliyet ~4x. Asıl çözüm
  beklentisi: tek konuşmacılı voice-lock fine-tuning (FAZ 9'da ölçülecek).
- **Upstream config uyuşmazlığı:** FreyaTTS training YAML'ları d=768/depth=22
  diyor, yayınlanan model d=640/depth=16. Upstream `strict=False` ile sessizce
  bozulurdu; `src/finetune.py` mimariyi init `config.json`'dan okur, strict yükler.
- **voxcpm:** Sadece AudioVAE kullanılıyor; paket `--no-deps` kurulu,
  `tts_engine._register_voxcpm_namespace()` ağır `__init__`'i atlatıyor.
- **Model sözlüğü (92 karakter):** `!`, `₺`, büyük `Ğ`, küçük `q` yok →
  `tts_engine.fit_to_vocab` eşliyor (`!`→`.`).
- **Kısa klipler:** Validation ve latent hazırlama alt sınırı 0.5 s
  (upstream varsayılanı 1.0 s; kısa ifadeler kaybolmasın diye).
- **CPU hızı:** RTF ~1.7 (Ryzen 5 5600H). İlk sentezde numba JIT ~4 s ek süre.

## Ortam

- Proje: `C:\Users\NY - Destek\Desktop\tts_modeli\turkish-tts-poc`
- Python 3.11.9 (kullanıcı kapsamı, `%LOCALAPPDATA%\Programs\Python\Python311`),
  venv: `.venv` (`.\.venv\Scripts\python.exe`)
- Model dosyaları `models/` (Git dışında); FreyaTTS kaynak kodu
  `third_party/FreyaTTS` @ `146d36c` (Git dışında). Yeni makinede kurulum: README.
- Testler: `python -m pytest tests` → 136 test geçiyor.

## Dokümanlar

- `README.md` – kurulum ve tüm komutlar
- `docs/dataset_report.md` – FAZ 5 dataset/lisans raporu
- `docs/recording_guide.md` – pilot kayıt rehberi + KVKK kontrol listesi
- `evaluation/manual_evaluation.csv` – dinleme değerlendirme formu (kullanıcı doldurur)
