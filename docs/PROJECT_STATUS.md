# Proje Durumu – Türkçe TTS PoC

**Son güncelleme:** 2026-09-30 · **Son commit:** `3ec2501` · **Repo:** https://github.com/cansuyildirimmm/turkish-tts-poc

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
| 5 Dataset araştırması | ✅ | Ticari kullanıma uygun tek konuşmacılı hazır Türkçe dataset yok → **şirket-içi pilot kayıt** kararı. `docs/dataset_report.md` |
| 6 Dataset validation | 🟡 Kod hazır | Pilot kayıt bekleniyor |
| 7 Train/val/test split | 🟡 Kod hazır | Pilot kayıt bekleniyor |
| 8 Fine-tuning | 🟡 Kod hazır, **training başlatılmadı** | Kayıt + GPU ortamı + parametre onayı bekleniyor |
| 9 Base vs fine-tuned | 🟡 Script hazır | Base ile test edildi (`comparison/`) |
| 10 FastAPI | ⏳ | Model kalitesi onaylanınca |
| 11 Docker | ⏳ | |
| 12 Azure/ERP mimarisi | ⏳ | Sadece öneri; Azure servis tipi henüz bilinmiyor |

## Şu an beklenenler (kullanıcı tarafı)

1. **Hukuki ön koşullar:** KVKK aydınlatma metni, açık rıza, hak devri
   sözleşmesi, sesli rıza beyanı (`docs/recording_guide.md` bölüm 0).
2. **Pilot kayıt:** 420 cümle (`recording/pilot_script.tsv`, `read_text`
   sütunu), 48 kHz / 24-bit mono WAV → `datasets/pilot/wavs/<id>.wav`.
3. **GPU ortamı seçimi:** Öneri: şirketin Azure hesabında geçici GPU VM
   (veri şirket içinde kalır). Alternatif: kiralık GPU (KVKK değerlendirmesi
   gerekir) veya şirket içi NVIDIA makine.
4. ~~Kısaltma okunuşları~~ ✅ 2026-10-01'de onaylandı: ERP "e re pe",
   CRM "si ar em", KPI "ke pe i", PDF "pe de ef". CRM `outputs/base/27.wav`
   dinlenerek teyit edildi (önceki "ce er me" duyumu normalize edilmemiş
   `base_raw` çıktısından).

## Kayıtlar gelince yapılacaklar (sırayla)

```powershell
python src/validate_dataset.py      # rapor: reports/dataset_report.txt/.json -> kullanıcıya göster
python src/split_dataset.py         # datasets/pilot/splits/*.jsonl + split_info.json
python src/prepare_latents.py       # datasets/pilot/latents/  (GPU makinesinde daha hızlı)
python src/finetune.py --config configs/finetune_pilot.json --plan   # parametreleri göster, ONAY AL
python src/finetune.py --config configs/finetune_pilot.json          # sadece onaydan sonra
python src/compare_models.py --fine-tuned checkpoints/pilot_sft/best
```

Önerilen pilot parametreleri (`configs/finetune_pilot.json`, henüz onaylanmadı):
batch 16, lr 5e-5, 1500 adım (~65 epoch), warmup 100 + cosine, AdamW
(0.9, 0.95) wd 0.01, grad clip 1.0, bf16, λ_dur 0.1, val her 50 / ckpt her
250 adım, early stopping patience 6. Tahmini VRAM ~7 GB, A10/4090'da ~10–20 dk
(kaba tahmin). ~25 dk veriyle overfitting riski → best checkpoint + early stopping.

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
- Testler: `python -m pytest tests` → 122 test geçiyor.

## Dokümanlar

- `README.md` – kurulum ve tüm komutlar
- `docs/dataset_report.md` – FAZ 5 dataset/lisans raporu
- `docs/recording_guide.md` – pilot kayıt rehberi + KVKK kontrol listesi
- `evaluation/manual_evaluation.csv` – dinleme değerlendirme formu (kullanıcı doldurur)
