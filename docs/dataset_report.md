# FAZ 5 – Türkçe TTS Dataset Araştırması

**Tarih:** 2026-09-30
**Karar:** Şirket-içi pilot kayıt ile devam edilecek (bkz. [recording_guide.md](recording_guide.md)).

> Bu doküman teknik bir ön incelemedir, hukuki görüş değildir. Seçilen veri
> kaynağı ve sözleşmeler üretim öncesinde hukuk birimince onaylanmalıdır.
> "Hugging Face'te bulunuyor" ifadesi ticari kullanım izni olarak kabul edilmemiştir.

## 1. Fine-tuning gereksinimi

FreyaTTS-small'un fine-tuning aşaması ("voice lock" SFT) **tek konuşmacılı** bir
korpus üzerinde yapılır. Model konuşmacı koşullaması içermez; ses, başlangıç
gürültüsünden gelir ve SFT modeli hedef sese kilitler. Bu nedenle:

- Çok konuşmacılı datasetler voice-lock için uygun değildir.
- Fine-tuning sonrası model, eğitildiği konuşmacının sesiyle konuşur.
- Gözlemlenen "cümle sonunda erkek sesine kayma" sorununun mimari çözümü,
  tutarlı tek konuşmacılı veriyle SFT'dir (FAZ 9'da ölçülecek, garanti değil).

**Format:** JSONL manifest `{"audio": yol, "text": metin}`, 1–14 sn klipler,
herhangi bir sample rate (16 kHz'e çevrilir; çıktı 48 kHz olduğundan kaynak
kalite önemlidir). Metin modelin 92 karakterlik sözlüğüne uymalıdır.

## 2. Karşılaştırma

### Teknik olarak uygun, lisans/haklar engel

| Dataset | Lisans | Ticari | Kapsam | Not |
|---|---|---|---|---|
| [marytts/dfki-ot-data](https://github.com/marytts/dfki-ot-data) | CC BY-NC-SA 4.0 | Hayır | ~1,6 sa, tek erkek, 44,1 kHz stüdyo | Teknik olarak ideal, ticari olmayan kullanımla sınırlı |
| [Anilosan15/Turkish_TTS_Data](https://huggingface.co/datasets/Anilosan15/Turkish_TTS_Data) | Yok | Hayır | 30,6k klip, tek kadın, 48 kHz | Kaynak ve konuşmacı rızası belirsiz |
| [erenfazlioglu/turkishvoicedataset](https://huggingface.co/datasets/erenfazlioglu/turkishvoicedataset) | CC BY-NC 4.0 | Hayır | 130k klip | **Sentetik** (Microsoft TTS); Microsoft şartları çıktıların başka AI eğitiminde kullanımını yasaklar |

### Hukuki riski yüksek, kullanılmamalı

- Sesli kitap / YouTube kaynaklı setler (Mazlum Kiper seti,
  serdarcaglar/turkish-tts-audiobooks, Mert-H): bağlantılı haklar (5846 s.
  Kanun), kişilik hakları, KVKK (ses kişisel veri; tanımlama amaçlı
  kullanımda biyometrik veri).
- Sentetik setler (Taklaxbr, SynDataLab omnivoice-tr, VoiceHub): kaynak modelin
  kullanım şartları veya klonlanan kişilerin hakları belirsiz.

### Lisans uygun, voice-lock için uygun değil

| Dataset | Lisans | Ticari | Sorun |
|---|---|---|---|
| [ISSAI TSC](https://issai.nu.edu.kz/turkic-asr/) (218 sa) | CC BY 4.0 (HF README: MIT, çelişkili) | Muhtemelen | Çok konuşmacılı yayın sesi, 16 kHz; kaynak kayıtların hakları belirsiz |
| [FLEURS tr](https://huggingface.co/datasets/google/fleurs) | CC BY 4.0 | Evet | Çok konuşmacılı, 16 kHz, ~12 sa |
| Common Voice TR (136 sa) | CC0 | Kısıtlı | 1.829 konuşmacı; şartlar konuşmacı kimliğinin tespitini yasaklar |
| [MediaSpeech TR](https://www.openslr.org/108/) | CC BY 4.0 | Belirsiz | YouTube kaynaklı, telif video sahiplerinde |
| LDC (ODTÜ Mikrofon, Broadcast News) | LDC sözleşmesi | Ücretli üyelikle | Çok konuşmacılı, 16 kHz |
| [YODAS2](https://huggingface.co/datasets/espnet/yodas2) | CC BY 3.0 | Belirsiz | YouTube, konuşmacı rızası yok |
| LibriVox Türkçe | Kamu malı | — | Yalnızca birkaç kısa şiir; Nâzım Hikmet Türkiye'de 2033'e kadar telifli |
| Emilia, Granary, VoxPopuli, MLS | — | — | Türkçe yok |

### Satın alınabilir (doğrulanmadı)

- Appen 700 sa Türkçe TTS (HF'de örnek; ticari satın alma).
- Commencis Türkçe TTS kayıtları (HF'de erişim kısıtlı).

### Yalnızca değerlendirme

- [freyavoice/freya-tr-eval](https://huggingface.co/datasets/freyavoice/freya-tr-eval): CC BY 4.0, 495 cümle, yalnızca metin.
- FLEURS tr.
- Bu cümleler ve `evaluation/sentences.txt` kayıt senaryolarına dahil edilmemelidir.

## 3. Öneri: şirket-içi kayıt

| Konu | Öneri |
|---|---|
| Hacim | Pilot: ~30–60 dk. Üretim: 2–5 sa (~1.500–4.000 klip). Microsoft: min. 300, önerilen 1.000–2.000 cümle |
| Format | 48 kHz / 24-bit mono WAV, SNR > 35 dB, tepe −6…−3 dBFS |
| Senaryo | ~%50 genel, ~%50 ERP (terim, sayı, tarih, para, kısaltma); tüm Türkçe harfler; soru ve ünlem cümleleri |
| Hukuk | Hak devri sözleşmesi (bağlantılı haklar dahil), KVKK açık rıza + aydınlatma metni, rıza geri çekilmesi hükmü, taklit amaçlı kullanım yasağı, sesli rıza beyanı |
| Maliyet (doğrulanmamış) | ~3–6 hafta, ~5–20 bin USD (en büyük kalem seslendirme sanatçısı hak devri bedeli) |

## 4. Açık konular

- ISSAI TSC lisans çelişkisi çözülmedi.
- Appen / Commencis koşulları doğrulanmadı.
- Maliyet/süre tahminleri doğrulanmadı.
- Voice-lock SFT'nin ses kaymasını çözdüğü henüz kanıtlanmadı.
