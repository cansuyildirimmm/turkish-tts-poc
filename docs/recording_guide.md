# Pilot Ses Kaydı Rehberi

Pilot kaydın amacı, FAZ 6–9 pipeline'ını (doğrulama, bölme, fine-tuning,
karşılaştırma) gerçek ve hakları temiz bir veriyle test etmektir. Sonuç iyi
olursa aynı rehberle asıl stüdyo kaydına geçilir.

**Kapsam:** 420 cümle, tahmini ~24 dk net konuşma (duraklamalar ve tekrarlarla
~1,5–2 saat kayıt süresi, 2 oturuma bölünebilir).

## 0. Kayıttan ÖNCE (zorunlu)

Kayıt, aşağıdakiler tamamlanmadan başlamamalıdır. Metinlerin son hali hukuk
birimince hazırlanmalıdır.

- [ ] **KVKK aydınlatma metni** konuşmacıya verildi (amaç: sentetik ses modeli
      geliştirme; saklama süresi; aktarım yapılmayacağı; haklar).
- [ ] **Açık rıza** yazılı ve ayrı bir belge olarak alındı (sesin işlenmesi,
      model eğitimi ve modelin şirket ürünlerinde kullanımı).
- [ ] **Hak devri / kullanım sözleşmesi** imzalandı (kayıtlar ve bağlantılı
      haklar; kapsam, süre, bölge).
- [ ] **Rızanın geri çekilmesi** durumunda kayıtların ve modelin ne olacağı
      yazılı olarak belirlendi.
- [ ] Sesin **taklit/yanıltma amaçlı kullanılmayacağı** taahhüt edildi.
- [ ] Konuşmacı ilk kayıt olarak **sesli rıza beyanını** okudu
      (`consent.wav`, aşağıdaki örnek metin hukukça uyarlanmalıdır):

  > "Ben, [ad soyad], [tarih] tarihinde, [şirket adı] tarafından sentetik ses
  > modeli geliştirilmesi amacıyla ses kayıtlarımın kullanılmasına, imzaladığım
  > açık rıza metni kapsamında onay veriyorum."

## 1. Ortam

- Küçük, eşyalı (halı, perde, dolap) ve **sessiz** bir oda. Boş, yankılı
  odalardan kaçının.
- Klima, vantilatör, bilgisayar fanı gibi sürekli gürültü kaynakları kapalı.
- Telefon sessizde; kapıya "kayıt var" notu.
- Tüm oturumlarda **aynı oda, aynı mikrofon, aynı mesafe, aynı ayarlar**.

## 2. Ekipman

- Harici mikrofon (USB kondenser veya dinamik, kardioid). **Laptop dahili
  mikrofonu kullanmayın.**
- Pop filtresi; ağız–mikrofon mesafesi ~15–20 cm, sabit.
- Kapalı kulaklık (seviyeyi dinlemek için).

## 3. Kayıt ayarları (ör. Audacity, ücretsiz)

| Ayar | Değer |
|---|---|
| Sample rate | **48000 Hz** |
| Kanal | **Mono** |
| Bit derinliği | **24-bit** PCM (en az 16-bit) |
| Dosya | WAV |
| Seviye | En yüksek noktalar −6 ile −3 dBFS arası; **kırpılma (clipping) olmamalı** |
| Efekt | **Gürültü azaltma, kompresör, EQ, normalize UYGULAMAYIN** (ham kayıt) |

## 4. Okuma

- Okunacak metin: `recording/pilot_script.tsv` dosyasındaki **`read_text`**
  sütunu. (Sayılar ve kısaltmalar yazıyla verilmiştir; tam olarak yazıldığı
  gibi okuyun, ör. "yüzde on beş", "Ka de ve".)
- Doğal, sakin, **sabit tempo ve ton**; haber spikeri gibi abartılı değil.
- Soru ve ünlem cümlelerinde doğal tonlama.
- Her dosyada **tek cümle**; başta ve sonda ~0,2 sn sessizlik.
- Yanlış okuma, takılma, öksürük, sayfa sesi olursa **cümleyi baştan
  tekrar** kaydedin; hatalı versiyonu saklamayın.
- Her oturumun başında `g001`–`g005` cümlelerini "referans" olarak okuyun ve
  önceki oturumla ton/seviye olarak eşleştirin (bu referans kayıtları
  `ref_<oturum>_<id>.wav` adıyla ayrıca saklayın).
- 45–60 dakikada bir mola verin; ses yorgunluğu kaliteyi düşürür.

## 5. Dosya adlandırma ve teslim

```
datasets/pilot/
├── wavs/
│   ├── g001.wav
│   ├── g002.wav
│   └── ...            (dosya adı = pilot_script.tsv'deki id)
├── consent.wav
└── session_notes.txt  (tarih, oda, mikrofon modeli, ayarlar, okunamayan id'ler)
```

- Okunamayan/atlanan cümleler sorun değildir; `session_notes.txt`'ye yazın.
- `datasets/` klasörü **Git'e gönderilmez**. Kayıtlar kişisel veridir;
  yalnızca şirket içi yetkili ve güvenli depolamada paylaşılmalıdır.

## 6. Sonraki adım

Kayıtlar `datasets/pilot/` altına konduktan sonra FAZ 6 doğrulama
pipeline'ı çalıştırılır: eksik/bozuk WAV, sample rate, kanal, süre, sessizlik,
kırpılma, transkript eşleşmesi vb. kontrol edilir ve
`reports/dataset_report.txt` / `.json` üretilir. Veri sessizce değiştirilmez
veya silinmez.
