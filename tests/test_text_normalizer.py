import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from text_normalizer import attach_suffix, number_to_words, normalize  # noqa: E402


@pytest.mark.parametrize("n, expected", [
    (0, "sıfır"),
    (1, "bir"),
    (10, "on"),
    (19, "on dokuz"),
    (100, "yüz"),
    (101, "yüz bir"),
    (200, "iki yüz"),
    (1000, "bin"),
    (1001, "bin bir"),
    (2026, "iki bin yirmi altı"),
    (5200, "beş bin iki yüz"),
    (12550, "on iki bin beş yüz elli"),
    (100000, "yüz bin"),
    (1250000, "bir milyon iki yüz elli bin"),
    (2000000000, "iki milyar"),
])
def test_number_to_words(n, expected):
    assert number_to_words(n) == expected


@pytest.mark.parametrize("base, suffix, possessive, expected", [
    ("yirmi", "dir", False, "yirmidir"),
    ("otuz", "da", False, "otuzda"),
    ("beş", "te", False, "beşte"),
    ("beş", "e", False, "beşe"),
    ("iki", "ye", False, "ikiye"),
    ("iki", "nin", False, "ikinin"),
    ("altı", "dır", False, "altıdır"),
    ("kuruş", "tir", False, "kuruştur"),        # suffix written for '75'tir'
    ("Türk lirası", "dir", True, "Türk lirasıdır"),
    ("Türk lirası", "ye", True, "Türk lirasına"),
    ("Türk lirası", "de", True, "Türk lirasında"),
    ("Türk lirası", "den", True, "Türk lirasından"),
    ("Türk lirası", "nin", True, "Türk lirasının"),
    ("Türk lirası", "yi", True, "Türk lirasını"),
    ("ka de ve", "yi", False, "ka de veyi"),
    ("ka de ve", "nin", False, "ka de venin"),
    ("yirmi", "", False, "yirmi"),
])
def test_attach_suffix(base, suffix, possessive, expected):
    assert attach_suffix(base, suffix, possessive) == expected


@pytest.mark.parametrize("text, expected", [
    # spec example
    ("KDV %20'dir. Toplam borç ₺12.550,75'tir.",
     "Ka de ve yüzde yirmidir. Toplam borç on iki bin beş yüz elli Türk lirası yetmiş beş kuruştur."),
    # currency
    ("Öğrencinin kalan borcu 5.200 TL'dir.", "Öğrencinin kalan borcu beş bin iki yüz Türk lirasıdır."),
    ("Fatura tutarı 1.250.000 TL olarak hesaplandı.",
     "Fatura tutarı bir milyon iki yüz elli bin Türk lirası olarak hesaplandı."),
    ("Tutar 100 TL'ye düşürüldü.", "Tutar yüz Türk lirasına düşürüldü."),
    ("Tutar ₺250 oldu.", "Tutar iki yüz elli Türk lirası oldu."),
    ("Tutar 99,90 ₺.", "Tutar doksan dokuz Türk lirası doksan kuruş."),
    ("Tutar 10,00 TL.", "Tutar on Türk lirası."),
    ("Tutar 0,5 TL.", "Tutar sıfır Türk lirası elli kuruş."),
    ("TL cinsinden girilmelidir.", "Türk lirası cinsinden girilmelidir."),
    # percent
    ("Tahsilat oranı %87,5 olarak gerçekleşti.", "Tahsilat oranı yüzde seksen yedi virgül beş olarak gerçekleşti."),
    ("İndirim % 15'tir.", "İndirim yüzde on beştir."),
    # dates
    ("Ödeme tarihi 30.09.2026'dır.", "Ödeme tarihi otuz Eylül iki bin yirmi altıdır."),
    ("Tarih: 01/01/2027", "Tarih: bir Ocak iki bin yirmi yedi"),
    ("Son ödeme tarihi 15 Ekim 2026'dır.", "Son ödeme tarihi on beş Ekim iki bin yirmi altıdır."),
    ("Geçersiz 45.13.2026 tarihi.", "Geçersiz kırk beş on üç iki bin yirmi altı tarihi."),  # not a calendar date
    # times
    ("Toplantı saat 14:30'da başlayacaktır.", "Toplantı saat on dört otuzda başlayacaktır."),
    ("Saat 09:05'te", "Saat dokuz sıfır beşte"),
    ("Saat 18:00", "Saat on sekiz"),
    # decimals / plain numbers
    ("Oran 3,05 oldu.", "Oran üç virgül sıfır beş oldu."),
    ("2026'da 3 şube açıldı.", "İki bin yirmi altıda üç şube açıldı."),
    ("Kod 0006 girildi.", "Kod sıfır sıfır sıfır altı girildi."),
    ("Müşteri no 12345678.", "Müşteri no bir iki üç dört beş altı yedi sekiz."),
    # IBAN
    ("IBAN numarası TR12 0006 1005 1978 6457 8413 26 olarak kaydedildi.",
     "İban numarası te re, on iki, sıfır sıfır sıfır altı, bir sıfır sıfır beş, bir dokuz yedi sekiz, "
     "altı dört beş yedi, sekiz dört bir üç, iki altı olarak kaydedildi."),
    ("IBAN bilgilerini kontrol ediniz.", "İban bilgilerini kontrol ediniz."),
    # abbreviations
    ("CRM modülünde yeni bir kayıt var.", "Si ar em modülünde yeni bir kayıt var."),
    ("ERP sistemi üzerinden cari hesap seçebilirsiniz.", "E re pe sistemi üzerinden cari hesap seçebilirsiniz."),
    ("KDV'yi ve SGK'nın payını hesaplayın.", "Ka de veyi ve se ge kanın payını hesaplayın."),
    ("e-Fatura ve e-Arşiv başarıyla oluşturuldu.", "E fatura ve e arşiv başarıyla oluşturuldu."),
    ("Faturalar, raporlar vb. gönderildi.", "Faturalar, raporlar ve benzeri gönderildi."),
    # foreign words
    ("Dashboard üzerindeki KPI değerleri", "Deşbord üzerindeki ke pe i değerleri"),
    ("Raporu Excel formatında export edebilirsiniz.", "Raporu Eksel formatında eksport edebilirsiniz."),
    # untouched text
    ("Yeni öğrenci kaydı oluşturabilirsiniz.", "Yeni öğrenci kaydı oluşturabilirsiniz."),
    ("Ankara'daki şubeye gönderildi.", "Ankara'daki şubeye gönderildi."),
    ("  Fazla   boşluk  ", "Fazla boşluk"),
])
def test_normalize(text, expected):
    assert normalize(text) == expected


def test_curly_apostrophe():
    assert normalize("KDV %20’dir.") == "Ka de ve yüzde yirmidir."


def test_words_containing_abbreviation_untouched():
    assert normalize("KDVSİZ ve TLX kodları") == "KDVSİZ ve TLX kodları"


def test_deterministic():
    text = "KDV %20'dir. Toplam borç ₺12.550,75'tir. Tarih 30.09.2026."
    assert len({normalize(text) for _ in range(5)}) == 1


def test_original_not_modified():
    original = "Toplam borç ₺12.550,75'tir."
    copy = str(original)
    normalize(original)
    assert original == copy


def test_idempotent_on_normalized_output():
    once = normalize("KDV %20'dir. Toplam borç ₺12.550,75'tir. Saat 14:30'da.")
    assert normalize(once) == once


def test_sentence_start_capitalized_after_expansion():
    assert normalize("Tamam. KDV %20'dir. ERP açık.") == "Tamam. Ka de ve yüzde yirmidir. E re pe açık."
